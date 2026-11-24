"""PDF ingestion: extract -> chunk -> embed -> store, idempotently."""

import hashlib
import logging
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.chunking import chunk_pages
from app.config import Settings
from app.embeddings import Embedder
from app.models import Chunk, Document
from app.pdf import InvalidPDFError, extract_pages

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    document_id: int
    filename: str
    status: Literal["created", "updated", "unchanged"]
    page_count: int
    chunk_count: int


def _chunk_count(session: Session, document_id: int) -> int:
    return session.scalar(select(func.count()).where(Chunk.document_id == document_id)) or 0


def _unchanged(session: Session, filename: str, sha256: str, model: str) -> IngestResult | None:
    doc = session.scalar(select(Document).where(Document.filename == filename))
    if doc and doc.sha256 == sha256 and doc.embedding_model == model:
        return IngestResult(
            doc.id, filename, "unchanged", doc.page_count, _chunk_count(session, doc.id)
        )
    return None


def ingest_pdf(
    session: Session, embedder: Embedder, settings: Settings, filename: str, data: bytes
) -> IngestResult:
    """Ingest one PDF. Safe to call repeatedly and concurrently with the same file.

    * same filename + same bytes + same embedding model -> "unchanged", nothing written
    * same filename, different bytes (or model)       -> "updated", chunks replaced atomically
    * new filename                                     -> "created"
    """
    sha256 = hashlib.sha256(data).hexdigest()

    # Cheap pre-check so a re-upload doesn't pay for parsing and embedding.
    if result := _unchanged(session, filename, sha256, embedder.model_name):
        session.rollback()
        return result
    session.rollback()  # don't hold a transaction open during the slow work below

    pages = extract_pages(data, settings.max_pdf_pages)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise InvalidPDFError("no extractable text (scanned PDFs need OCR, which is out of scope)")
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])

    # Authoritative, race-safe section. Inserting first (ON CONFLICT DO NOTHING) and then
    # locking the row means two concurrent uploads of one filename are serialised by Postgres.
    values = {"sha256": sha256, "embedding_model": embedder.model_name, "page_count": len(pages)}
    document_id = session.scalar(
        pg_insert(Document)
        .values(filename=filename, **values)
        .on_conflict_do_nothing(index_elements=["filename"])
        .returning(Document.id)
    )
    if document_id is not None:
        status: Literal["created", "updated"] = "created"
    else:
        doc = session.scalars(
            select(Document).where(Document.filename == filename).with_for_update()
        ).one()
        if doc.sha256 == sha256 and doc.embedding_model == embedder.model_name:
            # A concurrent request ingested the same bytes first.
            result = IngestResult(
                doc.id, filename, "unchanged", doc.page_count, _chunk_count(session, doc.id)
            )
            session.rollback()
            return result
        document_id, status = doc.id, "updated"
        session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        for key, value in values.items():
            setattr(doc, key, value)

    session.execute(
        insert(Chunk),
        [
            {
                "document_id": document_id,
                "chunk_index": index,
                "page_number": chunk.page_number,
                "content": chunk.text,
                "embedding": vector,
            }
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ],
    )
    session.commit()
    logger.info("ingested %s: %s, %d chunks", filename, status, len(chunks))
    return IngestResult(document_id, filename, status, len(pages), len(chunks))
