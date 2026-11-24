"""Vector search over chunks with pgvector."""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings import Embedder
from app.models import Chunk, Document


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    page_number: int
    chunk_index: int
    content: str
    score: float  # cosine similarity, 1.0 = identical direction


def search(
    session: Session,
    embedder: Embedder,
    query: str,
    top_k: int,
    filename: str | None = None,
) -> list[RetrievedChunk]:
    query_vector = embedder.embed_query(query)

    # HNSW is approximate: ef_search is the size of the candidate list it explores.
    # Higher = better recall, slower. It must be at least top_k to return top_k rows.
    session.execute(select(func.set_config("hnsw.ef_search", str(max(40, 2 * top_k)), True)))
    if filename is not None:
        # With a WHERE clause the index may find too few matching rows; pgvector >= 0.8
        # can keep scanning until it has enough ("iterative index scans").
        session.execute(select(func.set_config("hnsw.iterative_scan", "relaxed_order", True)))

    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(Chunk, Document.filename, distance)
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance)
        .limit(top_k)
    )
    if filename is not None:
        stmt = stmt.where(Document.filename == filename)

    rows = session.execute(stmt).all()
    results = [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=doc_filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=round(1.0 - float(dist), 4),
        )
        for chunk, doc_filename, dist in rows
    ]
    session.rollback()  # end the read transaction (also resets the SET LOCAL values)
    return results
