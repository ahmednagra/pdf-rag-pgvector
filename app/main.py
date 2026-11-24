"""HTTP API."""

import logging
import os
from dataclasses import asdict
from typing import Annotated

import openai
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from app.answer import answer_question
from app.chat import run_chat
from app.config import Settings, get_settings
from app.db import get_session
from app.embeddings import Embedder, get_embedder
from app.ingest import ingest_pdf
from app.llm import LLM, get_llm
from app.models import Chunk, Document
from app.pdf import InvalidPDFError
from app.retrieval import search
from app.schemas import (
    AnswerResponse,
    AskRequest,
    ChatResponse,
    DocumentOut,
    IngestResponse,
    SearchHit,
    SearchRequest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF RAG with pgvector",
    description="Ingest PDFs, search them semantically, and get cited answers.",
    version="0.1.0",
)

SessionDep = Annotated[Session, Depends(get_session)]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
LLMDep = Annotated[LLM | None, Depends(get_llm)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@app.exception_handler(openai.APIError)
def _provider_error(_request: Request, exc: openai.APIError) -> JSONResponse:
    # Log the provider's error type, never the request (it could contain document text).
    logger.warning("LLM/embedding provider error: %s", type(exc).__name__)
    return JSONResponse(status_code=502, content={"detail": "upstream model provider error"})


@app.get("/health")
def health(session: SessionDep) -> dict[str, str]:
    session.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse)
def upload_document(
    session: SessionDep,
    embedder: EmbedderDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="A text-based PDF")],
) -> IngestResponse:
    filename = os.path.basename(file.filename or "")[:255]
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "filename must end in .pdf")
    data = file.file.read(settings.max_upload_bytes + 1)  # bounded read
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "file too large")
    try:
        result = ingest_pdf(session, embedder, settings, filename, data)
    except InvalidPDFError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return IngestResponse(**asdict(result))


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(session: SessionDep) -> list[DocumentOut]:
    rows = session.execute(
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.filename)
    ).all()
    return [
        DocumentOut(
            id=doc.id,
            filename=doc.filename,
            page_count=doc.page_count,
            chunk_count=count,
            embedding_model=doc.embedding_model,
        )
        for doc, count in rows
    ]


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, session: SessionDep) -> None:
    deleted = session.execute(delete(Document).where(Document.id == document_id)).rowcount
    session.commit()
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")


@app.post("/search", response_model=list[SearchHit])
def search_chunks(
    body: SearchRequest, session: SessionDep, embedder: EmbedderDep, settings: SettingsDep
) -> list[SearchHit]:
    hits = search(
        session, embedder, body.query, body.top_k or settings.default_top_k, body.filename
    )
    return [SearchHit(**asdict(hit)) for hit in hits]


@app.post("/answer", response_model=AnswerResponse)
def answer(
    body: AskRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    settings: SettingsDep,
) -> AnswerResponse:
    result = answer_question(
        session, embedder, llm, body.question, body.top_k or settings.default_top_k
    )
    return AnswerResponse(
        answer=result.answer,
        citations=[asdict(c) for c in result.citations],
        sources=[SearchHit(**asdict(s)) for s in result.sources],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    body: AskRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    llm: LLMDep,
    settings: SettingsDep,
) -> ChatResponse:
    if llm is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "tool calling needs a real LLM: set LLM_PROVIDER and LLM_API_KEY",
        )
    result = run_chat(session, embedder, llm, body.question, settings.max_tool_rounds)
    return ChatResponse(
        answer=result.answer,
        citations=[asdict(c) for c in result.citations],
        tool_trace=[asdict(t) for t in result.tool_trace],
    )
