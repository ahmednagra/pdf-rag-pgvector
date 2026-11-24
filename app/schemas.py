"""Request/response models for the HTTP API."""

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    page_count: int
    chunk_count: int


class DocumentOut(BaseModel):
    id: int
    filename: str
    page_count: int
    chunk_count: int
    embedding_model: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filename: str | None = None


class SearchHit(BaseModel):
    chunk_id: int
    filename: str
    page_number: int
    score: float
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class CitationOut(BaseModel):
    source: int
    chunk_id: int
    filename: str
    page_number: int
    snippet: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    sources: list[SearchHit]


class ToolTraceOut(BaseModel):
    tool: str
    arguments: dict
    result_summary: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    tool_trace: list[ToolTraceOut]
