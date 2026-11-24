"""Tool-calling ("agentic") RAG: the model decides what to search for, and how often.

/answer always retrieves once with the user's exact question. /chat instead gives the model
tools, so it can rephrase the query, search several times (e.g. for a comparison question),
or list the available documents first. The loop is bounded by MAX_TOOL_ROUNDS.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.answer import NO_ANSWER, SYSTEM_PROMPT, Citation, resolve_citations
from app.embeddings import Embedder
from app.llm import LLM, ToolCall
from app.models import Chunk, Document
from app.retrieval import RetrievedChunk, search

MAX_SEARCH_TOP_K = 8
MAX_QUERY_CHARS = 500

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Semantic search over the ingested PDFs. Returns numbered passages; cite them "
                "in your answer by their source number, e.g. [3]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "filename": {
                        "type": "string",
                        "description": "Optional: only search this document (from list_documents).",
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_TOP_K},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the ingested documents with their page counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

CHAT_PROMPT = (
    SYSTEM_PROMPT
    + "\n- Use the search_documents tool to find sources before answering. You may search more"
    " than once with different queries."
)


@dataclass
class ToolTraceEntry:
    tool: str
    arguments: dict[str, Any]
    result_summary: str


@dataclass
class ChatResult:
    answer: str
    citations: list[Citation]
    tool_trace: list[ToolTraceEntry] = field(default_factory=list)


class _ToolRunner:
    """Executes tool calls and keeps one global numbering of every passage the model has seen."""

    def __init__(self, session: Session, embedder: Embedder) -> None:
        self.session = session
        self.embedder = embedder
        self.sources: list[RetrievedChunk] = []
        self._number_of: dict[int, int] = {}  # chunk_id -> source number (1-based)

    def run(self, call: ToolCall) -> tuple[str, str]:
        """Return (content for the model, short summary for the trace)."""
        if call.name == "search_documents":
            return self._search(call.arguments)
        if call.name == "list_documents":
            return self._list_documents()
        return json.dumps({"error": f"unknown tool {call.name!r}"}), "unknown tool"

    def _search(self, args: dict[str, Any]) -> tuple[str, str]:
        query = str(args.get("query", "")).strip()[:MAX_QUERY_CHARS]
        if not query:
            return json.dumps({"error": "query must be a non-empty string"}), "invalid arguments"
        try:
            top_k = min(max(int(args.get("top_k", 5)), 1), MAX_SEARCH_TOP_K)
        except (TypeError, ValueError):
            top_k = 5
        filename = args.get("filename") or None
        results = search(self.session, self.embedder, query, top_k, filename=filename)

        passages = []
        for chunk in results:
            if chunk.chunk_id not in self._number_of:
                self.sources.append(chunk)
                self._number_of[chunk.chunk_id] = len(self.sources)
            passages.append(
                {
                    "source": self._number_of[chunk.chunk_id],
                    "filename": chunk.filename,
                    "page": chunk.page_number,
                    "text": chunk.content,
                }
            )
        return json.dumps({"results": passages}), f"{len(passages)} passages"

    def _list_documents(self) -> tuple[str, str]:
        rows = self.session.execute(
            select(Document.filename, Document.page_count, func.count(Chunk.id))
            .outerjoin(Chunk, Chunk.document_id == Document.id)
            .group_by(Document.id)
            .order_by(Document.filename)
        ).all()
        self.session.rollback()
        docs = [{"filename": f, "pages": p, "chunks": c} for f, p, c in rows]
        return json.dumps({"documents": docs}), f"{len(docs)} documents"


def run_chat(
    session: Session, embedder: Embedder, llm: LLM, question: str, max_rounds: int
) -> ChatResult:
    runner = _ToolRunner(session, embedder)
    trace: list[ToolTraceEntry] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": CHAT_PROMPT},
        {"role": "user", "content": question},
    ]

    answer = ""
    for round_number in range(max_rounds + 1):
        # The last round sets tool_choice="none", forcing a final text answer. (Tools stay
        # declared because some providers reject tool messages in history without them.)
        final_round = round_number == max_rounds
        if final_round:
            # Not every provider documents tool_choice="none", so also ask in plain words.
            messages.append(
                {"role": "user", "content": "Answer now from the sources above. Do not call tools."}
            )
        response = llm.chat(messages, tools=TOOLS, tool_choice="none" if final_round else None)
        if final_round or not response.tool_calls:
            answer = (response.content or "").strip()
            break
        messages.append(response.as_message())
        for call in response.tool_calls:
            content, summary = runner.run(call)
            trace.append(ToolTraceEntry(call.name, call.arguments, summary))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    answer = answer or NO_ANSWER
    return ChatResult(answer, resolve_citations(answer, runner.sources), trace)
