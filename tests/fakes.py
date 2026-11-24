"""Deterministic stand-ins for the embedding model and the LLM, so tests need no network."""

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

from app.config import EMBEDDING_DIM
from app.llm import LLMResponse, ToolCall
from app.text import content_tokens


class HashEmbedder:
    """Bag-of-words hashed into EMBEDDING_DIM buckets, L2-normalised.

    Texts sharing words get similar vectors, which is all the retrieval tests need.
    """

    model_name = "test:hash-bow"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIM
        for token in content_tokens(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % EMBEDDING_DIM
            vector[bucket] += 1.0
        if not any(vector):
            vector[0] = 1.0  # cosine distance is undefined for a zero vector
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLM:
    """Returns scripted responses in order and records every request it received."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "tool_choice": tool_choice}
        )
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


def tool_call(name: str, call_id: str = "call_1", **arguments: Any) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(call_id, name, arguments, json.dumps(arguments))],
    )
