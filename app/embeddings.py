"""Embedding providers. All of them return EMBEDDING_DIM-dimensional vectors."""

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from app.config import EMBEDDING_DIM, Settings, get_settings


class Embedder(Protocol):
    model_name: str  # stored on each document, e.g. "fastembed:BAAI/bge-small-en-v1.5"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _check_dims(vectors: list[list[float]]) -> list[list[float]]:
    for vector in vectors:
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(f"embedding has {len(vector)} dims, expected {EMBEDDING_DIM}")
    return vectors


class FastEmbedEmbedder:
    """Local ONNX model (no API key). bge-small-en-v1.5: 384 dims, 512-token input limit."""

    def __init__(self, model: str, batch_size: int, cache_dir: str | None) -> None:
        from fastembed import TextEmbedding  # imported lazily: loading the model takes seconds

        self.model_name = f"fastembed:{model}"
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.embed(list(texts), batch_size=self._batch_size)
        return _check_dims([v.tolist() for v in vectors])

    def embed_query(self, text: str) -> list[float]:
        # bge models expect queries and passages to be embedded slightly differently.
        return _check_dims([v.tolist() for v in self._model.query_embed(text)])[0]


class OpenAIEmbedder:
    """OpenAI text-embedding-3-*, truncated to EMBEDDING_DIM via the `dimensions` parameter."""

    def __init__(self, model: str, api_key: str, batch_size: int) -> None:
        from openai import OpenAI

        self.model_name = f"openai:{model}:{EMBEDDING_DIM}"
        self._client = OpenAI(api_key=api_key, timeout=30.0, max_retries=3)
        self._model = model
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            response = self._client.embeddings.create(
                model=self._model,
                input=list(texts[start : start + self._batch_size]),
                dimensions=EMBEDDING_DIM,
            )
            vectors.extend(item.embedding for item in response.data)
        return _check_dims(vectors)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        if settings.embedding_api_key is None:
            raise RuntimeError("EMBEDDING_API_KEY is required for EMBEDDING_PROVIDER=openai")
        return OpenAIEmbedder(
            settings.embedding_model,
            settings.embedding_api_key.get_secret_value(),
            settings.embedding_batch_size,
        )
    return FastEmbedEmbedder(
        settings.embedding_model, settings.embedding_batch_size, settings.model_cache_dir
    )


@lru_cache
def get_embedder() -> Embedder:
    return build_embedder(get_settings())
