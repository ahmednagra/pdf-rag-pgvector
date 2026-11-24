"""Application settings, loaded from environment variables (and `.env` for local runs)."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The vector column is declared as vector(384) in the migration. Every embedding provider
# we support is configured to produce exactly this many dimensions.
EMBEDDING_DIM = 384


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5433/rag"

    # Embeddings. "fastembed" runs a small ONNX model locally, so no API key is needed.
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_api_key: SecretStr | None = None
    embedding_batch_size: int = 64
    model_cache_dir: str | None = None

    # Answer generation. "extractive" is an offline baseline (no LLM, no key).
    llm_provider: Literal["extractive", "openai", "gemini", "anthropic"] = "extractive"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None  # override the provider's OpenAI-compatible endpoint
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_max_output_tokens: int = 600
    # None = provider default: 0 for OpenAI/Gemini; omitted for Anthropic, whose recent models
    # reject temperature < 1 on the OpenAI-compatible endpoint.
    llm_temperature: float | None = None

    # Chunking (characters). See README "Chunking" for the reasoning.
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Request bounds.
    max_upload_bytes: int = 20 * 1024 * 1024
    max_pdf_pages: int = 300
    default_top_k: int = 5
    max_top_k: int = 20
    max_tool_rounds: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
