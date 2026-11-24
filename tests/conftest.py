import os
from collections.abc import Iterator

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from alembic import command
from app.config import Settings, get_settings
from app.db import get_session
from app.embeddings import get_embedder
from app.llm import get_llm
from app.main import app
from tests.fakes import HashEmbedder

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5433/rag_test"
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    """A migrated test database. Runs downgrade -> upgrade, so the migration is tested too."""
    engine = create_engine(TEST_DATABASE_URL)
    try:
        engine.connect().close()
    except OperationalError:
        pytest.skip(f"test database not reachable at {TEST_DATABASE_URL} (docker compose up db)")
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE documents, chunks RESTART IDENTITY CASCADE"))
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as session:
        yield session


@pytest.fixture
def settings() -> Settings:
    return Settings(chunk_size=300, chunk_overlap=50, database_url=TEST_DATABASE_URL)


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def make_client(session_factory, embedder, settings):
    """Build a TestClient with a given LLM (None = extractive baseline)."""

    def _session():
        with session_factory() as session:
            yield session

    def build(llm=None) -> TestClient:
        app.dependency_overrides[get_session] = _session
        app.dependency_overrides[get_embedder] = lambda: embedder
        app.dependency_overrides[get_llm] = lambda: llm
        app.dependency_overrides[get_settings] = lambda: settings
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()
