from sqlalchemy import text

from app.ingest import ingest_pdf
from app.retrieval import search
from tests.pdfs import make_pdf


def ingest_two_docs(session, embedder, settings):
    ingest_pdf(
        session,
        embedder,
        settings,
        "kettle.pdf",
        make_pdf(
            [
                "The kettle battery charges in four hours of full sun.",
                "Descale the kettle monthly with citric acid.",
            ]
        ),
    )
    ingest_pdf(
        session,
        embedder,
        settings,
        "library.pdf",
        make_pdf(["Library books can be borrowed for twenty one days."]),
    )


def test_most_relevant_chunk_ranks_first(session, embedder, settings):
    ingest_two_docs(session, embedder, settings)
    hits = search(session, embedder, "how do I descale the kettle with citric acid", top_k=3)
    assert (hits[0].filename, hits[0].page_number) == ("kettle.pdf", 2)
    assert hits[0].score >= hits[-1].score


def test_filename_filter(session, embedder, settings):
    ingest_two_docs(session, embedder, settings)
    hits = search(session, embedder, "kettle battery", top_k=5, filename="library.pdf")
    assert {h.filename for h in hits} == {"library.pdf"}


def test_top_k_limits_results(session, embedder, settings):
    ingest_two_docs(session, embedder, settings)
    assert len(search(session, embedder, "kettle", top_k=1)) == 1


def test_hnsw_index_exists(session):
    indexdef = session.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_embedding_hnsw'")
    ).scalar_one()
    assert "hnsw" in indexdef and "vector_cosine_ops" in indexdef


def test_planner_uses_hnsw_index_for_vector_search(session):
    # With few rows Postgres would rather scan the table, so disable that for this check.
    session.execute(text("SET LOCAL enable_seqscan = off"))
    vector = "[" + ",".join(["0.1"] * 384) + "]"
    plan = (
        session.execute(
            text("EXPLAIN SELECT id FROM chunks ORDER BY embedding <=> CAST(:v AS vector) LIMIT 5"),
            {"v": vector},
        )
        .scalars()
        .all()
    )
    assert any("ix_chunks_embedding_hnsw" in line for line in plan)
