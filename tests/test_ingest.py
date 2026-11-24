import threading

import pytest
from sqlalchemy import func, select

from app.ingest import ingest_pdf
from app.models import Chunk, Document
from app.pdf import InvalidPDFError
from tests.pdfs import make_pdf

PAGES = [
    "The Solar Kettle K2 boils one litre of water in nine minutes. " * 3,
    "Error code E3 means the water level is below the MIN line. " * 3,
]


def chunk_rows(session):
    return session.execute(select(Chunk.id, Chunk.page_number).order_by(Chunk.id)).all()


def test_ingest_creates_document_and_chunks(session, embedder, settings):
    result = ingest_pdf(session, embedder, settings, "kettle.pdf", make_pdf(PAGES))
    assert result.status == "created"
    assert result.page_count == 2
    assert result.chunk_count == len(chunk_rows(session)) > 0
    assert {page for _, page in chunk_rows(session)} == {1, 2}


def test_reingesting_identical_file_is_a_noop(session, embedder, settings):
    data = make_pdf(PAGES)
    first = ingest_pdf(session, embedder, settings, "kettle.pdf", data)
    before = chunk_rows(session)
    second = ingest_pdf(session, embedder, settings, "kettle.pdf", data)
    assert second.status == "unchanged"
    assert second.document_id == first.document_id
    assert chunk_rows(session) == before  # same rows, same ids: nothing was rewritten


def test_changed_file_replaces_chunks(session, embedder, settings):
    first = ingest_pdf(session, embedder, settings, "kettle.pdf", make_pdf(PAGES))
    old_ids = {cid for cid, _ in chunk_rows(session)}
    second = ingest_pdf(session, embedder, settings, "kettle.pdf", make_pdf(PAGES[:1]))
    assert second.status == "updated"
    assert second.document_id == first.document_id
    new = chunk_rows(session)
    assert old_ids.isdisjoint(cid for cid, _ in new)
    assert {page for _, page in new} == {1}
    assert session.scalar(select(func.count()).select_from(Document)) == 1


def test_concurrent_uploads_of_same_file_create_one_document(session_factory, embedder, settings):
    data = make_pdf(PAGES)
    statuses: list[str] = []

    def upload():
        with session_factory() as s:
            statuses.append(ingest_pdf(s, embedder, settings, "kettle.pdf", data).status)

    threads = [threading.Thread(target=upload) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert statuses.count("created") == 1
    assert set(statuses) <= {"created", "unchanged"}
    with session_factory() as s:
        assert s.scalar(select(func.count()).select_from(Document)) == 1
        counts = s.execute(
            select(Chunk.chunk_index, func.count()).group_by(Chunk.chunk_index)
        ).all()
        assert all(n == 1 for _, n in counts)


def test_rejects_non_pdf(session, embedder, settings):
    with pytest.raises(InvalidPDFError):
        ingest_pdf(session, embedder, settings, "notes.pdf", b"hello, I am not a PDF")


def test_rejects_pdf_without_text(session, embedder, settings):
    with pytest.raises(InvalidPDFError, match="no extractable text"):
        ingest_pdf(session, embedder, settings, "blank.pdf", make_pdf([""]))


def test_rejects_too_many_pages(session, embedder, settings):
    settings.max_pdf_pages = 1
    with pytest.raises(InvalidPDFError, match="limit"):
        ingest_pdf(session, embedder, settings, "big.pdf", make_pdf(PAGES))
