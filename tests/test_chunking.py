import pytest

from app.chunking import chunk_pages, normalize, split_text


def test_normalize_joins_wrapped_lines_and_hyphenation():
    assert (
        normalize("The bat-\ntery is\n  full.\n\nNext para.")
        == "The battery is full.\n\nNext para."
    )


def test_chunks_respect_size_and_keep_every_sentence():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(40)]
    chunks = split_text(" ".join(sentences), chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    for sentence in sentences:
        assert any(sentence in c for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"Fact {i} is here." for i in range(30))
    chunks = split_text(text, chunk_size=100, chunk_overlap=30)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_sentence = previous.rsplit(". ", 1)[-1]
        assert current.startswith(last_sentence)


def test_overlong_sentence_and_word_are_split():
    text = "word " * 100 + "x" * 250
    chunks = split_text(text, chunk_size=100, chunk_overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(c for c in chunks if set(c) == {"x"}) == "x" * 250


def test_empty_text_gives_no_chunks():
    assert split_text("   \n\n ", chunk_size=100, chunk_overlap=10) == []


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (0, 0), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", chunk_size=size, chunk_overlap=overlap)


def test_chunk_pages_tracks_page_numbers():
    chunks = chunk_pages(["First page.", "", "Third page."], chunk_size=100, chunk_overlap=10)
    assert [(c.page_number, c.text) for c in chunks] == [(1, "First page."), (3, "Third page.")]
