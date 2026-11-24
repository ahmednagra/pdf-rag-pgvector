from app.evaluation import groundedness, is_abstention, retrieval_rank


def test_retrieval_rank():
    results = [
        {"filename": "a.pdf", "page_number": 1},
        {"filename": "b.pdf", "page_number": 2},
    ]
    assert retrieval_rank(results, "b.pdf", 2) == 2
    assert retrieval_rank(results, "b.pdf", 3) is None


def test_groundedness_supported_and_unsupported_sentences():
    passages = {1: "The battery charges in about 4 hours in full sun."}
    good = groundedness("The battery charges in about 4 hours [1].", passages)
    assert (good.sentences, good.supported, good.cited) == (1, 1, 1)

    invented = groundedness("The kettle is painted bright purple with gold stripes [1].", passages)
    assert invented.supported == 0 and invented.cited == 1

    uncited = groundedness("The battery charges in about 4 hours.", passages)
    assert uncited.score == 0.0 and uncited.citation_rate == 0.0


def test_groundedness_ignores_citations_to_unknown_sources():
    result = groundedness("The battery charges in 4 hours [7].", {1: "battery charges 4 hours"})
    assert result.cited == 0


def test_is_abstention():
    assert is_abstention("I don't know based on the provided documents.")
    assert not is_abstention("It takes 4 hours [1].")


def test_citation_after_full_stop_belongs_to_the_previous_sentence():
    passages = {1: "The battery charges in about 4 hours.", 2: "Descale monthly with citric acid."}
    result = groundedness(
        "It charges in 4 hours. [1] Descale monthly with citric acid. [2]", passages
    )
    assert (result.sentences, result.cited, result.supported) == (2, 2, 2)
