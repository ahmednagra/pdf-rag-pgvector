"""Evaluation metrics, kept free of I/O so they are unit-tested (see tests/test_evaluation.py).

Groundedness here is a transparent *lexical* proxy, not an LLM judge: a sentence of the answer
counts as supported when most of its content words appear in the passages it cites. It catches
invented numbers and names; it cannot catch a paraphrase that changes the meaning.
"""

from dataclasses import dataclass

from app.text import CITATION, content_tokens, sentences

SUPPORT_THRESHOLD = 0.6


def retrieval_rank(results: list[dict], expected_file: str, expected_page: int) -> int | None:
    """1-based rank of the first result from the expected file and page, or None."""
    for rank, result in enumerate(results, start=1):
        if result["filename"] == expected_file and result["page_number"] == expected_page:
            return rank
    return None


@dataclass
class Groundedness:
    sentences: int
    supported: int
    cited: int

    @property
    def score(self) -> float:
        return self.supported / self.sentences if self.sentences else 1.0

    @property
    def citation_rate(self) -> float:
        return self.cited / self.sentences if self.sentences else 1.0


def groundedness(answer: str, cited_passages: dict[int, str]) -> Groundedness:
    """cited_passages maps source number -> passage text for the citations in the answer."""
    total = supported = cited = 0
    for sentence in sentences(answer):
        numbers = [int(n) for n in CITATION.findall(sentence) if int(n) in cited_passages]
        words = content_tokens(CITATION.sub("", sentence))
        if not words:
            continue
        total += 1
        if not numbers:
            continue
        cited += 1
        evidence = set().union(*(content_tokens(cited_passages[n]) for n in numbers))
        if len(words & evidence) / len(words) >= SUPPORT_THRESHOLD:
            supported += 1
    return Groundedness(total, supported, cited)


def is_abstention(answer: str) -> bool:
    return "i don't know" in answer.lower() or "i do not know" in answer.lower()
