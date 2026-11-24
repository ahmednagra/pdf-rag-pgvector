"""Tiny text helpers shared by the extractive baseline and the evaluation metrics."""

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does for from has have how i if in
    into is it its may must of on or our should so than that the their them then there
    these they this to too was we were what when where which who why will with would you
    your about after all also any before each more most no not only other over per such
    up out very""".split()
)
_WORD = re.compile(r"[a-z0-9]+(?:[.,:][0-9]+)*")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
CITATION = re.compile(r"\[(\d+)\]")


def content_tokens(text: str) -> set[str]:
    """Lower-cased words minus stopwords; numbers like 4,000 or 10:00 stay whole."""
    return {t for t in _WORD.findall(text.lower()) if t not in STOPWORDS and len(t) > 1}


_CITATION_AFTER_STOP = re.compile(r"([.!?])((?:\s*\[\d+\])+)")


def sentences(text: str) -> list[str]:
    """Split into sentences, keeping a trailing citation with its sentence.

    "Fact one. [1] Fact two [2]." is read as "Fact one [1]." and "Fact two [2]."
    """
    text = _CITATION_AFTER_STOP.sub(lambda m: f"{m.group(2)}{m.group(1)}", text.strip())
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
