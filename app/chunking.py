"""Split page text into overlapping chunks that respect sentence boundaries.

Chunks never cross a page boundary, so every chunk has exactly one page number to cite.
"""

import re
from dataclasses import dataclass

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class TextChunk:
    page_number: int  # 1-based
    text: str


def normalize(text: str) -> str:
    """Undo PDF line wrapping: join hyphenated breaks, collapse whitespace, keep paragraphs."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    paragraphs = [" ".join(p.split()) for p in _PARAGRAPH_BREAK.split(text)]
    return "\n\n".join(p for p in paragraphs if p)


def _units(text: str, chunk_size: int) -> list[str]:
    """Sentences, with any sentence longer than chunk_size split on words (or characters)."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        for sentence in _SENTENCE_END.split(paragraph.strip()):
            if not sentence:
                continue
            if len(sentence) <= chunk_size:
                units.append(sentence)
                continue
            piece = ""
            for word in sentence.split():
                while len(word) > chunk_size:  # pathological "word" (e.g. a long URL)
                    if piece:
                        units.append(piece)
                        piece = ""
                    units.append(word[:chunk_size])
                    word = word[chunk_size:]
                candidate = f"{piece} {word}" if piece else word
                if len(candidate) > chunk_size:
                    units.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                units.append(piece)
    return units


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack sentences into chunks of at most chunk_size characters.

    Each new chunk starts with the trailing sentences of the previous chunk (up to
    chunk_overlap characters) so a fact that straddles a boundary is not lost.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("need chunk_size > 0 and 0 <= chunk_overlap < chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    for unit in _units(normalize(text), chunk_size):
        if current and len(" ".join([*current, unit])) > chunk_size:
            chunks.append(" ".join(current))
            carry: list[str] = []
            for previous in reversed(current):
                if len(" ".join([previous, *carry])) > chunk_overlap:
                    break
                carry.insert(0, previous)
            # Drop the overlap if it would not leave room for the new sentence.
            current = carry if len(" ".join([*carry, unit])) <= chunk_size else []
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[str], chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    return [
        TextChunk(page_number=page_number, text=chunk)
        for page_number, page_text in enumerate(pages, start=1)
        for chunk in split_text(page_text, chunk_size, chunk_overlap)
    ]
