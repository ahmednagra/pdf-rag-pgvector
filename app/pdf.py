"""PDF text extraction."""

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class InvalidPDFError(ValueError):
    """The upload is not a readable, text-based PDF."""


def extract_pages(data: bytes, max_pages: int) -> list[str]:
    """Return the text of each page (index 0 = page 1). Raises InvalidPDFError."""
    if not data.startswith(b"%PDF-"):
        raise InvalidPDFError("file is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise InvalidPDFError("encrypted PDFs are not supported")
        if len(reader.pages) > max_pages:
            raise InvalidPDFError(f"PDF has {len(reader.pages)} pages; the limit is {max_pages}")
        return [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as exc:
        raise InvalidPDFError(f"could not read PDF: {exc}") from exc
