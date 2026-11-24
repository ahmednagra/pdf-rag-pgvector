"""Build small PDFs in memory for tests."""

from fpdf import FPDF


def make_pdf(pages: list[str]) -> bytes:
    pdf = FPDF()
    for body in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        if body:
            pdf.multi_cell(0, 6, body)
    return bytes(pdf.output())
