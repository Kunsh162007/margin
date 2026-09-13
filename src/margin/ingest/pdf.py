"""PDF reading with pdfium (BSD/Apache licensed, unlike AGPL PyMuPDF).

A page whose text layer holds fewer than ``MIN_TEXT_CHARS`` characters is
treated as scanned and sent to OCR. The threshold is low on purpose: a real
text page carries hundreds of characters, and a scanned one usually carries
none or only a page number.
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

from margin.ingest.images import OcrEngine
from margin.ingest.sections import split_blocks
from margin.ingest.types import Block, Document, TocEntry

MIN_TEXT_CHARS = 40
RENDER_SCALE = 2.0  # 144 dpi: enough for OCR of body text
MAX_TOC_DEPTH = 4


def _toc(pdf: pdfium.PdfDocument) -> tuple[TocEntry, ...]:
    entries = []
    for bookmark in pdf.get_toc(max_depth=MAX_TOC_DEPTH):
        title = (bookmark.get_title() or "").strip()
        if not title:
            continue
        dest = bookmark.get_dest()
        index = dest.get_index() if dest is not None else None
        entries.append(TocEntry(title=title, level=bookmark.level + 1, page=index + 1 if index is not None else None))
    return tuple(entries)


def _page_text(page: pdfium.PdfPage) -> str:
    textpage = page.get_textpage()
    try:
        return textpage.get_text_bounded()
    finally:
        textpage.close()


def read_pdf(path: Path, doc_id: str, ocr: OcrEngine | None) -> Document:
    pdf = pdfium.PdfDocument(str(path))
    try:
        blocks: list[Block] = []
        ocr_pages: list[int] = []
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                text = _page_text(page)
                if len(text.strip()) < MIN_TEXT_CHARS and ocr is not None:
                    text = ocr.read(page.render(scale=RENDER_SCALE).to_pil()).text
                    ocr_pages.append(index + 1)
                    blocks.extend(split_blocks(text, index + 1, "ocr"))
                else:
                    blocks.extend(split_blocks(text, index + 1))
            finally:
                page.close()
        title = (pdf.get_metadata_dict().get("Title") or "").strip() or path.stem
        return Document(doc_id, str(path), "pdf", title, len(pdf), tuple(blocks), _toc(pdf), tuple(ocr_pages))
    finally:
        pdf.close()
