"""One entry point for every supported file: detect, read, structure."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from margin.ingest import detect
from margin.ingest.images import OcrEngine, load_image
from margin.ingest.office import read_docx, read_pptx
from margin.ingest.pdf import read_pdf
from margin.ingest.sections import build_sections, split_blocks
from margin.ingest.types import Document, Section

HASH_CHUNK = 1024 * 1024
SUPPORTED = ("pdf", "docx", "pptx", "png", "jpeg", "webp", "text")


def content_id(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_document(path: Path, ocr: OcrEngine | None = None, use_ocr: bool = True, formulas: Any | None = None) -> Document:
    """`use_ocr=False` skips OCR for scanned PDF pages; images always need it. `formulas` reads drawn PDF equations."""
    if not path.is_file():
        raise FileNotFoundError(path)
    kind = detect.sniff(path)
    doc_id = content_id(path)
    ocr = ocr or OcrEngine()
    if kind == "pdf":
        return read_pdf(path, doc_id, ocr if use_ocr else None, formulas)
    if kind == "docx":
        return read_docx(path, doc_id)
    if kind == "pptx":
        return read_pptx(path, doc_id)
    if kind in ("png", "jpeg", "webp"):
        result = ocr.read(load_image(path))
        return Document(doc_id, str(path), kind, path.stem, 1, tuple(split_blocks(result.text, 1, "ocr")), ocr_pages=(1,))
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(doc_id, str(path), "text", path.stem, 0, tuple(split_blocks(text, None)))


def ingest_file(path: Path, ocr: OcrEngine | None = None, formulas: Any | None = None) -> tuple[Document, list[Section]]:
    doc = read_document(path, ocr, formulas=formulas)
    return doc, build_sections(doc)
