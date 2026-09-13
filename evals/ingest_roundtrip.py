"""Round-trip real textbooks through ingestion and measure what survives.

    uv run python -m evals.ingest_roundtrip D:/corpus/anatomy-and-physiology-2e.pdf [...]

The books are not stored in the repository (size and licence); only the
measurements are. What is measured, per book:

* **anchored_rate** — share of PDF bookmarks whose title was found inside the
  text of their page, rather than falling back to the page's first block. Low
  values mean sections start at the wrong place.
* **empty_section_rate** — sections with no body text. A few are normal
  (a chapter heading immediately followed by its first subsection); many mean
  anchoring collapsed.
* **char_coverage** — characters that ended up inside some section, over all
  characters read. Anything below 1.0 was dropped.
* **text_page_rate** and **ocr_pages** — how much of the book had a text layer.
* **pages_per_s** — reading speed on this machine, without OCR.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from margin.ingest.pipeline import read_document
from margin.ingest.sections import build_sections
from margin.ingest.types import Document

OUT_DIR = Path(__file__).parent / "results" / "ingest"
_SQUASH = re.compile(r"[^a-z0-9]+")


def _squash(text: str) -> str:
    return _SQUASH.sub("", text.lower())


def anchored_rate(doc: Document) -> float | None:
    if not doc.toc:
        return None
    by_page: dict[int, str] = {}
    for block in doc.blocks:
        if block.page is not None:
            by_page[block.page] = by_page.get(block.page, "") + _squash(block.text)
    found = sum(1 for e in doc.toc if e.page is not None and _squash(e.title) and _squash(e.title) in by_page.get(e.page, ""))
    return found / len(doc.toc)


def measure(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    doc = read_document(path, use_ocr=False)
    read_s = time.perf_counter() - started
    sections = build_sections(doc)
    total_s = time.perf_counter() - started

    block_chars = sum(len(b.text) for b in doc.blocks)
    section_chars = sum(len(s.text) for s in sections) + sum(len(s.title) for s in sections)
    pages_with_text = {b.page for b in doc.blocks if b.page is not None and b.text.strip()}
    return {
        "file": path.name,
        "pages": doc.page_count,
        "toc_entries": len(doc.toc),
        "sections": len(sections),
        "anchored_rate": anchored_rate(doc),
        "empty_section_rate": sum(1 for s in sections if not s.text.strip()) / len(sections) if sections else None,
        "char_coverage": min(1.0, section_chars / block_chars) if block_chars else None,
        "text_page_rate": len(pages_with_text) / doc.page_count if doc.page_count else None,
        "ocr_pages": len(doc.ocr_pages),
        "seconds": round(total_s, 2),
        "pages_per_s": round(doc.page_count / read_s, 1) if read_s else None,
        "sample_sections": [{"id": s.id, "title": s.title, "pages": [s.page_start, s.page_end], "chars": len(s.text)} for s in sections[:12]],
        "run_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", type=Path)
    args = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf in args.pdfs:
        result = measure(pdf)
        (OUT_DIR / f"{pdf.stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {k: v for k, v in result.items() if k != "sample_sections"}
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
