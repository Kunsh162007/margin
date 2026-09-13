"""Build a retrieval gold set from a textbook's own end-of-chapter questions.

    uv run python -m evals.datasets.build_retrieval_gold D:/corpus/university-physics-volume-1_-_WEB.pdf

OpenStax *University Physics Volume 1* lists its Conceptual Questions and
Problems under the heading of the section they test ("1.2 Units and
Standards", then "7 . Identify some advantages of metric units."). Every
question therefore arrives with its correct section, and a retrieval system is
scored on whether it finds that section — no hand labelling, and no judgement
calls by the person who built the system.

Source: OpenStax, University Physics Volume 1, CC BY 4.0,
https://openstax.org/details/books/university-physics-volume-1. The questions
are redistributed under that licence with this attribution.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from margin.ingest.pdf import read_pdf
from margin.ingest.sections import build_sections
from margin.ingest.types import Block, Document

OUT = Path(__file__).parent / "retrieval_gold.jsonl"
# The PDF text layer often glues a list heading onto the end of the previous block
# ("... the cross product. Conceptual Questions"), so markers are matched at the end of
# a block too — but only after a sentence end and with exact capitals, so prose such as
# "strategies for solving problems" cannot switch modes.
MARKER = re.compile(r"(?:^|(?<=[.?!)\]”\"'])\s+)(Conceptual Questions|Additional Problems|Challenge Problems|Problems)\s*$")
TRANSITIONS = {"Conceptual Questions": "conceptual", "Problems": "problem", "Additional Problems": None, "Challenge Problems": None}
STOP_HEADINGS = ("key terms", "key equations", "summary", "chapter review")
SECTION_HEAD = re.compile(r"^(\d{1,2})\.(\d{1,2})\s+\S")
QUESTION_START = re.compile(r"(?:^|(?<=\s))(\d{1,3}) \. ")
MIN_QUESTION_CHARS = 30


def split_questions(text: str) -> list[tuple[int, str]]:
    marks = list(QUESTION_START.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end() : end].strip()
        if len(body) >= MIN_QUESTION_CHARS:
            out.append((int(m.group(1)), body))
    return out


def before_answer_key(doc: Document) -> tuple[Block, ...]:
    """Blocks before the Answer Key. It repeats every question number with the answer in its
    place; reading it would replace questions with answers under the same ids."""
    pages = [e.page for e in doc.toc if e.page and "answer key" in e.title.lower()]
    cut = min(pages) if pages else None
    return tuple(b for b in doc.blocks if cut is None or b.page is None or b.page < cut)


def first_occurrences(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: dict[object, dict[str, object]] = {}
    for row in rows:
        seen.setdefault(row["id"], row)
    return list(seen.values())


def split_marker(text: str) -> tuple[str, str | None]:
    """(text before a trailing list marker, the marker) — or (text, None) if there is none."""
    m = MARKER.search(text)
    return (text[: m.start()].strip(), m.group(1)) if m else (text, None)


def extract(blocks: tuple[Block, ...], book: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    kind: str | None = None
    section: str | None = None
    for block in blocks:
        body, marker = split_marker(block.text.strip())
        if body.lower().startswith(STOP_HEADINGS):
            kind, section, body = None, None, ""
        if body and kind is not None:
            head = SECTION_HEAD.match(body)
            if block.kind == "heading" and head:
                section = f"{head.group(1)}.{head.group(2)}"
            elif section is not None:
                chapter = int(section.split(".")[0])
                for number, question in split_questions(body):
                    rows.append({"id": f"{book}-ch{chapter}-q{number}", "book": book, "kind": kind, "chapter": chapter, "gold_section": section, "question": question})
        if marker is not None:
            kind, section = TRANSITIONS[marker], None
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--book", default="uphys1")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    doc = read_pdf(args.pdf, "gold", None)
    valid = {s.id for s in build_sections(doc)}
    blocks = before_answer_key(doc)
    rows = extract(blocks, args.book)
    unique = first_occurrences(rows)
    kept = [r for r in unique if r["gold_section"] in valid]
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    print(f"pages read {max((b.page or 0) for b in blocks)}, extracted {len(rows)}, unique {len(unique)}, with a real section {len(kept)} -> {args.out}")
    print("by kind:", dict(Counter(r["kind"] for r in kept)))
    print("by chapter:", dict(sorted(Counter(r["chapter"] for r in kept).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
