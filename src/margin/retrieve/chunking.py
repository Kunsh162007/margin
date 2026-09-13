"""Split sections into search-sized chunks.

Chunks are built from whole paragraphs up to about 1,200 characters (~300
tokens). A paragraph longer than the hard limit is split at sentence
boundaries, and only a single sentence longer than the limit is cut mid-way.
Chunks never cross a section boundary, so every search hit names one section.
"""

from __future__ import annotations

import re

from margin.ingest.types import Section
from margin.store.db import ChunkRecord

TARGET_CHARS = 1200
MAX_CHARS = 1800
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_EXERCISE_TITLE = re.compile(
    r"^(?:(?:review|critical thinking|interactive link|conceptual)\s+questions|(?:additional\s+|challenge\s+)?problems)$",
    re.IGNORECASE,
)


# Question lists inside a section ("Chapter Review" in OpenStax physics holds Key Terms,
# Summary, then Conceptual Questions and Problems with no bookmarks of their own). A marker
# counts only after a sentence or paragraph end and when a section number or question number
# follows, so prose like "strategies for solving Problems" cannot start an exercise tail.
_EXERCISE_MARKER = re.compile(
    r"(?:^|(?<=[.?!)\]”\"'])\s+|\n\n)"
    r"(Conceptual Questions|Review Questions|Critical Thinking Questions|Interactive Link Questions|Problems|Additional Problems|Challenge Problems)"
    r"(?=\s*(?:$|\d{1,2}\.\d{1,2}\s|\d{1,3} \. ))"
)


def is_exercise_section(title: str) -> bool:
    return bool(_EXERCISE_TITLE.match(title.strip()))


def split_exercise_tail(text: str) -> tuple[str, str]:
    """(teaching text, question lists) — the second is empty when the section has no question list."""
    match = _EXERCISE_MARKER.search(text)
    if match is None:
        return text, ""
    return text[: match.start()].rstrip(), text[match.start() :].lstrip()


def _pieces(text: str) -> list[str]:
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(para) <= MAX_CHARS:
            pieces.append(para)
            continue
        for sentence in _SENTENCE.split(para):
            while len(sentence) > MAX_CHARS:
                pieces.append(sentence[:MAX_CHARS])
                sentence = sentence[MAX_CHARS:]
            if sentence.strip():
                pieces.append(sentence.strip())
    return pieces


def _pack(text: str) -> list[str]:
    texts: list[str] = []
    buffer: list[str] = []
    size = 0
    for piece in _pieces(text):
        if buffer and size + len(piece) > TARGET_CHARS:
            texts.append("\n\n".join(buffer))
            buffer, size = [], 0
        buffer.append(piece)
        size += len(piece) + 2
    if buffer:
        texts.append("\n\n".join(buffer))
    return texts


def chunk_section(section: Section) -> list[ChunkRecord]:
    """Teaching text and any trailing question list are packed separately, so no chunk mixes them."""
    if is_exercise_section(section.title):
        parts = [(t, True) for t in _pack(section.text)]
    else:
        teaching, questions = split_exercise_tail(section.text)
        parts = [(t, False) for t in _pack(teaching)] + [(t, True) for t in _pack(questions)]
    return [ChunkRecord(section.id, i, text, section.page_start, section.page_end, flag) for i, (text, flag) in enumerate(parts)]
