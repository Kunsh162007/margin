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


def is_exercise_section(title: str) -> bool:
    return bool(_EXERCISE_TITLE.match(title.strip()))


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


def chunk_section(section: Section) -> list[ChunkRecord]:
    texts: list[str] = []
    buffer: list[str] = []
    size = 0
    for piece in _pieces(section.text):
        if buffer and size + len(piece) > TARGET_CHARS:
            texts.append("\n\n".join(buffer))
            buffer, size = [], 0
        buffer.append(piece)
        size += len(piece) + 2
    if buffer:
        texts.append("\n\n".join(buffer))
    exercise = is_exercise_section(section.title)
    return [ChunkRecord(section.id, i, t, section.page_start, section.page_end, exercise) for i, t in enumerate(texts)]
