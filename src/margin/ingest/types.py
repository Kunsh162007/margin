"""The shapes every reader produces, whatever the file format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BlockKind = Literal["heading", "paragraph", "table", "notes", "ocr"]


@dataclass(frozen=True)
class Block:
    text: str
    page: int | None  # 1-based page or slide; None where the format has no pages (DOCX)
    kind: BlockKind = "paragraph"
    level: int | None = None  # heading depth, 1 = chapter


@dataclass(frozen=True)
class TocEntry:
    title: str
    level: int
    page: int | None


@dataclass(frozen=True)
class Document:
    id: str  # content hash: adding the same file twice is a no-op
    source: str
    kind: str
    title: str
    page_count: int
    blocks: tuple[Block, ...]
    toc: tuple[TocEntry, ...] = ()
    ocr_pages: tuple[int, ...] = ()  # pages whose text came from OCR, so answers can say so
    formula_pages: tuple[int, ...] = ()  # pages whose drawn equations were read into the text as LaTeX

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


@dataclass(frozen=True)
class Section:
    id: str  # "4.2" when the book numbers its headings, otherwise positional
    title: str
    level: int
    path: tuple[str, ...]  # ancestor titles, outermost first
    page_start: int | None
    page_end: int | None
    text: str
