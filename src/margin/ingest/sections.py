"""Recover structure: split raw text into blocks, then blocks into sections.

Sections are anchored on *blocks*, not pages. A textbook section usually starts
mid-page, so page-level boundaries would put the end of 4.1 inside 4.2. For PDFs
with bookmarks, each bookmark is located inside its page's text; without
bookmarks, numbered heading lines ("4.2 Kirchhoff's laws") are the anchors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from margin.ingest.types import Block, BlockKind, Document, Section, TocEntry

NUMBERED_HEADING = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][^\n]{2,90})$")
CHAPTER_HEADING = re.compile(r"^(?:chapter|unit|module|part)\s+(?P<num>\d{1,2}|[IVXL]{1,6})\b[\s:.\-–]*(?P<title>[^\n]{0,90})$", re.IGNORECASE)
HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
BLANK_LINE = re.compile(r"\n\s*\n")
MAX_HEADING_WORDS = 14
_SQUASH = re.compile(r"[^a-z0-9]+")


def _squash(text: str) -> str:
    return _SQUASH.sub("", text.lower())


def heading_of(line: str) -> tuple[str | None, str, int] | None:
    """(number, title, level) if the line looks like a heading, else None."""
    line = line.strip()
    if not line or len(line.split()) > MAX_HEADING_WORDS or line.endswith((".", ",", ";")):
        return None
    if m := NUMBERED_HEADING.match(line):
        return m["num"], m["title"].strip(), m["num"].count(".") + 1
    if m := CHAPTER_HEADING.match(line):
        return m["num"], (m["title"].strip() or line), 1
    return None


def split_blocks(text: str, page: int | None, kind: BlockKind = "paragraph") -> list[Block]:
    text = HYPHEN_BREAK.sub(r"\1", text.replace("\r\n", "\n").replace("\r", "\n"))
    blocks: list[Block] = []
    for chunk in BLANK_LINE.split(text):
        paragraph: list[str] = []
        for line in chunk.split("\n"):
            head = heading_of(line)
            if head is None:
                if line.strip():
                    paragraph.append(line.strip())
                continue
            if paragraph:
                blocks.append(Block(" ".join(paragraph), page, kind))
                paragraph = []
            num, title, level = head
            blocks.append(Block(f"{num} {title}" if num else title, page, "heading", level))
        if paragraph:
            blocks.append(Block(" ".join(paragraph), page, kind))
    return blocks


@dataclass(frozen=True)
class _Anchor:
    index: int
    title: str
    level: int
    number: str | None


def _toc_anchors(blocks: tuple[Block, ...], toc: tuple[TocEntry, ...]) -> list[_Anchor]:
    anchors: list[_Anchor] = []
    cursor = 0
    for entry in toc:
        target = _squash(entry.title)
        on_page = [i for i in range(cursor, len(blocks)) if entry.page is None or blocks[i].page == entry.page]
        hit = next((i for i in on_page if target and target in _squash(blocks[i].text)[: len(target) + 12]), None)
        index = hit if hit is not None else (on_page[0] if on_page else None)
        if index is None:
            continue
        head = heading_of(entry.title)
        anchors.append(_Anchor(index, entry.title.strip(), entry.level, head[0] if head else None))
        cursor = index
    return anchors


def _heading_anchors(blocks: tuple[Block, ...]) -> list[_Anchor]:
    anchors = []
    for i, block in enumerate(blocks):
        if block.kind == "heading":
            head = heading_of(block.text)
            anchors.append(_Anchor(i, block.text, block.level or 1, head[0] if head else None))
    return anchors


def _unique(candidate: str, used: set[str]) -> str:
    uid, n = candidate, 2
    while uid in used:
        uid, n = f"{candidate}~{n}", n + 1
    used.add(uid)
    return uid


def strip_title(text: str, title: str) -> str:
    """Remove `title` from the start of `text`, comparing letters and digits only.

    PDF text layers often run a heading straight into its first paragraph
    ("Key Terms abdominopelvic cavity ..."). Dropping the whole block would lose
    the paragraph; returning it unchanged would repeat the heading in the body.
    """
    target = _squash(title)
    if not target or not _squash(text).startswith(target):
        return text
    seen = 0
    for i, ch in enumerate(text):
        if _SQUASH.fullmatch(ch.lower()) is None:
            seen += 1
            if seen == len(target):
                return text[i + 1 :].lstrip(" \t\n:.-–—")
    return ""


def _body(blocks: tuple[Block, ...], start: int, end: int, title: str | None) -> str:
    texts = [b.text for b in blocks[start:end]]
    if title is not None and texts:
        texts[0] = strip_title(texts[0], title)
    return "\n\n".join(t for t in texts if t.strip())


def build_sections(doc: Document) -> list[Section]:
    blocks = doc.blocks
    if not blocks:
        return []
    anchors = _toc_anchors(blocks, doc.toc) if doc.toc else _heading_anchors(blocks)
    if not anchors:
        pages = [b.page for b in blocks if b.page is not None]
        return [Section("all", doc.title, 1, (), min(pages, default=None), max(pages, default=None), _body(blocks, 0, len(blocks), None))]

    sections: list[Section] = []
    used: set[str] = set()
    if anchors[0].index > 0:
        sections.append(Section(_unique("front", used), "Front matter", 1, (), blocks[0].page, blocks[anchors[0].index - 1].page, _body(blocks, 0, anchors[0].index, None)))

    stack: list[_Anchor] = []
    for k, anchor in enumerate(anchors):
        end = max(anchor.index + 1, anchors[k + 1].index) if k + 1 < len(anchors) else len(blocks)
        # two bookmarks anchored on the same block: the earlier one owns no text
        if k + 1 < len(anchors) and anchors[k + 1].index == anchor.index:
            end = anchor.index
        while stack and stack[-1].level >= anchor.level:
            stack.pop()
        path = tuple(a.title for a in stack)
        stack.append(anchor)
        sid = _unique(anchor.number or f"s{k + 1}", used)
        last_page = blocks[end - 1].page if end > anchor.index else blocks[anchor.index].page
        sections.append(Section(sid, anchor.title, anchor.level, path, blocks[anchor.index].page, last_page, _body(blocks, anchor.index, end, anchor.title)))
    return sections
