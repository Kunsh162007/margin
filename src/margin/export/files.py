"""Write saved notes out as Markdown, HTML or an Anki deck — one path for the CLI, the agent and the UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from margin.export.html import to_html
from margin.notes.writer import SectionNotes, load_saved, to_markdown
from margin.practice.flashcards import cards_from_notes, export_anki
from margin.store.db import Workspace

FORMATS = ("markdown", "html", "anki")
_SUFFIX = {"markdown": "md", "html": "html", "anki": "apkg"}


@dataclass(frozen=True)
class Exported:
    path: Path
    sections: int
    cards: int


def saved_notes(ws: Workspace, sections: list[tuple[str, str, str]]) -> list[SectionNotes]:
    return [n for doc_id, sid, title in sections if (n := load_saved(ws, doc_id, sid, title))]


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in name).strip("-") or "all"


def export_notes(notes: list[SectionNotes], fmt: str, out_dir: Path, name: str, mermaid_js: str | None = None) -> Exported:
    if fmt not in FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; choose from {', '.join(FORMATS)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"margin-{safe_name(name)}.{_SUFFIX[fmt]}"
    if fmt == "anki":
        cards = [c for n in notes for c in cards_from_notes(n.markdown, n.sid)]
        return Exported(export_anki(cards, f"Margin::{name}", path), len(notes), len(cards))
    text = to_markdown(notes) if fmt == "markdown" else to_html(notes, f"Notes: {name}", mermaid_js)
    path.write_text(text, encoding="utf-8")
    return Exported(path, len(notes), 0)
