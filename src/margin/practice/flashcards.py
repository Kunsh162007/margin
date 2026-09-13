"""Flashcards from saved notes, FSRS scheduling, and Anki export.

Cards are extracted from ``**term**: definition`` lines the notes writer already
produces, so making cards needs no model call and cannot invent a definition the
verified notes do not contain. Scheduling uses FSRS (``fsrs``), the algorithm
modern Anki uses. Export writes a real ``.apkg`` with ids derived from the deck
name, so re-exporting updates the same deck instead of creating a duplicate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import genanki
from fsrs import Card, Rating, Scheduler

_TERM_LINE = re.compile(r"\*\*([^*\n]{2,60}?)\*\*\s*(?:\([^)]{1,40}\))?\s*[:—–-]\s*(.{8,300})")
_MODEL_ID = 1_607_392_319  # fixed: changing it makes Anki treat every card as new
_MODEL = genanki.Model(
    _MODEL_ID, "Margin card", fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Section"}],
    templates=[{"name": "Card", "qfmt": "{{Front}}", "afmt": "{{FrontSide}}<hr id=answer>{{Back}}<br><small>{{Section}}</small>"}],
)


@dataclass(frozen=True)
class Flashcard:
    front: str
    back: str
    section: str


def cards_from_notes(markdown: str, section: str) -> list[Flashcard]:
    cards: dict[str, Flashcard] = {}
    for line in markdown.splitlines():
        match = _TERM_LINE.search(line)
        if match is None:
            continue
        term, definition = match.group(1).strip().rstrip(":"), match.group(2).strip()
        if term.lower() not in cards and not definition.startswith("**"):
            cards[term.lower()] = Flashcard(term, definition, section)
    return list(cards.values())


def new_schedule() -> str:
    return json.dumps(Card().to_dict(), default=str)


def review(schedule: str, rating: str) -> tuple[str, datetime]:
    """Apply a review ("again", "hard", "good", "easy") and return the new schedule and due time."""
    card, _ = Scheduler().review_card(Card.from_dict(json.loads(schedule)), Rating[rating.strip().capitalize()])
    return json.dumps(card.to_dict(), default=str), card.due


def _stable_id(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) % (1 << 31)


def export_anki(cards: list[Flashcard], deck_name: str, path: Path) -> Path:
    deck = genanki.Deck(_stable_id(deck_name), deck_name)
    for card in cards:
        deck.add_note(genanki.Note(model=_MODEL, fields=[card.front, card.back, card.section], guid=genanki.guid_for(deck_name, card.front)))
    path.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(path))
    return path
