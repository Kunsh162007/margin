"""Turn past-paper questions into a weight for every book section.

Each question carries its marks (1 if the paper does not print them), boosted
by up to 50% for the most recent year, because recent papers predict the next
one best, and that weight goes to the question's best-matching section.

Spreading it over the top three matches (60/25/15) looked safer, because
search ranks the right section first only 37% of the time but has it in the top
five 60% of the time. Measured on synthetic papers over five seeds
(``evals/blueprint_eval.py``), it was worse: chapter shares were
further from the truth in 5 of 5 seeds and less true weight was captured in 4
of 5. Spreading weight smears every question across neighbouring sections. The
split is kept as an option, not the default.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Protocol

from margin.blueprint.paper import PaperQuestion
from margin.retrieve.search import Hit

BEST_MATCH: tuple[float, ...] = (1.0,)
RANK_SPLIT: tuple[float, ...] = (0.6, 0.25, 0.15)  # measured worse; kept for comparison
DEFAULT_MARKS = 1
RECENCY_BOOST = 0.5
SEARCH_DEPTH = 10


class SearcherLike(Protocol):
    def search(self, query: str, top_k: int = 5, scope: str | None = None, rerank: bool = False) -> list[Hit]: ...


@dataclass(frozen=True)
class SectionWeight:
    doc_id: str
    sid: str
    title: str
    weight: float
    share: float
    questions: tuple[str, ...]  # questions whose best match is this section
    marks: int


@dataclass(frozen=True)
class Blueprint:
    sections: tuple[SectionWeight, ...]
    questions: int
    unmatched: int

    def top(self, n: int) -> tuple[SectionWeight, ...]:
        return self.sections[:n]


def blueprint_to_json(blueprint: Blueprint) -> str:
    return json.dumps({"questions": blueprint.questions, "unmatched": blueprint.unmatched, "sections": [asdict(s) for s in blueprint.sections]})


def blueprint_from_json(text: str) -> Blueprint:
    data = json.loads(text)
    rows = tuple(SectionWeight(**{**s, "questions": tuple(s["questions"])}) for s in data["sections"])
    return Blueprint(rows, data["questions"], data["unmatched"])


def chapter_of(sid: str) -> str:
    """'4.2' -> '4'. Sections without a numbered id stand alone."""
    head = sid.split(".")[0]
    return head if head.isdigit() else sid


def chapter_shares(blueprint: Blueprint) -> list[tuple[str, float]]:
    """Share of exam weight per chapter, largest first. The blueprint is most reliable at this level."""
    shares: dict[str, float] = {}
    for row in blueprint.sections:
        key = chapter_of(row.sid)
        shares[key] = shares.get(key, 0.0) + row.share
    return sorted(((k, round(v, 4)) for k, v in shares.items()), key=lambda kv: -kv[1])


def question_weight(question: PaperQuestion, years: tuple[int, int] | None) -> float:
    weight = float(question.marks or DEFAULT_MARKS)
    if years and question.year is not None and years[1] > years[0]:
        weight *= 1 + RECENCY_BOOST * (question.year - years[0]) / (years[1] - years[0])
    return weight


def _distinct_sections(hits: list[Hit], limit: int) -> list[tuple[str, str, str]]:
    seen: list[tuple[str, str, str]] = []
    for hit in hits:
        key = (hit.row.doc_id, hit.row.sid, hit.row.section_title)
        if all(key[:2] != s[:2] for s in seen):
            seen.append(key)
        if len(seen) == limit:
            break
    return seen


def build_blueprint(searcher: SearcherLike, questions: list[PaperQuestion], split: tuple[float, ...] = BEST_MATCH) -> Blueprint:
    known = [q.year for q in questions if q.year is not None]
    years = (min(known), max(known)) if known else None
    weights: dict[tuple[str, str], float] = {}
    titles: dict[tuple[str, str], str] = {}
    marks: dict[tuple[str, str], int] = {}
    owned: dict[tuple[str, str], list[str]] = {}
    unmatched = 0

    for question in questions:
        targets = _distinct_sections(searcher.search(question.text, top_k=SEARCH_DEPTH), len(split))
        if not targets:
            unmatched += 1
            continue
        shares = split[: len(targets)]
        total = sum(shares)
        weight = question_weight(question, years)
        for (doc_id, sid, title), share in zip(targets, shares):
            key = (doc_id, sid)
            weights[key] = weights.get(key, 0.0) + weight * share / total
            titles[key] = title
        best = targets[0][:2]
        marks[best] = marks.get(best, 0) + (question.marks or 0)
        owned[best] = [*owned.get(best, []), question.key]

    grand = sum(weights.values()) or 1.0
    rows = sorted(
        (SectionWeight(d, s, titles[(d, s)], round(w, 4), round(w / grand, 4), tuple(owned.get((d, s), [])), marks.get((d, s), 0)) for (d, s), w in weights.items()),
        key=lambda r: (-r.weight, r.sid),
    )
    return Blueprint(tuple(rows), len(questions), unmatched)
