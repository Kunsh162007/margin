"""Remember every marked answer, bring mistakes back, and rank weak topics.

Each marked answer is stored with its question and marking scheme, so a missed
question can be asked again later without the model writing it again. When it
comes back is decided by FSRS, the scheduler the flashcards use, with the marks
as the rating: under half is "again", half or more is "hard", full marks is
"good". Learning steps are off, so a missed question returns the next day at the
earliest rather than a minute later in the same sitting. A question answered in
full leaves the retry list.

The student may name why an answer lost marks — a misunderstood concept, a
misread question, a calculation slip, a missing keyword, or no attempt — the
error-log habit exam coaching teaches. It is optional and never guessed.

Weak topics rank practised sections by exam weight × the share of marks lost on
the latest attempt at each question. Without a blueprint every section weighs
the same; with one, a section the papers never ask about is not ranked.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from fsrs import Card, Rating, Scheduler

from margin.blueprint.weights import Blueprint
from margin.practice.grading import Grade
from margin.practice.questions import Question
from margin.store.db import Workspace

CAUSES = ("concept", "misread", "calculation", "keyword", "no-attempt")
_SCHEDULER = Scheduler(learning_steps=(), relearning_steps=(), enable_fuzzing=False)


@dataclass(frozen=True)
class Attempt:
    id: int
    question: Question
    marks_awarded: int
    max_marks: int
    missing: tuple[str, ...]
    cause: str | None
    due_at: datetime
    answered_at: datetime

    @property
    def lost_marks(self) -> bool:
        return self.marks_awarded < self.max_marks


@dataclass(frozen=True)
class SectionMastery:
    sid: str
    questions: int
    marks: int
    max_marks: int
    score: float  # share of marks won on the latest attempt at each question


@dataclass(frozen=True)
class WeakTopic:
    sid: str
    title: str
    score: float
    weight: float
    priority: float


def rating_for(marks_awarded: int, max_marks: int) -> Rating:
    if max_marks > 0 and marks_awarded >= max_marks:
        return Rating.Good
    if max_marks > 0 and 2 * marks_awarded >= max_marks:
        return Rating.Hard
    return Rating.Again


def _attempt(row: dict) -> Attempt:
    q = json.loads(row["question_json"])
    question = Question(q["section"], q["kind"], q["question"], q["answer"], q["marks"], tuple(q["marking_points"]))
    return Attempt(row["id"], question, row["marks_awarded"], row["max_marks"], tuple(json.loads(row["missing_json"])), row["cause"],
                   datetime.fromisoformat(row["due_at"]), datetime.fromisoformat(row["answered_at"]))


def record_attempt(ws: Workspace, question: Question, grade: Grade, now: datetime | None = None) -> Attempt:
    now = now or datetime.now(timezone.utc)
    previous = ws.latest_attempt(question.question)
    card = Card.from_dict(json.loads(previous["schedule_json"])) if previous else Card(due=now)
    card, _ = _SCHEDULER.review_card(card, rating_for(grade.marks_awarded, grade.max_marks), review_datetime=now)
    row = {
        "sid": question.section, "question": question.question, "question_json": json.dumps(asdict(question)),
        "marks_awarded": grade.marks_awarded, "max_marks": grade.max_marks, "missing_json": json.dumps(list(grade.missing)), "cause": None,
        "schedule_json": json.dumps(card.to_dict(), default=str), "due_at": card.due.isoformat(), "answered_at": now.isoformat(),
    }
    return _attempt({**row, "id": ws.add_attempt(row)})


def set_cause(ws: Workspace, attempt_id: int, cause: str) -> bool:
    if cause not in CAUSES:
        raise ValueError(f"cause must be one of: {', '.join(CAUSES)}")
    return ws.set_attempt_cause(attempt_id, cause)


def latest_attempts(ws: Workspace) -> list[Attempt]:
    latest: dict[str, dict] = {}
    for row in ws.attempts():
        latest[row["question"]] = row  # rows arrive oldest first
    return [_attempt(row) for row in latest.values()]


def due_retries(ws: Workspace, now: datetime | None = None) -> list[Attempt]:
    """Questions whose latest attempt lost marks and whose retry is due, soonest first."""
    now = now or datetime.now(timezone.utc)
    return sorted((a for a in latest_attempts(ws) if a.lost_marks and a.due_at <= now), key=lambda a: (a.due_at, a.id))


def mastery(ws: Workspace) -> dict[str, SectionMastery]:
    totals: dict[str, list[int]] = {}
    for a in latest_attempts(ws):
        count, marks, maximum = totals.get(a.question.section, [0, 0, 0])
        totals[a.question.section] = [count + 1, marks + a.marks_awarded, maximum + a.max_marks]
    return {sid: SectionMastery(sid, n, m, mx, round(m / mx, 4) if mx else 0.0) for sid, (n, m, mx) in totals.items()}


def weak_topics(ws: Workspace, blueprint: Blueprint | None = None, limit: int | None = None) -> list[WeakTopic]:
    weights = {row.sid: (row.share, row.title) for row in blueprint.sections} if blueprint else None
    topics = []
    for sid, m in mastery(ws).items():
        weight, title = weights.get(sid, (0.0, "")) if weights is not None else (1.0, "")
        priority = round(weight * (1 - m.score), 4)
        if priority > 0:
            topics.append(WeakTopic(sid, title, m.score, weight, priority))
    return sorted(topics, key=lambda t: (-t.priority, t.sid))[:limit]
