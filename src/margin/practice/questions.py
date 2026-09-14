"""Exam-style questions from one section, each checked before a student sees it.

Questions are generated under a JSON schema, so every one arrives with a model
answer, its marks and a marking scheme. Three checks then run in code:

1. **answerable** — at least 60% of the model answer's content words occur in
   the section, so the answer comes from the book rather than the model's memory;
2. **not a copy** — word overlap with every existing question from the book
   stays below 70%, so the student is not handed the textbook's own exercises;
3. **fits the request** — the marks are what was asked for.

A question that fails is kept with its reasons, so the rejection rate can be
measured instead of silently hidden. Both text checks are lexical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from margin.ingest.formulas import UNCERTAIN
from margin.notes.verify import content_stems
from margin.practice.maths import relations, relations_consistent

if TYPE_CHECKING:  # the agent executor imports this module
    from margin.agent.loop import ClientLike

QuestionType = Literal["short", "long", "numerical", "mcq"]
ANSWERABLE_SHARE = 0.6
COPY_SIMILARITY = 0.7
MIN_ANSWER_WORDS = 3
MAX_TOKENS = 2048
PROMPT = (
    "You write exam questions from one textbook section. Every question must be answerable from the section alone. "
    "For each question give a model answer written from the section, the marks, and a marking scheme with one point per mark. "
    "Do not copy questions that already appear in the book; test the ideas in your own words."
)


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    answer: str
    marks: int = Field(ge=1, le=25)
    marking_points: list[str] = Field(min_length=1, max_length=10)


class _QuestionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[_Question] = Field(min_length=1, max_length=10)


@dataclass(frozen=True)
class Question:
    section: str
    kind: str
    question: str
    answer: str
    marks: int
    marking_points: tuple[str, ...]


@dataclass(frozen=True)
class Checked:
    question: Question
    accepted: bool
    reasons: tuple[str, ...]


def similarity(a: str, b: str) -> float:
    left, right = content_stems(a), content_stems(b)
    return len(left & right) / len(left | right) if left and right else 0.0


def check(question: Question, source: str, existing: tuple[str, ...], marks_wanted: int) -> Checked:
    reasons: list[str] = []
    answer_stems, source_stems = content_stems(question.answer), content_stems(source)
    if len(answer_stems) >= MIN_ANSWER_WORDS and len(answer_stems & source_stems) / len(answer_stems) < ANSWERABLE_SHARE:
        reasons.append("model answer is not supported by the section")
    if len(answer_stems) < MIN_ANSWER_WORDS and question.kind != "numerical":
        reasons.append("model answer is too short to check")
    if not relations_consistent(question.answer, relations(source)):
        reasons.append("model answer contradicts a formula in the section")
    if existing and max(similarity(question.question, e) for e in existing) >= COPY_SIMILARITY:
        reasons.append("near-copy of a question already in the book")
    if question.marks != marks_wanted:
        reasons.append(f"worth {question.marks} marks, {marks_wanted} were asked for")
    return Checked(question, not reasons, tuple(reasons))


def generate_questions(client: ClientLike, section_text: str, sid: str, count: int, marks: int, kind: QuestionType = "short",
                       existing: tuple[str, ...] = ()) -> list[Checked]:
    request = f"Section {sid}\n\n{section_text}\n\nWrite exactly {count} {kind} questions worth {marks} marks each."
    reply = client.chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": request}],
                        json_schema=_QuestionSet.model_json_schema(), max_tokens=MAX_TOKENS)
    try:
        parsed = _QuestionSet.model_validate(json.loads(reply.content))
    except (json.JSONDecodeError, ValidationError):
        return []
    clean = lambda text: text.replace(UNCERTAIN, "").strip()  # noqa: E731 — the model may copy the book's uncertainty mark
    questions = [Question(sid, kind, clean(q.question), clean(q.answer), q.marks, tuple(clean(p) for p in q.marking_points)) for q in parsed.questions[:count]]
    return [check(q, section_text, existing, marks) for q in questions]


MAX_PER_CALL = 10  # the schema's limit on one reply


def generate_for_sections(client: ClientLike, sections: list[tuple[str, str]], count: int, marks: int, kind: QuestionType = "short",
                          on_section: Callable[[str, list[Checked]], None] | None = None) -> list[Checked]:
    """Spread ``count`` questions over ``(sid, text)`` sections, earlier sections first, at most 10 per model call."""
    usable = [(sid, text) for sid, text in sections if text.strip()]
    if not usable or count < 1:
        return []
    per_section, extra = divmod(count, len(usable))
    checked: list[Checked] = []
    for i, (sid, text) in enumerate(usable):
        wanted = per_section + (1 if i < extra else 0)
        batch: list[Checked] = []
        while wanted > 0:
            take = min(wanted, MAX_PER_CALL)
            batch += generate_questions(client, text, sid, take, marks, kind)
            wanted -= take
        checked += batch
        if on_section and batch:
            on_section(sid, batch)
    return checked
