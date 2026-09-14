"""Mark a student's answer against a marking scheme, point by point.

The model makes one narrow judgement per marking point — does the answer state
this, in meaning if not in words — and returns exactly one true or false per
point, enforced by the JSON schema. The mark is then computed in code from those
judgements. The model never chooses the number, so the same judgements always
give the same mark, and every mark can be explained by the points it counted.

A marking point that is only a formula ("States $M = gR^2/G$") is also met in
code when the answer holds an equivalent formula (``maths.py``): a small model
does not reliably see that g = GM/R² is the same relation. Code can add a point
the model missed; it never takes one away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from margin.practice.maths import points_met_by_formula

if TYPE_CHECKING:  # the agent executor imports this module
    from margin.agent.loop import ClientLike

MAX_TOKENS = 1024
PROMPT = (
    "You are marking a student's exam answer. For each marking point, answer true only if the student's answer states it "
    "(the same meaning counts; exact words are not needed) and false otherwise. Then write one sentence of feedback "
    "naming the most important thing that is missing, or confirming the answer is complete."
)


@dataclass(frozen=True)
class Grade:
    marks_awarded: int
    max_marks: int
    points_met: tuple[bool, ...]
    missing: tuple[str, ...]
    feedback: str
    points_by_formula: tuple[bool, ...] = ()  # points credited by an equivalent formula


def marks_for(points_met: tuple[bool, ...], max_marks: int) -> int:
    return round(max_marks * sum(points_met) / len(points_met)) if points_met else 0


def _schema(points: int) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["points_met", "feedback"],
        "properties": {
            "points_met": {"type": "array", "items": {"type": "boolean"}, "minItems": points, "maxItems": points},
            "feedback": {"type": "string"},
        },
    }


def grade_answer(client: ClientLike, question: str, marking_points: tuple[str, ...], student_answer: str, max_marks: int) -> Grade:
    if not student_answer.strip() or not marking_points:
        return Grade(0, max_marks, tuple(False for _ in marking_points), tuple(marking_points), "No answer was given.")
    points = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(marking_points))
    request = f"Question ({max_marks} marks): {question}\n\nMarking points:\n{points}\n\nStudent's answer:\n{student_answer}"
    reply = client.chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": request}], json_schema=_schema(len(marking_points)), max_tokens=MAX_TOKENS)
    try:
        data = json.loads(reply.content)
        met = tuple(bool(x) for x in data.get("points_met", []))
        feedback = str(data.get("feedback", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        met, feedback = (), "The answer could not be marked automatically."
    by_formula = points_met_by_formula(marking_points, student_answer)
    met = tuple(model or code for model, code in zip((met + (False,) * len(marking_points))[: len(marking_points)], by_formula))
    missing = tuple(p for p, ok in zip(marking_points, met) if not ok)
    return Grade(marks_for(met, max_marks), max_marks, met, missing, feedback, by_formula)
