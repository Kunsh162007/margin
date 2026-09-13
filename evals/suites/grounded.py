"""Grounded answering: answer from a passage, and refuse when the passage has no answer.

Refusal is scored in both directions. A model that always refuses scores 1.0 on
refusal and 0.0 on answers; one that never refuses does the reverse. Both
numbers are needed to see which failure a model has.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from evals.matchers import squash
from margin.runtime.client import ClientError, LlamaClient

SENTINEL = "NOT_IN_SOURCE"
SYSTEM_PROMPT = (
    "Answer the question using only the passage. Give the shortest complete answer. "
    f"If the passage does not contain the answer, reply exactly: {SENTINEL}"
)


def _score(case: dict[str, Any], answer: str) -> dict[str, Any]:
    """A refusal is a reply that *is* the sentinel. An answer with the sentinel tacked on is
    "mixed": the product cannot tell which the model meant, so it counts as wrong either way."""
    text, sentinel = squash(answer), squash(SENTINEL)
    refused = text.startswith(sentinel)
    mixed = not refused and sentinel in text
    if case["answerable"]:
        correct = not refused and not mixed and any(squash(a) in text for a in case["answers"])
    else:
        correct = refused
    return {"refused": refused, "mixed": mixed, "correct": correct}


def run(client: LlamaClient, cases: list[dict[str, Any]], passages: dict[str, dict[str, Any]]) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    rows = []
    for case in cases:
        text = passages[case["passage"]]["text"]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Passage:\n{text}\n\nQuestion: {case['question']}"}]
        try:
            res = client.chat(messages, max_tokens=160)
        except ClientError as exc:
            rows.append({"id": case["id"], "answerable": case["answerable"], "correct": False, "refused": False, "error": str(exc)[:300], "latency_s": 0.0})
            continue
        rows.append({"id": case["id"], "answerable": case["answerable"], **_score(case, res.content), "answer": res.content[:300], "latency_s": res.wall_s})

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    ok = [r for r in rows if "error" not in r]
    return {
        "grounded.answer_acc": mean(r["correct"] for r in answerable) if answerable else None,
        "grounded.refusal_acc": mean(r["correct"] for r in unanswerable) if unanswerable else None,
        "grounded.false_refusal_rate": mean(r["refused"] for r in answerable) if answerable else None,
        "grounded.mixed_refusal_rate": mean(r.get("mixed", False) for r in rows) if rows else None,
        "grounded.latency_mean_s": mean(r["latency_s"] for r in ok) if ok else None,
    }, rows
