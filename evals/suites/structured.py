"""Structured study material: diagrams, tables, formula sheets, questions and notes.

JSON outputs are requested with a JSON schema, which llama.cpp turns into a
grammar, so validity should be near 1.0 for every model. What separates models
is *content*: are the passage's key terms present (coverage), and do the
words used come from the passage rather than from memory (grounding)?

Both content measures are lexical. They reward the right words, not correct
reasoning, and are labelled as such wherever they are reported.
"""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evals.matchers import content_stems, squash, support_ratio
from margin.agent.tools import MakeFlowchart, MakeFormulaSheet, MakeGlossary, MakeMindmap, MakeTable, MakeTimeline
from margin.runtime.client import ClientError, LlamaClient

SYSTEM_PROMPT = "You turn textbook passages into study material. Use only information that is in the passage."
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    answer: str
    marks: int = Field(ge=1, le=25)


class QuestionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[_Question] = Field(min_length=3, max_length=3)


SCHEMAS: dict[str, type[BaseModel]] = {
    "flowchart": MakeFlowchart, "table": MakeTable, "mindmap": MakeMindmap, "formula": MakeFormulaSheet,
    "timeline": MakeTimeline, "glossary": MakeGlossary, "questions": QuestionSet,
}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _coverage(text: str, key_terms: list[str]) -> float | None:
    if not key_terms:
        return None
    haystack = squash(text)
    return mean(1.0 if squash(t) in haystack else 0.0 for t in key_terms)


def _score_json(case: dict[str, Any], raw: str, source: set[str]) -> dict[str, Any]:
    schema = SCHEMAS[case["task"]]
    try:
        data = json.loads(raw)
        schema.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return {"valid": False, "coverage": 0.0 if case["key_terms"] else None, "grounding": 0.0}
    ratios = [r for s in _strings(data) if (r := support_ratio(s, source)) is not None]
    row: dict[str, Any] = {"valid": True, "coverage": _coverage(raw, case["key_terms"]), "grounding": mean(ratios) if ratios else 0.0}
    if case["task"] == "flowchart":
        ids = {s["id"] for s in data["steps"]}
        row["edges_valid"] = all(e["source"] in ids and e["target"] in ids for e in data["edges"])
    if case["task"] == "questions":
        answers = [support_ratio(q["answer"], source) for q in data["questions"]]
        row["answers_grounded"] = mean(1.0 if (a or 0) >= 0.6 else 0.0 for a in answers)
    return row


def _score_notes(case: dict[str, Any], text: str, source: set[str]) -> dict[str, Any]:
    sentences = [s for s in _SENTENCE.split(text) if len(content_stems(s)) >= 3]
    supported = [1.0 if (support_ratio(s, source) or 0) >= 0.5 else 0.0 for s in sentences]
    return {"valid": bool(text.strip()), "coverage": _coverage(text, case["key_terms"]), "sentence_support": mean(supported) if supported else 0.0}


def _mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean(float(v) for v in vals) if vals else None


def run(client: LlamaClient, cases: list[dict[str, Any]], passages: dict[str, dict[str, Any]]) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    rows = []
    for case in cases:
        text = passages[case["passage"]]["text"]
        source = content_stems(text)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Passage:\n{text}\n\nTask: {case['instruction']}"}]
        is_notes = case["task"] == "notes"
        schema = None if is_notes else SCHEMAS[case["task"]].model_json_schema()
        try:
            res = client.chat(messages, json_schema=schema, max_tokens=1200)
        except ClientError as exc:
            rows.append({"id": case["id"], "task": case["task"], "valid": False, "error": str(exc)[:300], "latency_s": 0.0})
            continue
        scored = _score_notes(case, res.content, source) if is_notes else _score_json(case, res.content, source)
        rows.append({"id": case["id"], "task": case["task"], **scored, "latency_s": res.wall_s, "gen_tps": res.gen_tps, "output": res.content[:1500]})

    json_rows = [r for r in rows if r["task"] not in ("notes", "questions")]
    notes = [r for r in rows if r["task"] == "notes"]
    questions = [r for r in rows if r["task"] == "questions"]
    flow = [r for r in rows if r["task"] == "flowchart" and r.get("valid")]
    ok = [r for r in rows if "error" not in r]
    return {
        "structured.schema_valid_rate": _mean_of([r for r in rows if r["task"] != "notes"], "valid"),
        "structured.key_term_coverage": _mean_of(json_rows, "coverage"),
        "structured.grounding": _mean_of(json_rows, "grounding"),
        "structured.flowchart_edges_valid_rate": _mean_of(flow, "edges_valid"),
        "questions.answer_grounded_rate": _mean_of(questions, "answers_grounded"),
        "notes.key_term_coverage": _mean_of(notes, "coverage"),
        "notes.sentence_support_rate": _mean_of(notes, "sentence_support"),
        "structured.latency_mean_s": _mean_of(ok, "latency_s"),
    }, rows
