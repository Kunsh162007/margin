"""Measure the formula checker and the formula-aware question check, without a model.

    uv run python -m evals.maths_eval

1. **MATH-500** (500 competition answers, fetched and hash-checked by
   ``evals.datasets.fetch``). Measured: answers that parse; each answer against
   SymPy's own LaTeX for it (a consistency check of the parser, not independent
   truth); plain-number answers against their decimal; and two changes built in
   text, never by the parser — ``2(A)`` and ``(A)+1`` — which must not match.
2. **Physics relations** (``datasets/formula_pairs.jsonl``): 32 rearrangements and
   near-misses from introductory mechanics, labelled by hand by the author of the
   checker, so they test intent rather than give an independent score.
3. **Saved practice questions** (``results/practice/uphys1-questions.json``)
   re-checked against their sections now that formula spans and "(1 mark for …)"
   notes no longer count as unsupported words. Needs the workspace built by
   ``retrieval_eval``; skipped without it.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from evals.datasets import fetch
from margin import config
from margin.practice.maths import equivalent, parse
from margin.practice.questions import Question, check
from margin.store.db import Workspace

HERE = Path(__file__).parent
PAIRS = HERE / "datasets" / "formula_pairs.jsonl"
GOLD = HERE / "datasets" / "retrieval_gold.jsonl"
SAVED_QUESTIONS = HERE / "results" / "practice" / "uphys1-questions.json"
OUT_DIR = HERE / "results" / "maths"
SECTION_CHARS = 6000
_PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?|-?\.\d+")


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _latex(answer: str) -> str | None:
    import sympy

    parsed = parse(answer)
    if parsed is None:
        return None
    kind, value = parsed
    if kind == "eq":
        return f"{sympy.latex(value[0])} = {sympy.latex(value[1])}"
    return sympy.latex(sympy.Tuple(*value) if kind == "tuple" else value)


def math500() -> dict[str, object]:
    rows = [json.loads(line) for line in (fetch.ensure("math500") / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    answers = [r["answer"] for r in rows]
    parsed = [a for a in answers if parse(a) is not None]
    round_trip = [a for a in parsed if (latex := _latex(a)) is not None and equivalent(a, latex)]
    numbers = [a for a in answers if _PLAIN_NUMBER.fullmatch(a.strip())]
    decimals = [a for a in numbers if equivalent(a, repr(float(a)))]
    changed = [(a, f"2\\left({a}\\right)") for a in parsed if a.strip() != "0"] + [(a, f"\\left({a}\\right)+1") for a in parsed]
    false_matches = [pair for pair in changed if equivalent(*pair)]
    return {
        "answers": len(answers), "parsed": len(parsed), "parse_rate": _rate(len(parsed), len(answers)),
        "round_trip_match_rate": _rate(len(round_trip), len(parsed)), "decimal_match_rate": _rate(len(decimals), len(numbers)),
        "changed_pairs": len(changed), "false_match_rate": _rate(len(false_matches), len(changed)),
        "unparsed_examples": [a for a in answers if parse(a) is None][:15], "false_match_examples": false_matches[:10],
    }


def physics_pairs() -> dict[str, object]:
    pairs = [json.loads(line) for line in PAIRS.read_text(encoding="utf-8").splitlines() if line.strip()]
    wrong = [p for p in pairs if equivalent(p["a"], p["b"]) != p["equivalent"]]
    return {"pairs": len(pairs), "accuracy": _rate(len(pairs) - len(wrong), len(pairs)), "wrong": wrong}


def saved_questions() -> dict[str, object] | None:
    ws_path = config.paths().workspaces_dir / "eval-uphys1.db"
    if not ws_path.exists() or not SAVED_QUESTIONS.exists():
        return None
    saved = json.loads(SAVED_QUESTIONS.read_text(encoding="utf-8"))
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    flipped = []
    with Workspace.open(ws_path) as ws:
        for item in saved:
            sid = item["section"]
            text = (ws.section_text(ws.find_section(sid)[0], sid, include_exercises=False) or "")[:SECTION_CHARS]
            book = tuple(g["question"] for g in gold if g["gold_section"] == sid)
            question = Question(sid, "short", item["question"], item["answer"], 2, tuple(item["marking_points"]))
            now = check(question, text, book, 2)
            if now.accepted != item["accepted"]:
                flipped.append({"section": sid, "question": item["question"], "was": item["accepted"], "now": now.accepted, "reasons": list(now.reasons)})
    return {"questions": len(saved), "accepted_before": sum(i["accepted"] for i in saved),
            "accepted_now": sum(i["accepted"] for i in saved) + sum(1 if f["now"] else -1 for f in flipped), "flipped": flipped}


def main() -> int:
    summary = {"math500": math500(), "physics_pairs": physics_pairs(), "saved_questions": saved_questions(),
               "datasets": {"math500": [f.sha256 for f in fetch.SETS["math500"].files]}, "run_at": datetime.now().strftime("%Y%m%d_%H%M%S")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
