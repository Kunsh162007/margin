"""Measure question generation and answer marking with the default model.

    uv run python -m evals.practice_eval

For six University Physics sections, the default model writes three 2-mark
short questions each. Measured per question: arrived schema-valid, model answer
supported by the section, near-copy of one of the book's own questions for that
section (taken from the retrieval gold set), accepted by all three checks.

Marking is checked without a judge model. Every accepted question is marked
three times: with its own model answer, with only the first half of its marking
points written out, and with an off-topic answer. A sound marker ranks them
complete >= half >= off-topic with complete > off-topic; the share of questions
where that order holds is reported, along with the average marks for each.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

from margin import config, hardware
from margin.practice.grading import grade_answer
from margin.practice.questions import generate_questions
from margin.runtime import models
from margin.runtime.binaries import ensure_llama_server
from margin.runtime.client import LlamaClient
from margin.runtime.server import LlamaServer, ServerConfig
from margin.store.db import Workspace

SIDS = ["3.4", "5.3", "6.2", "10.6", "13.2", "14.5"]
COUNT, MARKS = 3, 2
OFF_TOPIC = "Photosynthesis in plants converts light energy into chemical energy stored in glucose."
GOLD = Path(__file__).parent / "datasets" / "retrieval_gold.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "practice"


def main() -> int:
    paths = config.paths().ensure()
    ws_path = paths.workspaces_dir / "eval-uphys1.db"
    if not ws_path.exists():
        print(f"no workspace at {ws_path}; run evals.retrieval_eval first")
        return 1
    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    hw = hardware.detect()
    backend = hardware.choose_backend(hw)
    spec = models.get(models.DEFAULT_GPU_MODEL if backend != "cpu" else models.DEFAULT_CPU_MODEL)
    cfg = ServerConfig(exe=ensure_llama_server(paths.bin_dir, hw.os, hw.arch, backend), model=spec.path(paths.models_dir),
                       gpu=backend != "cpu", threads=hardware.default_threads(hw), ctx=16384, chat_template_file=spec.chat_template_path())
    generated, grading = [], []
    with Workspace.open(ws_path) as ws, LlamaServer(cfg, paths.logs_dir / f"practice-eval-{spec.id}.log") as server:
        client = LlamaClient(server.base_url)
        for sid in SIDS:
            doc_id, _ = ws.find_section(sid)
            text = (ws.section_text(doc_id, sid, include_exercises=False) or "")[:6000]
            book_questions = tuple(g["question"] for g in gold if g["gold_section"] == sid)
            checked = generate_questions(client, text, sid, COUNT, MARKS, "short", book_questions)
            generated.append((sid, checked))
            print(f"{sid}: {len(checked)} questions, accepted {sum(c.accepted for c in checked)}, reasons {[r for c in checked for r in c.reasons]}", flush=True)
            for c in (c for c in checked if c.accepted):
                q = c.question
                half = "; ".join(q.marking_points[: max(1, len(q.marking_points) // 2)])
                marks = [grade_answer(client, q.question, q.marking_points, answer, q.marks).marks_awarded for answer in (q.answer, half, OFF_TOPIC)]
                grading.append(marks)
        client.close()

    all_checked = [c for _, cs in generated for c in cs]
    reasons = [r for c in all_checked for r in c.reasons]
    summary = {
        "model": spec.id, "backend": backend, "sections": len(SIDS), "requested": COUNT * len(SIDS),
        "schema_valid_sections": sum(1 for _, cs in generated if cs), "questions": len(all_checked),
        "accepted": sum(c.accepted for c in all_checked),
        "answerable_rate": round(mean(0.0 if any("not supported" in r for r in c.reasons) else 1.0 for c in all_checked), 4) if all_checked else None,
        "copy_rate": round(mean(1.0 if any("near-copy" in r for r in c.reasons) else 0.0 for c in all_checked), 4) if all_checked else None,
        "marks_mismatch": sum(1 for r in reasons if "were asked for" in r),
        "graded_questions": len(grading),
        "grading_order_ok_rate": round(mean(1.0 if full >= half >= off and full > off else 0.0 for full, half, off in grading), 4) if grading else None,
        "mean_marks": {"complete": round(mean(g[0] for g in grading), 2), "half": round(mean(g[1] for g in grading), 2), "off_topic": round(mean(g[2] for g in grading), 2)} if grading else None,
        "run_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    print(json.dumps(summary, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "uphys1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    samples = [{"section": sid, "accepted": c.accepted, "reasons": list(c.reasons), "question": c.question.question, "answer": c.question.answer,
                "marking_points": list(c.question.marking_points)} for sid, cs in generated for c in cs]
    (OUT_DIR / "uphys1-questions.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
