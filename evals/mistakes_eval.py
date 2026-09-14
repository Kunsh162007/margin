"""Run the mistake log with the real model: mark, bring back, retry, clear.

    uv run python -m evals.mistakes_eval

On a fresh copy of the eval workspace (built by ``retrieval_eval``), the default
model writes 2-mark questions for three University Physics sections through
``Study``, exactly as the interface does. Every accepted question is answered
off-topic. None may be due again the same day; two days later every question that
lost marks must come back through ``load_retries``. Each is then answered with its
own model answer; a question marked in full must leave the retry list. The
network is blocked while the model runs.

Measured: questions accepted, off-topic answers that lost marks, retries due the
same day (must be 0), retries after two days, full marks on the retry, and
questions cleared after full marks.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from evals.netguard import blocked_network
from margin import config, hardware
from margin.runtime import models
from margin.runtime.binaries import ensure_llama_server
from margin.runtime.client import LlamaClient
from margin.runtime.server import LlamaServer, ServerConfig
from margin.study import Study

SIDS = ("3.4", "5.3", "13.2")
COUNT, MARKS = 2, 2
OFF_TOPIC = "Photosynthesis in plants converts light energy into chemical energy stored in glucose."
OUT_DIR = Path(__file__).parent / "results" / "mistakes"


def main() -> int:
    paths = config.paths().ensure()
    source = paths.workspaces_dir / "eval-uphys1.db"
    if not source.exists():
        print(f"no workspace at {source}; run evals.retrieval_eval first")
        return 1
    ws_path = paths.workspaces_dir / "eval-mistakes.db"
    shutil.copyfile(source, ws_path)
    hw = hardware.detect()
    backend = hardware.choose_backend(hw)
    spec = models.get(models.DEFAULT_GPU_MODEL if backend != "cpu" else models.DEFAULT_CPU_MODEL)
    cfg = ServerConfig(exe=ensure_llama_server(paths.bin_dir, hw.os, hw.arch, backend), model=spec.path(paths.models_dir),
                       gpu=backend != "cpu", threads=hardware.default_threads(hw), ctx=16384, chat_template_file=spec.chat_template_path())

    now = [datetime.now(timezone.utc)]
    asked: dict[str, dict] = {}
    with LlamaServer(cfg, paths.logs_dir / f"mistakes-eval-{spec.id}.log") as server, blocked_network() as attempts:
        client = LlamaClient(server.base_url)
        study = Study(ws_path, paths.logs_dir / "mistakes-eval-exports", lambda: client, clock=lambda: now[0])
        for sid in SIDS:
            questions, _ = study.generate_questions(sid, COUNT, MARKS)
            for i, q in enumerate(questions):
                grade = study.mark(i, OFF_TOPIC)
                asked[q.question] = {"section": sid, "question": q.question, "max": q.marks, "off_topic_marks": grade.marks_awarded}
                print(f"{sid}: off-topic answer {grade.marks_awarded}/{q.marks}", flush=True)
        due_same_day = len(study.mistakes())
        now[0] += timedelta(days=2)
        retries = study.load_retries()
        for i, q in enumerate(retries):
            grade = study.mark(i, q.answer)
            asked[q.question]["retry_marks"] = grade.marks_awarded
            print(f"retry {q.section}: model answer {grade.marks_awarded}/{q.marks}", flush=True)
        now[0] += timedelta(days=400)
        still_due = {q.question for q in study.load_retries()}
        client.close()

    rows = list(asked.values())
    lost = [r for r in rows if r["off_topic_marks"] < r["max"]]
    full = [r for r in rows if r.get("retry_marks") == r["max"]]
    summary = {
        "model": spec.id, "backend": backend, "questions": len(rows), "off_topic_lost_marks": len(lost),
        "due_same_day": due_same_day, "retries_after_two_days": len(retries), "retry_full_marks": len(full),
        "cleared_after_full_marks": sum(1 for r in full if r["question"] not in still_due),
        "partial_retries_still_due": sum(1 for r in lost if r not in full and r["question"] in still_due),
        "network_attempts": len(attempts), "run_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    ok = (bool(rows) and due_same_day == 0 and len(retries) == len(lost) and summary["cleared_after_full_marks"] == len(full)
          and summary["partial_retries_still_due"] == len(lost) - len(full) and not attempts)
    summary["passed"] = ok
    print(json.dumps(summary, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "uphys1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "uphys1-attempts.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
