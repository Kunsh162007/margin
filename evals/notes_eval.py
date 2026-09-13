"""Write real notes with the default model and measure them.

    uv run python -m evals.notes_eval                # GPU if present, else CPU

Six University Physics sections from six chapters are written by the default
model (Gemma 4 E4B) through the real loop, executor, verifier and checkpoint
store, in the eval workspace built by ``retrieval_eval``. The run is split: the
first three sections are written, the process "stops", and a second run over
all six must reuse those three and write only the rest — resume after an
interruption between sections.

Measured: share of note sentences the section supports (lexical, see
``notes/verify.py``), sentences dropped, visuals made and whether they render,
seconds per written section, and whether resume reused exactly the finished
sections. The notes themselves are saved next to the JSON for reading.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

from margin import config, hardware
from margin.agent.executor import ToolExecutor
from margin.notes.writer import SectionNotes, to_markdown, write_notes
from margin.runtime import models
from margin.runtime.binaries import ensure_llama_server
from margin.runtime.client import LlamaClient
from margin.runtime.server import LlamaServer, ServerConfig
from margin.store.db import Workspace
from margin.visuals.render import problems

SIDS = ["3.4", "5.3", "6.2", "10.6", "13.2", "14.5"]
FIRST_RUN = 3
OUT_DIR = Path(__file__).parent / "results" / "notes"


def main() -> int:
    paths = config.paths().ensure()
    ws_path = paths.workspaces_dir / "eval-uphys1.db"
    if not ws_path.exists():
        print(f"no workspace at {ws_path}; run evals.retrieval_eval first")
        return 1
    hw = hardware.detect()
    backend = hardware.choose_backend(hw)
    spec = models.get(models.DEFAULT_GPU_MODEL if backend != "cpu" else models.DEFAULT_CPU_MODEL)
    cfg = ServerConfig(exe=ensure_llama_server(paths.bin_dir, hw.os, hw.arch, backend), model=spec.path(paths.models_dir),
                       gpu=backend != "cpu", threads=hardware.default_threads(hw), ctx=16384, chat_template_file=spec.chat_template_path())
    log: list[tuple[str, bool, SectionNotes]] = []

    with Workspace.open(ws_path) as ws, LlamaServer(cfg, paths.logs_dir / f"notes-eval-{spec.id}.log") as server:
        ws.clear_notes(SIDS)
        client = LlamaClient(server.base_url, tools_template_kwarg=spec.tools_template_kwarg)
        executor = ToolExecutor(ws)
        record = lambda sid, notes, resumed: (log.append((sid, resumed, notes)), print(f"{sid}: {'reused' if resumed else f'{notes.seconds:.1f} s'}, kept {notes.kept}, dropped {notes.dropped}, visuals {len(notes.visuals)}", flush=True))
        write_notes(client, ws, sids=SIDS[:FIRST_RUN], executor=executor, on_progress=record)
        print("-- second run over all sections --", flush=True)
        second = write_notes(client, ws, sids=SIDS, executor=executor, on_progress=record)
        client.close()

    second_run = log[FIRST_RUN:]
    reused = [sid for sid, resumed, _ in second_run if resumed]
    written = [n for _, resumed, n in log if not resumed]
    kept, dropped = sum(n.kept for n in written), sum(n.dropped for n in written)
    visuals = [v for n in written for v in n.visuals]
    summary = {
        "model": spec.id, "backend": backend, "sections": len(second),
        "support_rate": round(kept / max(1, kept + dropped), 4), "sentences_kept": kept, "sentences_dropped": dropped,
        "visuals": len(visuals), "visual_kinds": sorted({v.kind for v in visuals}), "visuals_render_ok": sum(1 for v in visuals if not problems(v)),
        "sections_with_visuals": sum(1 for n in written if n.visuals), "seconds_per_section": round(mean(n.seconds for n in written), 1) if written else None,
        "resumed_ok": reused == SIDS[:FIRST_RUN] and len(written) == len(SIDS), "run_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    print(json.dumps(summary, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "uphys1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "uphys1-notes.md").write_text(to_markdown(second), encoding="utf-8")
    return 0 if summary["resumed_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
