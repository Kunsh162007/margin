"""Speed and memory on fixed prompts.

Prompt caching is turned off here so every repeat does the full work; the other
suites leave it on, as the product does. Each measurement is repeated and the
median kept, because single timings on a laptop swing with background load.
"""

from __future__ import annotations

import shutil
import subprocess
from statistics import median
from typing import Any

from margin.runtime.client import LlamaClient
from margin.runtime.server import LlamaServer

REPEATS = 3
GEN_PROMPT = "Explain Kirchhoff's two circuit laws to a first-year student in about 150 words."


def gpu_memory_used_gb() -> float | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True).stdout
        return float(out.strip().splitlines()[0]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def run(client: LlamaClient, server: LlamaServer, passages: dict[str, dict[str, Any]], vram_before_gb: float | None) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    long_text = "\n\n".join(p["text"] for p in passages.values())
    long_messages = [{"role": "user", "content": f"{long_text}\n\nIn one sentence, which subjects do these passages cover?"}]
    gen_messages = [{"role": "user", "content": GEN_PROMPT}]

    long_runs = [client.chat(long_messages, max_tokens=48, cache_prompt=False) for _ in range(REPEATS)]
    gen_runs = [client.chat(gen_messages, max_tokens=200, cache_prompt=False) for _ in range(REPEATS)]

    vram_after = gpu_memory_used_gb()
    vram = vram_after - vram_before_gb if vram_after is not None and vram_before_gb is not None else None
    rows = [{"kind": "long_prompt", "prompt_tokens": r.prompt_tokens, "prompt_tps": r.prompt_tps, "prompt_ms": r.prompt_ms} for r in long_runs]
    rows += [{"kind": "generation", "completion_tokens": r.completion_tokens, "gen_tps": r.gen_tps} for r in gen_runs]
    return {
        "speed.load_s": server.load_seconds,
        "speed.rss_gb": server.rss_gb(),
        "speed.vram_gb": vram,
        "speed.prompt_tps": median(r.prompt_tps for r in long_runs),
        "speed.ttft_long_s": median(r.prompt_ms for r in long_runs) / 1000,
        "speed.long_prompt_tokens": float(long_runs[0].prompt_tokens),
        "speed.gen_tps": median(r.gen_tps for r in gen_runs),
    }, rows
