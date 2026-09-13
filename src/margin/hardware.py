"""Detect what this machine can run, and pick an inference backend for it.

The backend decides which prebuilt llama.cpp binary is downloaded. A GPU is
used when one is found, and the CPU build is always a valid fallback: Margin
must work on a laptop with no GPU at all.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass

import psutil

ENV_BACKEND = "MARGIN_BACKEND"
BACKENDS = ("cpu", "cuda", "vulkan", "metal")


@dataclass(frozen=True)
class Gpu:
    vendor: str
    name: str
    vram_gb: float


@dataclass(frozen=True)
class Hardware:
    os: str  # "windows" | "linux" | "macos"
    arch: str  # "x64" | "arm64"
    physical_cores: int
    logical_cores: int
    ram_gb: float
    gpu: Gpu | None


def _os_name() -> str:
    system = platform.system().lower()
    return {"darwin": "macos"}.get(system, system)


def _arch() -> str:
    machine = platform.machine().lower()
    return "arm64" if machine in ("arm64", "aarch64") else "x64"


def _nvidia_gpu() -> Gpu | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    name, _, mem = first.rpartition(",")
    try:
        return Gpu(vendor="nvidia", name=name.strip(), vram_gb=round(float(mem) / 1024, 1))
    except ValueError:
        return None


def detect() -> Hardware:
    return Hardware(
        os=_os_name(),
        arch=_arch(),
        physical_cores=psutil.cpu_count(logical=False) or 1,
        logical_cores=psutil.cpu_count(logical=True) or 1,
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        gpu=_nvidia_gpu(),
    )


def choose_backend(hw: Hardware, override: str | None = None) -> str:
    """Pick the llama.cpp build for this machine.

    llama.cpp publishes CUDA builds for Windows only; on Linux the portable GPU
    build is Vulkan. Apple Silicon builds include Metal.
    """
    override = override or os.environ.get(ENV_BACKEND)
    if override:
        if override not in BACKENDS:
            raise ValueError(f"{ENV_BACKEND} must be one of {BACKENDS}, got {override!r}")
        return override
    if hw.os == "macos" and hw.arch == "arm64":
        return "metal"
    if hw.gpu is not None and hw.gpu.vendor == "nvidia":
        return "cuda" if hw.os == "windows" else "vulkan"
    return "cpu"


def default_threads(hw: Hardware) -> int:
    """Physical cores: token generation is memory-bound and hyperthreads slow it down."""
    return max(1, hw.physical_cores)
