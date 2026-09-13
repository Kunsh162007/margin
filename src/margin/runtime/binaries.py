"""Fetch a prebuilt llama.cpp server for this platform.

The build is pinned. llama.cpp ships several releases a day and chat-template
and tool-call parsing change between them; an unpinned runtime would make every
evaluation number unrepeatable.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

from margin.runtime.download import download

LLAMA_CPP_BUILD = "b10936"
RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{build}/{asset}"
CUDA_VERSION = "12.4"


class UnsupportedPlatform(RuntimeError):
    pass


def asset_names(os_name: str, arch: str, backend: str, build: str = LLAMA_CPP_BUILD) -> list[str]:
    """Release assets needed for (os, arch, backend). CUDA on Windows also needs the runtime DLLs."""
    prefix = f"llama-{build}-bin"
    if os_name == "windows":
        if backend == "cuda":
            return [f"{prefix}-win-cuda-{CUDA_VERSION}-{arch}.zip", f"cudart-llama-bin-win-cuda-{CUDA_VERSION}-{arch}.zip"]
        if backend in ("cpu", "vulkan"):
            return [f"{prefix}-win-{backend}-{arch}.zip"]
    if os_name == "linux":
        if backend == "vulkan":
            return [f"{prefix}-ubuntu-vulkan-{arch}.tar.gz"]
        if backend == "cpu":
            return [f"{prefix}-ubuntu-{arch}.tar.gz"]
    if os_name == "macos" and backend in ("metal", "cpu"):
        return [f"{prefix}-macos-{arch}.tar.gz"]
    raise UnsupportedPlatform(f"no llama.cpp build for os={os_name} arch={arch} backend={backend}")


def _extract(archive: Path, into: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(into, filter="data")


def find_server(root: Path) -> Path | None:
    for name in ("llama-server.exe", "llama-server"):
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    return None


def ensure_llama_server(bin_dir: Path, os_name: str, arch: str, backend: str) -> Path:
    target = bin_dir / f"{LLAMA_CPP_BUILD}-{backend}"
    existing = find_server(target) if target.exists() else None
    if existing:
        return existing

    staging = bin_dir / f".{LLAMA_CPP_BUILD}-{backend}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    for asset in asset_names(os_name, arch, backend):
        archive = download(RELEASE_URL.format(build=LLAMA_CPP_BUILD, asset=asset), bin_dir / "downloads" / asset)
        _extract(archive, staging)

    server = find_server(staging)
    if server is None:
        raise UnsupportedPlatform(f"llama-server not found in {LLAMA_CPP_BUILD} {backend} build")
    if os_name != "windows":
        server.chmod(0o755)
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    found = find_server(target)
    if found is None:
        raise UnsupportedPlatform(f"llama-server missing after install into {target}")
    return found
