"""Browser code embedded in exported HTML, downloaded once by ``margin setup``.

An exported page must draw its diagrams with no network (D15), so Mermaid is
written into the file instead of loaded from a CDN. The download is pinned by
version and SHA-256: a file that does not match is deleted, never embedded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from margin import config
from margin.runtime.download import DownloadError, download

MERMAID_VERSION = "12.0.0"
MERMAID_URL = f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/mermaid.min.js"
MERMAID_SHA256 = "28fca7ae6ebc7ed7bb63bde63136a74bfef14f296a57e403657eeb8b32836073"


def mermaid_path(home: Path | None = None) -> Path:
    return (home or config.paths().home) / "assets" / f"mermaid-{MERMAID_VERSION}.min.js"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_mermaid(home: Path | None = None) -> Path:
    path = mermaid_path(home)
    if path.exists() and _sha256(path) == MERMAID_SHA256:
        return path
    path.unlink(missing_ok=True)
    download(MERMAID_URL, path, label="diagram renderer")
    if _sha256(path) != MERMAID_SHA256:
        path.unlink()
        raise DownloadError(f"the diagram renderer did not match its pinned checksum (mermaid {MERMAID_VERSION})")
    return path


def mermaid_script(home: Path | None = None) -> str | None:
    """The renderer's source if setup has fetched it, else None (diagrams are then shown as text)."""
    path = mermaid_path(home)
    return path.read_text(encoding="utf-8") if path.exists() else None
