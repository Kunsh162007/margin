"""Resumable file download with a progress bar.

Model files are gigabytes; a dropped connection must not restart from zero.
Bytes go to ``<name>.part`` and the file is renamed only when complete, so a
half-downloaded model is never mistaken for a finished one.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

CHUNK = 1024 * 1024


class DownloadError(RuntimeError):
    pass


def download(url: str, dest: Path, *, label: str | None = None, timeout: float = 60.0) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    try:
        with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout) as resp:
            if resp.status_code == 416:  # already complete
                part.rename(dest)
                return dest
            if resp.status_code not in (200, 206):
                raise DownloadError(f"{url} returned HTTP {resp.status_code}")
            if resp.status_code == 200:
                offset = 0  # server ignored the range; start over
            total = int(resp.headers.get("content-length", 0)) + offset
            _stream_to(resp, part, offset, total, label or dest.name)
    except httpx.HTTPError as exc:
        raise DownloadError(f"download of {label or dest.name} failed: {exc}") from exc

    part.rename(dest)
    return dest


def _stream_to(resp: httpx.Response, part: Path, offset: int, total: int, label: str) -> None:
    mode = "ab" if offset else "wb"
    columns = ("[progress.description]{task.description}", BarColumn(), DownloadColumn(), TransferSpeedColumn())
    with Progress(*columns, transient=True) as progress, part.open(mode) as fh:
        task = progress.add_task(label, total=total or None, completed=offset)
        for chunk in resp.iter_bytes(CHUNK):
            fh.write(chunk)
            progress.advance(task, len(chunk))
