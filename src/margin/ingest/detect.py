"""Identify a file by its bytes, not its name.

Extensions lie: phone photos arrive as ``.JPG``, ``.jpeg`` or no extension at
all, and a ``.pdf`` downloaded from a portal is sometimes an HTML error page.
Office files are ZIP archives, told apart by the part names inside.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

HEAD_BYTES = 16


class UnsupportedFile(ValueError):
    pass


def sniff(path: Path) -> str:
    with path.open("rb") as fh:
        head = fh.read(4096)
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"PK\x03\x04"):
        return _office_kind(path)
    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        raise UnsupportedFile(f"{path.name} is a HEIC photo; export it as JPEG or PNG first")
    if _is_text(head):
        return "text"
    raise UnsupportedFile(f"{path.name}: unrecognised file type")


def _office_kind(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile as exc:
        raise UnsupportedFile(f"{path.name}: damaged ZIP/Office file") from exc
    if "word/document.xml" in names:
        return "docx"
    if "ppt/presentation.xml" in names:
        return "pptx"
    raise UnsupportedFile(f"{path.name}: ZIP archive that is not a Word or PowerPoint file")


def _is_text(head: bytes) -> bool:
    if not head or b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # a multi-byte character cut at the 4096-byte boundary is still text
        return exc.start >= len(head) - 3
    return True
