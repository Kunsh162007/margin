"""One SQLite file per workspace.

Text, the FTS5 keyword index, vectors, notes and metadata live in the same
file, so a workspace is backed up by copying one file and cannot end up with an
index that disagrees with its text. Vectors are float32 blobs searched exactly
in numpy: a student's library is tens of thousands of chunks, which exact search
handles in milliseconds.

Exercise text (review questions, problem sets) is flagged. Default searches
skip it: a student searching the book wants the explanation, not the list of
questions about it — and an evaluation built from those questions would
otherwise retrieve the questions themselves.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from margin.ingest.types import Document, Section

SCHEMA_VERSION = "2"
_TOKEN = re.compile(r"[A-Za-z0-9]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
    id TEXT PRIMARY KEY, source TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL,
    pages INTEGER NOT NULL, ocr_pages INTEGER NOT NULL, added_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sections(
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, sid TEXT NOT NULL, ord INTEGER NOT NULL,
    title TEXT NOT NULL, level INTEGER NOT NULL, path TEXT NOT NULL, page_start INTEGER, page_end INTEGER,
    is_exercise INTEGER NOT NULL, PRIMARY KEY (doc_id, sid));
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, sid TEXT NOT NULL,
    ord INTEGER NOT NULL, text TEXT NOT NULL, page_start INTEGER, page_end INTEGER, is_exercise INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_by_section ON chunks(doc_id, sid, ord);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
CREATE TABLE IF NOT EXISTS vectors(
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE, model TEXT NOT NULL, vec BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS notes(
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE, sid TEXT NOT NULL, markdown TEXT NOT NULL,
    visuals_json TEXT NOT NULL, kept INTEGER NOT NULL, dropped INTEGER NOT NULL, seconds REAL NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, sid));
"""


@dataclass(frozen=True)
class ChunkRecord:
    sid: str
    ord: int
    text: str
    page_start: int | None
    page_end: int | None
    is_exercise: bool = False


@dataclass(frozen=True)
class ChunkRow:
    id: int
    doc_id: str
    doc_title: str
    sid: str
    section_title: str
    text: str
    page_start: int | None
    page_end: int | None


def fts_query(text: str) -> str | None:
    """Quote every word and OR them: raw student text ("Kirchhoff's law?") is not valid FTS5 syntax."""
    tokens = _TOKEN.findall(text)
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Workspace:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @classmethod
    def open(cls, path: Path) -> "Workspace":
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
        conn.commit()
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # documents -----------------------------------------------------------------

    def has_document(self, doc_id: str) -> bool:
        return self._conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone() is not None

    def add_document(self, doc: Document, sections: list[Section], chunks: list[ChunkRecord], vectors: np.ndarray, model: str, exercise_sids: set[str]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        with self._conn:
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc.id,))
            self._conn.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)", (doc.id, doc.source, doc.kind, doc.title, doc.page_count, len(doc.ocr_pages), _now()))
            self._conn.executemany(
                "INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(doc.id, s.id, i, s.title, s.level, json.dumps(list(s.path)), s.page_start, s.page_end, int(s.id in exercise_sids)) for i, s in enumerate(sections)],
            )
            for chunk, vec in zip(chunks, vectors):
                cur = self._conn.execute(
                    "INSERT INTO chunks (doc_id, sid, ord, text, page_start, page_end, is_exercise) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (doc.id, chunk.sid, chunk.ord, chunk.text, chunk.page_start, chunk.page_end, int(chunk.is_exercise)),
                )
                self._conn.execute("INSERT INTO vectors VALUES (?, ?, ?)", (cur.lastrowid, model, np.asarray(vec, dtype=np.float32).tobytes()))

    def remove_document(self, doc_id: str) -> bool:
        with self._conn:
            return self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,)).rowcount > 0

    def documents(self) -> list[dict[str, object]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM documents ORDER BY added_at")]

    # sections and chunks ---------------------------------------------------------

    def teaching_sections(self) -> list[tuple[str, str, str]]:
        """(doc_id, sid, title) of sections with explanatory text, in book order."""
        rows = self._conn.execute(
            "SELECT s.doc_id, s.sid, s.title FROM sections s JOIN documents d ON d.id = s.doc_id "
            "WHERE s.is_exercise = 0 AND EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = s.doc_id AND c.sid = s.sid AND c.is_exercise = 0) "
            "ORDER BY d.added_at, s.ord"
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def find_section(self, sid: str) -> tuple[str, str] | None:
        row = self._conn.execute(
            "SELECT s.doc_id, s.title FROM sections s JOIN documents d ON d.id = s.doc_id WHERE s.sid = ? ORDER BY d.added_at LIMIT 1", (sid,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def chunk_count(self, include_exercises: bool = False) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks WHERE ? OR is_exercise = 0", (int(include_exercises),)).fetchone()[0]

    def fts(self, query: str, limit: int, include_exercises: bool = False) -> list[tuple[int, float]]:
        match = fts_query(query)
        if match is None:
            return []
        rows = self._conn.execute(
            "SELECT c.id, bm25(chunks_fts) FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND (? OR c.is_exercise = 0) ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, int(include_exercises), limit),
        ).fetchall()
        return [(int(r[0]), -float(r[1])) for r in rows]  # bm25() is lower-is-better

    def vectors(self, model: str, include_exercises: bool = False) -> tuple[np.ndarray, np.ndarray]:
        rows = self._conn.execute(
            "SELECT v.chunk_id, v.vec FROM vectors v JOIN chunks c ON c.id = v.chunk_id "
            "WHERE v.model = ? AND (? OR c.is_exercise = 0) ORDER BY v.chunk_id",
            (model, int(include_exercises)),
        ).fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        return ids, np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])

    def chunk_rows(self, ids: list[int]) -> list[ChunkRow]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            "SELECT c.id, c.doc_id, d.title AS doc_title, c.sid, s.title AS section_title, c.text, c.page_start, c.page_end "
            f"FROM chunks c JOIN documents d ON d.id = c.doc_id JOIN sections s ON s.doc_id = c.doc_id AND s.sid = c.sid WHERE c.id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {r["id"]: ChunkRow(**dict(r)) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def section_text(self, doc_id: str, sid: str, include_exercises: bool = True) -> str | None:
        rows = self._conn.execute(
            "SELECT text FROM chunks WHERE doc_id = ? AND sid = ? AND (? OR is_exercise = 0) ORDER BY ord", (doc_id, sid, int(include_exercises))
        ).fetchall()
        return "\n\n".join(r[0] for r in rows) if rows else None

    # notes ------------------------------------------------------------------------

    def save_notes(self, doc_id: str, sid: str, markdown: str, visuals_json: str, kept: int, dropped: int, seconds: float) -> None:
        with self._conn:
            self._conn.execute("INSERT OR REPLACE INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (doc_id, sid, markdown, visuals_json, kept, dropped, seconds, _now()))

    def load_notes(self, doc_id: str, sid: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT markdown, visuals_json, kept, dropped, seconds FROM notes WHERE doc_id = ? AND sid = ?", (doc_id, sid)).fetchone()
        return dict(row) if row else None

    def clear_notes(self, sids: list[str] | None = None) -> int:
        with self._conn:
            if sids is None:
                return self._conn.execute("DELETE FROM notes").rowcount
            return self._conn.execute(f"DELETE FROM notes WHERE sid IN ({','.join('?' * len(sids))})", sids).rowcount
