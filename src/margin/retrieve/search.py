"""Index files into a workspace and search them.

Search is hybrid: FTS5 keyword ranking and dense vector ranking are fused with
reciprocal rank fusion, then a cross-encoder reranks the fused pool. Keyword
search catches the exact terms textbooks are full of ("Thévenin", "nephron");
vectors catch paraphrases ("why do bridges sag"); fusion by rank needs no score
calibration between the two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from margin.ingest.images import OcrEngine
from margin.ingest.pipeline import ingest_file
from margin.retrieve.chunking import chunk_section, is_exercise_section
from margin.store.db import ChunkRow, Workspace

RRF_K = 60
CANDIDATE_POOL = 50
RERANK_POOL = 20
_SCOPE_PREFIX = re.compile(r"^(?:ch(?:apter)?\.?\s*)", re.IGNORECASE)


class EmbedderLike(Protocol):
    name: str

    def passages(self, texts: list[str]) -> np.ndarray: ...

    def query(self, text: str) -> np.ndarray: ...


class RerankerLike(Protocol):
    def scores(self, query: str, texts: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class Hit:
    row: ChunkRow
    score: float


@dataclass(frozen=True)
class IndexResult:
    doc_id: str
    title: str
    sections: int
    chunks: int
    skipped: bool


def rrf(*rankings: list[int], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def in_scope(row: ChunkRow, scope: str | None) -> bool:
    """'ch4' or 'chapter 4' matches sections 4 and 4.x; '4.2' matches 4.2 and 4.2.x; a document id matches that document."""
    if not scope:
        return True
    target = _SCOPE_PREFIX.sub("", scope.strip())
    return row.doc_id == scope or row.sid == target or row.sid.startswith(target + ".")


def index_document(ws: Workspace, path: Path, embedder: EmbedderLike, ocr: OcrEngine | None = None) -> IndexResult:
    doc, sections = ingest_file(path, ocr)
    if ws.has_document(doc.id):
        return IndexResult(doc.id, doc.title, 0, 0, skipped=True)
    chunks = [chunk for section in sections for chunk in chunk_section(section)]
    vectors = embedder.passages([c.text for c in chunks]) if chunks else np.empty((0, 0), dtype=np.float32)
    exercise_sids = {s.id for s in sections if is_exercise_section(s.title)}
    ws.add_document(doc, sections, chunks, vectors, embedder.name, exercise_sids)
    return IndexResult(doc.id, doc.title, len(sections), len(chunks), skipped=False)


class Searcher:
    def __init__(self, ws: Workspace, embedder: EmbedderLike, reranker: RerankerLike | None = None):
        self._ws = ws
        self._embedder = embedder
        self._reranker = reranker
        self._cache_key: int | None = None
        self._ids = np.empty(0, dtype=np.int64)
        self._matrix = np.empty((0, 0), dtype=np.float32)

    def _vectors(self) -> tuple[np.ndarray, np.ndarray]:
        key = self._ws.chunk_count()
        if key != self._cache_key:
            self._ids, self._matrix = self._ws.vectors(self._embedder.name)
            self._cache_key = key
        return self._ids, self._matrix

    def dense(self, query: str, limit: int = CANDIDATE_POOL) -> list[tuple[int, float]]:
        ids, matrix = self._vectors()
        if not len(ids):
            return []
        sims = matrix @ self._embedder.query(query)
        top = np.argsort(-sims)[:limit]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def search(self, query: str, top_k: int = 5, scope: str | None = None, rerank: bool = False) -> list[Hit]:
        """Hybrid search. Reranking is opt-in: on the textbook gold set it lowered hit@5 from
        0.596 to 0.559 and cost ~850 ms per query on CPU against ~44 ms (DESIGN.md D24)."""
        lexical = [cid for cid, _ in self._ws.fts(query, CANDIDATE_POOL)]
        semantic = [cid for cid, _ in self.dense(query)]
        fused = rrf(lexical, semantic)
        fused_scores = dict(fused)
        rows = [r for r in self._ws.chunk_rows([cid for cid, _ in fused]) if in_scope(r, scope)][:RERANK_POOL]
        if rerank and self._reranker is not None and rows:
            ranked = sorted(zip(rows, self._reranker.scores(query, [r.text for r in rows])), key=lambda rs: -rs[1])
        else:
            ranked = [(r, fused_scores[r.id]) for r in rows]
        return [Hit(row, float(score)) for row, score in ranked[:top_k]]
