"""Embedding and reranking with fastembed (ONNX Runtime, CPU, no PyTorch).

bge-small-en-v1.5 (384 dimensions, ~67 MB) was measured in the previous project
against larger 768-dimension models and beat two of three. On this laptop's CPU
it embeds ~236 passages per second, so a 1,300-page textbook indexes in about
20 seconds. The MiniLM cross-encoder reranks ~340 pairs per second.

Models load on first use; `margin setup` fetches them so later runs are offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from margin import config

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 64


def cache_dir() -> Path:
    return config.paths().home / "embeddings"


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)


class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, cache: Path | None = None):
        self.name = model_name
        self._cache = cache
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.name, cache_dir=str(self._cache or cache_dir()))
        return self._model

    def passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return _normalise(np.asarray(list(self._load().embed(texts, batch_size=BATCH_SIZE)), dtype=np.float32))

    def query(self, text: str) -> np.ndarray:
        vector = next(iter(self._load().query_embed(text)))
        return _normalise(np.asarray(vector, dtype=np.float32)[None, :])[0]


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL, cache: Path | None = None):
        self.name = model_name
        self._cache = cache
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self.name, cache_dir=str(self._cache or cache_dir()))
        return self._model

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._load().rerank(query, texts)] if texts else []
