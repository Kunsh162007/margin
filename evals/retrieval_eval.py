"""Score retrieval on a textbook against its own end-of-chapter questions.

    uv run python -m evals.retrieval_eval D:/corpus/university-physics-volume-1_-_WEB.pdf

The book is indexed once into ``MARGIN_HOME/workspaces/eval-<book>.db`` and
reused. Four arms run on the same index, so differences are due to the ranking
method alone:

* ``keyword`` — FTS5 BM25 only
* ``vector`` — bge-small cosine only
* ``hybrid`` — both, fused by reciprocal rank
* ``hybrid+rerank`` — hybrid pool reranked by the MiniLM cross-encoder

A hit means a top-k chunk belongs to the gold section. Exercise sections are
excluded from the index's default search, so a question cannot find itself.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from margin import config
from margin.retrieve.embed import Embedder, Reranker
from margin.retrieve.search import CANDIDATE_POOL, Searcher, index_document, rrf
from margin.store.db import Workspace

GOLD = Path(__file__).parent / "datasets" / "retrieval_gold.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "retrieval"
TOP = 10


def load_gold(book: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["book"] == book]


def score(ranked_sids: list[str], gold: str) -> dict[str, float]:
    rank = next((i + 1 for i, sid in enumerate(ranked_sids[:TOP]) if sid == gold), None)
    chapter = gold.split(".")[0]
    return {
        "hit@1": float(rank == 1),
        "hit@5": float(rank is not None and rank <= 5),
        "mrr@10": 1.0 / rank if rank else 0.0,
        "chapter_hit@5": float(any(sid.split(".")[0] == chapter for sid in ranked_sids[:5])),
    }


def summarise(per_question: list[tuple[dict[str, Any], dict[str, float]]]) -> dict[str, Any]:
    keys = ("hit@1", "hit@5", "mrr@10", "chapter_hit@5")
    by_kind: dict[str, list[dict[str, float]]] = defaultdict(list)
    for case, s in per_question:
        by_kind[case["kind"]].append(s)
    overall = {k: round(mean(s[k] for _, s in per_question), 4) for k in keys}
    return {**overall, "n": len(per_question), "by_kind": {kind: {k: round(mean(s[k] for s in rows), 4) for k in keys} | {"n": len(rows)} for kind, rows in by_kind.items()}}


def run_arm(name: str, ranker: Callable[[str], list[str]], gold: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    per_question = [(case, score(ranker(case["question"]), case["gold_section"])) for case in gold]
    result = summarise(per_question)
    result["ms_per_query"] = round(1000 * (time.perf_counter() - started) / max(1, len(gold)), 1)
    print(f"{name:>14}: hit@1 {result['hit@1']:.3f}  hit@5 {result['hit@5']:.3f}  mrr@10 {result['mrr@10']:.3f}  chapter@5 {result['chapter_hit@5']:.3f}  {result['ms_per_query']} ms/query")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--book", default="uphys1")
    args = ap.parse_args(argv)

    gold = load_gold(args.book)
    if not gold:
        print(f"no gold questions for book {args.book!r}; build them with evals.datasets.build_retrieval_gold")
        return 1
    embedder, reranker = Embedder(), Reranker()
    ws_path = config.paths().ensure().workspaces_dir / f"eval-{args.book}.db"
    with Workspace.open(ws_path) as ws:
        started = time.perf_counter()
        indexed = index_document(ws, args.pdf, embedder)
        index_seconds = round(time.perf_counter() - started, 1)
        chunks = ws.chunk_count()
        print(f"index: {chunks} searchable chunks ({'reused' if indexed.skipped else f'built in {index_seconds} s'})")
        searcher = Searcher(ws, embedder, reranker)

        def sids(ids: list[int]) -> list[str]:
            return [row.sid for row in ws.chunk_rows(ids)]

        arms = {
            "keyword": lambda q: sids([cid for cid, _ in ws.fts(q, TOP)]),
            "vector": lambda q: sids([cid for cid, _ in searcher.dense(q, TOP)]),
            "hybrid": lambda q: sids([cid for cid, _ in rrf([c for c, _ in ws.fts(q, CANDIDATE_POOL)], [c for c, _ in searcher.dense(q)])][:TOP]),
            "hybrid+rerank": lambda q: [h.row.sid for h in searcher.search(q, top_k=TOP)],
        }
        results = {name: run_arm(name, ranker, gold) for name, ranker in arms.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = {"book": args.book, "questions": len(gold), "chunks": chunks, "index_seconds": None if indexed.skipped else index_seconds, "arms": results, "run_at": datetime.now().strftime("%Y%m%d_%H%M%S")}
    (OUT_DIR / f"{args.book}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
