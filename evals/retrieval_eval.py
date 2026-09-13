"""Score retrieval on a textbook against its own end-of-chapter questions.

    uv run python -m evals.retrieval_eval D:/corpus/university-physics-volume-1_-_WEB.pdf

The book is indexed once into ``MARGIN_HOME/workspaces/eval-<book>.db`` and
reused. Four arms run on the same index, so differences are due to the ranking
method alone:

* ``keyword`` — FTS5 BM25 only
* ``vector`` — bge-small cosine only
* ``hybrid`` — both, fused by reciprocal rank
* ``hybrid+rerank`` — hybrid pool reranked by the MiniLM cross-encoder

A hit means a top-k chunk belongs to the gold section. Exercise text is
excluded from default search, so a question cannot find itself.

**Leak guard.** Each arm reports the section title that most often ranks first.
If one title takes more than 20% of first places, the run prints a warning:
that pattern — not a low score — is what exposed exercise text leaking into the
index the first time this ran.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from margin import config
from margin.retrieve.embed import Embedder, Reranker
from margin.retrieve.search import CANDIDATE_POOL, Searcher, index_document, rrf
from margin.store.db import ChunkRow, Workspace

GOLD = Path(__file__).parent / "datasets" / "retrieval_gold.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "retrieval"
TOP = 10
LEAK_SHARE = 0.20

Ranker = Callable[[str], list[ChunkRow]]


def load_gold(book: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r["book"] == book]


def score(ranked: list[ChunkRow], gold: str) -> dict[str, float]:
    sids = [r.sid for r in ranked[:TOP]]
    rank = next((i + 1 for i, sid in enumerate(sids) if sid == gold), None)
    chapter = gold.split(".")[0]
    return {
        "hit@1": float(rank == 1),
        "hit@5": float(rank is not None and rank <= 5),
        "mrr@10": 1.0 / rank if rank else 0.0,
        "chapter_hit@5": float(any(sid.split(".")[0] == chapter for sid in sids[:5])),
    }


def summarise(per_question: list[tuple[dict[str, Any], dict[str, float]]]) -> dict[str, Any]:
    keys = ("hit@1", "hit@5", "mrr@10", "chapter_hit@5")
    by_kind: dict[str, list[dict[str, float]]] = defaultdict(list)
    for case, s in per_question:
        by_kind[case["kind"]].append(s)
    overall = {k: round(mean(s[k] for _, s in per_question), 4) for k in keys}
    kinds = {kind: {k: round(mean(s[k] for s in rows), 4) for k in keys} | {"n": len(rows)} for kind, rows in by_kind.items()}
    return {**overall, "n": len(per_question), "by_kind": kinds}


def leak_check(first_titles: Counter[str], n: int) -> dict[str, Any]:
    title, count = first_titles.most_common(1)[0] if first_titles else ("", 0)
    share = count / n if n else 0.0
    return {"top1_title": title, "top1_share": round(share, 4), "leak_warning": share > LEAK_SHARE}


def run_arm(name: str, ranker: Ranker, gold: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    per_question = []
    first_titles: Counter[str] = Counter()
    for case in gold:
        ranked = ranker(case["question"])
        if ranked:
            first_titles[ranked[0].section_title] += 1
        per_question.append((case, score(ranked, case["gold_section"])))
    result = summarise(per_question) | leak_check(first_titles, len(gold))
    result["ms_per_query"] = round(1000 * (time.perf_counter() - started) / max(1, len(gold)), 1)
    print(
        f"{name:>14}: hit@1 {result['hit@1']:.3f}  hit@5 {result['hit@5']:.3f}  mrr@10 {result['mrr@10']:.3f}  "
        f"chapter@5 {result['chapter_hit@5']:.3f}  {result['ms_per_query']} ms/query  "
        f"most common #1: {result['top1_title'][:30]!r} {result['top1_share']:.0%}",
        flush=True,
    )
    if result["leak_warning"]:
        print(f"   WARNING: one section title ranks first for {result['top1_share']:.0%} of questions — check for leaked exercise text", flush=True)
    return result


def build_arms(ws: Workspace, searcher: Searcher) -> dict[str, Ranker]:
    return {
        "keyword": lambda q: ws.chunk_rows([cid for cid, _ in ws.fts(q, TOP)]),
        "vector": lambda q: ws.chunk_rows([cid for cid, _ in searcher.dense(q, TOP)]),
        "hybrid": lambda q: ws.chunk_rows([cid for cid, _ in rrf([c for c, _ in ws.fts(q, CANDIDATE_POOL)], [c for c, _ in searcher.dense(q)])][:TOP]),
        "hybrid+rerank": lambda q: [h.row for h in searcher.search(q, top_k=TOP)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--book", default="uphys1")
    ap.add_argument("--arms", default="keyword,vector,hybrid,hybrid+rerank")
    args = ap.parse_args(argv)

    gold = load_gold(args.book)
    if not gold:
        print(f"no gold questions for book {args.book!r}; build them with evals.datasets.build_retrieval_gold")
        return 1
    embedder = Embedder()
    ws_path = config.paths().ensure().workspaces_dir / f"eval-{args.book}.db"
    with Workspace.open(ws_path) as ws:
        started = time.perf_counter()
        indexed = index_document(ws, args.pdf, embedder)
        index_seconds = round(time.perf_counter() - started, 1)
        chunks, all_chunks = ws.chunk_count(), ws.chunk_count(include_exercises=True)
        print(f"index: {chunks} searchable of {all_chunks} chunks ({'reused' if indexed.skipped else f'built in {index_seconds} s'})", flush=True)
        searcher = Searcher(ws, embedder, Reranker())
        arms = build_arms(ws, searcher)
        results = {name: run_arm(name, arms[name], gold) for name in args.arms.split(",")}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "book": args.book, "questions": len(gold), "chunks_searchable": chunks, "chunks_total": all_chunks,
        "index_seconds": None if indexed.skipped else index_seconds, "arms": results, "run_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    (OUT_DIR / f"{args.book}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 1 if any(r["leak_warning"] for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
