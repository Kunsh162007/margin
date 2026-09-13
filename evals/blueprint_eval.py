"""Score the exam blueprint on synthetic past papers whose true weights are known.

    uv run python -m evals.blueprint_eval                     # 5 seeds, eval-uphys1 workspace
    uv run python -m evals.blueprint_eval --seeds 7

Real past papers have no ground truth for "which section does this question
test". So papers are built from the 1,770 labelled University Physics
questions: an examiner favours 6 of the 17 chapters, each paper draws 12
questions from those chapters, conceptual questions carry 2–3 marks and
problems 5–10, and years run 2016–2025. Papers are rendered as text in the
formats real papers use — ``1. … (5 marks)``, ``Q3. (a) … [2 marks]`` with a
``(b)`` line, a page footer — then read back by Margin's own parser.

Two things are scored:

* **parsing** — questions recovered, marks and year read correctly;
* **the blueprint** — for two weighting arms (best match only, and a 60/25/15
  split over the top three sections): how many of the true top-10 sections the
  blueprint's top 10 contains, what share of the true weight those sections
  hold, how far the chapter distribution is from the truth (half the L1
  distance: 0 perfect, 1 disjoint), and how many of the examiner's favourite
  chapters appear in the blueprint's top 6 chapters.

Each seed draws a different examiner and different papers. Results are reported
per seed and as mean, min and max, because one seed of 20 papers is too few to
choose a default on.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from evals.matchers import squash
from margin import config
from margin.blueprint.paper import PaperQuestion, parse_paper
from margin.blueprint.weights import RANK_SPLIT, Blueprint, SearcherLike, build_blueprint, question_weight
from margin.retrieve.embed import Embedder
from margin.retrieve.search import Searcher
from margin.store.db import Workspace

GOLD = Path(__file__).parent / "datasets" / "retrieval_gold.jsonl"
OUT_DIR = Path(__file__).parent / "results" / "blueprint"
PAPERS = 20
QUESTIONS_PER_PAPER = 12
EXAM_CHAPTERS = 6
FIRST_YEAR = 2016
SEED = 7
DEFAULT_SEEDS = "7,11,13,17,19"
MARKS_BY_KIND = {"conceptual": (2, 3), "problem": (5, 8, 10)}
ARMS = {"best-match-only": (1.0,), "split-top3": RANK_SPLIT}
TOP_SECTIONS = 10
METRICS = ("top10_overlap", "weight_captured_top10", "chapter_distance", "favourite_chapters_found")


def make_papers(gold: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict[str, Any]], list[int]]:
    by_chapter: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for q in gold:
        by_chapter[q["chapter"]].append(q)
    favourites = sorted(rng.sample(sorted(by_chapter), EXAM_CHAPTERS))
    pool = [q for ch in favourites for q in by_chapter[ch]]
    papers = []
    for i in range(PAPERS):
        picks = rng.sample(pool, QUESTIONS_PER_PAPER)
        papers.append({"year": FIRST_YEAR + i % 10, "questions": [{"gold": q, "marks": rng.choice(MARKS_BY_KIND[q["kind"]])} for q in picks]})
    return papers, favourites


def render_paper(paper: dict[str, Any]) -> str:
    lines = [f"UNIVERSITY EXAMINATION {paper['year']}", "PHYSICS I — PAPER 1", "Answer ALL questions.", ""]
    qs, i, number = paper["questions"], 0, 1
    while i < len(qs):
        if number % 3 == 0 and i + 1 < len(qs):
            lines.append(f"Q{number}. (a) {qs[i]['gold']['question']} [{qs[i]['marks']} marks]")
            lines.append(f"(b) {qs[i + 1]['gold']['question']} [{qs[i + 1]['marks']} marks]")
            i += 2
        else:
            lines.append(f"{number}. {qs[i]['gold']['question']} ({qs[i]['marks']} marks)")
            i += 1
        number += 1
        lines.append("")
        if number == 5:
            lines.append("Page 1 of 2")
    lines.append("END OF PAPER")
    return "\n".join(lines)


_LEADING_PART = re.compile(r"^\s*\(\s*(?:[a-h]|[ivx]{1,4})\s*\)\s*", re.IGNORECASE)


def _key(text: str) -> str:
    """Match on the opening words, ignoring a leading "(a)": a textbook question that starts
    with "(a)" is correctly read by the parser as part (a), with the marker removed."""
    return squash(_LEADING_PART.sub("", text))[:60]


def score_parsing(papers: list[dict[str, Any]], parsed: list[list[PaperQuestion]]) -> dict[str, float]:
    total = found = marks_ok = year_ok = 0
    for paper, questions in zip(papers, parsed):
        by_key = {_key(q.text): q for q in questions}
        for truth in paper["questions"]:
            total += 1
            match = by_key.get(_key(truth["gold"]["question"]))
            if match is None:
                continue
            found += 1
            marks_ok += int(match.marks == truth["marks"])
            year_ok += int(match.year == paper["year"])
    return {"question_recall": round(found / total, 4), "marks_accuracy": round(marks_ok / max(1, found), 4), "year_accuracy": round(year_ok / max(1, found), 4), "questions": total}


def true_weights(papers: list[dict[str, Any]]) -> dict[str, float]:
    years = (FIRST_YEAR, max(p["year"] for p in papers))
    weights: dict[str, float] = defaultdict(float)
    for paper in papers:
        for truth in paper["questions"]:
            q = PaperQuestion("x", truth["gold"]["question"], truth["marks"], paper["year"])
            weights[truth["gold"]["gold_section"]] += question_weight(q, years)
    return dict(weights)


def _chapter_shares(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values()) or 1.0
    shares: dict[str, float] = defaultdict(float)
    for sid, w in weights.items():
        shares[sid.split(".")[0]] += w / total
    return dict(shares)


def score_blueprint(truth: dict[str, float], blueprint: Blueprint, favourites: list[int]) -> dict[str, float]:
    true_top = [sid for sid, _ in sorted(truth.items(), key=lambda kv: -kv[1])[:TOP_SECTIONS]]
    predicted = {row.sid: row.weight for row in blueprint.sections}
    pred_top = [row.sid for row in blueprint.top(TOP_SECTIONS)]
    true_ch, pred_ch = _chapter_shares(truth), _chapter_shares(predicted)
    chapters = set(true_ch) | set(pred_ch)
    pred_top_chapters = [ch for ch, _ in sorted(pred_ch.items(), key=lambda kv: -kv[1])[:EXAM_CHAPTERS]]
    return {
        "top10_overlap": round(len(set(true_top) & set(pred_top)) / TOP_SECTIONS, 4),
        "weight_captured_top10": round(sum(truth.get(s, 0.0) for s in pred_top) / sum(truth.values()), 4),
        "chapter_distance": round(0.5 * sum(abs(true_ch.get(c, 0.0) - pred_ch.get(c, 0.0)) for c in chapters), 4),
        "favourite_chapters_found": round(len({str(c) for c in favourites} & set(pred_top_chapters)) / EXAM_CHAPTERS, 4),
        "unmatched_questions": blueprint.unmatched,
    }


def run_seed(seed: int, gold: list[dict[str, Any]], searcher: SearcherLike) -> dict[str, Any]:
    papers, favourites = make_papers(gold, random.Random(seed))
    parsed = [parse_paper(render_paper(p), source=f"paper{i + 1}") for i, p in enumerate(papers)]
    truth = true_weights(papers)
    questions = [q for paper in parsed for q in paper]
    arms = {arm: score_blueprint(truth, build_blueprint(searcher, questions, split), favourites) for arm, split in ARMS.items()}
    return {"seed": seed, "favourite_chapters": favourites, "parsing": score_parsing(papers, parsed), "arms": arms}


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in ARMS:
        summary[arm] = {m: {"mean": round(mean(r["arms"][arm][m] for r in runs), 4), "min": min(r["arms"][arm][m] for r in runs), "max": max(r["arms"][arm][m] for r in runs)} for m in METRICS}
    better = {m: sum(1 for r in runs if _better(m, r["arms"]["best-match-only"][m], r["arms"]["split-top3"][m])) for m in METRICS}
    summary["best-match-only_wins_by_metric"] = better
    summary["parsing_recall_mean"] = round(mean(r["parsing"]["question_recall"] for r in runs), 4)
    return summary


def _better(metric: str, a: float, b: float) -> bool:
    return a < b if metric == "chapter_distance" else a > b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default="uphys1")
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    args = ap.parse_args(argv)

    gold = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold = [g for g in gold if g["book"] == args.book]
    ws_path = config.paths().workspaces_dir / f"eval-{args.book}.db"
    if not ws_path.exists():
        print(f"no workspace at {ws_path}; run evals.retrieval_eval first")
        return 1
    runs = []
    with Workspace.open(ws_path) as ws:
        searcher = Searcher(ws, Embedder())
        for seed in (int(s) for s in args.seeds.split(",")):
            run = run_seed(seed, gold, searcher)
            runs.append(run)
            print(f"seed {seed}: parsing {run['parsing']['question_recall']:.3f}  " + "  ".join(f"{arm}: {res}" for arm, res in run["arms"].items()), flush=True)
    summary = aggregate(runs)
    print(json.dumps(summary, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = {"book": args.book, "papers_per_seed": PAPERS, "summary": summary, "runs": runs, "run_at": datetime.now().strftime("%Y%m%d_%H%M%S")}
    (OUT_DIR / f"{args.book}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
