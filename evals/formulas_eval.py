"""Measure the formula reader on equations people transcribed, without the language model.

    uv run python -m evals.formulas_eval

OmniDocBench (fetched and hash-checked by ``evals.datasets.fetch``) labels every
displayed equation on its pages with LaTeX. Every English displayed equation on
its book, exam-paper and textbook pages is cropped from the page image by its
labelled box and read by the formula reader Margin uses for PDFs.

Measured, overall and per page source:

- **exact** — the reading equals the label once spacing commands are removed;
- **edit distance** — character edit distance over the longer string, the
  benchmark's own measure (0 is identical);
- **same maths** — among labels the SymPy checker can read, the share whose
  reading it judges equivalent (what the question and note checks rely on);
- **plausible** — readings ``plausible`` keeps; the rest are dropped at ingestion.

Limits: the boxes are the labellers', so this scores reading, not finding;
finding is measured separately on a real textbook. The pages are photographs and
scans as well as rendered PDFs, which is harder than a PDF crop.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean

from PIL import Image

from evals.datasets import fetch
from margin import config
from margin.ingest.formulas import MIN_CONFIDENCE, installed_reader, plausible
from margin.practice.maths import equivalent, parse

OUT_DIR = Path(__file__).parent / "results" / "formulas"
SOURCES = ("book", "exam_paper", "colorful_textbook")
_SPACING = re.compile(r"\\(?:left|right|displaystyle|textstyle|quad|qquad)\b|\\[,;:! ]|~|\s+")


def normalise(latex: str) -> str:
    return _SPACING.sub("", latex.strip().removeprefix("$$").removesuffix("$$").strip("$"))


def edit_distance(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1] / max(len(a), len(b))


def equations(folder: Path) -> list[dict]:
    pages = json.loads((folder / "OmniDocBench.json").read_text(encoding="utf-8"))
    found = []
    for page in pages:
        attrs = page["page_info"]["page_attribute"]
        if attrs.get("data_source") not in SOURCES or attrs.get("language") != "english":
            continue
        for det in page["layout_dets"]:
            if det["category_type"] == "equation_isolated" and det.get("latex") and not det.get("ignore"):
                xs, ys = det["poly"][0::2], det["poly"][1::2]
                found.append({"image": page["page_info"]["image_path"], "source": attrs["data_source"],
                              "box": (min(xs), min(ys), max(xs), max(ys)), "label": det["latex"]})
    return found


def _share(rows: list[dict], key: str) -> float | None:
    return round(mean(bool(r[key]) for r in rows), 4) if rows else None


def score(rows: list[dict]) -> dict[str, float | int | None]:
    comparable = [r for r in rows if parse(normalise(r["label"])) is not None]
    misread = [r for r in comparable if not r["same_maths"]]
    correct = [r for r in comparable if r["same_maths"]]
    return {
        "equations": len(rows),
        "exact_rate": _share(rows, "exact"),
        "mean_edit_distance": round(mean(r["distance"] for r in rows), 4) if rows else None,
        "sympy_comparable": len(comparable),
        "sympy_match_rate": _share(comparable, "same_maths"),
        "plausible_rate": _share(rows, "plausible"),
        "uncertain_rate": _share(rows, "uncertain"),  # readings marked so formula checks ignore them
        "misreads_marked_rate": _share(misread, "uncertain"),  # of checkable misreads, the share the mark keeps out of checks
        "correct_marked_rate": _share(correct, "uncertain"),  # of correct readings, the share checks lose to the mark
    }


def main() -> int:
    reader = installed_reader(config.paths().models_dir)
    if reader is None:
        print("the formula reader is not installed; run: margin setup")
        return 1
    folder = fetch.ensure("omnidocbench")
    rows = []
    items = equations(folder)
    for n, item in enumerate(items, 1):
        with Image.open(folder / "images" / item["image"]) as page:
            reading, confidence = reader.read_scored(page.convert("RGB").crop(tuple(int(v) for v in item["box"])))
        label = normalise(item["label"])
        rows.append({**item, "reading": reading, "exact": normalise(reading) == label, "distance": edit_distance(normalise(reading), label),
                     "same_maths": equivalent(reading, label), "plausible": plausible(reading),
                     "confidence": confidence, "uncertain": confidence < MIN_CONFIDENCE})
        if n % 100 == 0:
            print(f"{n}/{len(items)}", flush=True)
    summary = {
        "dataset": "OmniDocBench (English displayed equations on book, exam-paper and textbook pages)", "min_confidence": MIN_CONFIDENCE,
        "overall": score(rows), "by_source": {s: score([r for r in rows if r["source"] == s]) for s in SOURCES},
        "run_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    print(json.dumps(summary, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "omnidocbench.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
