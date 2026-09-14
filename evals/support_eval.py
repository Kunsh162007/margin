"""Measure the note-sentence support check on RAGTruth, without a model.

    uv run python -m evals.support_eval

RAGTruth holds answers written by six language models from retrieved passages,
with every unsupported or contradicting span marked by annotators. Its QA task
is the closest public match to notes written from a textbook section. Each
answer is split into sentences the way notes are; a sentence counts as
hallucinated when an annotated span overlaps it. Sentences with fewer than three
content words are always kept by the checker and are left out of both rates.

Reported per check on the test split: **catch rate** (hallucinated sentences
dropped) and **false-drop rate** (clean sentences dropped).

- ``lexical`` — the shipped check (``notes/verify.py``);
- ``meaning`` — supported when the sentence's bge-small embedding has cosine at
  least τ with some sentence of the passages;
- ``lexical_or_meaning`` — kept when either supports it.

τ is chosen on the first 1,000 training answers by maximising catch rate minus
false-drop rate for ``lexical_or_meaning``, then fixed for the test split.

**Decision rule, written before the test split was scored:** the combined check
replaces the lexical one only if, on the test split, it cuts the false-drop rate
by at least a quarter (relative) and lowers the catch rate by no more than 2
points.

Limits: the passages are web search snippets, not textbook sections, and the
labels mark spans in model answers, not in revision notes.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from evals.datasets import fetch
from margin.notes.verify import _SENTENCE, MIN_CONTENT_WORDS, content_stems, is_supported
from margin.retrieve.embed import Embedder

OUT_DIR = Path(__file__).parent / "results" / "support"
TRAIN_ANSWERS = 1000
THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.01), 2)
MAX_FALSE_DROP_CUT = 0.25
MAX_CATCH_LOSS = 0.02
_LINE = re.compile(r"[^\n]+")
_PASSAGE_LABEL = re.compile(r"passage \d+:", re.IGNORECASE)


@dataclass(frozen=True)
class Sentence:
    text: str
    source_id: str
    hallucinated: bool
    lexical: bool


def _sentences(text: str) -> list[tuple[str, int, int]]:
    found = []
    for line in _LINE.finditer(text):
        cursor = 0
        for part in (s for s in _SENTENCE.split(line.group(0)) if s.strip()):
            start = line.group(0).find(part, cursor)
            cursor = start + len(part)
            found.append((part.strip(), line.start() + start, line.start() + cursor))
    return found


def load(split: str, limit: int | None) -> tuple[list[Sentence], dict[str, str]]:
    folder = fetch.ensure("ragtruth")
    sources = {}
    for line in (folder / "source_info.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["task_type"] == "QA":
            sources[row["source_id"]] = _PASSAGE_LABEL.sub(" ", row["source_info"]["passages"])
    answers = [r for r in map(json.loads, (folder / "response.jsonl").read_text(encoding="utf-8").splitlines())
               if r["split"] == split and r["source_id"] in sources][:limit]
    sentences = []
    for answer in answers:
        stems = content_stems(sources[answer["source_id"]])
        for text, start, end in _sentences(answer["response"]):
            if len(content_stems(text)) < MIN_CONTENT_WORDS:
                continue
            hallucinated = any(label["start"] < end and label["end"] > start for label in answer["labels"])
            sentences.append(Sentence(text, answer["source_id"], hallucinated, is_supported(text, stems)))
    return sentences, {sid: sources[sid] for sid in {s.source_id for s in sentences}}


def best_cosines(sentences: list[Sentence], sources: dict[str, str], embedder: Embedder) -> np.ndarray:
    """Highest cosine between each sentence and any sentence of its own passages."""
    source_sentences = {sid: [t for t, _, _ in _sentences(text)] or [text] for sid, text in sources.items()}
    source_vectors = {sid: embedder.passages(texts) for sid, texts in source_sentences.items()}
    vectors = embedder.passages([s.text for s in sentences])
    return np.array([float(np.max(source_vectors[s.source_id] @ v)) for s, v in zip(sentences, vectors)])


def rates(sentences: list[Sentence], kept: np.ndarray) -> dict[str, float]:
    bad = np.array([s.hallucinated for s in sentences])
    return {"catch_rate": round(float(np.mean(~kept[bad])), 4), "false_drop_rate": round(float(np.mean(~kept[~bad])), 4)}


def choose_threshold(sentences: list[Sentence], cosines: np.ndarray) -> float:
    lexical = np.array([s.lexical for s in sentences])
    scores = [(r["catch_rate"] - r["false_drop_rate"], t) for t in THRESHOLDS for r in [rates(sentences, lexical | (cosines >= t))]]
    return float(max(scores)[1])


def main() -> int:
    embedder = Embedder()
    train, train_sources = load("train", TRAIN_ANSWERS)
    tau = choose_threshold(train, best_cosines(train, train_sources, embedder))
    test, test_sources = load("test", None)
    cosines = best_cosines(test, test_sources, embedder)
    lexical = np.array([s.lexical for s in test])
    checks = {"lexical": rates(test, lexical), "meaning": rates(test, cosines >= tau), "lexical_or_meaning": rates(test, lexical | (cosines >= tau))}
    base, combined = checks["lexical"], checks["lexical_or_meaning"]
    adopt = (combined["false_drop_rate"] <= base["false_drop_rate"] * (1 - MAX_FALSE_DROP_CUT)
             and combined["catch_rate"] >= base["catch_rate"] - MAX_CATCH_LOSS)
    summary = {
        "dataset": "RAGTruth QA", "sha256": [f.sha256 for f in fetch.SETS["ragtruth"].files], "embedder": embedder.name,
        "train_answers": TRAIN_ANSWERS, "train_sentences": len(train), "tau": tau,
        "test_sentences": len(test), "test_hallucinated": int(sum(s.hallucinated for s in test)), "checks": checks,
        "adopt_lexical_or_meaning": adopt, "run_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    print(json.dumps(summary, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ragtruth.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
