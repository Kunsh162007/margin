"""Keep only note sentences the source section supports.

A sentence is supported when at least half of its content words (stemmed to
six letters, stopwords removed) occur in the section. Sentences with fewer than
three content words — headings, "Key points:", a formula — are kept. This is a
*lexical* check: it catches invented material written in new words, not a
wrong claim made with the passage's own words, and it is reported as lexical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUPPORT_THRESHOLD = 0.5
MIN_CONTENT_WORDS = 3
STEM_LEN = 6
_WORD = re.compile(r"[a-z][a-z-]{3,}")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*(])")
_BULLET = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)")
STOPWORDS = frozenset(
    "that this from into their there which when where what while have been being each only also than then "
    "them they over under about after before most more less such other some will would should could does "
    "your with these those very used uses using make made same both between within without because".split()
)


@dataclass(frozen=True)
class Verified:
    markdown: str
    kept: int
    dropped: int
    dropped_sentences: tuple[str, ...]


def content_stems(text: str) -> set[str]:
    return {w[:STEM_LEN] for w in _WORD.findall(text.lower()) if w not in STOPWORDS}


def is_supported(sentence: str, source_stems: set[str]) -> bool:
    stems = content_stems(sentence)
    return len(stems) < MIN_CONTENT_WORDS or len(stems & source_stems) / len(stems) >= SUPPORT_THRESHOLD


def verify_notes(markdown: str, source: str) -> Verified:
    source_stems = content_stems(source)
    lines: list[str] = []
    kept = dropped = 0
    removed: list[str] = []
    in_code = False
    for line in markdown.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if in_code or line.strip().startswith(("```", "#", "|")) or not line.strip():
            lines.append(line)
            continue
        prefix_match = _BULLET.match(line)
        prefix = prefix_match.group(1) if prefix_match else ""
        body = line[len(prefix):]
        sentences = [s for s in _SENTENCE.split(body) if s.strip()]
        good = [s for s in sentences if is_supported(s, source_stems)]
        kept += len(good)
        dropped += len(sentences) - len(good)
        removed += [s for s in sentences if s not in good]
        if good:
            lines.append(prefix + " ".join(good))
    return Verified("\n".join(lines).strip(), kept, dropped, tuple(removed))
