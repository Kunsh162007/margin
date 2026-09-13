"""Argument matchers and text normalisation shared by the suites.

Expected tool arguments are written as plain values or as small operators:

    {"$contains": "kirchhoff"}           substring (after squashing), str or list of str
    {"$one_of": [a, b]}                   any alternative matches
    {"$range": [lo, hi]}                  numeric, inclusive
    {"$min_items": 3}                     list length
    {"$items_contain": ["a", "b"]}        every term appears somewhere in the list
    {"$optional": value}                  key may be absent; if present it must match
    {"$any": true}                        key must be present, any value

"Squashing" lowercases and removes everything but letters and digits, so
"SYN-ACK", "syn ack" and "SynAck" compare equal, and "wL²/8" equals "wl^2/8".
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

_TRANSLATE = str.maketrans({"²": "2", "³": "3"})
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[a-z][a-z-]{3,}")
STEM_LEN = 6
STOPWORDS = frozenset(
    "that this from into their there which when where what while have been being each only also than then "
    "them they over under about after before most more less such other some will would should could does "
    "your with these those very used uses using make made same both between within without because".split()
)


def squash(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return _NON_ALNUM.sub("", text.lower().translate(_TRANSLATE))


def content_stems(text: str) -> set[str]:
    return {w[:STEM_LEN] for w in _WORD.findall(text.lower()) if w not in STOPWORDS}


def support_ratio(text: str, source_stems: set[str]) -> float | None:
    """Share of the text's content words that also occur in the source. None if the text has none."""
    stems = content_stems(text)
    if not stems:
        return None
    return len(stems & source_stems) / len(stems)


_SPACE = re.compile(r"\s+")


def _ocr_normalise(text: str) -> str:
    return _SPACE.sub(" ", re.sub(r"[^a-z0-9 ]+", " ", text.lower().translate(_TRANSLATE))).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate after lowercasing and collapsing punctuation to spaces. Capped at 1.0."""
    ref, hyp = _ocr_normalise(reference), _ocr_normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return min(1.0, prev[-1] / len(ref))


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _operator(expected: Any) -> tuple[str, Any] | None:
    if isinstance(expected, dict) and len(expected) == 1:
        key = next(iter(expected))
        if key.startswith("$"):
            return key, expected[key]
    return None


def match_value(expected: Any, actual: Any) -> bool:
    op = _operator(expected)
    if op is not None:
        return _match_operator(op[0], op[1], actual)
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    exp_num, act_num = _as_number(expected), _as_number(actual)
    if exp_num is not None:
        return act_num is not None and abs(exp_num - act_num) < 1e-9
    if isinstance(expected, str):
        return isinstance(actual, str) and squash(expected) == squash(actual)
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(map(match_value, expected, actual))
    if isinstance(expected, dict):
        return isinstance(actual, dict) and match_args(expected, actual)
    return expected == actual


def _match_operator(name: str, arg: Any, actual: Any) -> bool:
    if name == "$any":
        return True
    if name == "$contains":
        haystack = squash(actual)
        terms = arg if isinstance(arg, list) else [arg]
        return all(squash(t) in haystack for t in terms)
    if name == "$one_of":
        return any(match_value(alt, actual) for alt in arg)
    if name == "$range":
        num = _as_number(actual)
        return num is not None and arg[0] <= num <= arg[1]
    if name == "$min_items":
        return isinstance(actual, list) and len(actual) >= arg
    if name == "$items_contain":
        return isinstance(actual, list) and all(squash(t) in squash(actual) for t in arg)
    if name == "$all":
        return all(match_value(sub, actual) for sub in arg)
    raise ValueError(f"unknown matcher {name}")


def match_args(expected: dict[str, Any], actual: dict[str, Any] | None) -> bool:
    if actual is None:
        return False
    for key, exp in expected.items():
        op = _operator(exp)
        if op is not None and op[0] == "$optional":
            if key not in actual or actual[key] is None:
                continue
            exp = op[1]
        if key not in actual:
            return False
        if not match_value(exp, actual[key]):
            return False
    return True


def match_calls(expected: list[dict[str, Any]], actual: list[tuple[str, dict[str, Any] | None]]) -> tuple[bool, bool]:
    """(right tools chosen, right tools with right arguments). Order-insensitive."""
    if Counter(e["name"] for e in expected) != Counter(name for name, _ in actual):
        return False, False
    unused = list(range(len(actual)))
    for exp in expected:
        hit = next((i for i in unused if actual[i][0] == exp["name"] and match_args(exp.get("args", {}), actual[i][1])), None)
        if hit is None:
            return True, False
        unused.remove(hit)
    return True, True
