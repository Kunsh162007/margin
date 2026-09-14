"""Download the public gold sets the evals use, and prove they are the expected bytes.

    uv run python -m evals.datasets.fetch             # every set
    uv run python -m evals.datasets.fetch ragtruth    # one set

Nothing here is committed: some sets carry licences that forbid redistribution
or changes, so each set is downloaded into ``MARGIN_HOME/datasets/<name>/`` and
every file's SHA-256 must equal the value pinned below. A result can then name
exactly the data it was measured on. A file that changed upstream is deleted and
reported; update its hash only after checking what changed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from margin import config
from margin.runtime.download import download

RAGTRUTH_RAW = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
OMNIDOCBENCH = "https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main"
# Names and hashes only — no OmniDocBench content is kept in this repository.
OMNIDOCBENCH_IMAGES = json.loads((Path(__file__).parent / "omnidocbench_images.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    sha256: str


@dataclass(frozen=True)
class GoldSet:
    name: str
    description: str
    licence: str
    source: str
    files: tuple[RemoteFile, ...]


SETS: dict[str, GoldSet] = {
    "ragtruth": GoldSet(
        "ragtruth", "Answers from six language models with word-level hallucination spans (Niu et al., 2024)", "MIT",
        "https://github.com/ParticleMedia/RAGTruth",
        (RemoteFile("response.jsonl", f"{RAGTRUTH_RAW}/response.jsonl", "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073"),
         RemoteFile("source_info.jsonl", f"{RAGTRUTH_RAW}/source_info.jsonl", "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b")),
    ),
    "math500": GoldSet(
        "math500", "500 MATH test problems with final answers (Hendrycks et al., 2021; split of Lightman et al., 2023)", "MIT",
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
        (RemoteFile("test.jsonl", "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
                    "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"),),
    ),
    "omnidocbench": GoldSet(
        "omnidocbench", "Document pages with every element labelled, formulas in LaTeX (Ouyang et al., 2025); the pages holding "
        "English displayed equations from books, exam papers and textbooks", "research use only, not for commercial use",
        "https://huggingface.co/datasets/opendatalab/OmniDocBench",
        (RemoteFile("OmniDocBench.json", f"{OMNIDOCBENCH}/OmniDocBench.json", "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496"),
         *(RemoteFile(f"images/{m['name']}", f"{OMNIDOCBENCH}/images/{quote(m['name'])}", m["sha256"]) for m in OMNIDOCBENCH_IMAGES)),
    ),
}


class ChecksumMismatch(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure(name: str, root: Path | None = None, downloader: Callable[..., Path] = download) -> Path:
    """The folder holding set ``name``, downloading any missing file and verifying every file's hash."""
    if name not in SETS:
        raise KeyError(f"unknown gold set {name!r}; known: {', '.join(sorted(SETS))}")
    folder = (root or config.paths().datasets_dir) / name
    for remote in SETS[name].files:
        dest = folder / remote.name
        if not dest.exists():
            downloader(remote.url, dest, label=f"{name}/{remote.name}")
        actual = sha256(dest)
        if actual != remote.sha256:
            dest.unlink()
            raise ChecksumMismatch(f"{name}/{remote.name}: expected sha256 {remote.sha256[:12]}…, got {actual[:12]}…. "
                                   "The file changed upstream; check what changed before pinning the new hash.")
    return folder


def main(argv: list[str] | None = None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or sorted(SETS)
    for name in names:
        gold = SETS[name]
        print(f"{name}: {ensure(name)}  ({gold.licence}; {gold.source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
