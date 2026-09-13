"""The `margin` command.

    margin            set up on first run, then start
    margin setup      download the runtime and the model for this machine
    margin doctor     show what Margin detected and what is installed
    margin models     list the models Margin can run
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from margin import __version__, config, hardware
from margin.runtime import models
from margin.runtime.binaries import LLAMA_CPP_BUILD, UnsupportedPlatform, ensure_llama_server, find_server
from margin.runtime.download import DownloadError, download

console = Console()
GOLD = "#d4a72c"


def _selection(args: argparse.Namespace) -> tuple[hardware.Hardware, str, models.ModelSpec]:
    hw = hardware.detect()
    backend = hardware.choose_backend(hw, getattr(args, "backend", None))
    model_id = getattr(args, "model", None)
    spec = models.get(model_id) if model_id else models.recommended(backend, hw.ram_gb, hw.gpu.vram_gb if hw.gpu else None)
    return hw, backend, spec


def cmd_doctor(args: argparse.Namespace) -> int:
    hw, backend, spec = _selection(args)
    paths = config.paths()
    server = find_server(paths.bin_dir / f"{LLAMA_CPP_BUILD}-{backend}") if (paths.bin_dir / f"{LLAMA_CPP_BUILD}-{backend}").exists() else None
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style=GOLD)
    table.add_column()
    table.add_row("version", __version__)
    table.add_row("system", f"{hw.os} {hw.arch}, {hw.physical_cores} cores, {hw.ram_gb} GB RAM")
    table.add_row("gpu", f"{hw.gpu.name} ({hw.gpu.vram_gb} GB)" if hw.gpu else "none found — running on CPU")
    table.add_row("backend", backend)
    table.add_row("home", str(paths.home))
    table.add_row("runtime", f"llama.cpp {LLAMA_CPP_BUILD} " + ("installed" if server else "not installed"))
    table.add_row("model", f"{spec.id} ({spec.size_gb} GB) " + ("installed" if spec.path(paths.models_dir).exists() else "not installed"))
    console.print(table)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    models_dir = config.paths().models_dir
    table = Table(title="Models")
    for col in ("id", "family", "params", "download", "vision", "license", "installed"):
        table.add_column(col)
    for m in models.CANDIDATES:
        table.add_row(m.id, m.family, f"{m.params_b}B", f"{m.size_gb} GB", "yes" if m.vision else "", m.license, "yes" if m.path(models_dir).exists() else "")
    console.print(table)
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    hw, backend, spec = _selection(args)
    paths = config.paths().ensure()
    console.print(f"[{GOLD}]Setting up Margin[/] for {backend.upper()} with {spec.id} ({spec.size_gb} GB download)")
    try:
        exe = ensure_llama_server(paths.bin_dir, hw.os, hw.arch, backend)
        download(spec.url, spec.path(paths.models_dir), label=spec.id)
    except (DownloadError, UnsupportedPlatform) as exc:
        console.print(f"[red]Setup failed:[/] {exc}")
        return 1
    if not _fetch_search_models():
        return 1
    console.print(f"[{GOLD}]Ready.[/] runtime {exe.name}, model {spec.file}, search models cached")
    return 0


def _fetch_search_models() -> bool:
    """Download the embedding and reranking models now, so searching later needs no network."""
    from margin.retrieve.embed import Embedder, Reranker

    try:
        Embedder().passages(["warm-up"])
        Reranker().scores("warm-up", ["warm-up"])
    except Exception as exc:  # fastembed surfaces network and file errors as several types
        console.print(f"[red]Could not fetch search models:[/] {exc}")
        return False
    return True


def _workspace_path():
    return config.paths().ensure().workspaces_dir / "default.db"


def cmd_add(args: argparse.Namespace) -> int:
    """Index files into the default workspace."""
    from pathlib import Path

    from margin.ingest.detect import UnsupportedFile
    from margin.retrieve import embed
    from margin.retrieve.search import index_document
    from margin.store.db import Workspace

    embedder, failures = embed.Embedder(), 0
    with Workspace.open(_workspace_path()) as ws:
        for name in args.files:
            try:
                result = index_document(ws, Path(name), embedder)
            except (FileNotFoundError, UnsupportedFile) as exc:
                console.print(f"[red]skipped[/] {name}: {exc}")
                failures += 1
                continue
            status = "already added" if result.skipped else f"{result.sections} sections, {result.chunks} chunks"
            console.print(f"[{GOLD}]{result.title}[/]  {status}")
    return 1 if failures else 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search the default workspace."""
    from margin.retrieve import embed
    from margin.retrieve.search import Searcher
    from margin.store.db import Workspace

    with Workspace.open(_workspace_path()) as ws:
        if ws.chunk_count() == 0:
            console.print("Nothing to search yet. Add files with: margin add <file> ...")
            return 1
        searcher = Searcher(ws, embed.Embedder(), None if args.no_rerank else embed.Reranker())
        hits = searcher.search(args.query, top_k=args.top_k, scope=args.scope)
    if not hits:
        console.print("No matches.")
        return 0
    table = Table(box=None)
    for col in ("section", "title", "pages", "passage"):
        table.add_column(col)
    for hit in hits:
        r = hit.row
        pages = "" if r.page_start is None else (f"{r.page_start}" if r.page_start == r.page_end else f"{r.page_start}-{r.page_end}")
        table.add_row(r.sid, r.section_title[:40], pages, " ".join(r.text.split())[:160])
    console.print(table)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Show what Margin reads from a file: type, pages, OCR pages and the section tree."""
    from pathlib import Path

    from margin.ingest.detect import UnsupportedFile
    from margin.ingest.pipeline import ingest_file

    try:
        doc, sections = ingest_file(Path(args.file))
    except (FileNotFoundError, UnsupportedFile) as exc:
        console.print(f"[red]Cannot read:[/] {exc}")
        return 1
    console.print(f"[{GOLD}]{doc.title}[/]  {doc.kind}, {doc.page_count} pages, {doc.char_count:,} characters, {len(sections)} sections")
    if doc.ocr_pages:
        console.print(f"text from OCR on {len(doc.ocr_pages)} page(s): {', '.join(map(str, doc.ocr_pages[:20]))}")
    table = Table(box=None)
    for col in ("id", "section", "pages", "chars"):
        table.add_column(col)
    for s in sections:
        pages = "" if s.page_start is None else (f"{s.page_start}" if s.page_start == s.page_end else f"{s.page_start}-{s.page_end}")
        table.add_row(s.id, "  " * (s.level - 1) + s.title[:70], pages, f"{len(s.text):,}")
    console.print(table)
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    _, backend, spec = _selection(args)
    paths = config.paths()
    installed = (paths.bin_dir / f"{LLAMA_CPP_BUILD}-{backend}").exists() and spec.path(paths.models_dir).exists()
    if not installed and cmd_setup(args) != 0:
        return 1
    console.print(f"[{GOLD}]Margin {__version__}[/] is installed. The study interface arrives in the next release; run `margin doctor` to check your setup.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="margin", description="Offline study assistant.")
    parser.add_argument("--version", action="version", version=f"margin {__version__}")
    sub = parser.add_subparsers(dest="command")
    for name, fn, help_text in (
        ("setup", cmd_setup, "download the runtime and model"),
        ("doctor", cmd_doctor, "show detected hardware and install status"),
        ("models", cmd_models, "list available models"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=fn)
        if name in ("setup", "doctor"):
            p.add_argument("--model", choices=list(models.BY_ID))
            p.add_argument("--backend", choices=hardware.BACKENDS)
    inspect = sub.add_parser("inspect", help="show how a file is read and split into sections")
    inspect.add_argument("file")
    inspect.set_defaults(func=cmd_inspect)
    add = sub.add_parser("add", help="add books, notes, slides or photos to your library")
    add.add_argument("files", nargs="+")
    add.set_defaults(func=cmd_add)
    search = sub.add_parser("search", help="search your library")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--scope", help="chapter or section, e.g. ch4 or 4.2")
    search.add_argument("--no-rerank", action="store_true", help="skip the reranker (faster, less precise)")
    search.set_defaults(func=cmd_search)
    parser.set_defaults(func=cmd_start)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("\nstopped")
        return 130


if __name__ == "__main__":
    sys.exit(main())
