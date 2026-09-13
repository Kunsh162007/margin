# Margin

An offline study assistant. Give it your reference books, notes, slides and
photos, plus past exam papers. It works out which parts of the book the exam
actually asks about, writes notes on those parts with flowcharts, tables and
formula sheets, and sets practice questions in the style of your paper.

Runs entirely on your own computer, on a CPU or a GPU.

## Install

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/Kunsh162007/margin/main/scripts/install.ps1 | iex
```

macOS and Linux:

```sh
curl -LsSf https://raw.githubusercontent.com/Kunsh162007/margin/main/scripts/install.sh | sh
```

Already have [uv](https://docs.astral.sh/uv/)? `uv tool install git+https://github.com/Kunsh162007/margin`

## Run

```sh
margin
```

The first run detects your hardware, downloads the matching llama.cpp runtime,
the Gemma 4 E4B model and the search models, then starts. `margin doctor` shows
what was detected; `margin models` lists the models you can switch to.

```sh
margin add biology-textbook.pdf lecture-slides.pptx whiteboard.jpg
margin search "how does the nephron filter blood" --scope ch25
margin inspect biology-textbook.pdf      # see how a file is split into sections
margin blueprint paper-2023.pdf paper-2024.jpg   # which chapters past papers weigh most
margin notes ch25 --out kidney-notes.md          # verified notes with a diagram per section
margin notes ch25 --no-visuals                   # faster on a CPU: notes only
```

Notes are checked sentence by sentence against the book, and every finished
section is saved as it is written — stop at any time and run the same command
again to carry on.

Books, Word files, slides and JPEG/PNG photos are supported; scanned pages and
photos are read with OCR. Search combines keyword and meaning-based matching;
`--rerank` adds a slower second pass that did not improve results on the
textbook benchmark, so it is off by default.

Set `MARGIN_HOME` to keep models on a different drive.

## Evaluation

The evaluation suite is optional and lives in `evals/`. See
[`docs/DESIGN.md`](docs/DESIGN.md) for what it measures and why.

```sh
uv run python -m evals.run --model qwen3.5-4b          # smoke tier
uv run python -m evals.bench_models                    # every model, every backend
```
