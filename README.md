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

The first run detects your hardware, downloads the matching llama.cpp runtime
and a model sized for your machine, then starts. `margin doctor` shows what was
detected; `margin models` lists the models you can switch to.

Set `MARGIN_HOME` to keep models on a different drive.

## Evaluation

The evaluation suite is optional and lives in `evals/`. See
[`docs/DESIGN.md`](docs/DESIGN.md) for what it measures and why.

```sh
uv run python -m evals.run --model qwen3.5-4b          # smoke tier
uv run python -m evals.bench_models                    # every model, every backend
```
