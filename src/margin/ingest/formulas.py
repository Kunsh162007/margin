"""Recover equations that a PDF draws as shapes instead of text.

Textbooks such as OpenStax typeset equations as vector drawings, so the text
layer jumps from "we obtain the scalar equation" straight to the next sentence and
reads "the magnitude of  in Newton's law" with the symbol missing. Two kinds of
region are read with Pix2Text's formula recogniser (MIT licence, ONNX, CPU, no
PyTorch):

- **displayed** — a group of shapes touching no text, placed as its own
  ``$$...$$`` paragraph before the first character drawn below it;
- **inline** — shapes sitting on a line of text between two words, cropped from the
  letter on its left to the letter on its right and placed as ``$...$`` at that
  character position.

A textbook draws the same symbols thousands of times (University Physics volume 1:
17,127 inline groups, 3,636 distinct shapes), so a region is read once per distinct
drawing — its outlines and their offsets — and the reading is kept in a cache under
``MARGIN_HOME``, so re-reading a book reuses what was read before. Crops that catch
prose come back as nonsense; ``plausible`` drops readings whose text commands hold
whole words.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from margin.runtime.download import DownloadError, download

Box = tuple[float, float, float, float]  # left, bottom, right, top in PDF points

PAD = 3.0
JOIN = 2.0  # shapes this close vertically belong to one displayed equation
MIN_WIDTH, MIN_HEIGHT = 20.0, 6.0
MAX_SHAPE_HEIGHT = 80.0
MAX_WIDTH_SHARE = 0.8  # wider shapes are rules and boxes, not equations
MARGIN = 40.0  # page furniture (icons, tabs) lives in the outer margin
INLINE_PAD = 4.0
INLINE_JOIN = 2.5  # shapes this close along a line belong to one inline formula
INLINE_MERGE_GAP = 12.0  # pieces this far apart with no letter between them are one formula
MIN_INLINE_HEIGHT = 1.5  # thinner shapes are underlines
RENDER_SCALE = 3
IMAGE_SIZE = 384
MAX_TOKENS = 256
START_TOKEN = END_TOKEN = 2
MAX_LATEX_CHARS = 600
# Geometric-mean token probability below which a reading is marked uncertain. Chosen on half of an OmniDocBench
# sample as the highest value marking no more than 1 in 10 correct readings; on the other half it marked 22% of
# misreads and 10% of correct readings. Confidence separates misreads only weakly; see evals/formulas_eval.py.
MIN_CONFIDENCE = 0.99
UNCERTAIN = "%uncertain\n"  # a LaTeX comment: renders the same, but formula checks cannot parse it, so a doubtful reading never rejects anything
MODEL_DIR_NAME = "pix2text-mfr"
READER_VERSION = f"{MODEL_DIR_NAME}/1"  # change when the model or the image preparation changes: cached readings go stale
CACHE_FILE = "formula-cache.db"
MODEL_URL = "https://huggingface.co/breezedeus/pix2text-mfr/resolve/main/{name}"
MODEL_FILES = {
    "encoder_model.onnx": "bd8d5c322792e9ec45793af5569e9748f82a3d728a9e00213dbfc56c1486f37d",
    "decoder_model.onnx": "fd0f92d7a012f3dae41e1ac79421aea0ea888b5a66cb3f9a004e424f82f3daed",
    "tokenizer.json": "3e2ab757277d22639bec28c9d7972e352d3d1dba223051fa674002dc5ab64df3",
}

_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\s+|.", re.DOTALL)
_COMMAND = re.compile(r"\\[A-Za-z]+")
_WORDY = re.compile(r"\\(?:text[a-z]*|mathrm|operatorname)\s*\{([^{}]*)\}")


def _hit(a: Box, b: Box) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def equation_regions(texts: list[Box], paths: list[Box], width: float, height: float) -> list[Box]:
    """Padded boxes around groups of drawn shapes that touch no text, top of the page first."""
    lone = sorted((p for p in paths if p[2] - p[0] < width * MAX_WIDTH_SHARE and p[3] - p[1] < MAX_SHAPE_HEIGHT
                   and not any(_hit(p, t) for t in texts)), key=lambda p: -p[3])
    bands: list[Box] = []
    for p in lone:
        if bands and p[3] >= bands[-1][1] - JOIN and p[1] <= bands[-1][3] + JOIN:
            left, bottom, right, top = bands[-1]
            bands[-1] = (min(left, p[0]), min(bottom, p[1]), max(right, p[2]), max(top, p[3]))
        else:
            bands.append(p)
    regions = []
    for left, bottom, right, top in bands:
        if right - left < MIN_WIDTH or top - bottom < MIN_HEIGHT or left < MARGIN or bottom < MARGIN or top > height - MARGIN:
            continue
        box = (float(left - PAD), float(bottom - PAD), float(right + PAD), float(top + PAD))
        if not any(_hit(box, t) for t in texts):
            regions.append(box)
    return regions


def _lines(chars: list[Box]) -> list[Box]:
    lines: list[Box] = []
    for c in sorted(chars, key=lambda c: -c[3]):
        for i, (left, bottom, right, top) in enumerate(lines):
            if min(top, c[3]) - max(bottom, c[1]) > 0.5 * min(top - bottom, c[3] - c[1]):
                lines[i] = (min(left, c[0]), min(bottom, c[1]), max(right, c[2]), max(top, c[3]))
                break
        else:
            lines.append(c)
    return lines


def inline_regions(chars: list[Box], paths: list[Box], display: list[Box], width: float) -> list[tuple[Box, Box]]:
    """``(shapes, crop)`` for groups of shapes sitting on a text line; the crop spans to the letters either side."""
    lines = _lines(chars)
    candidates = []
    for p in paths:
        height, middle = p[3] - p[1], (p[1] + p[3]) / 2
        if height < MIN_INLINE_HEIGHT or p[2] - p[0] > width * 0.5 or any(_hit(p, d) for d in display) or any(_hit(p, c) for c in chars):
            continue
        line = next((ln for ln in lines if ln[1] - 2 <= middle <= ln[3] + 2 and ln[0] - 5 <= p[0] <= ln[2] + 5
                     and height <= 2.5 * (ln[3] - ln[1])), None)
        if line is not None:
            candidates.append((p, line))
    groups: list[tuple[Box, Box]] = []
    for p, line in sorted(candidates, key=lambda c: (-c[1][3], c[0][0])):
        if groups and groups[-1][1] == line and p[0] <= groups[-1][0][2] + INLINE_JOIN:
            g = groups[-1][0]
            groups[-1] = ((min(g[0], p[0]), min(g[1], p[1]), max(g[2], p[2]), max(g[3], p[3])), line)
        else:
            groups.append((p, line))
    joined: list[tuple[Box, Box]] = []
    for g, line in groups:  # "R_E = r" is often drawn as three pieces with spaces between: read it as one formula
        if joined and joined[-1][1] == line:
            previous = joined[-1][0]
            letters_between = [c for c in chars if c[1] < line[3] and c[3] > line[1] and c[0] >= previous[2] - 0.5 and c[2] <= g[0] + 0.5]
            if g[0] - previous[2] <= INLINE_MERGE_GAP and not letters_between:
                joined[-1] = ((min(previous[0], g[0]), min(previous[1], g[1]), max(previous[2], g[2]), max(previous[3], g[3])), line)
                continue
        joined.append((g, line))
    regions = []
    for g, line in joined:
        on_line = [c for c in chars if c[1] < line[3] and c[3] > line[1]]
        left = max([c[2] for c in on_line if c[2] <= g[0]] + [g[0] - INLINE_PAD])
        right = min([c[0] for c in on_line if c[0] >= g[2]] + [g[2] + INLINE_PAD])
        regions.append((g, (float(left), float(line[1] - 1), float(right), float(line[3] + 1))))
    return regions


def place_formulas(text: str, chars: list[tuple[int, Box]], display: list[tuple[Box, str]], inline: list[tuple[Box, str]]) -> str:
    """Page text with each formula at the character position where it was drawn.

    ``chars`` pairs an index into ``text`` with that character's box. A displayed formula becomes its own
    paragraph before the first character drawn below it; an inline one goes before the first character to
    its right on the same line, or after the line's last character.
    """
    inserts: list[tuple[int, int, str]] = []
    for box, latex in display:
        below = [i for i, c in chars if (c[1] + c[3]) / 2 < box[1]]
        inserts.append((min(below) if below else len(text), 0, f"$${latex}$$"))
    for box, latex in inline:
        middle = (box[0] + box[2]) / 2
        same_line = [(i, c) for i, c in chars if c[1] < box[3] and c[3] > box[1]]
        if not same_line:
            continue
        right = [i for i, c in same_line if c[0] >= middle]
        inserts.append((min(right) if right else max(i for i, _ in same_line) + 1, 1, f"${latex}$"))
    out = text
    for at, _, snippet in sorted(inserts, key=lambda x: (-x[0], x[1])):
        before, after = out[:at], out[at:]
        if snippet.startswith("$$"):
            lead = "" if not before else ("\n" if before.endswith("\n") else "\n\n")
            snippet = lead + snippet + ("\n\n" if after else "")
        else:
            snippet = ("" if not before or before[-1].isspace() else " ") + snippet + ("" if not after or after[0].isspace() else " ")
        out = before + snippet + after
    return out


def _merge_by_bands(text_between: Callable[[float, float], str], display: list[tuple[Box, str]], height: float) -> str:
    """Fallback when text indices do not line up with character boxes: displayed formulas between text bands."""
    parts: list[str] = []
    cursor = height
    for (_, bottom, _, top), latex in sorted(display, key=lambda f: -f[0][3]):
        parts += [text_between(cursor, top), f"$${latex}$$"]
        cursor = bottom
    parts.append(text_between(cursor, 0.0))
    return "\n\n".join(p.strip() for p in parts if p.strip())


def clean_latex(raw: str) -> str:
    """The recogniser spaces every token; keep only the spaces LaTeX needs (``\\bf v``)."""
    tokens = _TOKEN.findall(raw.strip())
    out: list[str] = []
    for i, token in enumerate(tokens):
        if token.isspace():
            following = tokens[i + 1] if i + 1 < len(tokens) else ""
            if out and _COMMAND.fullmatch(out[-1]) and following[:1].isalpha():
                out.append(" ")
            continue
        out.append(token)
    return "".join(out)


def plausible(latex: str) -> bool:
    """False for empty or oversized readings, and for text commands holding whole words — prose read as maths."""
    if not latex.strip() or len(latex) > MAX_LATEX_CHARS:
        return False
    return not any(re.search(r"[A-Za-z]{5,}", group) for group in _WORDY.findall(latex))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_model(models_dir: Path, downloader: Callable[..., Path] = download) -> Path:
    """Download the recogniser's files into ``models_dir``, each checked against its pinned hash."""
    folder = models_dir / MODEL_DIR_NAME
    for name, digest in MODEL_FILES.items():
        dest = folder / name
        if dest.exists() and _sha256(dest) == digest:
            continue
        dest.unlink(missing_ok=True)
        downloader(MODEL_URL.format(name=name), dest, label=f"formula reader ({name})")
        if _sha256(dest) != digest:
            dest.unlink()
            raise DownloadError(f"the formula reader file {name} did not match its pinned checksum")
    return folder


class FormulaReader:
    """Pix2Text's formula recogniser, loaded on first use, with readings cached by the shapes they came from."""

    def __init__(self, model_dir: Path, cache_path: Path | None = None):
        self.model_dir = model_dir
        self.cache_path = cache_path
        self._loaded: tuple[Any, Any, Any] | None = None
        self._memory: dict[str, tuple[str, float]] = {}
        self._db: sqlite3.Connection | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            providers = ["CPUExecutionProvider"]
            self._loaded = (ort.InferenceSession(str(self.model_dir / "encoder_model.onnx"), providers=providers),
                            ort.InferenceSession(str(self.model_dir / "decoder_model.onnx"), providers=providers),
                            Tokenizer.from_file(str(self.model_dir / "tokenizer.json")))
        return self._loaded

    def read_scored(self, image: Image.Image) -> tuple[str, float]:
        """The LaTeX and the geometric mean of the probabilities of its tokens (1.0 is certain)."""
        import numpy as np

        encoder, decoder, tokenizer = self._load()
        pixels = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC), dtype=np.float32)
        hidden = encoder.run(None, {"pixel_values": ((pixels / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]})[0]
        ids, log_probs = [START_TOKEN], []
        for _ in range(MAX_TOKENS):
            logits = decoder.run(None, {"input_ids": np.array([ids], dtype=np.int64), "encoder_hidden_states": hidden})[0][0, -1]
            token = int(logits.argmax())
            log_probs.append(float(logits[token] - (logits.max() + np.log(np.exp(logits - logits.max()).sum()))))
            if token == END_TOKEN:
                break
            ids.append(token)
        return clean_latex(tokenizer.decode(ids[1:], skip_special_tokens=True)), math.exp(sum(log_probs) / len(log_probs))

    def read(self, image: Image.Image) -> str:
        return self.read_scored(image)[0]

    def _cache(self) -> sqlite3.Connection | None:
        if self._db is None and self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.cache_path)
            self._db.execute("CREATE TABLE IF NOT EXISTS readings(reader TEXT, key TEXT, latex TEXT, confidence REAL, PRIMARY KEY(reader, key))")
        return self._db

    def read_region(self, image: Image.Image, key: str) -> tuple[str, float]:
        """``read_scored`` for a region, reusing the reading of any identical drawing seen before."""
        if key in self._memory:
            return self._memory[key]
        db = self._cache()
        row = db.execute("SELECT latex, confidence FROM readings WHERE reader = ? AND key = ?", (READER_VERSION, key)).fetchone() if db else None
        reading = (row[0], row[1]) if row else self.read_scored(image)
        if db is not None and row is None:
            with db:
                db.execute("INSERT OR REPLACE INTO readings VALUES (?, ?, ?, ?)", (READER_VERSION, key, *reading))
        self._memory[key] = reading
        return reading


def installed_reader(models_dir: Path) -> FormulaReader | None:
    """A reader when ``margin setup`` has fetched the recogniser; None otherwise, and equations stay unread."""
    folder = models_dir / MODEL_DIR_NAME
    if not all((folder / name).exists() for name in MODEL_FILES):
        return None
    return FormulaReader(folder, models_dir.parent / CACHE_FILE)


def _outline(obj: Any) -> tuple[Any, ...]:
    import pypdfium2.raw as raw

    points = []
    for s in range(raw.FPDFPath_CountSegments(obj.raw)):
        segment = raw.FPDFPath_GetPathSegment(obj.raw, s)
        x, y = ctypes.c_float(), ctypes.c_float()
        raw.FPDFPathSegment_GetPoint(segment, x, y)
        points.append((raw.FPDFPathSegment_GetType(segment), round(x.value, 1), round(y.value, 1)))
    m = obj.get_matrix()
    return (round(m.a, 3), round(m.b, 3), round(m.c, 3), round(m.d, 3), round(m.e, 1), round(m.f, 1), tuple(points))


def shape_key(shapes: list[tuple[Box, Any]], region: Box) -> str:
    """The same drawing anywhere on any page gives the same key: outlines relative to the region's corner."""
    parts = []
    for box, obj in shapes:
        a, b, c, d, e, f, points = _outline(obj)
        parts.append((a, b, c, d, round(e - region[0], 1), round(f - region[1], 1), points))
    return hashlib.sha1(repr((sorted(parts, key=repr), round(region[2] - region[0], 1), round(region[3] - region[1], 1))).encode()).hexdigest()


def mark_uncertain(latex: str, confidence: float) -> str:
    """The reading as is when the reader was sure of it; otherwise behind a LaTeX comment that keeps it out of checks."""
    return latex if confidence >= MIN_CONFIDENCE else UNCERTAIN + latex


def _read(reader: Any, image: Image.Image, crop: Box, shapes: list[tuple[Box, Any]], height: float) -> str | None:
    pixels = image.crop((int(crop[0] * RENDER_SCALE), int((height - crop[3]) * RENDER_SCALE),
                         int(crop[2] * RENDER_SCALE), int((height - crop[1]) * RENDER_SCALE)))
    if hasattr(reader, "read_region"):
        latex, confidence = reader.read_region(pixels, shape_key(shapes, crop))
    else:
        latex, confidence = reader.read(pixels), 1.0
    return mark_uncertain(latex, confidence) if plausible(latex) else None


def page_text(page: Any, reader: Any) -> str | None:
    """A pdfium page's text with its drawn equations as LaTeX; None when it has none to add."""
    texts: list[Box] = []
    shapes: list[tuple[Box, Any]] = []
    for obj in page.get_objects(max_depth=3):
        if obj.type == 1:  # FPDF_PAGEOBJ_TEXT
            texts.append(tuple(float(v) for v in obj.get_bounds()))
        elif obj.type == 2:  # FPDF_PAGEOBJ_PATH
            shapes.append((tuple(float(v) for v in obj.get_bounds()), obj))
    width, height = page.get_size()
    display_boxes = equation_regions(texts, [b for b, _ in shapes], width, height)
    textpage = page.get_textpage()
    try:
        text = textpage.get_text_range()
        aligned = len(text) == textpage.count_chars()
        chars = [(i, box) for i, ch in enumerate(text) if aligned and not ch.isspace()
                 for box in [tuple(float(v) for v in textpage.get_charbox(i))] if box[2] > box[0]]
        inline_boxes = inline_regions([c for _, c in chars], [b for b, _ in shapes], display_boxes, width) if chars else []
        if not display_boxes and not inline_boxes:
            return None
        image = page.render(scale=RENDER_SCALE).to_pil()
        members = lambda region: [(b, o) for b, o in shapes if _hit(b, region)]  # noqa: E731
        display = [(box, latex) for box in display_boxes if (latex := _read(reader, image, box, members(box), height))]
        inline = [(group, latex) for group, crop in inline_boxes if (latex := _read(reader, image, crop, members(group), height))]
        if not display and not inline:
            return None
        if not aligned:
            return _merge_by_bands(lambda top, bottom: textpage.get_text_bounded(left=0, bottom=bottom, right=width, top=top), display, height)
        return place_formulas(text, chars, display, inline)
    finally:
        textpage.close()
