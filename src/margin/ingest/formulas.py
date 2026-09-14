"""Recover displayed equations that a PDF draws as shapes instead of text.

Textbooks such as OpenStax typeset their displayed equations as vector drawings,
so the text layer jumps from "we obtain the scalar equation" straight to the next
sentence and notes, questions and their checks never saw a formula. Here a region
of drawn shapes with no text inside or around it is cropped from the rendered
page and read by Pix2Text's formula recogniser (MIT licence, ONNX, CPU, no
PyTorch). The LaTeX goes into the page text between the lines above and below it
as a ``$$...$$`` block.

Regions must stay clear of text: a crop that also catches a line of prose comes
back as nonsense, and ``plausible`` drops any reading whose text commands hold
whole words. Inline symbols and very small shapes are left alone.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from margin.runtime.download import DownloadError, download

Box = tuple[float, float, float, float]  # left, bottom, right, top in PDF points

PAD = 3.0
JOIN = 2.0  # shapes this close vertically belong to one equation
MIN_WIDTH, MIN_HEIGHT = 20.0, 6.0
MAX_SHAPE_HEIGHT = 80.0
MAX_WIDTH_SHARE = 0.8  # wider shapes are rules and boxes, not equations
MARGIN = 40.0  # page furniture (icons, tabs) lives in the outer margin
RENDER_SCALE = 3
IMAGE_SIZE = 384
MAX_TOKENS = 256
START_TOKEN = END_TOKEN = 2
MAX_LATEX_CHARS = 600
MODEL_DIR_NAME = "pix2text-mfr"
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


def merge_page_text(text_between: Callable[[float, float], str], formulas: list[tuple[Box, str]], height: float) -> str:
    """Page text with each formula between the text above and below it. ``text_between(top, bottom)`` reads a band."""
    parts: list[str] = []
    cursor = height
    for (_, bottom, _, top), latex in sorted(formulas, key=lambda f: -f[0][3]):
        parts += [text_between(cursor, top), f"$${latex}$$"]
        cursor = bottom
    parts.append(text_between(cursor, 0.0))
    return "\n\n".join(p.strip() for p in parts if p.strip())


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
    """Pix2Text's formula recogniser, loaded on first use."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._loaded: tuple[Any, Any, Any] | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._loaded is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            providers = ["CPUExecutionProvider"]
            self._loaded = (ort.InferenceSession(str(self.model_dir / "encoder_model.onnx"), providers=providers),
                            ort.InferenceSession(str(self.model_dir / "decoder_model.onnx"), providers=providers),
                            Tokenizer.from_file(str(self.model_dir / "tokenizer.json")))
        return self._loaded

    def read(self, image: Image.Image) -> str:
        import numpy as np

        encoder, decoder, tokenizer = self._load()
        pixels = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC), dtype=np.float32)
        hidden = encoder.run(None, {"pixel_values": ((pixels / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]})[0]
        ids = [START_TOKEN]
        for _ in range(MAX_TOKENS):
            logits = decoder.run(None, {"input_ids": np.array([ids], dtype=np.int64), "encoder_hidden_states": hidden})[0]
            token = int(logits[0, -1].argmax())
            if token == END_TOKEN:
                break
            ids.append(token)
        return clean_latex(tokenizer.decode(ids[1:], skip_special_tokens=True))


def installed_reader(models_dir: Path) -> FormulaReader | None:
    """A reader when ``margin setup`` has fetched the recogniser; None otherwise, and equations stay unread."""
    folder = models_dir / MODEL_DIR_NAME
    return FormulaReader(folder) if all((folder / name).exists() for name in MODEL_FILES) else None


def page_text(page: Any, reader: Any) -> str | None:
    """A pdfium page's text with its drawn equations as ``$$LaTeX$$``; None when it has none to add."""
    texts: list[Box] = []
    paths: list[Box] = []
    for obj in page.get_objects(max_depth=3):
        if obj.type in (1, 2):  # FPDF_PAGEOBJ_TEXT, FPDF_PAGEOBJ_PATH
            (texts if obj.type == 1 else paths).append(tuple(float(v) for v in obj.get_bounds()))
    width, height = page.get_size()
    regions = equation_regions(texts, paths, width, height)
    if not regions:
        return None
    image = page.render(scale=RENDER_SCALE).to_pil()
    found = []
    for left, bottom, right, top in regions:
        crop = image.crop((int(left * RENDER_SCALE), int((height - top) * RENDER_SCALE), int(right * RENDER_SCALE), int((height - bottom) * RENDER_SCALE)))
        latex = reader.read(crop)
        if plausible(latex):
            found.append(((left, bottom, right, top), latex))
    if not found:
        return None
    textpage = page.get_textpage()
    try:
        return merge_page_text(lambda top, bottom: textpage.get_text_bounded(left=0, bottom=bottom, right=width, top=top), found, height)
    finally:
        textpage.close()
