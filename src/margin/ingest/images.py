"""Read text from photos and scanned pages with RapidOCR (ONNX, CPU).

Phone photos store their orientation in EXIF rather than in the pixels, so the
image is turned upright before OCR; otherwise a portrait photo is read sideways.
Very large photos are scaled down first: OCR accuracy stops improving well
below 12-megapixel resolution, and time keeps growing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image, ImageOps

MAX_SIDE_PX = 2500
PARAGRAPH_GAP = 1.6  # a vertical gap this many line-heights apart starts a new paragraph


@dataclass(frozen=True)
class OcrResult:
    text: str
    mean_score: float


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
    if max(img.size) > MAX_SIDE_PX:
        img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.Resampling.LANCZOS)
    return img


def _paragraphs(texts: tuple[str, ...], boxes: Any) -> str:
    if not texts:
        return ""
    tops = [float(min(p[1] for p in box)) for box in boxes]
    heights = [float(max(p[1] for p in box) - min(p[1] for p in box)) for box in boxes]
    line_height = median(heights) or 1.0
    out = [texts[0]]
    for i in range(1, len(texts)):
        sep = "\n\n" if tops[i] - tops[i - 1] > PARAGRAPH_GAP * line_height else "\n"
        out.append(sep + texts[i])
    return "".join(out)


class OcrEngine:
    """Created lazily: most documents never need OCR, and loading it costs ~3 s."""

    def __init__(self) -> None:
        self._engine: Any = None

    def _get(self) -> Any:
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def read(self, image: Image.Image) -> OcrResult:
        import numpy as np

        result = self._get()(np.asarray(image))
        texts = tuple(getattr(result, "txts", None) or ())
        scores = tuple(getattr(result, "scores", None) or ())
        boxes = getattr(result, "boxes", None)
        text = _paragraphs(texts, boxes) if texts and boxes is not None else " ".join(texts)
        return OcrResult(text=text, mean_score=sum(scores) / len(scores) if scores else 0.0)
