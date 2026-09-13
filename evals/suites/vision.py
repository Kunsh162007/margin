"""Reading photos of pages: the model's own transcription against a dedicated OCR engine.

Every run also scores RapidOCR (ONNX, CPU) on the same images. It does not
depend on the model, so it must reproduce exactly from run to run: that is the
control arm. It also answers the design question directly — if plain OCR reads
the photos as well as a vision model, the vision model is not needed for text.

RapidOCR runs with its default PP-OCRv6 small recogniser. Passing
``Rec.lang_type = LangRec.EN`` was tried: it loads the same model file and gives
byte-identical output, because PP-OCRv6 ships one multilingual small recogniser.
The default reads these English pages at about 3% character error rate.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from statistics import mean
from typing import Any

from evals.matchers import cer
from margin.runtime.client import ClientError, LlamaClient

PROMPT = "Transcribe all the text in this image exactly, in reading order. Output only the text."
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _data_url(path: Path) -> str:
    return f"data:{MIME[path.suffix.lower()]};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _rapidocr() -> Any | None:
    try:
        from rapidocr import RapidOCR
    except ImportError:
        return None
    return RapidOCR()


def _ocr_text(engine: Any, path: Path) -> str:
    result = engine(str(path))
    texts = getattr(result, "txts", None) or ()
    return " ".join(texts)


def _mean(rows: list[dict[str, Any]], key: str, style: str | None = None) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None and (style is None or r["style"] == style)]
    return mean(vals) if vals else None


def run(client: LlamaClient | None, cases: list[dict[str, Any]], passages: dict[str, dict[str, Any]], images_dir: Path) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    engine = _rapidocr()
    rows = []
    for case in cases:
        path = images_dir / case["file"]
        truth = passages[case["passage"]]["text"]
        row: dict[str, Any] = {"id": case["id"], "style": case["style"], "degradation": case["degradation"]}
        if client is not None:
            messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(path)}}, {"type": "text", "text": PROMPT}]}]
            try:
                res = client.chat(messages, max_tokens=600)
                row |= {"cer": cer(truth, res.content), "latency_s": res.wall_s, "output": res.content[:400]}
            except ClientError as exc:
                row |= {"cer": 1.0, "error": str(exc)[:300]}
        if engine is not None:
            started = time.perf_counter()
            text = _ocr_text(engine, path)
            row |= {"ocr_cer": cer(truth, text), "ocr_latency_s": time.perf_counter() - started}
        rows.append(row)

    return {
        "vision.cer_mean": _mean(rows, "cer"),
        "vision.cer_typed": _mean(rows, "cer", "typed"),
        "vision.cer_handwriting": _mean(rows, "cer", "handwriting"),
        "vision.latency_mean_s": _mean(rows, "latency_s"),
        "vision.error_rate": mean(1.0 if "error" in r else 0.0 for r in rows) if client is not None and rows else None,
        "vision.ocr_cer_mean": _mean(rows, "ocr_cer"),
        "vision.ocr_cer_typed": _mean(rows, "ocr_cer", "typed"),
        "vision.ocr_cer_handwriting": _mean(rows, "ocr_cer", "handwriting"),
        "vision.ocr_latency_mean_s": _mean(rows, "ocr_latency_s"),
    }, rows
