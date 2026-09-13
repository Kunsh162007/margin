"""Render the evaluation passages as photographed pages.

    uv run --group evals python -m evals.datasets.make_images

The images are synthetic: each passage is drawn in a typed or a handwriting-style
font, then degraded the way phone photos are — rotation, blur, uneven lighting,
sensor noise, JPEG compression. They approximate photos of printed pages and
neat handwritten notes. They are not real handwriting, and results on them are
reported as synthetic.

Output is deterministic (fixed seed, fixed fonts), so the dataset fingerprint
only changes when this script does. Fonts are read from the Windows font folder.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
OUT_DIR = HERE / "images"
FONT_DIR = Path("C:/Windows/Fonts")
PAPER = (247, 245, 237)
INK = (28, 30, 42)
PAGE_WIDTH = 1400
MARGIN = 70
FONT_SIZE = 30
SEED = 42

STYLES = (
    ("typed", "arial.ttf"), ("handwriting", "Inkfree.ttf"), ("typed", "times.ttf"), ("handwriting", "segoepr.ttf"),
    ("typed", "georgia.ttf"), ("handwriting", "comic.ttf"), ("typed", "calibri.ttf"),
)
DEGRADATIONS = ("clean", "rotate", "blur", "shadow", "noise", "jpeg")


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_px: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if font.getlength(trial) <= max_px:
            current = trial
        else:
            lines.append(current)
            current = word
    return lines + [current] if current else lines


def render(text: str, font_file: str) -> Image.Image:
    font = ImageFont.truetype(str(FONT_DIR / font_file), FONT_SIZE)
    lines = _wrap(text, font, PAGE_WIDTH - 2 * MARGIN)
    line_height = int(FONT_SIZE * 1.6)
    img = Image.new("RGB", (PAGE_WIDTH, 2 * MARGIN + line_height * len(lines)), PAPER)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((MARGIN, MARGIN + i * line_height), line, fill=INK, font=font)
    return img


def degrade(img: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    if kind == "rotate":
        return img.rotate(rng.uniform(-4, 4), expand=True, fillcolor=PAPER, resample=Image.Resampling.BICUBIC)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(1.3))
    if kind == "shadow":
        gradient = Image.linear_gradient("L").rotate(90).resize(img.size)
        shade = Image.merge("RGB", [gradient.point(lambda v: 150 + v * 105 // 255)] * 3)
        return Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), shade.convert("L"))
    if kind == "noise":
        noise = Image.effect_noise(img.size, 60).convert("RGB")
        return Image.blend(img, noise, 0.18)
    return img


def main() -> None:
    rng = random.Random(SEED)
    passages = [json.loads(line) for line in (HERE / "passages.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for i, passage in enumerate(passages):
        style, font_file = STYLES[i % len(STYLES)]
        kind = DEGRADATIONS[i % len(DEGRADATIONS)]
        img = degrade(render(passage["text"], font_file), kind, rng)
        as_jpeg = kind == "jpeg" or i % 2 == 1
        name = f"img-{i + 1:02d}.{'jpg' if as_jpeg else 'png'}"
        if as_jpeg:
            img.save(OUT_DIR / name, quality=35 if kind == "jpeg" else 80)
        else:
            img.save(OUT_DIR / name, optimize=True)
        rows.append({"id": f"img-{i + 1:02d}", "file": name, "passage": passage["id"], "style": style, "font": font_file, "degradation": kind})
    (HERE / "images.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} images to {OUT_DIR}")


if __name__ == "__main__":
    main()
