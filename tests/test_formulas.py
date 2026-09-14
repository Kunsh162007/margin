import hashlib

import pytest

from margin.ingest import formulas
from margin.ingest.formulas import clean_latex, equation_regions, merge_page_text, plausible

W, H = 612.0, 792.0


def test_the_formula_model_is_fetched_once_and_a_bad_download_is_removed(tmp_path, monkeypatch, no_network):
    from margin.runtime.download import DownloadError

    body = b"model bytes"
    monkeypatch.setattr(formulas, "MODEL_FILES", {"encoder_model.onnx": hashlib.sha256(body).hexdigest()})
    calls = []

    def fake_download(url, dest, *, label=None):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body if len(calls) == 1 else b"tampered")
        return dest

    folder = formulas.ensure_model(tmp_path, downloader=fake_download)
    assert (folder / "encoder_model.onnx").read_bytes() == body and formulas.installed_reader(tmp_path) is not None
    formulas.ensure_model(tmp_path, downloader=fake_download)
    assert len(calls) == 1
    (folder / "encoder_model.onnx").write_bytes(b"corrupted on disk")
    with pytest.raises(DownloadError):
        formulas.ensure_model(tmp_path, downloader=fake_download)
    assert not (folder / "encoder_model.onnx").exists() and formulas.installed_reader(tmp_path) is None


def _pdf_with_drawn_equation(path):
    import ctypes

    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(W, H)
    for text, y in (("The weight of an object near the surface is mg, so", 700), ("where r is the distance between the centres of mass.", 560)):
        obj = raw.FPDFPageObj_NewTextObj(pdf.raw, b"Helvetica", 12.0)
        encoded = (text + "\x00").encode("utf-16-le")
        raw.FPDFText_SetText(obj, ctypes.cast((ctypes.c_uint16 * (len(encoded) // 2)).from_buffer_copy(encoded), ctypes.POINTER(ctypes.c_uint16)))
        raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, 72, y)
        raw.FPDFPage_InsertObject(page.raw, obj)
    shape = raw.FPDFPageObj_CreateNewRect(250, 620, 100, 25)  # the equation, drawn rather than written
    raw.FPDFPageObj_SetFillColor(shape, 0, 0, 0, 255)
    raw.FPDFPath_SetDrawMode(shape, raw.FPDF_FILLMODE_ALTERNATE, 0)
    raw.FPDFPage_InsertObject(page.raw, shape)
    raw.FPDFPage_GenerateContent(page.raw)
    pdf.save(str(path))


class _FakeReader:
    def __init__(self):
        self.crops = []

    def read(self, image):
        self.crops.append(image.size)
        return "g=9.8"


def test_a_pdf_page_gets_its_drawn_equation_as_latex_between_its_lines(tmp_path):
    from margin.ingest.pdf import read_pdf

    path = tmp_path / "book.pdf"
    _pdf_with_drawn_equation(path)
    reader = _FakeReader()
    doc = read_pdf(path, "d1", None, formulas=reader)
    text = "\n".join(b.text for b in doc.blocks)
    assert len(reader.crops) == 1 and doc.formula_pages == (1,)
    assert text.index("mg, so") < text.index("$$g=9.8$$") < text.index("where r is")
    assert "$$" not in "\n".join(b.text for b in read_pdf(path, "d1", None).blocks)


def test_regions_are_drawn_shapes_clear_of_text_rules_icons_and_specks():
    texts = [(72, 700, 540, 712), (72, 600, 540, 612)]
    paths = [
        (275, 640, 300, 660), (302, 645, 336, 662),  # one equation drawn as two shapes
        (72, 338, 540, 339),  # a rule across the page
        (0, 0, 35, 35),  # a margin icon
        (100, 702, 110, 710),  # a shape inside a text line
        (300, 400, 305, 404),  # a speck
    ]
    assert equation_regions(texts, paths, W, H) == [(272.0, 637.0, 339.0, 665.0)]


def test_a_region_that_would_crop_into_text_is_skipped():
    texts = [(72, 663, 540, 675)]  # a text line just above the shapes, inside the padding
    assert equation_regions(texts, [(275, 640, 336, 662)], W, H) == []


def test_clean_latex_joins_tokens_but_keeps_command_spacing():
    raw = r"m g = G \, { \frac { m M _ { \mathrm { E } } } { r ^ { 2 } } }"
    assert clean_latex(raw) == r"mg=G\,{\frac{mM_{\mathrm{E}}}{r^{2}}}"
    assert clean_latex(r"\vec { \bf v } _ { 1 }") == r"\vec{\bf v}_{1}"


def test_plausible_rejects_text_read_as_maths():
    assert plausible(r"T=2\pi\sqrt{\frac{m}{k}}") and plausible(r"\mathrm{kg}\cdot\mathrm{m}^{2}/\mathrm{s}^{2}")
    assert not plausible(r"\textsf{d}R_{\textrm{E}},\textrm{wecanuseEquation13.2tofind}M_{\textrm{E}}.")
    assert not plausible(r"\begin{array}{c}{\mathrm{accuracywas}}\\{5500\,\mathrm{kg/m^{3}}.}\\\end{array}")
    assert not plausible("")


def test_formulas_are_placed_between_the_text_above_and_below_them():
    lines = {720: "The weight is mg.", 600: "where r is the distance.", 300: "End of page."}

    def text_between(top, bottom):
        return "\n".join(t for y, t in sorted(lines.items(), reverse=True) if bottom < y <= top)

    formulas = [((100, 400, 200, 420), "g=9.8"), ((272, 637, 339, 665), r"mg=G\frac{mM}{r^{2}}")]
    assert merge_page_text(text_between, formulas, H) == (
        "The weight is mg.\n\n$$mg=G\\frac{mM}{r^{2}}$$\n\nwhere r is the distance.\n\n$$g=9.8$$\n\nEnd of page."
    )
