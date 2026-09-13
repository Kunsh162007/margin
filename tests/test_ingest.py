from pathlib import Path

import pytest

from margin.ingest.detect import UnsupportedFile, sniff
from margin.ingest.sections import build_sections, heading_of, split_blocks
from margin.ingest.types import Block, Document, TocEntry

FONT = Path("C:/Windows/Fonts/arial.ttf")


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_sniff_uses_bytes_not_extension(tmp_path):
    assert sniff(_write(tmp_path, "notes.txt", b"%PDF-1.4 ...")) == "pdf"
    assert sniff(_write(tmp_path, "photo", b"\xff\xd8\xff\xe0rest")) == "jpeg"
    assert sniff(_write(tmp_path, "scan.jpg", b"\x89PNG\r\n\x1a\nrest")) == "png"
    assert sniff(_write(tmp_path, "readme", "Thévenin's theorem".encode())) == "text"
    with pytest.raises(UnsupportedFile):
        sniff(_write(tmp_path, "blob.bin", b"\x00\x01\x02\x03"))
    with pytest.raises(UnsupportedFile, match="HEIC"):
        sniff(_write(tmp_path, "IMG_0001.HEIC", b"\x00\x00\x00\x18ftypheic...."))


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("4.2 Kirchhoff's laws", ("4.2", "Kirchhoff's laws", 2)),
        ("Chapter 7: Timber structures", ("7", "Timber structures", 1)),
        ("3 Fluid Mechanics", ("3", "Fluid Mechanics", 1)),
        ("2.5 kg of water was heated.", None),
        ("The current in each branch is found.", None),
    ],
)
def test_heading_of(line, expected):
    assert heading_of(line) == expected


def test_split_blocks_joins_lines_repairs_hyphenation_and_finds_headings():
    text = "4.1 Ohm's law\nThe current is propor-\ntional to voltage.\n\n4.2 Kirchhoff's laws\nThe sum is zero."
    blocks = split_blocks(text, page=12)
    assert [b.kind for b in blocks] == ["heading", "paragraph", "heading", "paragraph"]
    assert blocks[1].text == "The current is proportional to voltage."
    assert blocks[2].level == 2 and all(b.page == 12 for b in blocks)


def _doc(blocks, toc=()):
    return Document("id", "src", "pdf", "Circuits", 3, tuple(blocks), tuple(toc))


def test_toc_sections_start_mid_page_not_at_page_boundaries():
    blocks = [
        Block("Preface text.", 1),
        Block("4.1 Ohm's law", 2, "heading", 2),
        Block("V equals IR.", 2),
        Block("4.2 Kirchhoff's laws", 2, "heading", 2),
        Block("Loop sums are zero.", 3),
    ]
    toc = [TocEntry("4.1 Ohm's law", 2, 2), TocEntry("4.2 Kirchhoff's laws", 2, 2)]
    sections = build_sections(_doc(blocks, toc))
    assert [s.id for s in sections] == ["front", "4.1", "4.2"]
    assert sections[1].text == "V equals IR."
    assert sections[2].text == "Loop sums are zero." and (sections[2].page_start, sections[2].page_end) == (2, 3)


def test_roundtrip_anchored_rate_counts_bookmarks_found_in_their_page_text():
    from evals.ingest_roundtrip import anchored_rate

    blocks = [Block("4.1 Ohm's law", 2, "heading", 2), Block("V = IR.", 2), Block("Kirchhoff sums", 3)]
    toc = [TocEntry("4.1 Ohm's law", 2, 2), TocEntry("4.2 Kirchhoff's laws", 2, 3), TocEntry("Appendix", 1, None)]
    assert anchored_rate(_doc(blocks, toc)) == 1 / 3
    assert anchored_rate(_doc(blocks)) is None


def test_title_merged_into_body_keeps_the_body_text():
    blocks = [
        Block("Key Terms abdominopelvic cavity division of the anterior cavity", 30),
        Block("Review Questions 1. Which cavity houses the viscera?", 31),
    ]
    toc = [TocEntry("Key Terms", 2, 30), TocEntry("Review Questions", 2, 31)]
    sections = build_sections(_doc(blocks, toc))
    assert sections[0].text == "abdominopelvic cavity division of the anterior cavity"
    assert sections[1].text == "1. Which cavity houses the viscera?"


def test_two_bookmarks_on_one_block_do_not_duplicate_text():
    blocks = [Block("PREFACE About OpenStax OpenStax is a nonprofit.", 1), Block("More preface.", 2)]
    toc = [TocEntry("Preface", 1, 1), TocEntry("About OpenStax", 2, 1)]
    sections = build_sections(_doc(blocks, toc))
    total = "".join(s.text for s in sections)
    assert total.count("nonprofit") == 1


def test_heading_sections_build_paths_and_unique_ids():
    blocks = [
        Block("Chapter 1 Circuits", 1, "heading", 1),
        Block("1.1 Basics", 1, "heading", 2),
        Block("Charge flows.", 1),
        Block("Summary", 2, "heading", 2),
        Block("Summary", 3, "heading", 2),
    ]
    sections = build_sections(_doc(blocks))
    assert sections[1].path == ("Chapter 1 Circuits",)
    ids = [s.id for s in sections]
    assert len(ids) == len(set(ids))


def test_unstructured_document_is_one_section():
    sections = build_sections(_doc([Block("Just text.", 1), Block("More.", 2)]))
    assert len(sections) == 1 and sections[0].page_end == 2


def test_docx_headings_and_tables_keep_document_order(tmp_path):
    docx = pytest.importorskip("docx")
    from margin.ingest.office import read_docx

    d = docx.Document()
    d.add_heading("Arteries and veins", level=1)
    d.add_paragraph("Arteries carry blood away from the heart.")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Vessel", "Wall"
    table.cell(1, 0).text, table.cell(1, 1).text = "Artery", "Thick"
    d.add_heading("Capillaries", level=2)
    path = tmp_path / "vessels.docx"
    d.save(path)

    doc = read_docx(path, "x")
    assert [b.kind for b in doc.blocks] == ["heading", "paragraph", "table", "heading"]
    assert "Artery | Thick" in doc.blocks[2].text
    assert [s.title for s in build_sections(doc)] == ["Arteries and veins", "Capillaries"]


def test_inspect_command_prints_sections_and_rejects_missing_files(tmp_path, capsys):
    docx = pytest.importorskip("docx")
    from margin.cli import main

    d = docx.Document()
    d.add_heading("Arteries", level=1)
    d.add_paragraph("Arteries carry blood away from the heart.")
    d.add_heading("Capillaries", level=1)
    d.add_paragraph("Capillaries exchange gases with tissue.")
    path = tmp_path / "vessels.docx"
    d.save(path)

    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "2 sections" in out and "Capillaries" in out
    assert main(["inspect", str(tmp_path / "missing.pdf")]) == 1


def test_pptx_slides_titles_and_notes(tmp_path):
    pptx = pytest.importorskip("pptx")
    from margin.ingest.office import read_pptx

    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Passive solar design"
    slide.placeholders[1].text = "South-facing glazing"
    slide.notes_slide.notes_text_frame.text = "Overhangs shade summer sun."
    path = tmp_path / "lecture.pptx"
    prs.save(path)

    doc = read_pptx(path, "x")
    assert doc.page_count == 1
    assert [b.kind for b in doc.blocks] == ["heading", "paragraph", "notes"]


def test_exif_rotated_photo_is_turned_upright(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    from margin.ingest.images import load_image

    img = Image.new("RGB", (400, 200), "white")
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    path = tmp_path / "phone.jpg"
    img.save(path, exif=exif)
    assert load_image(path).size == (200, 400)


@pytest.mark.skipif(not FONT.exists(), reason="needs a TrueType font")
def test_photo_ocr_end_to_end(tmp_path):
    pytest.importorskip("rapidocr")
    from PIL import Image, ImageDraw, ImageFont

    from margin.ingest.pipeline import ingest_file

    img = Image.new("RGB", (1200, 220), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(FONT), 40)
    draw.text((40, 40), "Flying buttresses carry outward thrust", fill="black", font=font)
    draw.text((40, 120), "to massive buttress piers.", fill="black", font=font)
    path = tmp_path / "board.png"
    img.save(path)

    doc, sections = ingest_file(path)
    assert doc.kind == "png" and doc.ocr_pages == (1,)
    assert "buttress" in sections[0].text.lower()
