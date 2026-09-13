import hashlib
import re
import zipfile

import pytest

from margin.export import assets
from margin.export.files import export_notes, saved_notes
from margin.export.html import CSP, render_markdown, to_html
from margin.notes.writer import SectionNotes, sections_in_scope
from margin.runtime.download import DownloadError
from margin.visuals.render import Rendered

FLOW = Rendered("flowchart", "mermaid", 'flowchart TD\n  A["Loop"] --> B["Sum <b>zero</b>"]')
TABLE = Rendered("table", "markdown", "| Law | Says |\n|---|---|\n| KVL | loop sums to zero |")


def _notes(markdown="# 4.1 Kirchhoff's laws\n- **Voltage law**: the loop sums to zero.", visuals=(FLOW, TABLE)):
    return [SectionNotes("d1", "4.1", "Kirchhoff's laws", markdown, visuals, 1, 0, 1.0)]


def test_markdown_escapes_html_and_keeps_latex_as_written():
    out = render_markdown("<script>alert(1)</script>\n\nOhm: $a_1*b_2*c$ and $$V = IR$$")
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert '<span class="math" title="$a_1*b_2*c$">a_1*b_2*c</span>' in out and "<em>" not in out
    assert '<span class="math" title="$$V = IR$$">V = IR</span>' in out


def test_page_is_self_contained_and_draws_diagrams_with_the_embedded_renderer():
    page = to_html(_notes(), "Notes: ch4", "window.mermaid={initialize(){}};")
    assert f'content="{CSP}"' in page and "default-src 'none'" in page
    assert '<pre class="mermaid">flowchart TD\n  A[&quot;Loop&quot;] --&gt; B[&quot;Sum &lt;b&gt;zero&lt;/b&gt;&quot;]</pre>' in page
    assert "<script>window.mermaid={initialize(){}};</script>" in page and "mermaid.initialize" in page
    assert "<table>" in page and "<th>Law</th>" in page
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", page)  # nothing loads from elsewhere


def test_page_without_renderer_says_how_to_get_one_and_embeds_nothing_without_diagrams():
    no_renderer = to_html(_notes(), "t", None)
    assert "<script" not in no_renderer and "margin setup" in no_renderer
    plain = to_html(_notes(visuals=(TABLE,)), "t", "window.mermaid={};")
    assert "<script" not in plain and "margin setup" not in plain


def test_mermaid_fences_inside_notes_become_diagrams():
    page = to_html(_notes("# 4.1\n\n```mermaid\nflowchart TD\n  A --> B\n```", visuals=()), "t", "x")
    assert '<pre class="mermaid">flowchart TD\n  A --&gt; B\n</pre>' in page


def test_export_files_write_markdown_html_and_anki(tmp_path):
    from tests.test_notes import _workspace

    with _workspace(tmp_path) as ws:
        ws.save_notes("d1", "4.1", "- **Voltage law**: voltage around a closed loop sums to zero.", "[]", 1, 0, 1.0)
        notes = saved_notes(ws, sections_in_scope(ws, "ch4"))
    assert [n.sid for n in notes] == ["4.1"]  # 4.2 has no saved notes
    md = export_notes(notes, "markdown", tmp_path / "out", "ch4")
    page = export_notes(notes, "html", tmp_path / "out", "ch4")
    deck = export_notes(notes, "anki", tmp_path / "out", "ch4")
    assert md.path.name == "margin-ch4.md" and "Voltage law" in md.path.read_text(encoding="utf-8")
    assert page.path.suffix == ".html" and deck.cards == 1 and zipfile.is_zipfile(deck.path)
    with pytest.raises(ValueError):
        export_notes(notes, "pdf", tmp_path, "ch4")


def test_renderer_download_is_verified_against_its_pinned_checksum(tmp_path, monkeypatch):
    def fake_download(url, dest, label=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"tampered")
        return dest

    monkeypatch.setattr(assets, "download", fake_download)
    with pytest.raises(DownloadError):
        assets.ensure_mermaid(tmp_path)
    assert not assets.mermaid_path(tmp_path).exists() and assets.mermaid_script(tmp_path) is None

    monkeypatch.setattr(assets, "MERMAID_SHA256", hashlib.sha256(b"tampered").hexdigest())
    assert assets.ensure_mermaid(tmp_path).read_bytes() == b"tampered"
