"""Notes as one self-contained HTML file.

Everything the page needs is inside it — styles, the diagram renderer, the notes —
and its Content-Security-Policy forbids loading anything else. The file opens
offline, can be sent to someone, and cannot fetch a URL that came from a book.

Markdown is rendered with raw HTML switched off, so text from documents and from
the model is always escaped. LaTeX is lifted out before Markdown runs (``a_1*b_2``
would otherwise become emphasis) and shown as written. Diagrams are Mermaid,
drawn by the browser; the renderer is embedded only when a page has diagrams.
"""

from __future__ import annotations

import html
import re
from datetime import date
from typing import TYPE_CHECKING

from markdown_it import MarkdownIt

if TYPE_CHECKING:  # the notes writer imports the agent, which exports through this module
    from margin.notes.writer import SectionNotes

CSP = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:"
_MD = MarkdownIt("commonmark", {"html": False}).enable("table")
_MATH = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$", re.S)
_PLACEHOLDER = re.compile(r"MARGINMATH(\d+)X")
_MERMAID_FENCE = re.compile(r'<pre><code class="language-mermaid">(.*?)</code></pre>', re.S)

CSS = """
:root{--ground:#fbfaf6;--ink:#1f1c16;--muted:#6b6457;--rule:#e6dfcf;--gold:#8f6a10;--gold-soft:#f3ead2;--code:#f1ede3}
@media (prefers-color-scheme: dark){:root{--ground:#14120e;--ink:#ebe5d6;--muted:#a39b8a;--rule:#332e24;--gold:#d4a72c;--gold-soft:#2a2415;--code:#1e1b15}}
*{box-sizing:border-box}
body{margin:0;padding:0 20px 4rem;background:var(--ground);color:var(--ink);font:17px/1.65 Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif}
header,nav,main{max-width:46rem;margin:0 auto}
header{padding:3rem 0 1.25rem;border-bottom:2px solid var(--gold)}
h1,h2,h3,h4,nav,table,.brand,.meta,.note{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1,h2,h3,h4{line-height:1.25;text-wrap:balance}
.brand{font-weight:650;font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--gold);margin:0 0 .6rem}
header h1{font-size:2rem;margin:0}
.meta{color:var(--muted);font-size:.85rem;margin:.5rem 0 0}
nav{padding:1.25rem 0;border-bottom:1px solid var(--rule);font-size:.9rem}
nav h2{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:0 0 .5rem}
nav ol{margin:0;padding-left:1.4rem;columns:2 15rem;column-gap:2rem}
a{color:var(--gold)}
section{padding:2rem 0;border-bottom:1px solid var(--rule)}
section h1{font-size:1.45rem;margin-top:0}
section h2{font-size:1.15rem}
section h3{font-size:1rem}
li::marker{color:var(--gold)}
table{border-collapse:collapse;display:block;overflow-x:auto;font-size:.88rem;line-height:1.45;margin:1.25rem 0}
th,td{border-bottom:1px solid var(--rule);padding:.45rem .7rem;text-align:left;vertical-align:top}
th{background:var(--gold-soft)}
code,pre,.math{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;font-size:.86em}
.math{background:var(--code);padding:0 .25em;border-radius:3px}
pre{background:var(--code);padding:1rem;overflow-x:auto;border-radius:4px}
figure.diagram{margin:1.5rem 0;padding:1rem;border:1px solid var(--rule);border-radius:6px;overflow-x:auto}
figure.diagram pre.mermaid{background:transparent;margin:0;text-align:center}
.note{font-size:.85rem;color:var(--muted);border-left:3px solid var(--gold);padding-left:.75rem;margin:1rem auto;max-width:46rem}
@media print{body{background:#fff;color:#000;font-size:11pt;padding:0}nav{display:none}section{break-before:page;border:0}figure.diagram{break-inside:avoid}}
"""

MERMAID_START = (
    "mermaid.initialize({startOnLoad:true,securityLevel:'strict',"
    "theme:window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'neutral'});"
)


def _math_span(latex: str) -> str:
    """LaTeX without its $ delimiters; the full source stays in the tooltip."""
    inner = latex.strip("$").strip()
    return f'<span class="math" title="{html.escape(latex)}">{html.escape(inner)}</span>'


def render_markdown(text: str) -> str:
    spans: list[str] = []

    def lift(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"MARGINMATH{len(spans) - 1}X"

    body = _MD.render(_MATH.sub(lift, text))
    body = _PLACEHOLDER.sub(lambda m: _math_span(spans[int(m.group(1))]), body)
    return _MERMAID_FENCE.sub(r'<figure class="diagram"><pre class="mermaid">\1</pre></figure>', body)


def _visual(kind_format: str, text: str) -> str:
    if kind_format == "mermaid":
        return f'<figure class="diagram"><pre class="mermaid">{html.escape(text)}</pre></figure>'
    return render_markdown(text)


def to_html(notes: list[SectionNotes], title: str, mermaid_js: str | None) -> str:
    toc, sections = [], []
    for i, n in enumerate(notes):
        label = n.title if n.title.startswith(n.sid) else f"{n.sid} {n.title}"  # many books put the number in the title
        toc.append(f'<li><a href="#s{i}">{html.escape(label)}</a></li>')
        body = render_markdown(n.markdown) + "".join(_visual(v.format, v.text) for v in n.visuals)
        sections.append(f'<section id="s{i}">{body}</section>')
    content = "".join(sections)
    has_diagrams = 'class="mermaid"' in content
    if has_diagrams and mermaid_js:
        scripts = f"<script>{mermaid_js}</script><script>{MERMAID_START}</script>"
    elif has_diagrams:
        scripts = '<p class="note">Diagrams are shown as their source text. Run <code>margin setup</code> once, then export again to draw them.</p>'
    else:
        scripts = ""
    meta = f"{len(notes)} section{'s' if len(notes) != 1 else ''} · exported {date.today():%d %B %Y}"
    return "".join([
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n',
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n',
        f'<meta http-equiv="Content-Security-Policy" content="{CSP}">\n',
        f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n",
        f'<header><p class="brand">Margin</p><h1>{html.escape(title)}</h1><p class="meta">{meta}</p></header>\n',
        f"<nav><h2>Contents</h2><ol>{''.join(toc)}</ol></nav>\n",
        f"<main>{content}</main>\n{scripts}\n</body>\n</html>\n",
    ])
