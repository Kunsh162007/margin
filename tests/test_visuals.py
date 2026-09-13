import pytest

from margin.visuals.render import Rendered, problems, render
from margin.visuals.repair import repair_flowchart

GLUCOSE = {
    "title": "Blood glucose",
    "steps": [
        {"id": "1", "label": "Glucose rises after meal"},
        {"id": "2", "label": "Beta cells release insulin"},
        {"id": "3", "label": "Blood glucose returns to normal"},
    ],
    "edges": [
        {"source": "Glucose rises after meal", "target": "Beta cells release insulin"},  # labels, not ids
        {"source": "2", "target": "3"},
        {"source": "2", "target": "3"},  # duplicate
        {"source": "3", "target": "3"},  # self-loop
        {"source": "Meal", "target": "Pancreas"},  # resolves to nothing
    ],
}


def test_repair_relinks_labels_and_drops_bad_edges():
    fixed, report = repair_flowchart(GLUCOSE)
    assert fixed["edges"] == [{"source": "1", "target": "2"}, {"source": "2", "target": "3"}]
    assert (report.edges_in, report.edges_relinked, report.edges_dropped, report.chained) == (5, 1, 3, False)
    assert not report.was_valid


def test_repair_chains_steps_when_no_edge_is_usable():
    data = {"steps": [{"id": "a", "label": "SYN"}, {"id": "b", "label": "SYN-ACK"}, {"id": "c", "label": "ACK"}], "edges": [{"source": "Client", "target": "Server"}]}
    fixed, report = repair_flowchart(data)
    assert report.chained and [(e["source"], e["target"]) for e in fixed["edges"]] == [("a", "b"), ("b", "c")]


def test_ambiguous_partial_label_is_not_guessed():
    data = {"steps": [{"id": "1", "label": "Atrial systole"}, {"id": "2", "label": "Ventricular systole"}], "edges": [{"source": "systole", "target": "2"}]}
    _, report = repair_flowchart(data)
    assert report.edges_dropped == 1 and report.edges_relinked == 0


def test_flowchart_escapes_hostile_labels_and_passes_the_check():
    data = {"steps": [{"id": "x", "label": 'Say "hi" ]] then {break}'}, {"id": "y", "label": "Done"}], "edges": [{"source": "x", "target": "y", "label": 'a "quoted" edge'}]}
    out = render("flowchart", data)
    assert out.format == "mermaid" and out.text.startswith("flowchart TD")
    assert '"Say #quot;hi#quot; )) then (break)"' in out.text
    assert problems(out) == []


def test_problems_catches_undeclared_nodes_and_ragged_tables():
    assert problems(Rendered("flowchart", "mermaid", 'flowchart TD\n    n0["A"]\n    n0 --> n7')) == ["undeclared nodes: ['n7']"]
    ragged = Rendered("table", "markdown", "| A | B |\n|---|---|\n| 1 | 2 | 3 |")
    assert problems(ragged) == ["1 table row(s) with the wrong number of cells"]


def test_table_pads_rows_escapes_pipes_and_coerces_flat_rows():
    out = render("table", {"title": "Orders", "columns": ["Order", "Capital", "Base"], "rows": [["Doric", "plain | cushion"], "Ionic"]})
    assert "| Doric | plain \\| cushion |  |" in out.text and "| Ionic |  |  |" in out.text
    assert problems(out) == []


def test_mindmap_timeline_formula_and_glossary_render():
    mind = render("mindmap", {"root": "Pharmacokinetics", "branches": [{"label": "Absorption", "children": ["into blood"]}, {"label": "Excretion", "children": []}]})
    assert mind.text.splitlines()[0] == "mindmap" and "into blood" in mind.text and problems(mind) == []
    line = render("timeline", {"title": "Penicillin: history", "events": [{"when": "1928", "what": "Fleming: mould kills bacteria"}]})
    assert "1928 : Fleming - mould kills bacteria" in line.text and problems(line) == []
    sheet = render("formula", {"title": "Beams", "formulas": [{"name": "Stress", "expression": "\\sigma = My/I", "variables": "M moment"}, {"name": "Ohm", "expression": "V = IR", "variables": "V volts"}]})
    assert "$\\sigma = My/I$" in sheet.text and "`V = IR`" in sheet.text
    gloss = render("glossary", {"terms": [{"term": "Pinnacle", "definition": "adds weight"}]})
    assert gloss.text == "- **Pinnacle** — adds weight"


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        render("piechart", {})
