import pytest

from evals.regression import FingerprintMismatch, classify, compare
from evals.spec import BY_ID, quality_score


def test_higher_is_better_bands():
    spec = BY_ID["agent.args_acc"]  # band 0.05 absolute
    assert classify(spec, 0.80, 0.90) == "PASS"
    assert classify(spec, 0.80, 0.76) == "PASS"
    assert classify(spec, 0.80, 0.72) == "SUSPECT"
    assert classify(spec, 0.80, 0.60) == "FAIL"


def test_lower_is_better_bands():
    spec = BY_ID["agent.hallucinated_tool_rate"]  # band 0.02
    assert classify(spec, 0.00, 0.01) == "PASS"
    assert classify(spec, 0.00, 0.10) == "FAIL"


def test_speed_metrics_never_gate():
    spec = BY_ID["speed.gen_tps"]
    assert classify(spec, 20.0, 5.0) == "INFO-FAIL"


def test_new_and_missing():
    spec = BY_ID["agent.args_acc"]
    assert classify(spec, None, 0.5) == "NEW"
    assert classify(spec, 0.5, None) == "MISSING"


def test_compare_refuses_different_fingerprints():
    with pytest.raises(FingerprintMismatch):
        compare({"fingerprint": "a", "metrics": {}}, {"fingerprint": "b", "metrics": {}})


def test_compare_verdict_is_worst_gated_status():
    base = {"fingerprint": "x", "metrics": {"agent.args_acc": 0.8, "speed.gen_tps": 20.0}}
    cur = {"fingerprint": "x", "metrics": {"agent.args_acc": 0.5, "speed.gen_tps": 21.0}}
    _, verdict = compare(cur, base)
    assert verdict == "FAIL"
    _, verdict = compare({"fingerprint": "x", "metrics": {"agent.args_acc": 0.8, "speed.gen_tps": 5.0}}, base)
    assert verdict == "PASS"


def _bench_doc(quality, **metrics):
    return {"quality_score": quality, "metrics": metrics}


def test_selection_rule_filters_cpu_by_latency_and_memory_then_ranks_by_quality():
    from evals.bench_models import select_defaults

    docs = {
        ("fast", "cuda"): _bench_doc(0.80, **{"speed.vram_gb": 1.0}),
        ("fast", "cpu"): _bench_doc(0.79, **{"agent.call_latency_mean_s": 3.0, "speed.rss_gb": 1.5}),
        ("best", "cuda"): _bench_doc(0.95, **{"speed.vram_gb": 6.0}),
        ("best", "cpu"): _bench_doc(0.94, **{"agent.call_latency_mean_s": 14.0, "speed.rss_gb": 6.0}),  # too slow on CPU
        ("mid", "cuda"): _bench_doc(0.90, **{"speed.vram_gb": 3.0}),
        ("mid", "cpu"): _bench_doc(0.90, **{"agent.call_latency_mean_s": 6.0, "speed.rss_gb": 3.0, "vision.rss_gb": 9.0}),  # vision too big
        ("huge", "cuda"): _bench_doc(0.99, **{"speed.vram_gb": 9.5}),  # does not fit 8 GB VRAM
    }
    assert select_defaults(docs, "cuda") == {"cpu": "fast", "gpu": "best"}


def test_selection_rule_uses_gpu_quality_not_cpu_smoke_quality():
    from evals.bench_models import select_defaults

    docs = {
        ("a", "cuda"): _bench_doc(0.70, **{"speed.vram_gb": 1.0}),
        ("a", "cpu"): _bench_doc(0.99, **{"agent.call_latency_mean_s": 2.0, "speed.rss_gb": 1.0}),
        ("b", "cuda"): _bench_doc(0.90, **{"speed.vram_gb": 2.0}),
        ("b", "cpu"): _bench_doc(0.60, **{"agent.call_latency_mean_s": 2.0, "speed.rss_gb": 2.0}),
    }
    assert select_defaults(docs, "cuda")["cpu"] == "b"


def test_design_doc_tables_are_replaced_between_markers_only(tmp_path):
    from evals.bench_models import TABLES_END, TABLES_START, sync_design_doc

    doc = tmp_path / "DESIGN.md"
    doc.write_text(f"# Title\nbefore\n{TABLES_START}\nold table\n{TABLES_END}\nafter\n", encoding="utf-8")
    assert sync_design_doc("| new | table |", doc)
    text = doc.read_text(encoding="utf-8")
    assert "old table" not in text and "| new | table |" in text
    assert text.startswith("# Title\nbefore\n") and text.endswith(f"{TABLES_END}\nafter\n")
    assert sync_design_doc("| newer |", doc) and doc.read_text(encoding="utf-8").count(TABLES_START) == 1

    no_markers = tmp_path / "plain.md"
    no_markers.write_text("nothing to replace", encoding="utf-8")
    assert not sync_design_doc("| x |", no_markers)
    assert no_markers.read_text(encoding="utf-8") == "nothing to replace"


def test_quality_score_ignores_missing():
    assert quality_score({"agent.args_acc": 1.0, "grounded.answer_acc": 0.5}) == 0.75
    assert quality_score({}) is None
