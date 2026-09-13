"""Every metric the suite reports: its direction, its tolerance, and whether it gates.

Quality metrics gate. Speed and memory are reported but never gate: they depend
on what else the machine is doing, and a guard that fires at random teaches
people to ignore it.

Bands are absolute for rates and relative for timings. A change within the band
is PASS, within twice the band SUSPECT, beyond that FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class MetricSpec:
    id: str
    direction: str  # "higher" or "lower" is better
    band: float
    relative: bool = False
    gated: bool = True
    description: str = ""


def _q(id: str, band: float, description: str, direction: str = "higher") -> MetricSpec:
    return MetricSpec(id, direction, band, description=description)


def _perf(id: str, direction: str, band: float, description: str) -> MetricSpec:
    return MetricSpec(id, direction, band, relative=True, gated=False, description=description)


SPECS: tuple[MetricSpec, ...] = (
    _q("agent.selection_acc", 0.05, "Right tool(s) chosen, single-turn cases"),
    _q("agent.args_acc", 0.05, "Right tool(s) with right arguments, single-turn cases"),
    _q("agent.parallel_acc", 0.10, "Several independent calls emitted in one turn, all correct"),
    _q("agent.multi_turn_acc", 0.10, "Correct call that depends on earlier conversation"),
    _q("agent.irrelevance_acc", 0.10, "No tool called when none fits"),
    _q("agent.schema_valid_rate", 0.03, "Emitted calls whose arguments validate against the schema"),
    _q("agent.hallucinated_tool_rate", 0.02, "Calls to tools that do not exist", "lower"),
    _q("agent.task_success_rate", 0.10, "Multi-step tasks completed: order, arguments, answer, termination"),
    _q("agent.trajectory_acc", 0.10, "Multi-step tasks whose required tools were called in order"),
    _q("agent.error_rate", 0.02, "Requests that failed outright", "lower"),
    MetricSpec("agent.redundant_calls_per_task", "lower", 0.5, gated=False, description="Tool calls beyond the required trajectory"),
    MetricSpec("agent.text_parsed_call_rate", "lower", 0.10, gated=False, description="Share of tool calls recovered from reply text because llama-server did not parse them"),
    _perf("agent.steps_mean", "lower", 0.25, "Model turns per task"),
    _perf("agent.call_latency_mean_s", "lower", 0.25, "Seconds per single-turn tool decision"),
    _q("grounded.answer_acc", 0.05, "Answerable questions answered correctly from the passage"),
    _q("grounded.refusal_acc", 0.10, "Unanswerable questions correctly refused"),
    _q("grounded.false_refusal_rate", 0.05, "Answerable questions wrongly refused", "lower"),
    _q("grounded.mixed_refusal_rate", 0.05, "Replies that give an answer and also the refusal sentinel", "lower"),
    _perf("grounded.latency_mean_s", "lower", 0.25, "Seconds per grounded answer"),
    _q("structured.schema_valid_rate", 0.03, "JSON outputs that validate"),
    _q("structured.key_term_coverage", 0.05, "Passage key terms present in diagrams and tables (lexical)"),
    _q("structured.grounding", 0.05, "Content words in diagrams and tables that occur in the passage (lexical)"),
    _q("structured.flowchart_edges_valid_rate", 0.05, "Flowcharts whose edges reference existing steps"),
    _q("questions.answer_grounded_rate", 0.10, "Generated model answers supported by the passage (lexical)"),
    _q("notes.key_term_coverage", 0.10, "Passage key terms present in notes (lexical)"),
    _q("notes.sentence_support_rate", 0.10, "Note sentences supported by the passage (lexical)"),
    _perf("structured.latency_mean_s", "lower", 0.25, "Seconds per structured output"),
    _q("vision.cer_mean", 0.03, "Character error rate transcribing photographed pages", "lower"),
    _q("vision.cer_typed", 0.03, "Character error rate, typed pages", "lower"),
    _q("vision.cer_handwriting", 0.05, "Character error rate, handwriting-style pages (synthetic)", "lower"),
    _q("vision.error_rate", 0.02, "Image requests that failed outright", "lower"),
    MetricSpec("vision.ocr_cer_mean", "lower", 0.01, gated=False, description="Control arm: RapidOCR character error rate on the same images"),
    MetricSpec("vision.ocr_cer_handwriting", "lower", 0.01, gated=False, description="Control arm: RapidOCR on handwriting-style pages"),
    _perf("vision.latency_mean_s", "lower", 0.25, "Seconds per image transcription by the model"),
    _perf("vision.ocr_latency_mean_s", "lower", 0.25, "Seconds per image for RapidOCR"),
    _perf("vision.rss_gb", "lower", 0.15, "Server resident memory with the vision projector loaded, GB"),
    _perf("vision.load_s", "lower", 0.5, "Seconds to load model plus vision projector"),
    _perf("speed.load_s", "lower", 0.5, "Seconds to load the model and become ready"),
    _perf("speed.rss_gb", "lower", 0.15, "Server process resident memory, GB"),
    _perf("speed.vram_gb", "lower", 0.15, "GPU memory taken by the model, GB"),
    _perf("speed.prompt_tps", "higher", 0.25, "Prompt tokens processed per second"),
    _perf("speed.ttft_long_s", "lower", 0.25, "Seconds to first token on a ~2k-token prompt"),
    _perf("speed.gen_tps", "higher", 0.25, "Tokens generated per second"),
)

BY_ID = {s.id: s for s in SPECS}

QUALITY_SCORE_IDS = (
    "agent.args_acc", "agent.irrelevance_acc", "agent.task_success_rate",
    "grounded.answer_acc", "grounded.refusal_acc",
    "structured.key_term_coverage", "structured.grounding",
    "questions.answer_grounded_rate", "notes.sentence_support_rate",
)


def quality_score(metrics: dict[str, float | None]) -> float | None:
    """Unweighted mean of the headline quality metrics. A summary for ranking, never a gate."""
    vals = [metrics[k] for k in QUALITY_SCORE_IDS if metrics.get(k) is not None]
    return mean(vals) if vals else None
