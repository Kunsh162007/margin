"""Registry of language models Margin knows how to run.

Every entry is a single-file GGUF at roughly 4-bit precision. Gemma uses the
Q4_0 files published alongside the model; the rest use Q4_K_M, the common
default for this size class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HF_URL = "https://huggingface.co/{repo}/resolve/main/{file}"
TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    family: str
    repo: str
    file: str
    size_gb: float
    params_b: float
    license: str
    mmproj: str | None = None  # vision projector file in the same repo, for image input
    # Chat templates disagree on where tools go. Most read the standard `tools` variable.
    # SmolLM3 reads its own template variable ("xml_tools"). Phi-4-mini's embedded template
    # cannot receive tools through llama-server at all, so Margin ships a replacement.
    tools_template_kwarg: str | None = None
    chat_template: str | None = None  # file in runtime/templates overriding the GGUF's template

    @property
    def vision(self) -> bool:
        return self.mmproj is not None

    @property
    def url(self) -> str:
        return HF_URL.format(repo=self.repo, file=self.file)

    @property
    def mmproj_url(self) -> str | None:
        return HF_URL.format(repo=self.repo, file=self.mmproj) if self.mmproj else None

    def path(self, models_dir: Path) -> Path:
        return models_dir / self.file

    def mmproj_path(self, models_dir: Path) -> Path | None:
        # Several repos name their projector "mmproj-F16.gguf"; prefix the id so they cannot collide.
        return models_dir / f"{self.id}.mmproj.gguf" if self.mmproj else None

    def chat_template_path(self) -> Path | None:
        return TEMPLATES_DIR / self.chat_template if self.chat_template else None


CANDIDATES: tuple[ModelSpec, ...] = (
    ModelSpec("lfm2.5-1.2b", "LiquidAI LFM2.5", "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "LFM2.5-1.2B-Instruct-Q4_K_M.gguf", 0.68, 1.2, "LFM Open License"),
    ModelSpec("qwen3.5-2b", "Qwen3.5", "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 1.19, 2.0, "Apache-2.0", "mmproj-F16.gguf"),
    ModelSpec("gemma4-e2b", "Gemma 4", "ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_0.gguf", 2.65, 2.3, "Gemma", "mmproj-gemma-4-E2B-it-Q8_0.gguf"),
    ModelSpec("granite4.1-3b", "IBM Granite 4.1", "ibm-granite/granite-4.1-3b-GGUF", "granite-4.1-3b-Q4_K_M.gguf", 1.96, 3.0, "Apache-2.0"),
    ModelSpec("smollm3-3b", "SmolLM3", "ggml-org/SmolLM3-3B-GGUF", "SmolLM3-Q4_K_M.gguf", 1.78, 3.1, "Apache-2.0", tools_template_kwarg="xml_tools"),
    ModelSpec("phi4-mini", "Phi-4-mini", "unsloth/Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct-Q4_K_M.gguf", 2.32, 3.8, "MIT", chat_template="phi4-mini.jinja"),
    ModelSpec("qwen3.5-4b", "Qwen3.5", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 2.55, 4.0, "Apache-2.0", "mmproj-F16.gguf"),
    ModelSpec("gemma4-e4b", "Gemma 4", "ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_0.gguf", 4.28, 4.5, "Gemma", "mmproj-gemma-4-E4B-it-Q8_0.gguf"),
    ModelSpec("qwen3.5-9b", "Qwen3.5", "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf", 5.29, 9.0, "Apache-2.0", "mmproj-F16.gguf"),
)

BY_ID = {m.id: m for m in CANDIDATES}

# CPU: chosen by the selection rule in ``evals/bench_models.py``. GPU: the rule picked granite4.1-3b
# by 0.001, a tie; the owner chose gemma4-e4b for both (one download, vision, parallel calls).
DEFAULT_CPU_MODEL = "gemma4-e4b"
DEFAULT_GPU_MODEL = "gemma4-e4b"
MIN_FREE_RAM_GB = 2.0


def recommended(backend: str, ram_gb: float, vram_gb: float | None) -> ModelSpec:
    """Default model for this machine; falls back to the smallest if the default will not fit."""
    choice = BY_ID[DEFAULT_CPU_MODEL if backend == "cpu" else DEFAULT_GPU_MODEL]
    budget = (vram_gb if backend != "cpu" and vram_gb else ram_gb - MIN_FREE_RAM_GB)
    if choice.size_gb * 1.3 > budget:
        return CANDIDATES[0]
    return choice


def get(model_id: str) -> ModelSpec:
    try:
        return BY_ID[model_id]
    except KeyError:
        raise KeyError(f"unknown model {model_id!r}; known: {', '.join(BY_ID)}") from None
