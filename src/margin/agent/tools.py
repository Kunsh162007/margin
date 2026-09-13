"""Tool definitions: the contract between the orchestrator and the model.

Each tool's parameters are a pydantic model. That one definition produces the
JSON schema the model sees and validates what the model sends back, so the
schema shown and the schema enforced can never drift apart.

Implementations are bound separately so the evaluation suite can score tool
*selection and arguments* without running retrieval or rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchBook(_Args):
    query: str = Field(description="What to look for, in the student's words or the book's terms.")
    top_k: int = Field(5, ge=1, le=20, description="How many passages to return.")
    scope: str | None = Field(None, description="Limit to a chapter or section id, e.g. 'ch4' or '4.2'.")


class ReadSection(_Args):
    section_id: str = Field(description="Section id as returned by search_book, e.g. '4.2'.")


class GetExamBlueprint(_Args):
    subject: str | None = Field(None, description="Subject or paper name; omit for the active workspace.")
    top_n: int = Field(10, ge=1, le=50, description="How many highest-weight topics to return.")


class FlowStep(_Args):
    id: str
    label: str


class FlowEdge(_Args):
    source: str
    target: str
    label: str | None = None


class MakeFlowchart(_Args):
    title: str
    steps: list[FlowStep] = Field(min_length=2)
    edges: list[FlowEdge] = Field(min_length=1)


class MakeTable(_Args):
    title: str
    columns: list[str] = Field(min_length=2)
    rows: list[list[str]] = Field(min_length=1)


class MindBranch(_Args):
    label: str
    children: list[str] = Field(default_factory=list)


class MakeMindmap(_Args):
    root: str
    branches: list[MindBranch] = Field(min_length=2)


class TimelineEvent(_Args):
    when: str
    what: str


class MakeTimeline(_Args):
    title: str
    events: list[TimelineEvent] = Field(min_length=2)


class Formula(_Args):
    name: str
    expression: str
    variables: str = Field(description="What each symbol means, with units.")


class MakeFormulaSheet(_Args):
    title: str
    formulas: list[Formula] = Field(min_length=1)


class Term(_Args):
    term: str
    definition: str


class MakeGlossary(_Args):
    terms: list[Term] = Field(min_length=1)


QuestionType = Literal["mcq", "short", "long", "numerical", "diagram", "case_study"]


class GenerateQuestions(_Args):
    topic: str
    count: int = Field(ge=1, le=20)
    marks: int = Field(ge=1, le=25, description="Marks per question.")
    question_type: QuestionType


class GradeAnswer(_Args):
    question: str
    student_answer: str
    max_marks: int = Field(ge=1, le=25)


class Calculate(_Args):
    expression: str = Field(description="Arithmetic expression, e.g. '0.5 * 2.2 * 3**2'. Supports + - * / ** sqrt log exp sin cos tan pi e.")


class CreateFlashcards(_Args):
    topic: str
    count: int = Field(10, ge=1, le=50)


class ExportNotes(_Args):
    format: Literal["markdown", "html", "pdf", "anki"]
    topic: str | None = Field(None, description="Limit the export to one topic; omit for everything.")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: type[_Args]

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.params.model_json_schema()}}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("search_book", "Search the student's uploaded books, notes and papers for passages about a topic.", SearchBook),
    ToolSpec("read_section", "Read the full text of one section of a book.", ReadSection),
    ToolSpec("get_exam_blueprint", "List the topics that carry the most marks in past exam papers.", GetExamBlueprint),
    ToolSpec("make_flowchart", "Draw a flowchart of a process, algorithm or sequence of stages.", MakeFlowchart),
    ToolSpec("make_table", "Build a table, e.g. to compare or classify things side by side.", MakeTable),
    ToolSpec("make_mindmap", "Draw a mind map of a topic and its subtopics.", MakeMindmap),
    ToolSpec("make_timeline", "Draw a timeline of dated or ordered events.", MakeTimeline),
    ToolSpec("make_formula_sheet", "Collect formulas with the meaning and units of every symbol.", MakeFormulaSheet),
    ToolSpec("make_glossary", "Build a glossary of terms and short definitions.", MakeGlossary),
    ToolSpec("generate_questions", "Write practice exam questions on a topic with answers and a marking scheme.", GenerateQuestions),
    ToolSpec("grade_answer", "Mark a student's answer to a question and explain what is missing.", GradeAnswer),
    ToolSpec("calculate", "Evaluate an arithmetic expression exactly. Use for any numerical working.", Calculate),
    ToolSpec("create_flashcards", "Make spaced-repetition flashcards on a topic.", CreateFlashcards),
    ToolSpec("export_notes", "Export the notes to a file format.", ExportNotes),
)

BY_NAME = {t.name: t for t in TOOLS}


def openai_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    selected = TOOLS if names is None else tuple(BY_NAME[n] for n in names)
    return [t.openai_schema() for t in selected]


def validate_arguments(name: str, arguments: dict[str, Any] | None) -> str | None:
    """Return None if the call is valid, otherwise a short reason."""
    spec = BY_NAME.get(name)
    if spec is None:
        return f"unknown tool {name!r}"
    if arguments is None:
        return "arguments were not valid JSON"
    try:
        spec.params.model_validate(arguments)
    except ValidationError as exc:
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3])
    return None
