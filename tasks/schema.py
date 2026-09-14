"""Data contract for one golden task, plus the loader.

Every field an agent's performance is graded against is declared here once
and enforced by Pydantic when golden_tasks.json is loaded - the same
"validate the fixture, not just the code" approach lead-router's
agent/schema.py takes for LeadIn.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_TASK_ID_MAX = 40
_PROMPT_MAX = 500
_ANSWER_MAX = 300
_NOTES_MAX = 500

Category = Literal["lookup", "refund_calc", "escalation_required", "nonexistent_id", "ambiguous"]
ExpectedAction = Literal["answer", "escalate"]
MatchType = Literal["exact", "numeric_tolerance", "contains_all"]


class GoldenTask(BaseModel):
    task_id: str = Field(min_length=1, max_length=_TASK_ID_MAX)
    category: Category
    prompt: str = Field(min_length=1, max_length=_PROMPT_MAX)
    expected_action: ExpectedAction
    expected_answer: str | None = Field(default=None, max_length=_ANSWER_MAX)
    match_type: MatchType
    tolerance: float | None = Field(default=None, ge=0)
    required_substrings: list[str] | None = None
    notes: str = Field(min_length=1, max_length=_NOTES_MAX)
    max_steps_override: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def _match_type_has_its_required_fields(self) -> GoldenTask:
        needs_numeric = self.match_type == "numeric_tolerance"
        if needs_numeric and (self.expected_answer is None or self.tolerance is None):
            raise ValueError("numeric_tolerance requires both expected_answer and tolerance")
        if self.match_type == "contains_all" and not self.required_substrings:
            raise ValueError("contains_all requires a non-empty required_substrings list")
        needs_exact_answer = self.match_type == "exact" and self.expected_action == "answer"
        if needs_exact_answer and self.expected_answer is None:
            raise ValueError("match_type 'exact' + expected_action 'answer' needs expected_answer")
        return self


_TASKS_PATH = Path(__file__).resolve().parent / "golden_tasks.json"


def load_golden_tasks(path: Path | None = None) -> list[GoldenTask]:
    raw = json.loads((path or _TASKS_PATH).read_text())
    return [GoldenTask(**entry) for entry in raw]
