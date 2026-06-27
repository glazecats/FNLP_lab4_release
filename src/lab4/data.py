from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    id: int
    field: str
    question: str
    answer: str | None = None
    subfield: str | None = None
    theorem: str | None = None
    unit: str | None = None


def load_questions(path: str | Path = "student_zh.json") -> list[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Question(**item) for item in data]


def select_questions(
    questions: list[Question],
    *,
    ids: str | None = None,
    limit: int | None = None,
) -> list[Question]:
    selected = questions
    if ids:
        wanted = {int(part.strip()) for part in ids.split(",") if part.strip()}
        selected = [q for q in selected if q.id in wanted]
    if limit is not None:
        selected = selected[:limit]
    return selected

