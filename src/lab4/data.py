from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Question:
    id: int
    field: str
    question: str
    answer: str | None = None
    subfield: str | None = None
    theorem: str | None = None
    unit: str | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "Question":
        return cls(
            id=int(item["id"]),
            field=str(item["field"]),
            question=str(item["question"]),
            answer=item.get("answer"),
            subfield=item.get("subfield"),
            theorem=item.get("theorem"),
            unit=item.get("unit"),
        )


def load_questions(path: str | Path) -> list[Question]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [Question.from_dict(item) for item in json.load(f)]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_submission(path: str | Path, questions: list[Question], answers: dict[int, str]) -> None:
    missing = [q.id for q in questions if q.id not in answers]
    if missing:
        raise ValueError(f"Missing predictions for ids: {missing[:10]}")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        for q in questions:
            writer.writerow({"id": q.id, "answer": answers[q.id]})

