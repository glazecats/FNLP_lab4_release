from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import load_questions
from lab4.extract import looks_invalid_answer


def validate_submission(submission: str | Path, data: str | Path = "student_zh.json") -> list[str]:
    questions = load_questions(data)
    expected_ids = [str(q.id) for q in questions]
    errors: list[str] = []
    with Path(submission).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["id", "answer"]:
            errors.append(f"header must be id,answer; got {reader.fieldnames}")
            return errors
        rows = list(reader)
    if len(rows) != len(questions):
        errors.append(f"row count must be {len(questions)}; got {len(rows)}")
    got_ids = [row.get("id", "") for row in rows]
    if got_ids != expected_ids[: len(got_ids)]:
        errors.append("id order does not match data order")
    for i, row in enumerate(rows):
        answer = row.get("answer", "")
        reason = looks_invalid_answer(answer)
        if reason:
            errors.append(f"row {i + 2} id={row.get('id')}: {reason}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Kaggle submission format.")
    parser.add_argument("--submission", required=True)
    parser.add_argument("--data", default="student_zh.json")
    args = parser.parse_args()
    errors = validate_submission(args.submission, args.data)
    if errors:
        print("INVALID")
        for error in errors[:50]:
            print("-", error)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more errors")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()

