from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import load_questions, write_submission  # noqa: E402
from lab4.units import normalize_for_unit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "student_zh.json"))
    parser.add_argument("--submission", required=True)
    parser.add_argument("--submission-out", required=True)
    args = parser.parse_args()

    questions = load_questions(args.data)
    units = {question.id: question.unit for question in questions}
    with Path(args.submission).open("r", encoding="utf-8", newline="") as f:
        answers = {
            int(row["id"]): normalize_for_unit(row["answer"], units.get(int(row["id"])))
            for row in csv.DictReader(f)
        }
    write_submission(args.submission_out, questions, answers)
    print(f"Wrote {args.submission_out}")


if __name__ == "__main__":
    main()
