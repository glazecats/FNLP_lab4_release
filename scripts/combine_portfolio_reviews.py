from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import load_questions, write_submission  # noqa: E402


def load_submission(path: str | Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return {int(row["id"]): row["answer"] for row in csv.DictReader(f)}


def load_trace(path: str | Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    trace_path = Path(path)
    if not trace_path.exists():
        return rows
    with trace_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[int(row["id"])] = row
    return rows


def parse_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def close_enough(left: str, right: str, *, rtol: float) -> bool:
    left_number = parse_float(left)
    right_number = parse_float(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, rel_tol=rtol, abs_tol=0.0)
    )


def opposite_same_magnitude(left: str, right: str, *, rtol: float) -> bool:
    left_number = parse_float(left)
    right_number = parse_float(right)
    return (
        left_number is not None
        and right_number is not None
        and left_number * right_number < 0
        and math.isclose(abs(left_number), abs(right_number), rel_tol=rtol, abs_tol=0.0)
    )


def support_count(candidates: list[dict], answer: str, *, rtol: float) -> int:
    return sum(1 for candidate in candidates if close_enough(str(candidate.get("answer", "")), answer, rtol=rtol))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "student_zh.json"))
    parser.add_argument("--primary-submission", required=True)
    parser.add_argument("--base-submission", required=True)
    parser.add_argument("--base-trace", required=True)
    parser.add_argument("--calc-submission", required=True)
    parser.add_argument("--calc-trace", required=True)
    parser.add_argument("--submission-out", required=True)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    questions = load_questions(args.data)
    primary = load_submission(args.primary_submission)
    base = load_submission(args.base_submission)
    calc = load_submission(args.calc_submission)
    base_trace = load_trace(args.base_trace)
    calc_trace = load_trace(args.calc_trace)

    predictions = dict(base)
    decisions: list[dict] = []
    for question in questions:
        qid = question.id
        first_answer = primary.get(qid, "0")
        current = predictions.get(qid, "0")
        reason = ""

        if opposite_same_magnitude(current, first_answer, rtol=args.rtol):
            predictions[qid] = first_answer
            reason = "opposite-sign same-magnitude guard"

        calc_row = calc_trace.get(qid, {})
        calc_results = calc_row.get("calculation_results") or []
        calc_answer = calc.get(qid, "")
        current = predictions.get(qid, "0")
        candidates = base_trace.get(qid, {}).get("candidates") or []
        if (
            not calc_results
            and close_enough(calc_answer, first_answer, rtol=args.rtol)
            and not close_enough(current, first_answer, rtol=args.rtol)
            and support_count(candidates, first_answer, rtol=args.rtol) > support_count(candidates, current, rtol=args.rtol)
        ):
            predictions[qid] = first_answer
            reason = "calc-review no-expression fallback to better-supported primary group"

        if reason:
            decisions.append({"id": qid, "answer": predictions[qid], "reason": reason})

    write_submission(args.submission_out, questions, predictions)
    print(f"Wrote {args.submission_out}")
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
