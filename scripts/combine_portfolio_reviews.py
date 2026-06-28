from __future__ import annotations

import argparse
import csv
import json
import math
import re
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


def significant_digit_count(value: str) -> int:
    cleaned = str(value).strip()
    match = re.fullmatch(r"[-+]?(\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?", cleaned, flags=re.I)
    if not match:
        return 0
    mantissa = match.group(1)
    digits = mantissa.replace(".", "").lstrip("0") if "." in mantissa else mantissa.lstrip("0").rstrip("0")
    return max(1, len(digits))


def candidate_groups(candidates: list[dict], *, rtol: float) -> list[list[str]]:
    groups: list[list[str]] = []
    for candidate in candidates:
        answer = str(candidate.get("answer", ""))
        if parse_float(answer) is None:
            continue
        for group in groups:
            if close_enough(answer, group[0], rtol=rtol):
                group.append(answer)
                break
        else:
            groups.append([answer])
    return groups


def representative(group: list[str]) -> str:
    return max(group, key=significant_digit_count)


def calculation_value_support(answer: str, calculation_results: list[dict], *, rtol: float) -> int:
    count = 0
    for result in calculation_results:
        value = result.get("value") if isinstance(result, dict) else None
        if value is not None and close_enough(str(value), answer, rtol=rtol):
            count += 1
    return count


def precision_upgrade(current: str, candidates: list[dict], *, rtol: float, loose_rtol: float = 0.02) -> str | None:
    current_support = support_count(candidates, current, rtol=rtol)
    current_digits = significant_digit_count(current)
    for group in candidate_groups(candidates, rtol=rtol):
        answer = representative(group)
        if (
            len(group) > current_support
            and significant_digit_count(answer) > current_digits
            and not close_enough(current, answer, rtol=rtol)
            and close_enough(current, answer, rtol=loose_rtol)
        ):
            return answer
    return None


def references_supplied_values(question_text: str) -> bool:
    text = question_text.lower()
    return (
        ("\u6839\u636e" in question_text and "\u503c" in question_text)
        or "given value" in text
        or "given values" in text
        or "from the table" in text
        or "previous problem" in text
    )


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

        current = predictions.get(qid, "0")
        groups = candidate_groups(candidates, rtol=args.rtol)
        if calc_results and groups:
            supported_by_calculation = [
                (calculation_value_support(representative(group), calc_results, rtol=args.rtol), len(group), representative(group))
                for group in groups
            ]
            supported_by_calculation.sort(reverse=True)
            best_calc_support, best_candidate_support, best_answer = supported_by_calculation[0]
            current_calc_support = calculation_value_support(current, calc_results, rtol=args.rtol)
            if (
                best_calc_support >= 2
                and best_calc_support > current_calc_support
                and not opposite_same_magnitude(current, best_answer, rtol=args.rtol)
                and not close_enough(current, best_answer, rtol=args.rtol)
            ):
                predictions[qid] = best_answer
                reason = "multiple calculation expressions aligned with candidate group"

        current = predictions.get(qid, "0")
        upgraded = precision_upgrade(current, candidates, rtol=args.rtol)
        if upgraded is not None:
            predictions[qid] = upgraded
            reason = "precision upgrade to better-supported close candidate"

        current = predictions.get(qid, "0")
        groups = candidate_groups(candidates, rtol=args.rtol)
        if not calc_results and groups and references_supplied_values(question.question):
            best_group = max(groups, key=len)
            best_answer = representative(best_group)
            if (
                len(best_group) >= 3
                and support_count(candidates, current, rtol=args.rtol) <= 1
                and not close_enough(current, best_answer, rtol=args.rtol)
            ):
                predictions[qid] = best_answer
                reason = "referenced-data problem fallback to high-support candidate group"

        if reason:
            decisions.append({"id": qid, "answer": predictions[qid], "reason": reason})

    write_submission(args.submission_out, questions, predictions)
    print(f"Wrote {args.submission_out}")
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
