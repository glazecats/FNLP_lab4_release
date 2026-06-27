from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import load_questions  # noqa: E402


UNIT_RE = re.compile(r"\b(?:m/s|kg|m|s|nm|eV|MeV|J|K|mol|L|M|atm|Pa|Hz|W|V|A|C|N|rad)\b|°")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
BAD_TEXT_RE = re.compile(r"\b(?:answer|cannot|unknown|undetermined)\b|无法|不能")
BAD_MATH_RE = re.compile(r"[≈≃]|(?<!\\)\*|(?:^|[^A-Za-z])[-+]?\d+(?:\.\d+)?e[+-]?$", re.IGNORECASE)
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "student_zh.json"))
    parser.add_argument("--submission", required=True)
    args = parser.parse_args()

    questions = load_questions(args.data)
    expected_ids = [str(q.id) for q in questions]
    with Path(args.submission).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    errors: list[str] = []
    if not rows:
        errors.append("submission is empty")
    if rows and set(rows[0].keys()) != {"id", "answer"}:
        errors.append(f"columns should be exactly id,answer; got {list(rows[0].keys())}")
    if len(rows) != len(expected_ids):
        errors.append(f"row count should be {len(expected_ids)}; got {len(rows)}")

    seen_ids = [row.get("id", "") for row in rows]
    if seen_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(seen_ids), key=int)
        extra = sorted(set(seen_ids) - set(expected_ids))
        errors.append(f"id order or id set mismatch; missing={missing[:5]}, extra={extra[:5]}")

    empty = [row["id"] for row in rows if not row.get("answer", "").strip()]
    unit_like = [row["id"] for row in rows if UNIT_RE.search(row.get("answer", ""))]
    text_like = [
        row["id"]
        for row in rows
        if CHINESE_RE.search(row.get("answer", "")) or BAD_TEXT_RE.search(row.get("answer", ""))
    ]
    equation_like = [row["id"] for row in rows if "=" in row.get("answer", "")]
    bad_math = [row["id"] for row in rows if BAD_MATH_RE.search(row.get("answer", ""))]
    multi_number = []
    for row in rows:
        answer = row.get("answer", "")
        if "\\" in answer:
            continue
        if len(NUMBER_RE.findall(answer)) >= 2 and any(ch.isspace() for ch in answer):
            multi_number.append(row["id"])
    if empty:
        errors.append(f"empty answers for ids: {empty[:10]}")
    if unit_like:
        errors.append(f"answers may contain units for ids: {unit_like[:10]}")
    if text_like:
        errors.append(f"answers may contain text instead of a final value for ids: {text_like[:10]}")
    if equation_like:
        errors.append(f"answers may contain equations or variable assignments for ids: {equation_like[:10]}")
    if bad_math:
        errors.append(f"answers may contain rough or incomplete math expressions for ids: {bad_math[:10]}")
    if multi_number:
        errors.append(f"answers may contain multiple numeric candidates for ids: {multi_number[:10]}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"OK: {args.submission} has {len(rows)} rows with columns id,answer")


if __name__ == "__main__":
    main()
