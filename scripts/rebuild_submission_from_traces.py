from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import load_questions, write_submission  # noqa: E402
from lab4.extract import extract_answer  # noqa: E402
from lab4.units import infer_target_unit, normalize_for_unit  # noqa: E402


def is_bad_verified_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    lowered = answer.lower()
    bad_phrases = (
        "无法确定",
        "无法计算",
        "无解",
        "缺少数据",
        "不能确定",
        "cannot",
        "unknown",
        "undetermined",
    )
    return (
        any(phrase in lowered for phrase in bad_phrases)
        or any("\u4e00" <= ch <= "\u9fff" for ch in answer)
        or not any(ch.isdigit() for ch in answer)
    )


def load_existing_submission(path: str | Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return {int(row["id"]): row["answer"] for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "student_zh.json"))
    parser.add_argument("--traces", required=True)
    parser.add_argument("--fill-from-submission", default="")
    parser.add_argument("--normalize-units", action="store_true", help="Also apply optional unit post-processing")
    parser.add_argument("--submission-out", required=True)
    args = parser.parse_args()

    questions = load_questions(args.data)
    units = {question.id: infer_target_unit(question.question, question.unit) for question in questions}
    answers: dict[int, str] = (
        load_existing_submission(args.fill_from_submission) if args.fill_from_submission else {}
    )
    with Path(args.traces).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            responses = [trace["response"] for trace in row.get("traces", []) if trace.get("response")]
            if responses:
                qid = int(row["id"])
                last_trace = row.get("traces", [])[-1]
                answer = extract_answer(responses[-1])
                draft_answer = last_trace.get("draft_answer")
                if draft_answer is not None and is_bad_verified_answer(answer):
                    answer = str(draft_answer)
                if is_bad_verified_answer(answer):
                    answer = "0"
                if args.normalize_units:
                    answer = normalize_for_unit(answer, units.get(qid))
                answers[qid] = answer
            else:
                answers[int(row["id"])] = str(row.get("answer", "0"))

    missing = [q.id for q in questions if q.id not in answers]
    if missing:
        raise ValueError(
            f"Trace file is missing ids: {missing[:10]}. "
            "Use --fill-from-submission to fill missing ids from an existing complete submission."
        )
    write_submission(args.submission_out, questions, answers)
    print(f"Wrote {args.submission_out}")


if __name__ == "__main__":
    main()
