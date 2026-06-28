from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.calc import evaluate_calculation_requests, parse_calculation_requests  # noqa: E402
from lab4.data import load_questions, write_submission  # noqa: E402
from lab4.extract import extract_answer  # noqa: E402
from lab4.llm import ChatClient, LLMConfig  # noqa: E402
from lab4.prompts import normalize_question_text_for_prompt  # noqa: E402
from lab4.units import infer_target_unit, normalize_for_unit  # noqa: E402


SYSTEM_PROMPT = """You are a strict physics and chemistry numeric-answer judge.
You will see one problem and two candidate solutions. Do not vote by length or style.
First reread the problem, identify the requested quantity and target unit, then solve independently.
Use the candidates only as hints to find possible mistakes.
Common mistakes to check: wrong target quantity, sign convention, percent conversion, unit prefix,
order of magnitude, OCR-lost exponent, per-particle vs per-mole quantity, and rounding too early.
Retrieved context or a candidate derivation may be irrelevant or misleading; trust the problem statement.
If both candidates are wrong, give your own numeric result.
The last line must be exactly FINAL_ANSWER: <number>, with no unit or explanation after it."""


CALC_CHECK_SYSTEM_PROMPT = """You are a calculation-check assistant.
You receive one problem plus two candidate solutions. Extract only complex numeric expressions whose
Python-evaluated value would help check a candidate's arithmetic. Do not solve symbolically.
Return JSON only: a list of objects with keys label, source, purpose, expression.
Use Python math syntax: *, /, **, sqrt(...), sin(radians(...)), exp(...), log(...), pi, e.
Inline all numeric constants; do not use variables or units. Return [] if there is nothing useful."""


def load_submission(path: str | Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return {int(row["id"]): row["answer"] for row in csv.DictReader(f)}


def load_traces(paths: list[str]) -> dict[int, str]:
    responses: dict[int, str] = {}
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                traces = row.get("traces") or []
                if traces and traces[-1].get("response"):
                    responses[int(row["id"])] = traces[-1]["response"]
                elif row.get("response"):
                    responses[int(row["id"])] = row["response"]
    return responses


def load_judge_trace(path: str | Path) -> dict[int, dict]:
    trace_path = Path(path)
    if not trace_path.exists():
        return {}
    rows: dict[int, dict] = {}
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


def significant_digit_count(value: str) -> int:
    cleaned = str(value).strip()
    match = re.fullmatch(r"[-+]?(\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?", cleaned, flags=re.I)
    if not match:
        return 0
    mantissa = match.group(1)
    if "." in mantissa:
        digits = mantissa.replace(".", "").lstrip("0")
    else:
        digits = mantissa.lstrip("0").rstrip("0")
    return max(1, len(digits))


def choose_more_precise_if_close(primary_answer: str, secondary_answer: str, *, rtol: float) -> str:
    if not close_enough(primary_answer, secondary_answer, rtol=rtol):
        return primary_answer
    primary_digits = significant_digit_count(primary_answer)
    secondary_digits = significant_digit_count(secondary_answer)
    if secondary_digits > primary_digits:
        return secondary_answer
    return primary_answer


def valid_numeric(value: str) -> bool:
    number = parse_float(value)
    return number is not None and number != 0


def format_calculation_results(results: list) -> str:
    if not results:
        return "No code-calculated expression results were supplied."
    lines = [
        "The following expression values were computed by local Python code. If an expression correctly represents the needed formula, its numeric value is reliable. If the expression/formula is wrong, ignore or repair it."
    ]
    for result in results:
        if result.value is None:
            lines.append(
                f"- {result.label} ({result.source}): {result.expression} -> ERROR: {result.error}; purpose: {result.purpose}"
            )
        else:
            lines.append(
                f"- {result.label} ({result.source}): {result.expression} = {result.value}; purpose: {result.purpose}"
            )
    return "\n".join(lines)


def build_prompt(
    question,
    primary_answer: str,
    secondary_answer: str,
    primary_response: str,
    secondary_response: str,
    calculation_results: list | None = None,
) -> str:
    question_text = normalize_question_text_for_prompt(question.question)
    target_unit = infer_target_unit(question_text, question.unit) or "as specified by the problem"
    return "\n".join(
        [
            f"Problem id: {question.id}",
            f"Field: {question.field}",
            f"Subfield: {question.subfield or ''}",
            f"Topic hint: {question.theorem or ''}",
            f"Problem: {question_text}",
            f"Target answer unit/form: {target_unit}",
            "",
            f"Candidate A answer: {primary_answer}",
            "Candidate A reasoning excerpt:",
            primary_response[-1800:],
            "",
            f"Candidate B answer: {secondary_answer}",
            "Candidate B reasoning excerpt:",
            secondary_response[-1800:],
            "",
            "Code-calculated expression checks:",
            format_calculation_results(calculation_results or []),
            "",
            "Now solve independently. If a candidate only has a final-format, unit-scale, or sign error, correct it.",
            "Return one submit-ready number on the final line.",
        ]
    )


def build_calc_check_prompt(question, primary_answer: str, secondary_answer: str, primary_response: str, secondary_response: str) -> str:
    question_text = normalize_question_text_for_prompt(question.question)
    target_unit = infer_target_unit(question_text, question.unit) or "as specified by the problem"
    return "\n".join(
        [
            f"Problem id: {question.id}",
            f"Problem: {question_text}",
            f"Target answer unit/form: {target_unit}",
            "",
            f"Candidate A answer: {primary_answer}",
            "Candidate A reasoning excerpt:",
            primary_response[-2200:],
            "",
            f"Candidate B answer: {secondary_answer}",
            "Candidate B reasoning excerpt:",
            secondary_response[-2200:],
            "",
            "Extract up to 6 useful numeric expressions from the reasoning. Prefer final arithmetic expressions that directly compute the requested quantity or an important intermediate value.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "student_zh.json"))
    parser.add_argument("--primary-submission", required=True)
    parser.add_argument("--secondary-submission", required=True)
    parser.add_argument("--primary-trace", action="append", default=[])
    parser.add_argument("--secondary-trace", action="append", default=[])
    parser.add_argument("--submission-out", required=True)
    parser.add_argument("--trace-out", required=True)
    parser.add_argument("--ids", default="", help="Comma-separated ids to judge; default judges all disagreements")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--resume-from-trace", action="store_true")
    parser.add_argument(
        "--calculation-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask a calculation-check role to extract expressions, evaluate them locally, and pass results to the judge",
    )
    args = parser.parse_args()

    questions = load_questions(args.data)
    questions_by_id = {question.id: question for question in questions}
    requested_ids = {int(item.strip()) for item in args.ids.split(",") if item.strip()}
    primary = load_submission(args.primary_submission)
    secondary = load_submission(args.secondary_submission)
    primary_traces = load_traces(args.primary_trace)
    secondary_traces = load_traces(args.secondary_trace)
    judged = load_judge_trace(args.trace_out) if args.resume_from_trace else {}

    predictions = dict(primary)
    for question in questions:
        if question.id in judged:
            continue
        primary_answer = primary.get(question.id, "")
        secondary_answer = secondary.get(question.id, "")
        chosen = choose_more_precise_if_close(primary_answer, secondary_answer, rtol=args.rtol)
        if chosen != primary_answer:
            predictions[question.id] = chosen

    for qid, row in judged.items():
        question = questions_by_id.get(qid)
        answer = str(row.get("answer", ""))
        if question is not None:
            unit = infer_target_unit(question.question, question.unit)
            response = str(row.get("response", ""))
            if response:
                answer = extract_answer(response, target_unit=unit)
            answer = normalize_for_unit(answer, unit, question.question)
            row["answer"] = answer
        if valid_numeric(answer) and not close_enough(answer, primary.get(qid, ""), rtol=args.rtol):
            predictions[qid] = answer

    to_judge = [
        question
        for question in questions
        if (not requested_ids or question.id in requested_ids)
        if question.id not in judged
        and not close_enough(primary.get(question.id, ""), secondary.get(question.id, ""), rtol=args.rtol)
    ]
    print(f"Judging {len(to_judge)} disagreements; {len(judged)} already completed")

    llm_config = LLMConfig()
    if args.disable_thinking:
        llm_config.enable_thinking = False
    if args.enable_thinking:
        llm_config.enable_thinking = True
    client = ChatClient(llm_config)

    def solve(question) -> dict:
        calculation_response = ""
        calculation_results = []
        if args.calculation_check:
            calculation_response = client.chat(
                [
                    {"role": "system", "content": CALC_CHECK_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_calc_check_prompt(
                            question,
                            primary.get(question.id, ""),
                            secondary.get(question.id, ""),
                            primary_traces.get(question.id, ""),
                            secondary_traces.get(question.id, ""),
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=min(args.max_tokens, 2048),
            )
            calculation_requests = parse_calculation_requests(calculation_response)
            calculation_results = evaluate_calculation_requests(calculation_requests)

        response = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_prompt(
                        question,
                        primary.get(question.id, ""),
                        secondary.get(question.id, ""),
                        primary_traces.get(question.id, ""),
                        secondary_traces.get(question.id, ""),
                        calculation_results,
                    ),
                },
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        unit = infer_target_unit(question.question, question.unit)
        answer = extract_answer(response, target_unit=unit)
        answer = normalize_for_unit(answer, unit, question.question)
        return {
            "id": question.id,
            "answer": answer,
            "primary_answer": primary.get(question.id, ""),
            "secondary_answer": secondary.get(question.id, ""),
            "calculation_response": calculation_response,
            "calculation_results": [result.__dict__ for result in calculation_results],
            "response": response,
        }

    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume_from_trace else "w"
    with trace_path.open(mode, encoding="utf-8") as f:
        if args.workers <= 1:
            for pos, question in enumerate(to_judge, start=1):
                row = solve(question)
                if valid_numeric(row["answer"]) and not close_enough(row["answer"], primary.get(question.id, ""), rtol=args.rtol):
                    predictions[question.id] = str(row["answer"])
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{pos}/{len(to_judge)}] id={question.id} answer={row['answer']}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_question = {executor.submit(solve, question): question for question in to_judge}
                for pos, future in enumerate(as_completed(future_to_question), start=1):
                    question = future_to_question[future]
                    row = future.result()
                    if valid_numeric(row["answer"]) and not close_enough(row["answer"], primary.get(question.id, ""), rtol=args.rtol):
                        predictions[question.id] = str(row["answer"])
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[{pos}/{len(to_judge)}] id={question.id} answer={row['answer']}", flush=True)

    for question in questions:
        predictions.setdefault(question.id, "0")
    write_submission(args.submission_out, questions, predictions)
    print(f"Wrote {args.submission_out}")
    print(f"Wrote {args.trace_out}")


if __name__ == "__main__":
    main()
