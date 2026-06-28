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


SYSTEM_PROMPT = """You are a strict physics and chemistry numeric-answer arbiter.
You will see one problem and several independently produced candidate answers.
Do not vote by source name, answer frequency, length, or confidence. Reread the problem and solve the requested quantity.
Use the candidate reasoning and code-calculated expression values only as evidence.
The code-calculated values are arithmetically reliable if their expressions/formulas represent the needed quantity; if an expression/formula is conceptually wrong, ignore it.
Check target quantity, target unit, sign convention, percent conversion, unit prefix, exponent/OCR slips, per-particle vs per-mole quantities, and premature rounding.
If the problem refers to given table values, a previous problem, or supplied constants that are not visible in the prompt, do not replace them with memorized data; prefer candidate derivations that clearly use the referenced supplied values.
Respect explicit wording such as diameter/radius, total/one-side distance, per molecule/per mole, and percent/fraction before applying a familiar textbook convention.
When converting unit prefixes, write the power-of-ten conversion mentally before choosing the final number.
Return enough digits to distinguish close candidates; do not round to significant figures unless the problem explicitly asks for that.
If candidates differ only by sign, do not guess from wording such as gained/lost; write the governing energy or direction equation and use the standard sign convention for the requested quantity.
The last line must be exactly FINAL_ANSWER: <number>, with no unit or explanation after it."""


CALC_CHECK_SYSTEM_PROMPT = """You are a calculation-check assistant.
Extract useful numeric expressions from the candidate reasoning. Return JSON only: a list of objects
with keys label, source, purpose, expression. Use Python math syntax: *, /, **, sqrt(...),
sin(radians(...)), exp(...), log(...), pi, e. Inline all numeric constants; do not use variables or units.
Prefer expressions that directly compute the requested final quantity or a key intermediate. Return [] if none."""


def load_submission(path: str | Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return {int(row["id"]): row["answer"] for row in csv.DictReader(f)}


def parse_labeled_path(raw: str) -> tuple[str, str]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), path.strip()
    path = Path(raw)
    return path.stem, raw


def load_traces(paths: list[str]) -> dict[int, list[dict[str, str]]]:
    responses: dict[int, list[dict[str, str]]] = {}
    for raw in paths:
        label, path_text = parse_labeled_path(raw)
        path = Path(path_text)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                qid = int(row["id"])
                snippets: list[tuple[str, str]] = []
                for trace in row.get("traces") or []:
                    for key in ("response", "draft_response", "rag_response", "no_rag_response"):
                        if trace.get(key):
                            snippets.append((key, str(trace[key])))
                if row.get("response"):
                    snippets.append(("response", str(row["response"])))
                for key, text in snippets[-3:]:
                    responses.setdefault(qid, []).append({"source": f"{label}:{key}", "response": text})
    return responses


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
    digits = mantissa.replace(".", "").lstrip("0") if "." in mantissa else mantissa.lstrip("0").rstrip("0")
    return max(1, len(digits))


def choose_representative(answers: list[tuple[str, str]]) -> str:
    return max(answers, key=lambda item: significant_digit_count(item[1]))[1]


def answer_groups(answers: list[tuple[str, str]], *, rtol: float) -> list[list[tuple[str, str]]]:
    groups: list[list[tuple[str, str]]] = []
    for item in answers:
        for group in groups:
            if close_enough(item[1], group[0][1], rtol=rtol):
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


def valid_numeric(value: str) -> bool:
    number = parse_float(value)
    return number is not None and number != 0


def opposite_same_magnitude(left: str, right: str, *, rtol: float) -> bool:
    left_number = parse_float(left)
    right_number = parse_float(right)
    return (
        left_number is not None
        and right_number is not None
        and left_number * right_number < 0
        and math.isclose(abs(left_number), abs(right_number), rel_tol=rtol, abs_tol=0.0)
    )


def format_calculation_results(results: list) -> str:
    if not results:
        return "No code-calculated expression results were supplied."
    lines = []
    for result in results:
        if result.value is None:
            lines.append(f"- {result.label} ({result.source}): {result.expression} -> ERROR: {result.error}; purpose: {result.purpose}")
        else:
            lines.append(f"- {result.label} ({result.source}): {result.expression} = {result.value}; purpose: {result.purpose}")
    return "\n".join(lines)


def build_candidate_block(qid: int, answers: list[tuple[str, str]], traces: list[dict[str, str]]) -> str:
    lines: list[str] = ["Candidate answers:"]
    for label, answer in answers:
        lines.append(f"- {label}: {answer}")
    if traces:
        lines.append("")
        lines.append("Reasoning excerpts:")
        for trace in traces[-8:]:
            snippet = trace["response"][-1000:].replace("\r", "")
            lines.append(f"[{trace['source']}]\n{snippet}")
    return "\n".join(lines)


def build_calc_prompt(question, answers: list[tuple[str, str]], traces: list[dict[str, str]]) -> str:
    question_text = normalize_question_text_for_prompt(question.question)
    target_unit = infer_target_unit(question_text, question.unit) or "as specified by the problem"
    return "\n".join(
        [
            f"Problem id: {question.id}",
            f"Problem: {question_text}",
            f"Target answer unit/form: {target_unit}",
            "",
            build_candidate_block(question.id, answers, traces),
        ]
    )


def build_judge_prompt(question, answers: list[tuple[str, str]], traces: list[dict[str, str]], calculation_results: list) -> str:
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
            build_candidate_block(question.id, answers, traces),
            "",
            "Code-calculated expression checks:",
            format_calculation_results(calculation_results),
            "",
            "Solve independently and choose or correct the final submit-ready number.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "student_zh.json"))
    parser.add_argument("--submission", action="append", required=True, help="label=path, may be repeated")
    parser.add_argument("--trace", action="append", default=[], help="label=path, may be repeated")
    parser.add_argument("--submission-out", required=True)
    parser.add_argument("--trace-out", required=True)
    parser.add_argument("--ids", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--resume-from-trace", action="store_true")
    parser.add_argument("--calculation-check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    questions = load_questions(args.data)
    questions_by_id = {question.id: question for question in questions}
    requested_ids = {int(item.strip()) for item in args.ids.split(",") if item.strip()}
    submissions = [(label, load_submission(path)) for label, path in map(parse_labeled_path, args.submission)]
    trace_map = load_traces(args.trace)

    predictions = {question.id: submissions[0][1].get(question.id, "0") for question in questions}
    completed: dict[int, dict] = {}
    trace_path = Path(args.trace_out)
    if args.resume_from_trace and trace_path.exists():
        with trace_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    qid = int(row["id"])
                    answer = str(row["answer"])
                    first_answer = submissions[0][1].get(qid, "0")
                    if opposite_same_magnitude(answer, first_answer, rtol=args.rtol):
                        answer = first_answer
                        row["answer"] = answer
                    completed[qid] = row
                    predictions[qid] = answer

    to_judge = []
    for question in questions:
        if requested_ids and question.id not in requested_ids:
            continue
        if question.id in completed:
            continue
        answers = [(label, values.get(question.id, "0")) for label, values in submissions]
        groups = answer_groups(answers, rtol=args.rtol)
        if len(groups) == 1:
            predictions[question.id] = choose_representative(groups[0])
        else:
            to_judge.append(question)
    print(f"Judging {len(to_judge)} portfolio disagreements; {len(completed)} already completed")

    llm_config = LLMConfig()
    if args.disable_thinking:
        llm_config.enable_thinking = False
    if args.enable_thinking:
        llm_config.enable_thinking = True
    client = ChatClient(llm_config)

    def solve(question) -> dict:
        answers = [(label, values.get(question.id, "0")) for label, values in submissions]
        traces = trace_map.get(question.id, [])
        calculation_response = ""
        calculation_results = []
        if args.calculation_check:
            calculation_response = client.chat(
                [
                    {"role": "system", "content": CALC_CHECK_SYSTEM_PROMPT},
                    {"role": "user", "content": build_calc_prompt(question, answers, traces)},
                ],
                temperature=0.0,
                max_tokens=min(args.max_tokens, 2048),
            )
            calculation_results = evaluate_calculation_requests(parse_calculation_requests(calculation_response))

        response = client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_prompt(question, answers, traces, calculation_results)},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        unit = infer_target_unit(question.question, question.unit)
        answer = extract_answer(response, target_unit=unit)
        answer = normalize_for_unit(answer, unit, question.question)
        first_answer = submissions[0][1].get(question.id, "0")
        if opposite_same_magnitude(answer, first_answer, rtol=args.rtol):
            answer = first_answer
        if not valid_numeric(answer):
            answer = first_answer
        return {
            "id": question.id,
            "answer": answer,
            "candidates": [{"label": label, "answer": value} for label, value in answers],
            "calculation_response": calculation_response,
            "calculation_results": [result.__dict__ for result in calculation_results],
            "response": response,
        }

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume_from_trace else "w"
    with trace_path.open(mode, encoding="utf-8") as f:
        if args.workers <= 1:
            for pos, question in enumerate(to_judge, start=1):
                row = solve(question)
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
