from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import load_questions, write_submission  # noqa: E402
from lab4.extract import extract_answer  # noqa: E402
from lab4.llm import ChatClient, LLMConfig  # noqa: E402
from lab4.units import infer_target_unit, normalize_for_unit  # noqa: E402


SYSTEM_PROMPT = """你是严格的大学物理/化学数值题仲裁器。你会看到同一题两个强候选。
请不要按候选投票，必须独立重读题干，确定目标量和目标单位，重新建立公式，检查单位、数量级、符号、百分数、10 的幂、每粒子/每摩尔。
若两个候选都错，请给出你独立计算的数值。最后一行严格写 FINAL_ANSWER: <number>，只能是一个数值，不写单位。"""


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


def valid_numeric(value: str) -> bool:
    number = parse_float(value)
    return number is not None and number != 0


def build_prompt(question, primary_answer: str, secondary_answer: str, primary_response: str, secondary_response: str) -> str:
    target_unit = infer_target_unit(question.question, question.unit) or "题目指定或无单位"
    return "\n".join(
        [
            f"题目编号：{question.id}",
            f"学科：{question.field}",
            f"子领域：{question.subfield or ''}",
            f"相关提示：{question.theorem or ''}",
            f"题目：{question.question}",
            f"目标答案单位/形式：{target_unit}",
            "",
            f"候选A答案：{primary_answer}",
            "候选A过程摘录：",
            primary_response[-1800:],
            "",
            f"候选B答案：{secondary_answer}",
            "候选B过程摘录：",
            secondary_response[-1800:],
            "",
            "请独立重算。若某候选只是最终格式、单位缩放或符号错，可以修正；不要因为候选过程看起来详细就相信它。最后只输出一个可提交数值。",
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--resume-from-trace", action="store_true")
    args = parser.parse_args()

    questions = load_questions(args.data)
    questions_by_id = {question.id: question for question in questions}
    primary = load_submission(args.primary_submission)
    secondary = load_submission(args.secondary_submission)
    primary_traces = load_traces(args.primary_trace)
    secondary_traces = load_traces(args.secondary_trace)
    judged = load_judge_trace(args.trace_out) if args.resume_from_trace else {}

    predictions = dict(primary)
    for qid, row in judged.items():
        answer = str(row.get("answer", ""))
        if valid_numeric(answer) and not close_enough(answer, primary.get(qid, ""), rtol=args.rtol):
            predictions[qid] = answer

    to_judge = [
        question
        for question in questions
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
