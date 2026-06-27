from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import Question, load_questions, write_submission  # noqa: E402
from lab4.extract import extract_answer  # noqa: E402
from lab4.llm import ChatClient, LLMConfig  # noqa: E402
from lab4.prompts import build_user_prompt, build_verifier_prompt, get_system_prompt, VERIFIER_SYSTEM_PROMPT  # noqa: E402
from lab4.query import expand_query  # noqa: E402
from lab4.retrieval import TextbookIndex, load_or_build_index  # noqa: E402
from lab4.units import infer_target_unit, normalize_for_unit  # noqa: E402


MIN_RAG_SCORE = 20.0


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
    return any(phrase in lowered for phrase in bad_phrases) or any("\u4e00" <= ch <= "\u9fff" for ch in answer)


def choose_answer(candidates: list[str]) -> str:
    counts = Counter(candidates)
    return counts.most_common(1)[0][0]


def load_existing_submission(path: str | Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return {int(row["id"]): row["answer"] for row in csv.DictReader(f)}


def load_existing_trace(path: str | Path) -> tuple[dict[int, str], set[int]]:
    predictions: dict[int, str] = {}
    completed_ids: set[int] = set()
    trace_path = Path(path)
    if not trace_path.exists():
        return predictions, completed_ids
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = int(row["id"])
            predictions[qid] = str(row["answer"])
            completed_ids.add(qid)
    return predictions, completed_ids


def solve_one(
    client: ChatClient,
    question: Question,
    *,
    index: TextbookIndex | None,
    method: str,
    samples: int,
    temperature: float,
    max_tokens: int,
    top_k: int,
    prompt_style: str,
    normalize_units: bool,
) -> dict:
    uses_rag = method in {"rag", "rag-verify"}
    uses_verify = method in {"verify", "rag-verify"}
    retrieved = index.search(expand_query(question), field=question.field, top_k=top_k) if index else []
    retrieved = [chunk for chunk in retrieved if chunk.score >= MIN_RAG_SCORE]
    system_prompt = get_system_prompt(prompt_style)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": build_user_prompt(
                question,
                retrieved if uses_rag else None,
                prompt_style=prompt_style,
            ),
        }
    )

    traces = []
    answers = []
    for sample_idx in range(samples):
        response = client.chat(
            messages,
            temperature=temperature if samples > 1 else min(temperature, 0.2),
            max_tokens=max_tokens,
        )
        answer = extract_answer(response)
        trace = {"sample": sample_idx, "response": response, "answer": answer}
        if uses_verify:
            verify_messages = [
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_verifier_prompt(
                        question,
                        first_response=response,
                        first_answer=answer,
                        context=retrieved if uses_rag else None,
                    ),
                },
            ]
            verified_response = client.chat(
                verify_messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            verified_answer = extract_answer(verified_response)
            if is_bad_verified_answer(verified_answer):
                verified_answer = answer
            if is_bad_verified_answer(verified_answer):
                verified_answer = "0"
            trace.update(
                {
                    "draft_response": response,
                    "draft_answer": answer,
                    "response": verified_response,
                    "answer": verified_answer,
                }
            )
            answer = verified_answer
        if normalize_units:
            answer = normalize_for_unit(answer, infer_target_unit(question.question, question.unit))
            trace["answer"] = answer
        traces.append(trace)
        answers.append(answer)

    final_answer = choose_answer(answers)
    return {
        "id": question.id,
        "field": question.field,
        "subfield": question.subfield,
        "theorem": question.theorem,
        "unit": question.unit,
        "question": question.question,
        "retrieved": [chunk.__dict__ for chunk in retrieved],
        "traces": traces,
        "answer": final_answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "student_zh.json"))
    parser.add_argument("--tex-dir", default=str(ROOT / "textbooks-tex"))
    parser.add_argument("--index-cache", default=str(ROOT / "cache" / "textbook_index.json"))
    parser.add_argument("--method", choices=["baseline", "rag", "verify", "rag-verify"], default="baseline")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent questions to solve")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--prompt-style", choices=["baseline"], default="baseline")
    parser.add_argument("--normalize-units", action="store_true", help="Post-process obvious 10^n/prefix unit scaling mistakes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="Comma-separated ids to run, for example: 13,161,192")
    parser.add_argument("--fill-from-submission", default="", help="Existing submission used to fill ids that are not rerun")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--enable-thinking", action="store_true", help="Use Qwen3 streaming thinking mode")
    parser.add_argument("--disable-thinking", action="store_true", help="Force enable_thinking=false even if LLM_ENABLE_THINKING=1")
    parser.add_argument("--trace-out", default=str(ROOT / "outputs" / "traces.jsonl"))
    parser.add_argument("--resume-from-trace", action="store_true", help="Append to trace-out and skip ids already present in it")
    parser.add_argument("--submission-out", default=str(ROOT / "outputs" / "submission.csv"))
    parser.add_argument(
        "--checkpoint-submission-out",
        default="",
        help="Optional path for a submission file refreshed after each completed question",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    questions = load_questions(args.data)
    requested_ids = {int(item.strip()) for item in args.ids.split(",") if item.strip()}
    if requested_ids:
        run_questions = [question for question in questions if question.id in requested_ids]
        missing_ids = sorted(requested_ids - {question.id for question in run_questions})
        if missing_ids:
            raise ValueError(f"Requested ids not found: {missing_ids}")
    else:
        run_questions = questions[: args.limit] if args.limit else questions

    trace_predictions: dict[int, str] = {}
    completed_ids: set[int] = set()
    if args.resume_from_trace:
        trace_predictions, completed_ids = load_existing_trace(args.trace_out)
        run_questions = [question for question in run_questions if question.id not in completed_ids]
        print(f"Resuming from {args.trace_out}: {len(completed_ids)} completed, {len(run_questions)} remaining")

    index = None
    if args.method in {"rag", "rag-verify"}:
        index = load_or_build_index(args.tex_dir, args.index_cache)

    llm_config = LLMConfig()
    if args.disable_thinking:
        llm_config.enable_thinking = False
    if args.enable_thinking:
        llm_config.enable_thinking = True
    client = ChatClient(llm_config)
    trace_path = Path(args.trace_out)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    predictions: dict[int, str] = {}
    if args.fill_from_submission:
        predictions.update(load_existing_submission(args.fill_from_submission))
    predictions.update(trace_predictions)
    checkpoint_path = args.checkpoint_submission_out or str(Path(args.submission_out).with_suffix(".partial.csv"))

    trace_mode = "a" if args.resume_from_trace else "w"
    def write_progress(f, result: dict, pos: int) -> None:
        predictions[int(result["id"])] = str(result["answer"])
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
        checkpoint_predictions = dict(predictions)
        for checkpoint_question in questions:
            checkpoint_predictions.setdefault(checkpoint_question.id, "0")
        write_submission(checkpoint_path, questions, checkpoint_predictions)
        print(f"[{pos}/{len(run_questions)}] id={result['id']} answer={result['answer']}", flush=True)

    def solve_question(question: Question) -> dict:
        return solve_one(
            client,
            question,
            index=index,
            method=args.method,
            samples=max(1, args.samples),
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            prompt_style=args.prompt_style,
            normalize_units=args.normalize_units,
        )

    with trace_path.open(trace_mode, encoding="utf-8") as f:
        if args.workers <= 1:
            for pos, question in enumerate(run_questions, start=1):
                write_progress(f, solve_question(question), pos)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_question = {executor.submit(solve_question, question): question for question in run_questions}
                for pos, future in enumerate(as_completed(future_to_question), start=1):
                    question = future_to_question[future]
                    try:
                        write_progress(f, future.result(), pos)
                    except Exception as exc:
                        print(f"[failed] id={question.id}: {exc}", flush=True)
                        raise

    if args.limit or requested_ids:
        for question in questions:
            predictions.setdefault(question.id, "0")

    write_submission(args.submission_out, questions, predictions)
    print(f"Wrote {args.submission_out}")
    print(f"Wrote {args.trace_out}")


if __name__ == "__main__":
    main()
