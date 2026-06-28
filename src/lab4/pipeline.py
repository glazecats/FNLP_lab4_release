from __future__ import annotations

import csv
import json
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .calculator import CalculatorError, evaluate_expression, format_number
from .data import Question, load_questions, select_questions
from .extract import equivalent_answer_text, extract_final_answer, extract_tool_expression, looks_invalid_answer
from .llm import LLMClient, Message
from .prompts import (
    ARBITER_SYSTEM,
    DIRECT_SYSTEM,
    RAG_CURATOR_SYSTEM,
    RAG_SOLVER_SYSTEM,
    VERIFIER_SYSTEM,
    arbiter_user_prompt,
    curator_user_prompt,
    direct_user_prompt,
    rag_solver_user_prompt,
    verifier_user_prompt,
)
from .retrieval import RetrievedChunk, TextbookIndex


class Solver:
    def __init__(
        self,
        *,
        client: LLMClient,
        index: TextbookIndex | None,
        top_k: int,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        max_tool_rounds: int = 5,
        max_verify_loops: int = 1,
    ) -> None:
        self.client = client
        self.index = index
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.max_tool_rounds = max_tool_rounds
        self.max_verify_loops = max_verify_loops

    def solve(self, question: Question, method: str) -> dict[str, Any]:
        if method == "baseline":
            direct = self._direct(question)
            return self._trace(question, method, direct.get("answer"), direct=direct)
        if method == "rag":
            rag = self._rag(question)
            return self._trace(question, method, rag.get("answer"), rag=rag, retrieval=rag.get("retrieval", []))
        if method == "verify":
            direct = self._direct(question)
            verified = self._verify(question, direct.get("answer") or "", self._history(direct))
            answer = verified.get("answer") or direct.get("answer")
            return self._trace(question, method, answer, direct=direct, verifier=verified)
        if method in {"dual", "rag-verify", "full"}:
            return self._full(question, method)
        raise ValueError(f"unknown method: {method}")

    def _full(self, question: Question, method: str) -> dict[str, Any]:
        feedback: str | None = None
        attempts = []
        for loop_index in range(self.max_verify_loops + 1):
            direct = self._direct(question, feedback=feedback)
            rag = self._rag(question, feedback=feedback)
            final = self._arbiter(question, direct, rag, feedback=feedback)

            history = "\n\n".join(
                [
                    "DIRECT_SOLVER:\n" + self._history(direct),
                    "RAG_SOLVER:\n" + self._history(rag),
                    "FINAL_CANDIDATE:\n" + self._history(final),
                ]
            )
            verifier = self._verify(question, final.get("answer") or "", history)
            attempts.append(
                {
                    "loop": loop_index,
                    "direct": direct,
                    "rag": rag,
                    "final": final,
                    "verifier": verifier,
                }
            )
            decision = verifier.get("decision")
            if decision in {"PASS", "FIX"}:
                answer = verifier.get("answer") or final.get("answer")
                return self._trace(
                    question,
                    method,
                    answer,
                    attempts=attempts,
                    direct=direct,
                    rag=rag,
                    arbiter=final,
                    verifier=verifier,
                    retrieval=rag.get("retrieval", []),
                )
            feedback = verifier.get("reason") or "Verifier requested a full redo."
        last = attempts[-1]
        answer = _fallback_after_failed_verification(last)
        return self._trace(
            question,
            method,
            answer,
            attempts=attempts,
            direct=last["direct"],
            rag=last["rag"],
            arbiter=last["final"],
            verifier=last["verifier"],
            retrieval=last["rag"].get("retrieval", []),
        )

    def _direct(self, question: Question, feedback: str | None = None) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": direct_user_prompt(question, feedback)},
        ]
        return self._run_role("direct", messages)

    def _rag(self, question: Question, feedback: str | None = None) -> dict[str, Any]:
        chunks = self.index.search(question, self.top_k) if self.index else []
        curator_messages = [
            {"role": "system", "content": RAG_CURATOR_SYSTEM},
            {"role": "user", "content": curator_user_prompt(question, chunks)},
        ]
        notes = self.client.chat(
            curator_messages,
            temperature=0.0,
            max_tokens=min(self.max_tokens, 2048),
            enable_thinking=self.enable_thinking,
        )
        solver_messages = [
            {"role": "system", "content": RAG_SOLVER_SYSTEM},
            {"role": "user", "content": rag_solver_user_prompt(question, notes, feedback)},
        ]
        result = self._run_role("rag", solver_messages)
        result["curator_notes"] = notes
        result["retrieval"] = [asdict(chunk) for chunk in chunks]
        return result

    def _arbiter(
        self,
        question: Question,
        direct: dict[str, Any],
        rag: dict[str, Any],
        feedback: str | None = None,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": ARBITER_SYSTEM},
            {"role": "user", "content": arbiter_user_prompt(question, direct, rag, feedback)},
        ]
        return self._run_role("arbiter", messages)

    def _verify(self, question: Question, candidate_answer: str, history: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": verifier_user_prompt(question, candidate_answer, history)},
        ]
        result = self._run_role("verifier", messages)
        response = result.get("transcript", "")
        decision = "PASS"
        for line in response.splitlines():
            marker = line.strip().upper()
            if marker in {"PASS", "FIX", "LOOP"}:
                decision = marker
                break
        result["decision"] = decision
        if decision == "LOOP":
            result["reason"] = _extract_reason(response)
        if decision in {"PASS", "FIX"} and not result.get("answer"):
            result["answer"] = candidate_answer
        if decision in {"PASS", "FIX"}:
            invalid_reason = looks_invalid_answer(result.get("answer") or "")
            if invalid_reason:
                if candidate_answer and not looks_invalid_answer(candidate_answer):
                    result["answer"] = candidate_answer
                    result["reason"] = f"Verifier returned an invalid replacement answer: {invalid_reason}"
                else:
                    result["answer"] = None
                    result["decision"] = "LOOP"
                    result["reason"] = f"Verifier returned an invalid final answer: {invalid_reason}"
        return result

    def _run_role(self, role: str, messages: list[Message]) -> dict[str, Any]:
        transcript: list[dict[str, str]] = []
        answer = None
        failed_tool_expressions: set[str] = set()
        for _ in range(self.max_tool_rounds + 1):
            response = self.client.chat(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                enable_thinking=self.enable_thinking,
            )
            transcript.append({"assistant": response})
            messages.append({"role": "assistant", "content": response})

            expression = extract_tool_expression(response)
            if expression:
                try:
                    calculation = evaluate_expression(expression)
                    tool_text = f"TOOL_RESULT: {calculation.expression} = {calculation.text}"
                except Exception as exc:
                    if expression in failed_tool_expressions:
                        transcript.append(
                            {
                                "tool": (
                                    f"TOOL_ERROR: repeated invalid expression {expression!r}; "
                                    "stopping this role's tool loop."
                                )
                            }
                        )
                        break
                    failed_tool_expressions.add(expression)
                    tool_text = f"TOOL_ERROR: {expression} failed with {exc}. Rewrite a valid Python expression."
                transcript.append({"tool": tool_text})
                messages.append({"role": "user", "content": tool_text + "\nContinue from the tool result."})
                continue

            answer = extract_final_answer(response)
            if answer:
                break

        return {
            "role": role,
            "answer": answer,
            "invalid_reason": looks_invalid_answer(answer or ""),
            "transcript": _format_transcript(transcript),
            "messages": messages,
        }

    def _trace(self, question: Question, method: str, answer: str | None, **extra: Any) -> dict[str, Any]:
        trace = {
            "id": question.id,
            "field": question.field,
            "subfield": question.subfield,
            "theorem": question.theorem,
            "unit": question.unit,
            "question": question.question,
            "method": method,
            "answer": answer or "",
            "invalid_reason": looks_invalid_answer(answer or ""),
            **extra,
        }
        trace["answer"] = _postprocess_trace_answer(question, trace) or ""
        trace["invalid_reason"] = looks_invalid_answer(trace["answer"])
        return trace

    @staticmethod
    def _history(result: dict[str, Any]) -> str:
        return f"answer: {result.get('answer')}\n{result.get('transcript', '')}"


def solve_questions(
    *,
    method: str,
    data_path: str | Path,
    submission_out: str | Path,
    trace_out: str | Path,
    top_k: int = 4,
    workers: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    enable_thinking: bool = False,
    ids: str | None = None,
    limit: int | None = None,
    resume_from_trace: bool = False,
    max_verify_loops: int = 1,
) -> None:
    all_questions = load_questions(data_path)
    selected = select_questions(all_questions, ids=ids, limit=limit)
    done = _load_existing_traces(trace_out) if resume_from_trace else {}
    to_run = [q for q in selected if q.id not in done]

    needs_rag = method in {"rag", "dual", "rag-verify", "full"}
    index = TextbookIndex.load_or_build() if needs_rag else None
    client = LLMClient()
    solver = Solver(
        client=client,
        index=index,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        max_verify_loops=max_verify_loops,
    )

    Path(trace_out).parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume_from_trace and Path(trace_out).exists() else "w"
    with Path(trace_out).open(mode, encoding="utf-8") as trace_file:
        if workers <= 1:
            for question in to_run:
                trace = _safe_solve(solver, question, method)
                trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_file.flush()
                done[question.id] = trace
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_safe_solve, solver, question, method): question for question in to_run}
                for future in as_completed(futures):
                    trace = future.result()
                    trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    trace_file.flush()
                    done[trace["id"]] = trace

    traces = [done.get(q.id) for q in selected]
    write_submission(selected, traces, submission_out)


def write_submission(questions: list[Question], traces: list[dict[str, Any] | None], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "answer"])
        for question, trace in zip(questions, traces, strict=True):
            answer = _postprocess_trace_answer(question, trace) if trace else ""
            writer.writerow([question.id, answer or ""])


def _safe_solve(solver: Solver, question: Question, method: str, retries: int = 2) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    for attempt in range(retries + 1):
        try:
            trace = solver.solve(question, method)
            if errors:
                trace["retry_errors"] = errors
            return trace
        except Exception as exc:  # trace failures so resume can target them later.
            errors.append(
                {
                    "attempt": str(attempt),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    return {
        "id": question.id,
        "field": question.field,
        "question": question.question,
        "method": method,
        "answer": "",
        "error": errors[-1]["error"] if errors else "unknown error",
        "retry_errors": errors,
        "traceback": errors[-1]["traceback"] if errors else "",
    }


def _fallback_after_failed_verification(attempt: dict[str, Any]) -> str | None:
    for role in ("direct", "rag", "final", "verifier"):
        answer = (attempt.get(role) or {}).get("answer")
        if answer and not looks_invalid_answer(answer):
            return answer
    return None


def _load_existing_traces(path: str | Path) -> dict[int, dict[str, Any]]:
    trace_path = Path(path)
    if not trace_path.exists():
        return {}
    done: dict[int, dict[str, Any]] = {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("answer"):
            done[int(item["id"])] = item
    return done


def _format_transcript(items: list[dict[str, str]]) -> str:
    lines = []
    for item in items:
        if "assistant" in item:
            lines.append("ASSISTANT:\n" + item["assistant"])
        if "tool" in item:
            lines.append(item["tool"])
    return "\n\n".join(lines)


def _extract_reason(response: str) -> str:
    for line in response.splitlines():
        if line.strip().upper().startswith("REASON"):
            return line.split(":", 1)[-1].strip()
    return response.strip()[:500]


POSITIVE_TARGET_TERMS = {
    "高度",
    "长度",
    "距离",
    "多远",
    "大小",
    "速率",
    "比例",
    "概率",
    "分数",
    "次数",
    "数量",
    "半径",
    "体积",
    "面积",
    "波长",
    "频率",
    "height",
    "length",
    "distance",
    "magnitude",
    "speed",
    "ratio",
    "probability",
    "fraction",
    "count",
    "radius",
    "volume",
    "area",
    "wavelength",
    "frequency",
}

SIGNED_TARGET_TERMS = {
    "方向",
    "位移",
    "矢量",
    "变化量",
    "势能",
    "自由能",
    "电势",
    "电压",
    "velocity",
    "displacement",
    "vector",
    "change",
    "potential",
    "free energy",
    "voltage",
}

def _postprocess_trace_answer(question: Question, trace: dict[str, Any]) -> str | None:
    answer = _postprocess_answer(question, trace.get("answer"))
    answer = _postprocess_calculable_expression(answer)
    answer = _postprocess_scaled_coefficient(question, answer)
    if looks_invalid_answer(answer or ""):
        answer = _numeric_answer_fallback(trace) or "0"
        answer = _postprocess_answer(question, answer)
        answer = _postprocess_calculable_expression(answer)
        answer = _postprocess_scaled_coefficient(question, answer)
    return answer


def _postprocess_answer(question: Question, answer: str | None) -> str | None:
    if not answer:
        return answer
    try:
        value = float(answer)
    except ValueError:
        return answer
    if value >= 0:
        return answer
    target_text = " ".join(
        part
        for part in [
            question.question,
            question.unit or "",
            question.subfield or "",
            question.theorem or "",
        ]
        if part
    ).lower()
    if any(term in target_text for term in SIGNED_TARGET_TERMS):
        return answer
    if any(term in target_text for term in POSITIVE_TARGET_TERMS):
        return answer[1:] if answer.startswith("-") else str(abs(value))
    return answer


def _postprocess_calculable_expression(answer: str | None) -> str | None:
    if not answer or _to_float(answer) is not None:
        return answer
    expression = _answer_to_calculator_expression(answer)
    if not expression:
        return answer
    try:
        return evaluate_expression(expression).text
    except CalculatorError:
        return answer


def _answer_to_calculator_expression(answer: str) -> str | None:
    expression = answer.strip()
    expression = expression.strip("$` ")
    expression = expression.replace("×", "*")
    expression = expression.replace(r"\times", "*").replace(r"\cdot", "*")
    expression = re.sub(r"10\s*\^\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", r"10**(\1)", expression)
    expression = expression.replace("{", "(").replace("}", ")")
    if not re.search(r"[+\-*/^()]|\b(?:sqrt|log|ln|log10|exp|sin|cos|tan|e|pi|k_B|kB|kB_eV|e_charge|c|h|hbar|N_A|R|F|G|g|epsilon0|amu|u|m_e|m_p|m_n)\b", expression):
        return None
    return expression


SCALED_COEFFICIENT_RE = re.compile(
    r"\b[xX]\b\s*(?:\*|×|\\times)\s*10\s*(?:\^|\*\*)\s*\{?\s*([-+]?\d+)\s*\}?"
)


def _postprocess_scaled_coefficient(question: Question, answer: str | None) -> str | None:
    value = _to_float(answer)
    if value is None or value == 0:
        return answer
    target_text = " ".join(
        part
        for part in [
            question.question,
            question.unit or "",
        ]
        if part
    )
    for match in SCALED_COEFFICIENT_RE.finditer(target_text):
        exponent = int(match.group(1))
        if exponent == 0:
            continue
        scale = 10.0**exponent
        coefficient = value / scale
        if 1 <= abs(coefficient) < 100:
            if exponent > 0 and abs(value) >= abs(scale):
                return format_number(coefficient)
            if exponent < 0 and abs(value) <= abs(scale):
                return format_number(coefficient)
    return answer


def _to_float(answer: str | None) -> float | None:
    if not answer:
        return None
    try:
        return float(answer)
    except ValueError:
        return None


def _numeric_answer_fallback(trace: dict[str, Any]) -> str | None:
    fallback = None
    for answer in _iter_answer_values(trace):
        if _to_float(answer) is not None:
            fallback = str(answer)
    return fallback


def _iter_answer_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "answer" and isinstance(item, str):
                yield item
            else:
                yield from _iter_answer_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_answer_values(item)
