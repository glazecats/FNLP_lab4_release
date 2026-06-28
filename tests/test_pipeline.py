import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import Question
from lab4.pipeline import (
    Solver,
    _fallback_after_failed_verification,
    _postprocess_answer,
    _postprocess_trace_answer,
    _safe_solve,
)


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, temperature=0.0, max_tokens=4096, enable_thinking=False):
        self.calls += 1
        if self.calls == 1:
            return "TOOL_CALC: sqrt(4)\nTOOL_RESULT: sqrt(4) = 999\nFINAL_ANSWER: 999"
        assert any("TOOL_RESULT: sqrt(4) = 2" in message["content"] for message in messages)
        return "FINAL_ANSWER: 2"


class FlakySolver:
    def __init__(self) -> None:
        self.calls = 0

    def solve(self, question, method):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary timeout")
        return {"id": question.id, "method": method, "answer": "42"}


class RepeatingBadToolClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, temperature=0.0, max_tokens=4096, enable_thinking=False):
        self.calls += 1
        return "TOOL_CALC: import scipy; scipy.optimize.fsolve(x, 1)"


class FixedResponseClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def chat(self, messages, *, temperature=0.0, max_tokens=4096, enable_thinking=False):
        return self.response


class PipelineTests(unittest.TestCase):
    def test_tool_calc_takes_precedence_over_model_tool_result(self) -> None:
        solver = Solver(
            client=FakeClient(),
            index=None,
            top_k=0,
            temperature=0,
            max_tokens=256,
            enable_thinking=False,
        )
        result = solver._run_role("test", [{"role": "user", "content": "x"}])
        self.assertEqual(result["answer"], "2")

    def test_repeated_bad_tool_expression_stops_role_loop(self) -> None:
        client = RepeatingBadToolClient()
        solver = Solver(
            client=client,
            index=None,
            top_k=0,
            temperature=0,
            max_tokens=256,
            enable_thinking=False,
            max_tool_rounds=5,
        )

        result = solver._run_role("test", [{"role": "user", "content": "x"}])

        self.assertIsNone(result["answer"])
        self.assertEqual(client.calls, 2)
        self.assertIn("repeated invalid expression", result["transcript"])

    def test_safe_solve_retries_transient_failures(self) -> None:
        question = Question(id=9, field="physics", question="Find the value.")
        solver = FlakySolver()

        result = _safe_solve(solver, question, "rag-verify", retries=1)

        self.assertEqual(result["answer"], "42")
        self.assertEqual(solver.calls, 2)
        self.assertIn("retry_errors", result)

    def test_failed_verification_fallback_avoids_rejected_final_first(self) -> None:
        attempt = {
            "direct": {"answer": "1.3038"},
            "rag": {"answer": "1.1562"},
            "final": {"answer": "1.1562"},
            "verifier": {"decision": "LOOP"},
        }

        self.assertEqual(_fallback_after_failed_verification(attempt), "1.3038")

    def test_postprocess_abs_for_magnitude_targets_only(self) -> None:
        height_question = Question(id=1, field="physics", question="What is the image height?")
        energy_question = Question(id=2, field="physics", question="Find the potential energy change.")

        self.assertEqual(_postprocess_answer(height_question, "-3.67"), "3.67")
        self.assertEqual(_postprocess_answer(energy_question, "-3.67"), "-3.67")

    def test_postprocess_uses_numeric_fallback_for_invalid_answer(self) -> None:
        question = Question(id=3, field="physics", question="Find the number.")
        trace = {
            "answer": "unknown",
            "direct": {"answer": "2"},
            "rag": {"answer": "2"},
        }

        self.assertEqual(_postprocess_trace_answer(question, trace), "2")

    def test_postprocess_scales_explicit_x_times_power_target(self) -> None:
        question = Question(id=4, field="physics", question="The radius is X * 10^3 m. What is X?")
        trace = {"answer": "8847.897"}

        self.assertEqual(_postprocess_trace_answer(question, trace), "8.847897")

    def test_postprocess_does_not_rescale_existing_coefficient(self) -> None:
        question = Question(id=5, field="physics", question="The radius is X * 10^3 m. What is X?")
        trace = {"answer": "8.847897"}

        self.assertEqual(_postprocess_trace_answer(question, trace), "8.847897")

    def test_postprocess_uses_zero_when_no_numeric_fallback_exists(self) -> None:
        question = Question(id=6, field="physics", question="Find the value.")
        trace = {
            "answer": "unknown",
            "direct": {"answer": "cannot determine"},
            "rag": {"answer": ""},
        }

        self.assertEqual(_postprocess_trace_answer(question, trace), "0")

    def test_postprocess_evaluates_calculable_final_expression(self) -> None:
        question = Question(id=7, field="physics", question="Find the value.")
        trace = {"answer": "e^(ln(4))"}

        self.assertEqual(_postprocess_trace_answer(question, trace), "4")

    def test_postprocess_keeps_expression_with_unknown_variable(self) -> None:
        question = Question(id=8, field="physics", question="Find the value.")
        trace = {"answer": r"\frac{3}{2} k_B T"}

        self.assertEqual(_postprocess_trace_answer(question, trace), r"\frac{3}{2} k_B T")

    def test_verifier_invalid_pass_forces_loop(self) -> None:
        solver = Solver(
            client=FixedResponseClient("PASS\nFINAL_ANSWER: 无法确定"),
            index=None,
            top_k=0,
            temperature=0,
            max_tokens=256,
            enable_thinking=False,
        )
        question = Question(id=9, field="physics", question="Find the value.")

        result = solver._verify(question, "无法确定", "history")

        self.assertEqual(result["decision"], "LOOP")
        self.assertIsNone(result["answer"])
        self.assertIn("invalid final answer", result["reason"])

    def test_verifier_invalid_replacement_keeps_valid_candidate(self) -> None:
        solver = Solver(
            client=FixedResponseClient("FIX\nFINAL_ANSWER: 无法确定"),
            index=None,
            top_k=0,
            temperature=0,
            max_tokens=256,
            enable_thinking=False,
        )
        question = Question(id=10, field="physics", question="Find the value.")

        result = solver._verify(question, "42", "history")

        self.assertEqual(result["decision"], "FIX")
        self.assertEqual(result["answer"], "42")
        self.assertIn("invalid replacement answer", result["reason"])


if __name__ == "__main__":
    unittest.main()
