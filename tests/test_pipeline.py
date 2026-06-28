import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import Question
from lab4.pipeline import Solver, _postprocess_answer, _postprocess_trace_answer


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, temperature=0.0, max_tokens=4096, enable_thinking=False):
        self.calls += 1
        if self.calls == 1:
            return "TOOL_CALC: sqrt(4)\nTOOL_RESULT: sqrt(4) = 999\nFINAL_ANSWER: 999"
        assert any("TOOL_RESULT: sqrt(4) = 2" in message["content"] for message in messages)
        return "FINAL_ANSWER: 2"


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

    def test_postprocess_uses_zero_when_no_numeric_fallback_exists(self) -> None:
        question = Question(id=4, field="physics", question="Find the value.")
        trace = {
            "answer": "unknown",
            "direct": {"answer": "cannot determine"},
            "rag": {"answer": ""},
        }

        self.assertEqual(_postprocess_trace_answer(question, trace), "0")


if __name__ == "__main__":
    unittest.main()
