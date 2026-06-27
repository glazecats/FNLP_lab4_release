import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import Question
from lab4.pipeline import Solver, _postprocess_answer


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
        height_question = Question(id=1, field="physics", question="像的高度是多少 cm？")
        energy_question = Question(id=2, field="physics", question="求势能变化量。")

        self.assertEqual(_postprocess_answer(height_question, "-3.67"), "3.67")
        self.assertEqual(_postprocess_answer(energy_question, "-3.67"), "-3.67")


if __name__ == "__main__":
    unittest.main()
