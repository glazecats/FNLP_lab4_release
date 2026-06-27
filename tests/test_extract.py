from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.extract import extract_answer  # noqa: E402


class ExtractAnswerTest(unittest.TestCase):
    def test_final_answer_marker(self):
        self.assertEqual(extract_answer("过程...\nFINAL_ANSWER: 3e8 m/s"), "3e8")

    def test_boxed_latex(self):
        self.assertEqual(extract_answer("Therefore \\boxed{\\frac{22}{\\sqrt{7}}}."), "\\frac{22}{\\sqrt{7}}")

    def test_final_answer_overrides_boxed(self):
        text = "\\boxed{P(\\theta \\geq 90^\\circ) \\approx 1.8 \\times 10^{-5}}\nFINAL_ANSWER: 1.8e-5"
        self.assertEqual(extract_answer(text), "1.8e-5")

    def test_uses_last_final_answer(self):
        text = "FINAL_ANSWER: 1\n检查后修正。\nFINAL_ANSWER: 2"
        self.assertEqual(extract_answer(text), "2")

    def test_normalize_common_math_strings(self):
        self.assertEqual(extract_answer("FINAL_ANSWER: sqrt(156.8)"), "\\sqrt{156.8}")
        self.assertEqual(extract_answer("FINAL_ANSWER: n=2"), "2")
        self.assertEqual(extract_answer("FINAL_ANSWER: -e^57.4"), "-e^{57.4}")
        self.assertEqual(extract_answer("FINAL_ANSWER: e58.02"), "e^{58.02}")
        self.assertEqual(extract_answer("FINAL_ANSWER: (59.5)^(1/3)"), "59.5^{1/3}")

    def test_uses_last_numeric_candidate(self):
        self.assertEqual(extract_answer("FINAL_ANSWER: 54.4 5247.07"), "5247.07")
        self.assertEqual(extract_answer("FINAL_ANSWER: 1.8e8 9.0e-4 1.8"), "1.8")

    def test_normalize_unicode_scientific_notation(self):
        self.assertEqual(extract_answer("FINAL_ANSWER: 4.74 × 10¹⁴"), "4.74e14")
        self.assertEqual(extract_answer("FINAL_ANSWER: 1.2816 × 10⁻¹⁸"), "1.2816e-18")

    def test_chinese_marker(self):
        self.assertEqual(extract_answer("最终答案：6.67 \\times 10^{-11}"), "6.67e-11")

    def test_markdown_and_latex_wrapped_final_answer(self):
        self.assertEqual(extract_answer("**FINAL_ANSWER: 4.357**"), "4.357")
        self.assertEqual(extract_answer("$$\n\\text{FINAL_ANSWER: } 3.2997\n$$"), "3.2997")


if __name__ == "__main__":
    unittest.main()
