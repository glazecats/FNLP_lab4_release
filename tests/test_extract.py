import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.extract import equivalent_answer_text, extract_final_answer, extract_tool_expression, looks_invalid_answer


class ExtractTests(unittest.TestCase):
    def test_extract_final_answer_last_marker(self) -> None:
        text = "work\nFINAL_ANSWER: 3 m\nrevision\nFINAL_ANSWER: 4.2e-3"
        self.assertEqual(extract_final_answer(text), "4.2e-3")

    def test_extract_tool_expression(self) -> None:
        self.assertEqual(extract_tool_expression("Need calculation\nTOOL_CALC: sqrt(4) * 3\n"), "sqrt(4) * 3")

    def test_equivalent_answer_text_compacts_latex_spaces(self) -> None:
        self.assertTrue(equivalent_answer_text(r"\frac { 1 } {2}", r"\frac{1}{2}"))

    def test_invalid_answer_flags_units(self) -> None:
        self.assertIsNotNone(looks_invalid_answer("3e8 m/s"))

    def test_invalid_answer_flags_chinese_refusal(self) -> None:
        self.assertIsNotNone(looks_invalid_answer("\u65e0\u6cd5\u8ba1\u7b97"))

    def test_invalid_answer_flags_unresolved_variable(self) -> None:
        self.assertIsNotNone(looks_invalid_answer(r"\frac{3}{2} k_B T"))

    def test_invalid_answer_allows_scientific_notation_and_latex_math(self) -> None:
        self.assertIsNone(looks_invalid_answer("6.32e25"))
        self.assertIsNone(looks_invalid_answer(r"\frac{22}{\sqrt{7}}"))

    def test_normalizes_numeric_fraction(self) -> None:
        self.assertEqual(extract_final_answer(r"FINAL_ANSWER: \frac{1}{4}"), "0.25")

    def test_normalizes_boxed_fraction(self) -> None:
        self.assertEqual(extract_final_answer(r"FINAL_ANSWER: \boxed{\frac{1}{4}}"), "0.25")

    def test_unwraps_boxed_numeric_answer(self) -> None:
        self.assertEqual(extract_final_answer(r"FINAL_ANSWER: \boxed{3.14}"), "3.14")

    def test_normalizes_signed_latex_fraction(self) -> None:
        self.assertEqual(extract_final_answer(r"FINAL_ANSWER: -\frac{11}{3}"), "-3.66666666667")

    def test_normalizes_percent_to_fraction(self) -> None:
        self.assertEqual(extract_final_answer(r"FINAL_ANSWER: 44.6%"), "0.446")


if __name__ == "__main__":
    unittest.main()
