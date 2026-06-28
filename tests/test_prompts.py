from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.prompts import normalize_question_text_for_prompt  # noqa: E402


class PromptTextNormalizationTest(unittest.TestCase):
    def test_repairs_lost_power_in_hyphenated_unit(self):
        self.assertIn("10^6-kg", normalize_question_text_for_prompt("一堆 106-kg 的页岩"))

    def test_leaves_regular_hyphenated_quantities_alone(self):
        self.assertEqual(normalize_question_text_for_prompt("一个 125-kg 的人"), "一个 125-kg 的人")


if __name__ == "__main__":
    unittest.main()
