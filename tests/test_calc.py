import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lab4.calc import evaluate_calculation_requests, parse_calculation_requests, safe_eval_expression  # noqa: E402
from run_pair_judge import choose_more_precise_if_close, significant_digit_count  # noqa: E402


class CalculationTests(unittest.TestCase):
    def test_safe_eval_math_expression(self):
        self.assertEqual(safe_eval_expression("sqrt(9) + sin(radians(30))"), "3.5")
        self.assertEqual(safe_eval_expression("6.7 * (10**6 * 0.00055) * 86400"), "318384000")

    def test_safe_eval_rejects_code(self):
        with self.assertRaises(Exception):
            safe_eval_expression("__import__('os').system('echo bad')")
        with self.assertRaises(Exception):
            safe_eval_expression("(1).__class__")

    def test_parse_and_evaluate_requests(self):
        response = '[{"label":"x","source":"A","purpose":"check","expression":"exp(0) + 2**3"}]'
        requests = parse_calculation_requests(response)
        results = evaluate_calculation_requests(requests)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].value, "9")


class PrecisionSelectionTests(unittest.TestCase):
    def test_significant_digits(self):
        self.assertEqual(significant_digit_count("8.7"), 2)
        self.assertEqual(significant_digit_count("8.679"), 4)
        self.assertEqual(significant_digit_count("3.20e8"), 3)
        self.assertEqual(significant_digit_count("3000"), 1)

    def test_choose_more_precise_when_close(self):
        self.assertEqual(choose_more_precise_if_close("8.7", "8.679", rtol=0.01), "8.679")
        self.assertEqual(choose_more_precise_if_close("8.679", "8.7", rtol=0.01), "8.679")
        self.assertEqual(choose_more_precise_if_close("8.7", "9.2", rtol=0.01), "8.7")


if __name__ == "__main__":
    unittest.main()
