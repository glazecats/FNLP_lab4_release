import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.calculator import evaluate_expression


class CalculatorTests(unittest.TestCase):
    def test_evaluate_expression_with_constants(self) -> None:
        result = evaluate_expression("sqrt(4) + log10(100) + pi*0")
        self.assertEqual(result.text, "4")

    def test_evaluate_scientific_expression(self) -> None:
        result = evaluate_expression("h*c/(500e-9)")
        self.assertGreater(result.value, 3.9e-19)
        self.assertLess(result.value, 4.1e-19)


if __name__ == "__main__":
    unittest.main()
