import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.calculator import CalculatorError, evaluate_expression


class CalculatorTests(unittest.TestCase):
    def test_evaluate_expression_with_constants(self) -> None:
        result = evaluate_expression("sqrt(4) + log10(100) + pi*0")
        self.assertEqual(result.text, "4")

    def test_evaluate_scientific_expression(self) -> None:
        result = evaluate_expression("h*c/(500e-9)")
        self.assertGreater(result.value, 3.9e-19)
        self.assertLess(result.value, 4.1e-19)

    def test_degree_helpers_and_math_prefix(self) -> None:
        self.assertAlmostEqual(evaluate_expression("sind(30)").value, 0.5)
        self.assertAlmostEqual(evaluate_expression("sin(30)").value, 0.5)
        self.assertAlmostEqual(evaluate_expression("sin(pi/6)").value, 0.5)
        self.assertAlmostEqual(evaluate_expression("math.asin(0.5)").value, evaluate_expression("asin(0.5)").value)

    def test_extra_math_aliases(self) -> None:
        self.assertAlmostEqual(evaluate_expression("ln(e)").value, 1.0)
        self.assertAlmostEqual(evaluate_expression("np.log(np.e)").value, 1.0)
        self.assertAlmostEqual(evaluate_expression("numpy.sqrt(9)").value, 3.0)
        self.assertAlmostEqual(evaluate_expression("tanh(atanh(0.5))").value, 0.5)

    def test_numeric_integral(self) -> None:
        self.assertAlmostEqual(evaluate_expression("integral(x**2, x, 0, 1)").value, 1 / 3, places=8)
        self.assertAlmostEqual(evaluate_expression("integrate(sin(x), x, 0, pi)").value, 2.0, places=8)

    def test_rejects_ambiguous_chained_exponent(self) -> None:
        with self.assertRaises(CalculatorError):
            evaluate_expression("(5.8e10)**3 ** 0.5")


if __name__ == "__main__":
    unittest.main()
