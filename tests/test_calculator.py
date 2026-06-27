from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.calculator import (  # noqa: E402
    CalculatorError,
    evaluate_expression,
    extract_calc_expr,
    format_calculated_value,
    looks_calculable,
    normalize_expression,
)


class CalculatorTest(unittest.TestCase):
    def test_extract_calc_expr(self):
        response = "work\nCALC_EXPR: 80/(2*0.27)*log(1/(1-0.95**2))\nFINAL_ANSWER: 344.87"
        self.assertEqual(extract_calc_expr(response), "80/(2*0.27)*log(1/(1-0.95**2))")

    def test_jump_distance_expression(self):
        value = evaluate_expression("80/(2*0.27)*log(1/(1-0.95^2))")
        self.assertAlmostEqual(value, 344.874503849, places=9)

    def test_degree_trig(self):
        value = evaluate_expression("degrees(asin((1.33/2.42)*sin(radians(13))))")
        self.assertAlmostEqual(value, 7.1, places=1)

    def test_model_inverse_sine_notation(self):
        expr = "sin-1(1.33/2.42 * sin(13°))"
        self.assertEqual(normalize_expression(expr), "asin_deg(1.33/2.42 * sin(radians(13)))")
        self.assertAlmostEqual(evaluate_expression(expr), 7.1, places=1)

    def test_looks_calculable(self):
        self.assertTrue(looks_calculable("0.53*2.46"))
        self.assertTrue(looks_calculable("sin-1(1.33/2.42 * sin(13°))"))
        self.assertFalse(looks_calculable("1.3038"))
        self.assertFalse(looks_calculable("\\frac{1}{2}"))

    def test_format_small_scientific_value(self):
        self.assertEqual(format_calculated_value(1.506531e-36), "1.506531e-36")

    def test_rejects_unsafe_expression(self):
        with self.assertRaises((CalculatorError, AttributeError, TypeError, ValueError, SyntaxError)):
            evaluate_expression("__import__('os').system('echo bad')")
        with self.assertRaises(CalculatorError):
            evaluate_expression("math.sqrt(4)")
        self.assertEqual(format_calculated_value(evaluate_expression("sqrt(4)")), "2")


if __name__ == "__main__":
    unittest.main()
