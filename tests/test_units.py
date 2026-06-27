from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.units import normalize_for_unit  # noqa: E402


class UnitNormalizationTest(unittest.TestCase):
    def test_positive_power_unit(self):
        self.assertEqual(normalize_for_unit("4.74e14", "$10^{14} \\mathrm{Hz}$"), "4.74")
        self.assertEqual(normalize_for_unit("4.74", "$10^{14} \\mathrm{Hz}$"), "4.74")
        self.assertEqual(normalize_for_unit("5.93 \\times 10^8", "$10^8 \\mathrm{m/s}$"), "5.93")
        self.assertEqual(normalize_for_unit("5.45", "$10^3 \\mathrm{K}$"), "5.45")
        self.assertEqual(normalize_for_unit("10^{39}", "$10^{39}$"), "1")

    def test_negative_power_unit(self):
        self.assertEqual(normalize_for_unit("1.2816e-18", "$10^{-18} \\mathrm{J}$"), "1.2816")
        self.assertEqual(normalize_for_unit("1.2816", "$10^{-18} \\mathrm{J}$"), "1.2816")
        self.assertEqual(normalize_for_unit("6.626 \\times 10^{-25}", "$10^{-25} \\mathrm{J}$"), "6.626")

    def test_no_scale(self):
        self.assertEqual(normalize_for_unit("7.1", "nm"), "7.1")

    def test_metric_prefix(self):
        self.assertEqual(normalize_for_unit("11.4e-6", "$\\mu \\mathrm{T}$"), "11.4")
        self.assertEqual(normalize_for_unit("5.166e-11", "$\\mathrm{pm}$"), "51.66")
        self.assertEqual(normalize_for_unit("0.0397", "$\\mathrm{mJ}$"), "39.7")
        self.assertEqual(normalize_for_unit("0.13886", "mm"), "0.13886")


if __name__ == "__main__":
    unittest.main()
