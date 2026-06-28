from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.units import infer_target_unit, normalize_for_unit  # noqa: E402


class UnitNormalizationTest(unittest.TestCase):
    def test_infer_target_unit_from_question_text(self):
        self.assertEqual(infer_target_unit("求速度。（单位：10 ^ 4 m/s）"), "10 ^ 4 m/s")
        self.assertEqual(infer_target_unit("事件视界半径为 X * 10^3 m，那么 X 是多少？"), "10^3 m")
        self.assertEqual(infer_target_unit("计算周期。请以地球日为单位给出答案。"), "地球日")
        self.assertEqual(infer_target_unit("任意题干", "$10^{14} \\mathrm{Hz}$"), "$10^{14} \\mathrm{Hz}$")

    def test_positive_power_unit(self):
        self.assertEqual(normalize_for_unit("4.74e14", "$10^{14} \\mathrm{Hz}$"), "4.74")
        self.assertEqual(normalize_for_unit("4.74", "$10^{14} \\mathrm{Hz}$"), "4.74")
        self.assertEqual(normalize_for_unit("5.93 \\times 10^8", "$10^8 \\mathrm{m/s}$"), "5.93")
        self.assertEqual(normalize_for_unit("5.45", "$10^3 \\mathrm{K}$"), "5.45")
        self.assertEqual(normalize_for_unit("10^{39}", "$10^{39}$"), "1")

    def test_infer_percent_target(self):
        self.assertEqual(infer_target_unit("从第三个偏振片射出的光强是原始光强的百分之多少？"), "%")

    def test_does_not_treat_intermediate_mass_unit_as_final_count_unit(self):
        text = "通过确定其中存在的 U-238 质量（以 kg 为单位），计算一天内发生的自发裂变次数。"
        self.assertIsNone(infer_target_unit(text))

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

    def test_percent_target(self):
        self.assertEqual(normalize_for_unit("0.1549", "%"), "15.49")
        self.assertEqual(normalize_for_unit("15.49", "%"), "15.49")

    def test_height_questions_use_magnitude(self):
        self.assertEqual(normalize_for_unit("-3.6667", "cm", "像的高度是多少 cm？"), "3.6667")
        self.assertEqual(normalize_for_unit("-0.5", None, "求极限的值"), "-0.5")

    def test_loss_questions_use_magnitude(self):
        self.assertEqual(normalize_for_unit("-330", "ps", "计算移动时钟每秒预期损失的时间"), "330")
        self.assertEqual(normalize_for_unit("-8", "L", "体积的减少量"), "8")

    def test_one_mole_joule_answer_scales_particle_energy(self):
        text = "计算一摩尔理想气体的平均平动动能，以 J 为单位"
        self.assertEqual(normalize_for_unit("6.07e-21", "J", text), "3655.439441")
        self.assertEqual(normalize_for_unit("6.07e-21", "kJ/mol", text), "6.07e-21")


if __name__ == "__main__":
    unittest.main()
