from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import Question  # noqa: E402
from lab4.query import expand_query  # noqa: E402


def make_question(text: str) -> Question:
    return Question(
        id=0,
        field="physics",
        question=text,
        answer=None,
        unit=None,
        subfield=None,
        theorem=None,
    )


class QueryExpansionTest(unittest.TestCase):
    def test_solar_geometry_terms(self):
        query = expand_query(make_question("夏至日正午，纬度 50°N 的太阳能收集器接收太阳辐射通量。"))
        self.assertIn("solar radiation flux", query)
        self.assertIn("latitude", query)

    def test_relativity_frame_terms(self):
        query = expand_query(make_question("两个事件相距 8.3 光分，一个惯性系中的观察者测量时间差。"))
        self.assertIn("Lorentz transformation", query)
        self.assertIn("spacetime interval", query)

    def test_mirror_and_gas_terms(self):
        mirror_query = expand_query(make_question("凸面镜前的物体，求像的放大倍数。"))
        self.assertIn("convex mirror", mirror_query)
        self.assertIn("magnification", mirror_query)
        gas_query = expand_query(make_question("计算氢气的平均分子速率。"))
        self.assertIn("mean molecular speed", gas_query)
        self.assertIn("molar mass", gas_query)


if __name__ == "__main__":
    unittest.main()
