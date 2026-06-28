import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.rebuild_submission_from_traces import load_traces, rebuild_submission


class RebuildSubmissionTests(unittest.TestCase):
    def test_load_traces_skips_corrupt_jsonl_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            traces = Path(tmp) / "traces.jsonl"
            traces.write_text(
                '{"id": 1, "answer": "old"}\n'
                '{"id": bad json\n'
                '{"id": 1, "answer": "new"}\n',
                encoding="utf-8",
            )

            loaded = load_traces(traces)

            self.assertEqual(loaded[1]["answer"], "new")

    def test_rebuild_submission_uses_current_trace_postprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            data = temp / "data.json"
            traces = temp / "traces.jsonl"
            submission = temp / "submission.csv"
            data.write_text(
                json.dumps(
                    [
                        {
                            "id": 1,
                            "field": "physics",
                            "question": "Find the fraction.",
                            "answer": None,
                            "subfield": None,
                            "theorem": None,
                            "unit": None,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            trace = {
                "id": 1,
                "field": "physics",
                "question": "Find the fraction.",
                "answer": r"\boxed{\frac{1}{4}}",
                "attempts": [
                    {
                        "final": {"answer": None},
                        "verifier": {"decision": "PASS", "answer": r"\boxed{\frac{1}{4}}"},
                    }
                ],
            }
            traces.write_text(json.dumps(trace, ensure_ascii=False) + "\n", encoding="utf-8")

            rebuild_submission(traces=traces, submission_out=submission, data=data)

            with submission.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows, [{"id": "1", "answer": "0.25"}])


if __name__ == "__main__":
    unittest.main()
