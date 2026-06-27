import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.validate_submission import validate_submission


class ValidateSubmissionTests(unittest.TestCase):
    def test_validate_submission_accepts_sample(self) -> None:
        temp = ROOT / "outputs" / "_test_validate"
        temp.mkdir(parents=True, exist_ok=True)
        data = temp / "data.json"
        data.write_text(
            '[{"id": 1, "field": "physics", "question": "q", "answer": null, "subfield": null, "theorem": null, "unit": null}]',
            encoding="utf-8",
        )
        submission = temp / "submission.csv"
        with submission.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "answer"])
            writer.writerow([1, "3.14"])
        self.assertEqual(validate_submission(submission, data), [])


if __name__ == "__main__":
    unittest.main()
