from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.data import load_questions
from lab4.pipeline import write_submission


def load_traces(path: str | Path) -> dict[int, dict[str, Any]]:
    traces: dict[int, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            trace = json.loads(line)
        except json.JSONDecodeError:
            continue
        traces[int(trace["id"])] = trace
    return traces


def rebuild_submission(*, traces: str | Path, submission_out: str | Path, data: str | Path = "student_zh.json") -> None:
    questions = load_questions(data)
    trace_map = load_traces(traces)
    ordered_traces = [trace_map.get(question.id) for question in questions]
    write_submission(questions, ordered_traces, submission_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a submission CSV from saved JSONL traces.")
    parser.add_argument("--traces", required=True)
    parser.add_argument("--submission-out", required=True)
    parser.add_argument("--data", default="student_zh.json")
    args = parser.parse_args()
    rebuild_submission(traces=args.traces, submission_out=args.submission_out, data=args.data)


if __name__ == "__main__":
    main()
