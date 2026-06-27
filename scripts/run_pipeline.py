from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab4.pipeline import solve_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FNLP Lab 4 solving pipeline.")
    parser.add_argument("--method", default="rag-verify", choices=["baseline", "rag", "verify", "dual", "rag-verify", "full"])
    parser.add_argument("--data", default="student_zh.json")
    parser.add_argument("--submission-out", default="outputs/submission.csv")
    parser.add_argument("--trace-out", default="outputs/traces.jsonl")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids")
    parser.add_argument("--resume-from-trace", action="store_true")
    parser.add_argument("--max-verify-loops", type=int, default=1)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true")
    thinking.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_thinking = bool(args.enable_thinking and not args.disable_thinking)
    solve_questions(
        method=args.method,
        data_path=args.data,
        submission_out=args.submission_out,
        trace_out=args.trace_out,
        top_k=args.top_k,
        workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        enable_thinking=enable_thinking,
        ids=args.ids,
        limit=args.limit,
        resume_from_trace=args.resume_from_trace,
        max_verify_loops=args.max_verify_loops,
    )


if __name__ == "__main__":
    main()
