from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd: list[str], *, cwd: Path) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def maybe_resume_args(path: Path, resume: bool) -> list[str]:
    if resume and path.exists():
        return ["--resume-from-trace"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final three-stage submission pipeline: "
            "concise solver, baseline-thinking solver, then pairwise judge."
        )
    )
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "final_pipeline"))
    parser.add_argument("--final-submission", default=str(ROOT / "outputs" / "final_pipeline_submission.csv"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume stages from existing trace files in output-dir when present.",
    )
    parser.add_argument(
        "--skip-solvers",
        action="store_true",
        help="Reuse existing primary/secondary submissions and only rerun the pairwise judge.",
    )
    parser.add_argument(
        "--calculation-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the pair-judge calculation checker that extracts expressions and evaluates them locally.",
    )
    args = parser.parse_args()

    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LAB4_MOCK_LLM") == "1"):
        raise RuntimeError("Set DASHSCOPE_API_KEY before running this reproduction script.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_submission = out_dir / "primary_concise_thinking_submission.csv"
    primary_trace = out_dir / "primary_concise_thinking_traces.jsonl"
    secondary_submission = out_dir / "secondary_baseline_thinking_submission.csv"
    secondary_trace = out_dir / "secondary_baseline_thinking_traces.jsonl"
    judge_trace = out_dir / "pair_judge_traces.jsonl"
    final_submission = Path(args.final_submission)

    common = [
        "--workers",
        str(args.workers),
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--enable-thinking",
    ]

    if not args.skip_solvers:
        run_step(
            [
                sys.executable,
                "scripts/run_pipeline.py",
                "--method",
                "baseline",
                "--prompt-style",
                "concise",
                "--submission-out",
                str(primary_submission),
                "--trace-out",
                str(primary_trace),
                *common,
                *maybe_resume_args(primary_trace, args.resume),
            ],
            cwd=ROOT,
        )
        run_step(
            [
                sys.executable,
                "scripts/run_pipeline.py",
                "--method",
                "baseline",
                "--prompt-style",
                "baseline",
                "--submission-out",
                str(secondary_submission),
                "--trace-out",
                str(secondary_trace),
                *common,
                *maybe_resume_args(secondary_trace, args.resume),
            ],
            cwd=ROOT,
        )

    run_step(
        [
            sys.executable,
            "scripts/run_pair_judge.py",
            "--primary-submission",
            str(primary_submission),
            "--secondary-submission",
            str(secondary_submission),
            "--primary-trace",
            str(primary_trace),
            "--secondary-trace",
            str(secondary_trace),
            "--submission-out",
            str(final_submission),
            "--trace-out",
            str(judge_trace),
            "--calculation-check" if args.calculation_check else "--no-calculation-check",
            *common,
            *maybe_resume_args(judge_trace, args.resume),
        ],
        cwd=ROOT,
    )
    print(f"\nFinal submission: {final_submission}", flush=True)


if __name__ == "__main__":
    main()
