from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def maybe_resume_args(trace_path: Path, resume: bool) -> list[str]:
    if resume and trace_path.exists():
        return ["--resume-from-trace"]
    return []


def require_api_key() -> None:
    if os.getenv("LAB4_MOCK_LLM") == "1":
        return
    if os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError("Set DASHSCOPE_API_KEY before running the final pipeline.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a from-scratch final submission pipeline. The script starts from the "
            "questions/textbooks/code, creates all candidate submissions in output-dir, "
            "then judges and combines them into one submission CSV."
        )
    )
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "final_from_scratch"))
    parser.add_argument("--final-submission", default=str(ROOT / "outputs" / "submission.csv"))
    parser.add_argument("--resume", action="store_true", help="Resume traces inside output-dir if they exist.")
    parser.add_argument("--compare", action="store_true", help="Run ../compare_accuracy.py when local gold is available.")
    parser.add_argument(
        "--run-portfolio",
        action="store_true",
        help="Also run the experimental portfolio stage after the main pair judge. This is slower and not the default final output.",
    )
    args = parser.parse_args()

    require_api_key()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    concise_submission = out_dir / "solver_concise_thinking_submission.csv"
    concise_trace = out_dir / "solver_concise_thinking_traces.jsonl"
    baseline_submission = out_dir / "solver_baseline_thinking_submission.csv"
    baseline_trace = out_dir / "solver_baseline_thinking_traces.jsonl"

    solver_common = [
        "--workers",
        str(args.workers),
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--enable-thinking",
    ]

    # Stage 1: full 230-question solver runs. These are the only raw candidates;
    # every later CSV is derived from files created in this output directory.
    run_step(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "--method",
            "baseline",
            "--prompt-style",
            "concise",
            "--submission-out",
            str(concise_submission),
            "--trace-out",
            str(concise_trace),
            "--checkpoint-submission-out",
            str(out_dir / "solver_concise_thinking.partial.csv"),
            *solver_common,
            *maybe_resume_args(concise_trace, args.resume),
        ]
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
            str(baseline_submission),
            "--trace-out",
            str(baseline_trace),
            "--checkpoint-submission-out",
            str(out_dir / "solver_baseline_thinking.partial.csv"),
            *solver_common,
            *maybe_resume_args(baseline_trace, args.resume),
        ]
    )

    pair_specs = [
        ("pair_calc_thinking_a", True, True),
        ("pair_calc_thinking_b", True, True),
        ("pair_nocalc_thinking", False, True),
        ("pair_calc_nothinking", True, False),
    ]
    pair_submissions: list[tuple[str, Path, Path]] = []
    for label, calculation_check, enable_thinking in pair_specs:
        pair_submission = out_dir / f"{label}_submission.csv"
        pair_trace = out_dir / f"{label}_traces.jsonl"
        pair_submissions.append((label, pair_submission, pair_trace))
        thinking_flag = "--enable-thinking" if enable_thinking else "--disable-thinking"
        calc_flag = "--calculation-check" if calculation_check else "--no-calculation-check"
        run_step(
            [
                sys.executable,
                "scripts/run_pair_judge.py",
                "--primary-submission",
                str(concise_submission),
                "--secondary-submission",
                str(baseline_submission),
                "--primary-trace",
                str(concise_trace),
                "--secondary-trace",
                str(baseline_trace),
                "--submission-out",
                str(pair_submission),
                "--trace-out",
                str(pair_trace),
                "--workers",
                str(args.workers),
                "--temperature",
                str(args.temperature),
                "--max-tokens",
                str(args.max_tokens),
                calc_flag,
                thinking_flag,
                *maybe_resume_args(pair_trace, args.resume),
            ]
        )

    final_submission = Path(args.final_submission)
    final_submission.parent.mkdir(parents=True, exist_ok=True)
    # The most reliable from-scratch result observed for this pipeline is the
    # first calculation-checked pair judge. Later portfolio passes are useful
    # experiments, but can overrule correct minority answers.
    shutil.copyfile(pair_submissions[0][1], final_submission)

    if not args.run_portfolio:
        run_step([sys.executable, "scripts/validate_submission.py", "--submission", str(final_submission)])

        compare_script = ROOT.parent / "compare_accuracy.py"
        gold_file = ROOT.parent / "student_zh.json"
        if args.compare and compare_script.exists() and gold_file.exists():
            run_step([sys.executable, str(compare_script), str(final_submission), "--gold", str(gold_file)])

        print(f"\nFinal submission: {final_submission}", flush=True)
        return

    portfolio_args: list[str] = []
    trace_args: list[str] = []
    for label, submission, trace in pair_submissions:
        portfolio_args.extend(["--submission", f"{label}={submission}"])
        trace_args.extend(["--trace", f"{label}={trace}"])

    portfolio_nocalc_submission = out_dir / "portfolio_nocalc_submission.csv"
    portfolio_nocalc_trace = out_dir / "portfolio_nocalc_traces.jsonl"
    portfolio_calc_submission = out_dir / "portfolio_calc_submission.csv"
    portfolio_calc_trace = out_dir / "portfolio_calc_traces.jsonl"

    run_step(
        [
            sys.executable,
            "scripts/run_portfolio_judge.py",
            *portfolio_args,
            *trace_args,
            "--submission-out",
            str(portfolio_nocalc_submission),
            "--trace-out",
            str(portfolio_nocalc_trace),
            "--workers",
            str(args.workers),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--no-calculation-check",
            "--enable-thinking",
            *maybe_resume_args(portfolio_nocalc_trace, args.resume),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/run_portfolio_judge.py",
            *portfolio_args,
            *trace_args,
            "--submission-out",
            str(portfolio_calc_submission),
            "--trace-out",
            str(portfolio_calc_trace),
            "--workers",
            str(args.workers),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--calculation-check",
            "--enable-thinking",
            *maybe_resume_args(portfolio_calc_trace, args.resume),
        ]
    )

    primary_pair_submission = pair_submissions[0][1]
    run_step(
        [
            sys.executable,
            "scripts/combine_portfolio_reviews.py",
            "--primary-submission",
            str(primary_pair_submission),
            "--base-submission",
            str(portfolio_calc_submission),
            "--base-trace",
            str(portfolio_calc_trace),
            "--calc-submission",
            str(portfolio_calc_submission),
            "--calc-trace",
            str(portfolio_calc_trace),
            "--submission-out",
            str(final_submission),
        ]
    )

    run_step([sys.executable, "scripts/validate_submission.py", "--submission", str(final_submission)])

    compare_script = ROOT.parent / "compare_accuracy.py"
    gold_file = ROOT.parent / "student_zh.json"
    if args.compare and compare_script.exists() and gold_file.exists():
        run_step([sys.executable, str(compare_script), str(final_submission), "--gold", str(gold_file)])

    print(f"\nFinal submission: {final_submission}", flush=True)


if __name__ == "__main__":
    main()
