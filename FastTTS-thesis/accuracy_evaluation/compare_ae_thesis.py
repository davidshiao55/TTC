#!/usr/bin/env python3
"""
Side-by-side accuracy comparison between FastTTS-AE and FastTTS-thesis results.

Runs evaluate.py on both AE and thesis result files for each (method, N),
then prints a comparison table.  AE files may have more than N completions
(overshoot bug); the evaluation uses ALL completions in each file.

Usage:
    python compare_ae_thesis.py
    python compare_ae_thesis.py --model_combo 1.5B-1.5B --dataset aime
    python compare_ae_thesis.py --agg_strategy prod
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

EVAL_SCRIPT = Path(__file__).parent / "evaluation" / "evaluate.py"

AE_BASE = Path("/TTC/FastTTS-AE/benchmarks/benchmark_results")
THESIS_BASE = Path("/TTC/FastTTS-thesis/benchmarks/benchmark_results")

METHODS = ["baseline", "spec_prefix"]


def find_result_files(result_dir: Path):
    """Find result JSONL files and extract configured N from filename."""
    files = {}
    if not result_dir.exists():
        return files
    for f in sorted(result_dir.glob("*_results.jsonl")):
        parts = f.stem.split("_")
        for p in parts:
            if p.startswith("n") and p[1:].isdigit():
                n = int(p[1:])
                files[n] = f
                break
    return files


def run_eval(result_file: Path, agg_strategy: str = "last"):
    """Run evaluate.py on a single file and return the result dict."""
    output_path = result_file.with_suffix(".compare.json")
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--data_name", "math",
        "--file_path", str(result_file),
        "--agg_strategy", agg_strategy,
        "--output", str(output_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(EVAL_SCRIPT.parent),
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:300]}")
        return None
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        output_path.unlink()
        return data.get("result")
    return None


def print_table(method, ae_results, thesis_results, all_n, agg_strategy):
    """Print comparison table for a single method."""
    method_label = "FastTTS" if method == "spec_prefix" else "Baseline"
    print(f"\n{'=' * 100}")
    print(f"  Method: {method_label} ({method})    |    agg_strategy: {agg_strategy}")
    print(f"{'=' * 100}")
    print(f"{'':>5} | {'--- AE (original) ---':^43} | {'--- Thesis (migrated) ---':^43}")
    print(f"{'N':>5} | {'actual':>6} {'Pass@N':>7} {'MajVot':>7} {'PRMMax':>7} {'PRMVot':>7} | "
          f"{'actual':>6} {'Pass@N':>7} {'MajVot':>7} {'PRMMax':>7} {'PRMVot':>7}")
    print("-" * 100)

    for n in all_n:
        ae = ae_results.get(n)
        th = thesis_results.get(n)

        def fmt(r):
            if r:
                return (f"{r['n']:>6} {r['pass_at_n']:>6.1f}% {r['majority_vote']:>6.1f}% "
                        f"{r['prm_max']:>6.1f}% {r['prm_vote']:>6.1f}%")
            return f"{'—':>6} {'—':>7} {'—':>7} {'—':>7} {'—':>7}"

        print(f"{n:>5} | {fmt(ae)} | {fmt(th)}")

    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Compare AE vs thesis accuracy")
    parser.add_argument("--model_combo", type=str, default="7B-1.5B")
    parser.add_argument("--dataset", type=str, default="aime")
    parser.add_argument("--agg_strategy", type=str, default="last",
                        choices=["last", "min", "prod", "mean"])
    args = parser.parse_args()

    print(f"Model combo: {args.model_combo}")
    print(f"Dataset:     {args.dataset}")
    print(f"Agg strategy: {args.agg_strategy}")

    for method in METHODS:
        ae_dir = AE_BASE / args.model_combo / args.dataset / method
        thesis_dir = THESIS_BASE / args.model_combo / args.dataset / method

        ae_files = find_result_files(ae_dir)
        thesis_files = find_result_files(thesis_dir)
        all_n = sorted(set(list(ae_files.keys()) + list(thesis_files.keys())))

        if not all_n:
            print(f"\n  [{method}] No result files found")
            print(f"    AE:     {ae_dir}")
            print(f"    Thesis: {thesis_dir}")
            continue

        ae_results = {}
        thesis_results = {}

        for n in all_n:
            if n in ae_files:
                print(f"Evaluating AE {method} n={n}...")
                ae_results[n] = run_eval(ae_files[n], args.agg_strategy)
            if n in thesis_files:
                print(f"Evaluating thesis {method} n={n}...")
                thesis_results[n] = run_eval(thesis_files[n], args.agg_strategy)

        print_table(method, ae_results, thesis_results, all_n, args.agg_strategy)

    print()
    print("Notes:")
    print("  - 'actual' = real completions in file (AE may exceed configured N due to overshoot)")
    print("  - Thesis files have exactly N completions (early exit + truncation fix)")
    print(f"  - All selection metrics use agg_strategy='{args.agg_strategy}'")


if __name__ == "__main__":
    main()
