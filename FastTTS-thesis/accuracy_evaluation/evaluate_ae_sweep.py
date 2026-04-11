#!/usr/bin/env python3
"""
Run the AE original evaluation method across all N-value result files.

For each N-value file (separate beam search run):
  1. np.prod(scores) per completion
  2. Select top_n completions by product score
  3. Extract answers, majority vote among selected
  4. Sweep top_n = [1, 2, 4, ..., actual_completions] and report best

This shows what the AE pipeline would report for each N if it had used
per-N files instead of only the n=512 file.

Usage:
    python evaluate_ae_sweep.py
    python evaluate_ae_sweep.py --result_dir /path/to/results --methods baseline,spec_prefix
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "evaluation"))
from grader import math_equal_process
from parser import extract_answer

AE_BASE = Path("/TTC/FastTTS-AE/benchmarks/benchmark_results")
THESIS_BASE = Path("/TTC/FastTTS-thesis/benchmarks/benchmark_results")


def ae_evaluate_problem(completions, step_scores, reference_answer, data_name, top_n):
    """AE method for a single problem at a given top_n."""
    prod_scores = [float(np.prod(s)) for s in step_scores]
    n_select = min(top_n, len(prod_scores))
    max_indices = np.argsort(prod_scores)[-n_select:]

    selected = [completions[i] for i in max_indices]
    preds = [extract_answer(c, data_name) for c in selected]
    preds = [p for p in preds if p]
    if not preds:
        return False

    pred = max(preds, key=lambda x: preds.count(x))
    result = math_equal_process((0, pred, str(reference_answer)))
    return bool(result)


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def find_result_files(result_dir):
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


def evaluate_file(file_path, data_name="math"):
    """Run AE sweep on a single file. Returns (actual_n, best_acc, best_top_n, all_results)."""
    samples = list(load_jsonl(file_path))
    if not samples:
        return 0, 0.0, 0, []

    if "idx" in samples[0]:
        samples = list({s["idx"]: s for s in samples}.values())
        samples.sort(key=lambda x: x["idx"])

    n_completions = [len(s["solutions"]["completions"][0]) for s in samples]
    actual_n = min(n_completions)

    # Sweep top_n: powers of 2 up to actual_n, plus actual_n itself
    top_n_values = sorted(set(
        [2**i for i in range(actual_n.bit_length()) if 2**i <= actual_n] + [actual_n]
    ))

    best_acc = -1.0
    best_top_n = 0
    all_results = []

    for top_n in top_n_values:
        corrects = []
        for sample in samples:
            completions = sample["solutions"]["completions"][0]
            step_scores = sample["solutions"]["scores"][0]
            reference = sample["reference_answer"]
            corrects.append(
                ae_evaluate_problem(completions, step_scores, reference, data_name, top_n)
            )
        acc = round(100.0 * sum(corrects) / len(corrects), 1)
        all_results.append({"top_n": top_n, "acc": acc})
        if acc > best_acc:
            best_acc = acc
            best_top_n = top_n

    return actual_n, best_acc, best_top_n, all_results


def main():
    parser = argparse.ArgumentParser(
        description="AE evaluation method swept across all N-value files"
    )
    parser.add_argument("--source", type=str, default="both",
                        choices=["ae", "thesis", "both"],
                        help="Which result set to evaluate")
    parser.add_argument("--model_combo", type=str, default="7B-1.5B")
    parser.add_argument("--dataset", type=str, default="aime")
    parser.add_argument("--methods", type=str, default="baseline,spec_prefix")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    sources = []
    if args.source in ("ae", "both"):
        sources.append(("AE", AE_BASE))
    if args.source in ("thesis", "both"):
        sources.append(("Thesis", THESIS_BASE))

    all_output = {}

    for source_name, base_dir in sources:
        print(f"\n{'='*80}")
        print(f"  {source_name}  ({base_dir})")
        print(f"{'='*80}")

        for method in methods:
            result_dir = base_dir / args.model_combo / args.dataset / method
            files = find_result_files(result_dir)

            if not files:
                print(f"\n  [{method}] No files found at {result_dir}")
                continue

            method_label = "FastTTS" if method == "spec_prefix" else "Baseline"
            print(f"\n  --- {method_label} ({method}) ---")
            print(f"  {'N':>5} {'actual':>7} {'best_acc':>9} {'best_top_n':>11}  top_n sweep")
            print(f"  {'-'*70}")

            for n_cfg in sorted(files.keys()):
                print(f"  Evaluating n={n_cfg}...", end="", flush=True)
                actual_n, best_acc, best_top_n, results = evaluate_file(files[n_cfg])
                sweep_str = "  ".join(
                    f"{r['top_n']}:{r['acc']:.0f}" for r in results
                )
                print(f"\r  {n_cfg:>5} {actual_n:>7} {best_acc:>8.1f}% {best_top_n:>11}  {sweep_str}")

                key = f"{source_name}/{method}/{n_cfg}"
                all_output[key] = {
                    "source": source_name,
                    "method": method,
                    "n_configured": n_cfg,
                    "n_actual": actual_n,
                    "best_acc": best_acc,
                    "best_top_n": best_top_n,
                    "sweep": results,
                }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_output, f, indent=2)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
