#!/usr/bin/env python3
"""
Replicate the original FastTTS-AE evaluation method exactly.

AE method (evaluate.py lines 34-41):
  1. Aggregate step scores via np.prod (product of all steps)
  2. Select top_n completions by product score
  3. Extract answers from selected completions
  4. Majority vote among extracted answers

This script runs the AE method at every top_n value and reports each,
plus the "best" (test-set-tuned) result that the original pipeline reported.

Usage:
    python evaluate_ae_original.py --file_path results.jsonl
    python evaluate_ae_original.py --file_path results.jsonl --top_n_values 1,8,32,64
"""

import argparse
import json
import sys
import os
import numpy as np
from concurrent.futures import TimeoutError
from pathlib import Path

from pebble import ProcessPool

# Add evaluation dir to path for imports
sys.path.insert(0, str(Path(__file__).parent / "evaluation"))
from grader import math_equal_process
from parser import extract_answer


def ae_evaluate_problem(completions, step_scores, reference_answer, data_name, top_n):
    """Replicate AE evaluate.py logic for a single problem at a given top_n.

    Returns True if the selected answer is correct.
    """
    # 1. Product-aggregate step scores
    prod_scores = [float(np.prod(s)) for s in step_scores]

    # 2. Select top_n by product score
    n_select = min(top_n, len(prod_scores))
    max_indices = np.argsort(prod_scores)[-n_select:]

    # 3. Extract answers from selected completions
    selected = [completions[i] for i in max_indices]
    preds = [extract_answer(c, data_name) for c in selected]

    # 4. Filter empty predictions (AE doesn't do this, but empty strings
    #    can't win majority vote anyway unless all are empty)
    preds_nonempty = [p for p in preds if p]
    if not preds_nonempty:
        return False

    # 5. Majority vote (AE line 41: max by count, order-dependent tiebreak)
    pred = max(preds_nonempty, key=lambda x: preds_nonempty.count(x))

    # 6. Check correctness
    result = math_equal_process((0, pred, str(reference_answer)))
    return bool(result)


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def evaluate(file_path, data_name="math", top_n_values=None):
    """Run AE-style evaluation at each top_n value."""
    samples = list(load_jsonl(file_path))
    if not samples:
        raise ValueError(f"No samples found in {file_path}")

    # Deduplicate
    if "idx" in samples[0]:
        samples = list({s["idx"]: s for s in samples}.values())
        samples.sort(key=lambda x: x["idx"])

    n_completions = [len(s["solutions"]["completions"][0]) for s in samples]
    max_n = min(n_completions)

    if top_n_values is None:
        top_n_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    top_n_values = [t for t in top_n_values if t <= max(n_completions)]

    print(f"Evaluating {len(samples)} problems")
    print(f"Completions per problem: {min(n_completions)}-{max(n_completions)}")
    print(f"top_n values: {top_n_values}")
    print()

    results = []
    best_acc = -1
    best_top_n = None

    for top_n in top_n_values:
        corrects = []
        for sample in samples:
            completions = sample["solutions"]["completions"][0]
            step_scores = sample["solutions"]["scores"][0]
            reference = sample["reference_answer"]
            correct = ae_evaluate_problem(
                completions, step_scores, reference, data_name, top_n
            )
            corrects.append(correct)

        acc = round(100.0 * sum(corrects) / len(corrects), 1)
        results.append({"top_n": top_n, "acc": acc, "num_problems": len(samples)})

        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_top_n = top_n
        print(f"  top_n={top_n:>4d}: acc={acc:5.1f}%")

    print(f"\n  Best: {best_acc}% at top_n={best_top_n}  (this is what AE reports)")

    return {
        "file_path": str(file_path),
        "method": "ae_original",
        "results": results,
        "best_acc": best_acc,
        "best_top_n": best_top_n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Replicate FastTTS-AE original evaluation method"
    )
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--top_n_values", type=str, default=None,
                        help="Comma-separated top_n values (default: powers of 2)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    top_n_values = None
    if args.top_n_values:
        top_n_values = [int(x.strip()) for x in args.top_n_values.split(",")]

    results = evaluate(args.file_path, args.data_name, top_n_values)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
