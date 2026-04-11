"""
Principled evaluation pipeline for FastTTS test-time compute scaling.

Evaluates ALL completions in a single result file. For multi-N scaling
curves, call this script once per N-value result file (each is a separate
beam search run with a different N).

Metrics:
  - Pass@N   : at least one of N completions is correct (OpenAI unbiased formula)
  - Majority Vote : most common extracted answer (no PRM)
  - PRM-Max  : single best completion by aggregate PRM score
  - PRM-Vote : group answers by equivalence, sum PRM scores per group, pick highest
               (matches Liu et al. _agg_prm_last_vote)

Usage:
  python evaluate.py --data_name math --file_path results.jsonl
  python evaluate.py --data_name math --file_path results.jsonl --agg_strategy last --output eval.json
"""

import argparse
import json
import numpy as np
from concurrent.futures import TimeoutError
from pathlib import Path
from typing import List, Dict, Any, Optional

from pebble import ProcessPool

from grader import math_equal, math_equal_process
from parser import extract_answer


# ---------------------------------------------------------------------------
# Pass@K  (OpenAI Codex formula, numerically stable)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k.

    From Chen et al., "Evaluating Large Language Models Trained on Code", 2021.
    https://arxiv.org/abs/2107.03374

    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# ---------------------------------------------------------------------------
# Score aggregation  (replicates search/utils.py:aggregate_scores)
# ---------------------------------------------------------------------------

def aggregate_scores(scores: List[float], strategy: str = "last") -> float:
    """Aggregate per-step PRM scores into a single value."""
    if not scores:
        return 0.0
    if strategy == "last":
        return scores[-1]
    elif strategy == "min":
        return min(scores)
    elif strategy == "prod":
        return float(np.prod(scores))
    elif strategy == "mean":
        return float(np.mean(scores))
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


# ---------------------------------------------------------------------------
# Answer grouping with math_equal  (timeout-safe)
# ---------------------------------------------------------------------------

def _group_answers(answers: List[str]) -> Dict[str, List[int]]:
    """Group answer indices by mathematical equivalence.

    Returns {canonical_answer: [indices]} where canonical_answer is the
    first-seen representative of each equivalence group.
    """
    groups: Dict[str, List[int]] = {}  # representative -> indices
    rep_list: List[str] = []  # ordered list of representatives

    for idx, ans in enumerate(answers):
        matched = False
        for rep in rep_list:
            try:
                if _answers_equal(ans, rep):
                    groups[rep].append(idx)
                    matched = True
                    break
            except Exception:
                continue

        if not matched:
            rep_list.append(ans)
            groups[ans] = [idx]

    return groups


def _answers_equal(a: str, b: str) -> bool:
    """Check if two answers are mathematically equal (with timeout)."""
    if a.strip().lower() == b.strip().lower():
        return True
    # math_equal with timeout=True uses call_with_timeout (1s) for symbolic checks
    return math_equal(a, b, timeout=True)


# ---------------------------------------------------------------------------
# Per-problem metric computation
# ---------------------------------------------------------------------------

def _compute_problem_metrics(
    completions: List[str],
    step_scores: List[List[float]],
    reference_answer: str,
    data_name: str,
    agg_strategy: str,
) -> Dict[str, Any]:
    """Compute all metrics for a single problem using ALL completions."""
    n = len(completions)

    if n == 0:
        return {
            "pass_at_n": 0.0,
            "majority_vote_correct": False,
            "prm_max_correct": False,
            "prm_vote_correct": False,
            "n": 0,
        }

    # 1. Extract answers from completions
    extracted = [extract_answer(c, data_name) for c in completions]

    # 2. Check correctness of each answer
    correct = []
    params = [(i, ext, str(reference_answer)) for i, ext in enumerate(extracted)]
    with ProcessPool(max_workers=min(4, n)) as pool:
        future = pool.map(math_equal_process, params, timeout=5)
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
                correct.append(bool(result))
            except StopIteration:
                break
            except (TimeoutError, Exception):
                correct.append(False)

    n_correct = sum(correct)

    # 3. Aggregate PRM scores
    agg = [aggregate_scores(s, agg_strategy) for s in step_scores]

    # --- Pass@N (computed on ALL completions, including unparseable) ---
    pass_n = pass_at_k(n, n_correct, n)

    # 4. Filter invalid answers before selection metrics
    #    (matching Liu et al. compute-optimal-tts: judge_ans filters INVALID_ANS)
    valid_indices = [i for i, ans in enumerate(extracted) if ans]
    if not valid_indices:
        return {
            "pass_at_n": pass_n,
            "majority_vote_correct": False,
            "prm_max_correct": False,
            "prm_vote_correct": False,
            "n": n,
        }

    valid_extracted = [extracted[i] for i in valid_indices]
    valid_correct = [correct[i] for i in valid_indices]
    valid_agg = [agg[i] for i in valid_indices]

    # 5. Group valid answers by equivalence
    groups = _group_answers(valid_extracted)

    # --- Majority Vote (pure count, no PRM) ---
    best_group_key = None
    best_count = -1
    for rep, indices in groups.items():
        count = len(indices)
        if count > best_count or (count == best_count and rep < (best_group_key or "")):
            best_count = count
            best_group_key = rep
    maj_idx = groups[best_group_key][0] if best_group_key else 0
    maj_correct = valid_correct[maj_idx]

    # --- PRM-Max (best single completion by PRM score) ---
    prm_max_idx = int(np.argmax(valid_agg))
    prm_max_correct = valid_correct[prm_max_idx]

    # --- PRM-Vote (group answers, sum PRM scores, pick highest-sum group) ---
    best_vote_key = None
    best_vote_score = -float("inf")
    for rep, indices in groups.items():
        group_score = sum(valid_agg[i] for i in indices)
        if group_score > best_vote_score or (
            group_score == best_vote_score and rep < (best_vote_key or "")
        ):
            best_vote_score = group_score
            best_vote_key = rep
    vote_idx = groups[best_vote_key][0] if best_vote_key else 0
    prm_vote_correct = valid_correct[vote_idx]

    return {
        "pass_at_n": pass_n,
        "majority_vote_correct": maj_correct,
        "prm_max_correct": prm_max_correct,
        "prm_vote_correct": prm_vote_correct,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def load_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Error parsing line: {line[:80]}...")
                continue


def evaluate(
    data_name: str,
    file_path: str,
    agg_strategy: str = "last",
    max_num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run multi-metric evaluation on a single result file.

    Evaluates ALL completions in the file — no subsampling.
    For multi-N scaling curves, call this once per N-value result file.

    Args:
        data_name: Dataset name (for answer extraction, e.g. "math")
        file_path: Path to JSONL results file
        agg_strategy: PRM score aggregation strategy ("last", "min", "prod", "mean")
        max_num_samples: Limit number of problems to evaluate

    Returns:
        Dict with aggregated metrics and per-problem details.
    """
    samples = list(load_jsonl(file_path))
    if not samples:
        raise ValueError(f"No samples found in {file_path}")

    # Deduplicate by idx if present
    if "idx" in samples[0]:
        samples = list({s["idx"]: s for s in samples}.values())
        samples.sort(key=lambda x: x["idx"])

    if max_num_samples:
        samples = samples[:max_num_samples]

    # Determine N (completions per problem)
    n_per_problem = [len(s["solutions"]["completions"][0]) for s in samples]
    n_min = min(n_per_problem)
    n_max = max(n_per_problem)

    print(f"Evaluating {len(samples)} problems, "
          f"completions per problem: {n_min}" +
          (f"-{n_max}" if n_min != n_max else ""))
    print(f"Aggregation strategy: {agg_strategy}")

    problem_results = []
    for sample in samples:
        completions = sample["solutions"]["completions"][0]
        step_scores = sample["solutions"]["scores"][0]
        reference = str(sample["reference_answer"])

        metrics = _compute_problem_metrics(
            completions, step_scores, reference,
            data_name, agg_strategy,
        )
        metrics["idx"] = sample.get("idx", sample.get("id", ""))
        problem_results.append(metrics)

    # Aggregate across problems
    num_problems = len(problem_results)
    result = {
        "n": n_min,
        "num_problems": num_problems,
        "agg_strategy": agg_strategy,
        "pass_at_n": round(
            100.0 * np.mean([r["pass_at_n"] for r in problem_results]), 1
        ),
        "majority_vote": round(
            100.0 * np.mean([r["majority_vote_correct"] for r in problem_results]), 1
        ),
        "prm_max": round(
            100.0 * np.mean([r["prm_max_correct"] for r in problem_results]), 1
        ),
        "prm_vote": round(
            100.0 * np.mean([r["prm_vote_correct"] for r in problem_results]), 1
        ),
    }

    print(
        f"  N={result['n']:>4d}: Pass@N={result['pass_at_n']:5.1f}%  "
        f"MajVote={result['majority_vote']:5.1f}%  "
        f"PRM-Max={result['prm_max']:5.1f}%  "
        f"PRM-Vote={result['prm_vote']:5.1f}%"
    )

    return {
        "file_path": str(file_path),
        "data_name": data_name,
        "agg_strategy": agg_strategy,
        "result": result,
        "per_problem": problem_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FastTTS results with multi-metric pipeline"
    )
    parser.add_argument("--data_name", type=str, default="math",
                        help="Dataset name for answer extraction")
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to JSONL results file")
    parser.add_argument("--agg_strategy", type=str, default="last",
                        choices=["last", "min", "prod", "mean"],
                        help="PRM score aggregation strategy")
    parser.add_argument("--max_num_samples", type=int, default=None,
                        help="Limit number of problems")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = evaluate(
        data_name=args.data_name,
        file_path=args.file_path,
        agg_strategy=args.agg_strategy,
        max_num_samples=args.max_num_samples,
    )

    # Print summary
    r = results["result"]
    print(f"\n{'='*60}")
    print(f"N={r['n']}: Pass@N={r['pass_at_n']}  MajVote={r['majority_vote']}  "
          f"PRM-Max={r['prm_max']}  PRM-Vote={r['prm_vote']}")
    print(f"{'='*60}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}")
