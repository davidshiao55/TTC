"""Full-metric evaluation utility for FastTTS results.

Reports all four metrics (Pass@N, Majority Vote, PRM-Max, PRM-Vote) for
every aggregation strategy (last, min, prod, mean) on a single result
file. Intended for ablation tables and methodological transparency —
the main thesis pipeline uses `evaluate.py` (pass@n + pass@1 only).

Usage:
  python evaluate_full.py --data_name math --file_path results.jsonl
  python evaluate_full.py --data_name math --file_path results.jsonl --output full_eval.json
"""

import argparse
import json
from concurrent.futures import TimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pebble import ProcessPool

from grader import math_equal_process
from parser import extract_answer


AGG_STRATEGIES: Tuple[str, ...] = ("last", "min", "prod", "mean")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProblemMetrics:
    pass_at_n: float
    majority_vote_correct: bool
    prm_max_correct: bool
    prm_vote_correct: bool
    n: int
    idx: Optional[str] = None


@dataclass
class AggregatedResult:
    n: int
    num_problems: int
    agg_strategy: str
    pass_at_n: float
    majority_vote: float
    prm_max: float
    prm_vote: float


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def aggregate_scores(scores: List[float], strategy: str) -> float:
    if not scores:
        return 0.0
    if strategy == "last":
        return scores[-1]
    if strategy == "min":
        return min(scores)
    if strategy == "prod":
        return float(np.prod(scores))
    if strategy == "mean":
        return float(np.mean(scores))
    raise ValueError(f"Unknown aggregation strategy: {strategy}")


def _group_answers(answers: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, ans in enumerate(answers):
        groups.setdefault(ans.strip(), []).append(idx)
    return groups


def _extract_and_check(
    completions: List[str],
    reference: str,
    data_name: str,
) -> Tuple[List[str], List[bool]]:
    extracted = [extract_answer(c, data_name) for c in completions]
    params = [(i, ext, reference) for i, ext in enumerate(extracted)]
    correct: List[bool] = []
    with ProcessPool(max_workers=min(4, len(completions))) as pool:
        future = pool.map(math_equal_process, params, timeout=5)
        iterator = future.result()
        while True:
            try:
                correct.append(bool(next(iterator)))
            except StopIteration:
                break
            except (TimeoutError, Exception):
                correct.append(False)
    return extracted, correct


def _pick_best(candidates, groups, fallback_idx: int = 0) -> int:
    best_score = -float("inf")
    best_key: Optional[str] = None
    for score, key in candidates:
        if score > best_score or (score == best_score and (best_key is None or key < best_key)):
            best_score = score
            best_key = key
    return groups[best_key][0] if best_key is not None else fallback_idx


# ---------------------------------------------------------------------------
# Per-problem evaluation for a single aggregation strategy
# ---------------------------------------------------------------------------

def _compute_problem_metrics_for_strategy(
    extracted: List[str],
    correct: List[bool],
    step_scores: List[List[float]],
    agg_strategy: str,
    pass_n: float,
) -> ProblemMetrics:
    """Compute all 4 metrics for one agg_strategy, given already-extracted answers."""
    n = len(extracted)
    if n == 0:
        return ProblemMetrics(0.0, False, False, False, 0)

    valid_indices = [i for i, ans in enumerate(extracted) if ans]
    if not valid_indices:
        return ProblemMetrics(pass_n, False, False, False, n)

    agg = [aggregate_scores(s, agg_strategy) for s in step_scores]
    valid_extracted = [extracted[i] for i in valid_indices]
    valid_correct = [correct[i] for i in valid_indices]
    valid_agg = [agg[i] for i in valid_indices]
    groups = _group_answers(valid_extracted)

    maj_cands = [(float(len(indices)), key) for key, indices in groups.items()]
    vote_cands = [
        (sum(valid_agg[i] for i in indices), key)
        for key, indices in groups.items()
    ]

    return ProblemMetrics(
        pass_at_n=pass_n,
        majority_vote_correct=valid_correct[_pick_best(maj_cands, groups)],
        prm_max_correct=valid_correct[int(np.argmax(valid_agg))],
        prm_vote_correct=valid_correct[_pick_best(vote_cands, groups)],
        n=n,
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def load_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _dedupe_and_sort(samples: List[dict]) -> List[dict]:
    if samples and "idx" in samples[0]:
        samples = list({s["idx"]: s for s in samples}.values())
        samples.sort(key=lambda x: x["idx"])
    return samples


def _aggregate(
    problem_results: List[ProblemMetrics],
    agg_strategy: str,
) -> AggregatedResult:
    return AggregatedResult(
        n=min(r.n for r in problem_results),
        num_problems=len(problem_results),
        agg_strategy=agg_strategy,
        pass_at_n=round(100.0 * np.mean([r.pass_at_n for r in problem_results]), 1),
        majority_vote=round(100.0 * np.mean([r.majority_vote_correct for r in problem_results]), 1),
        prm_max=round(100.0 * np.mean([r.prm_max_correct for r in problem_results]), 1),
        prm_vote=round(100.0 * np.mean([r.prm_vote_correct for r in problem_results]), 1),
    )


def evaluate_full(
    data_name: str,
    file_path: str,
    max_num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate all 4 metrics across all aggregation strategies on one file."""
    samples = _dedupe_and_sort(list(load_jsonl(file_path)))
    if not samples:
        raise ValueError(f"No samples found in {file_path}")
    if max_num_samples:
        samples = samples[:max_num_samples]

    n_per_problem = [len(s["solutions"]["completions"][0]) for s in samples]
    n_min, n_max = min(n_per_problem), max(n_per_problem)
    print(f"Evaluating {len(samples)} problems, completions per problem: {n_min}"
          + (f"-{n_max}" if n_min != n_max else ""))

    # Extract + grade once per problem (expensive); reuse across all strategies.
    per_problem_cache: List[Tuple[List[str], List[bool], List[List[float]], float, int, str]] = []
    for sample in samples:
        completions = sample["solutions"]["completions"][0]
        step_scores = sample["solutions"]["scores"][0]
        reference = str(sample["reference_answer"])
        extracted, correct = _extract_and_check(completions, reference, data_name)
        pass_n = pass_at_k(len(completions), sum(correct), len(completions))
        idx = sample.get("idx", sample.get("id", ""))
        per_problem_cache.append(
            (extracted, correct, step_scores, pass_n, len(completions), idx)
        )

    results_by_strategy: Dict[str, Any] = {}
    for strategy in AGG_STRATEGIES:
        problem_results: List[ProblemMetrics] = []
        for extracted, correct, step_scores, pass_n, n, idx in per_problem_cache:
            pm = _compute_problem_metrics_for_strategy(
                extracted, correct, step_scores, strategy, pass_n,
            )
            pm.idx = idx
            pm.n = n
            problem_results.append(pm)

        result = _aggregate(problem_results, strategy)
        results_by_strategy[strategy] = {
            "result": asdict(result),
            "per_problem": [asdict(r) for r in problem_results],
        }
        print(
            f"  [{strategy:>4s}] Pass@N={result.pass_at_n:5.1f}%  "
            f"MajVote={result.majority_vote:5.1f}%  "
            f"PRM-Max={result.prm_max:5.1f}%  "
            f"PRM-Vote={result.prm_vote:5.1f}%"
        )

    return {
        "file_path": str(file_path),
        "data_name": data_name,
        "by_agg_strategy": results_by_strategy,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full multi-metric × multi-strategy evaluation for FastTTS results",
    )
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_full(
        data_name=args.data_name,
        file_path=args.file_path,
        max_num_samples=args.max_num_samples,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}")
