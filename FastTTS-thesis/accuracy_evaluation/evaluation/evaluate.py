"""Principled evaluation pipeline for FastTTS test-time compute scaling.

Evaluates ALL completions in a single result file. For multi-N scaling
curves, call this script once per N-value result file (each is a separate
beam search run with a different N).

Metrics:
  - pass@n : at least one of N completions is correct (OpenAI unbiased formula)
  - pass@1 : PRM-Vote = group answers by equivalence, sum PRM scores per
             group, pick highest-sum group (matches Liu et al. _agg_prm_last_vote).
             Named pass@1 because it represents the single deployed answer.

For a richer dump with all 4 metrics (Pass@N, MajVote, PRM-Max, PRM-Vote)
across all aggregation strategies, use `evaluate_full.py`.

Usage:
  python evaluate.py --data_name math --file_path results.jsonl
  python evaluate.py --data_name math --file_path results.jsonl --agg_strategy last --output eval.json
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
    pass_at_1_correct: bool  # = PRM-Vote correctness
    n: int
    idx: Optional[str] = None


@dataclass
class AggregatedResult:
    n: int
    num_problems: int
    agg_strategy: str
    pass_at_n: float
    pass_at_1: float  # = PRM-Vote accuracy


# ---------------------------------------------------------------------------
# Pass@K  (OpenAI Codex formula, numerically stable)
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# ---------------------------------------------------------------------------
# Score aggregation  (replicates search/utils.py:aggregate_scores)
# ---------------------------------------------------------------------------

def aggregate_scores(scores: List[float], strategy: str = "last") -> float:
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
    raise ValueError(f"Unknown aggregation strategy: {strategy}; expected one of {AGG_STRATEGIES}")


# ---------------------------------------------------------------------------
# Answer grouping (exact string match — matches Liu et al. compute-optimal-tts)
# ---------------------------------------------------------------------------

def _group_answers(answers: List[str]) -> Dict[str, List[int]]:
    """Group answer indices by exact string match (after normalization)."""
    groups: Dict[str, List[int]] = {}
    for idx, ans in enumerate(answers):
        groups.setdefault(ans.strip(), []).append(idx)
    return groups


# ---------------------------------------------------------------------------
# Per-problem metric computation
# ---------------------------------------------------------------------------

def _extract_and_check(
    completions: List[str],
    reference: str,
    data_name: str,
) -> Tuple[List[str], List[bool]]:
    """Extract each completion's final answer and grade it against ``reference``."""
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


def _select_prm_vote(
    valid_correct: List[bool],
    valid_agg: List[float],
    groups: Dict[str, List[int]],
) -> bool:
    """PRM-Vote: group answers, sum PRM scores per group, pick highest."""
    best_score = -float("inf")
    best_key: Optional[str] = None
    for key, indices in groups.items():
        score = sum(valid_agg[i] for i in indices)
        if score > best_score or (score == best_score and (best_key is None or key < best_key)):
            best_score = score
            best_key = key
    idx = groups[best_key][0] if best_key is not None else 0
    return valid_correct[idx]


def _compute_problem_metrics(
    completions: List[str],
    step_scores: List[List[float]],
    reference_answer: str,
    data_name: str,
    agg_strategy: str,
) -> ProblemMetrics:
    """Compute pass@n and pass@1 (PRM-Vote) for a single problem."""
    n = len(completions)
    if n == 0:
        return ProblemMetrics(pass_at_n=0.0, pass_at_1_correct=False, n=0)

    extracted, correct = _extract_and_check(completions, str(reference_answer), data_name)
    agg = [aggregate_scores(s, agg_strategy) for s in step_scores]
    pass_n = pass_at_k(n, sum(correct), n)

    # Filter invalid answers before selection (matches Liu et al. judge_ans).
    valid_indices = [i for i, ans in enumerate(extracted) if ans]
    if not valid_indices:
        return ProblemMetrics(pass_at_n=pass_n, pass_at_1_correct=False, n=n)

    valid_extracted = [extracted[i] for i in valid_indices]
    valid_correct = [correct[i] for i in valid_indices]
    valid_agg = [agg[i] for i in valid_indices]
    groups = _group_answers(valid_extracted)

    return ProblemMetrics(
        pass_at_n=pass_n,
        pass_at_1_correct=_select_prm_vote(valid_correct, valid_agg, groups),
        n=n,
    )


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
        pass_at_1=round(100.0 * np.mean([r.pass_at_1_correct for r in problem_results]), 1),
    )


def evaluate(
    data_name: str,
    file_path: str,
    agg_strategy: str = "last",
    max_num_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run evaluation on a single result file, reporting pass@n and pass@1."""
    samples = _dedupe_and_sort(list(load_jsonl(file_path)))
    if not samples:
        raise ValueError(f"No samples found in {file_path}")
    if max_num_samples:
        samples = samples[:max_num_samples]

    n_per_problem = [len(s["solutions"]["completions"][0]) for s in samples]
    n_min, n_max = min(n_per_problem), max(n_per_problem)
    print(f"Evaluating {len(samples)} problems, completions per problem: {n_min}"
          + (f"-{n_max}" if n_min != n_max else ""))
    print(f"Aggregation strategy: {agg_strategy}")

    problem_results: List[ProblemMetrics] = []
    for sample in samples:
        metrics = _compute_problem_metrics(
            sample["solutions"]["completions"][0],
            sample["solutions"]["scores"][0],
            str(sample["reference_answer"]),
            data_name, agg_strategy,
        )
        metrics.idx = sample.get("idx", sample.get("id", ""))
        problem_results.append(metrics)

    result = _aggregate(problem_results, agg_strategy)
    print(f"  N={result.n:>4d}: pass@n={result.pass_at_n:5.1f}%  pass@1={result.pass_at_1:5.1f}%")

    return {
        "file_path": str(file_path),
        "data_name": data_name,
        "agg_strategy": agg_strategy,
        "result": asdict(result),
        "per_problem": [asdict(r) for r in problem_results],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FastTTS results (pass@n + pass@1 via PRM-Vote)",
    )
    parser.add_argument("--data_name", type=str, default="math",
                        help="Dataset name for answer extraction")
    parser.add_argument("--file_path", type=str, required=True,
                        help="Path to JSONL results file")
    parser.add_argument("--agg_strategy", type=str, default="last",
                        choices=list(AGG_STRATEGIES),
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

    r = results["result"]
    print(f"\n{'=' * 60}")
    print(f"N={r['n']}: pass@n={r['pass_at_n']}  pass@1={r['pass_at_1']}")
    print(f"{'=' * 60}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output_path}")
