# SPDX-License-Identifier: Apache-2.0
"""Stage 4 — per-bucket thread-count sweep.

Sweeps `cpu_num_threads ∈ {2, 4, 8, 16, 24}` against three
representative bucket sizes `B ∈ {1, 4, 16}` on a synthetic CPU GEMM
workload that mimics the Phase 1a MLP1 9% slice (`(N=3408, K=3584)`
BF16). Emits a per-bucket optimal table the Planner can ingest as
`cpu_num_threads_by_bucket`, plus the full grid as JSON for the
phase1c findings doc.

This is the first Stage 4 measurement; Stage 6 will re-run on the
real model + the §1.13b shape table.

Run:
    /opt/conda/envs/thesis/bin/python bench_thread_policy_sweep.py
    /opt/conda/envs/thesis/bin/python bench_thread_policy_sweep.py \
        --thread-counts 2 4 8 16 24 \
        --buckets 1 4 16 \
        --n-iters 80
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from vllm._cots_C import CotsCpuInfer

# Phase 1a §0.3.2 reference: MLP1 9% slice = (out=3408, in=3584) BF16.
MLP1_K_DEFAULT = 3584
MLP1_N_DEFAULT = 3408


def _bench_one(
    *, batch: int, k: int, n: int, n_threads: int, n_iters: int, warmup: int
) -> dict:
    """Time `n_iters` slab dispatches at the given (batch, n_threads)
    cell. Returns wall-clock stats per cycle in microseconds."""
    ci = CotsCpuInfer()
    ci.install(n_slabs=1, max_num_tokens=batch)
    x_pinned = torch.empty(batch, k, dtype=torch.bfloat16, pin_memory=True)
    y_pinned = torch.empty(batch, n, dtype=torch.bfloat16, pin_memory=True)
    w_cpu = torch.randn(n, k, dtype=torch.bfloat16, pin_memory=True)
    ci.populate_slab_qkv(
        task_id=0,
        n_threads=n_threads,
        x_pinned_ptr=x_pinned.data_ptr(),
        in_dim=k,
        y_pinned_ptr=y_pinned.data_ptr(),
        cpu_out_dim=n,
        w_cpu_ptr=w_cpu.data_ptr(),
        w_cpu_rows=n,
    )

    stream = torch.cuda.current_stream().cuda_stream

    def _one_cycle() -> None:
        ci.submit_on_stream(task_id=0, num_tokens=batch, cuda_stream=stream)
        ci.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()

    # Warmup so first-iter cost (oneDNN JIT, cache priming) doesn't
    # land in the measured medians.
    for _ in range(warmup):
        _one_cycle()

    samples_us: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        _one_cycle()
        samples_us.append((time.perf_counter_ns() - t0) / 1e3)
    assert not ci.has_error()
    return {
        "median_us": statistics.median(samples_us),
        "p90_us": (
            statistics.quantiles(samples_us, n=10)[8]
            if len(samples_us) >= 10
            else max(samples_us)
        ),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-counts", type=int, nargs="+", default=[2, 4, 8, 16, 24])
    parser.add_argument("--buckets", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--n-iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--k", type=int, default=MLP1_K_DEFAULT)
    parser.add_argument("--n", type=int, default=MLP1_N_DEFAULT)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — skipping thread-policy sweep", file=sys.stderr)
        return 0

    grid: dict[int, dict[int, dict]] = {}
    print()
    print("=" * 72)
    print("Stage 4 thread-policy sweep — batch × n_threads grid (μs/cycle)")
    print("=" * 72)
    header = "  B \\ t " + "".join(f"{t:>10}" for t in args.thread_counts)
    print(header)
    optimal: dict[int, int] = {}
    for batch in args.buckets:
        grid[batch] = {}
        row = f"  B={batch:<3}  "
        best_t = args.thread_counts[0]
        best_med = float("inf")
        for t in args.thread_counts:
            r = _bench_one(
                batch=batch,
                k=args.k,
                n=args.n,
                n_threads=t,
                n_iters=args.n_iters,
                warmup=args.warmup,
            )
            grid[batch][t] = r
            row += f"{r['median_us']:>10.1f}"
            if r["median_us"] < best_med:
                best_med = r["median_us"]
                best_t = t
        optimal[batch] = best_t
        print(row + f"   ← best t={best_t}")
    print("=" * 72)
    print()
    print("Optimal cpu_num_threads_by_bucket (suggested, from medians):")
    print(f"  {optimal!r}")
    print()
    print("Use this as a starting point for the Planner — the real")
    print("optima depend on workload + concurrent GPU dispatch (Stage 6")
    print("re-runs on the model with GPU work in the loop).")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "bench_thread_policy_sweep.json"
    out_path.write_text(
        json.dumps(
            {
                "args": vars(args) | {"results_dir": str(args.results_dir)},
                "shape": {"k": args.k, "n": args.n},
                "grid": {
                    str(b): {str(t): r for t, r in row.items()}
                    for b, row in grid.items()
                },
                "optimal_cpu_num_threads_by_bucket": optimal,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n  results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
