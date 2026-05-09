# SPDX-License-Identifier: Apache-2.0
"""Stage 2 substrate gate (deferred to Stage 3 once operators were
flipped end-to-end).

Question: at the same workload, does the C++ host-callback round-trip
(`x_pinned.copy_(x, non_blocking=True)` → `cudaLaunchHostFunc` enqueue
→ TaskQueue worker → `cudaLaunchHostFunc` sync → UVA-copy) cost AT
LEAST as little as Python's `executor.submit` + `future.result()`
round-trip in eager mode?

If "yes" — the substrate is at least neutral, and the headline §1.14
collapse (Stage 5 with graph capture eliminating Python operator-body
orchestration) can land on top of it without inheriting a substrate
regression.

If "no" — investigate before Stage 4 (bucket-aware thread policy)
changes the measurement landscape. Stage 4's gains would mask a
substrate slowdown.

We don't need a real model for this: a synthetic submit/wait microbench
with a fixed payload exercises the host-callback substrate in isolation
and is fully reproducible. End-to-end model-orch comparison is the
Stage 5 / 6 bench.

Acceptance gate (per the plan): `orch_eager_native ≤ orch_python`.
This script logs `(median, p90)` per cycle for each runner and a
verdict. Threshold for the assert: native median <= python median *
1.05 (5% slack to avoid flake on tiny absolute deltas).

Run:
    /opt/conda/envs/thesis/bin/python bench_dryrun_vs_real_native_eager.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _bench_runner(runner_kind: str, *, n_iters: int, dim: int) -> dict:
    """Time N submit/wait cycles on a single runner under
    `dry_run=True` semantics — the worker callback is a no-op so only
    the substrate orch round-trip is measured (no real GEMM time).
    This is the load-bearing comparison: real CPU GEMM time is
    workload-dependent and equal between runners (both call
    `at::linear` underneath with the same thread count); the substrate
    delta is the fixed cost the plan calls 'orch'.

    Both runners install one descriptor (`(0, 0, "qkv")`); the python
    path swaps the worker closure for `_cpu_dryrun_noop` (only
    `event.synchronize()` runs), the native path populates the slab as
    `kDryrunNoop` (worker dispatcher takes the noop branch).
    """
    from vllm.model_executor.offloader import cots

    runner: cots.PythonCotsRunner | cots.NativeCotsRunner
    # dry_run=True on BOTH so only the substrate orch round-trip is
    # measured — no real CPU GEMM contributing to wall-clock.
    if runner_kind == "python":
        runner = cots.PythonCotsRunner(dry_run=True)
    elif runner_kind == "native":
        runner = cots.NativeCotsRunner(dry_run=True)
    else:
        raise ValueError(runner_kind)

    # Workload buffers (used as pointer targets even when the worker
    # body is a noop).
    w_cpu = torch.randn(dim, dim, dtype=torch.bfloat16, pin_memory=True)
    x_pinned = torch.empty(dim, dim, dtype=torch.bfloat16, pin_memory=True)
    y_pinned = torch.empty(dim, dim, dtype=torch.bfloat16, pin_memory=True)
    x_gpu = torch.randn(dim, dim, dtype=torch.bfloat16, device="cuda")
    y_gpu = torch.empty(dim, dim, dtype=torch.bfloat16, device="cuda")
    # Two distinct dummy anchors per Phase 1c §design-decision 6: the
    # operator-facing `wait_and_uva(gpu_anchor_a, gpu_anchor_b)`
    # contract is "never alias the two mutates_args anchors" because
    # torch.compile / functionalization can fold aliased mutation
    # slots. The bench is eager-only so aliasing wouldn't break this
    # measurement, but modelling production usage faithfully here
    # keeps the bench from accidentally codifying a wrong pattern.
    dummy_anchor_a = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    dummy_anchor_b = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    op_descriptor = (0, 0, "qkv")

    if runner_kind == "python":
        # Python path: the runner's `dry_run=True` flag short-circuits
        # the closure to `_cpu_dryrun_noop`, so the registered closure
        # below is never actually called. We still install one so the
        # descriptor lookup succeeds.
        def _cb(
            event: torch.cuda.Event,
            x_p: torch.Tensor,
            y_p: torch.Tensor,
        ) -> None:
            event.synchronize()
            y_p.copy_(F.linear(x_p, w_cpu))

        runner.install({op_descriptor: _cb}, bucket_for_fallback=lambda _n: 0)
    else:  # native
        # Native path: dry_run=True overrides the slab kind to
        # kDryrunNoop at populate time (NativeSlabSpec.populate). The
        # full QKV spec is still constructed so the install path
        # exercises the same code on both runners.
        slab = cots._NativeSlabSpecQkv(
            op_descriptor=op_descriptor,
            n_threads=1,
            x_pinned_ptr=int(x_pinned.data_ptr()),
            in_dim=dim,
            y_pinned_ptr=int(y_pinned.data_ptr()),
            cpu_out_dim=dim,
            w_cpu_ptr=int(w_cpu.data_ptr()),
            w_cpu_rows=dim,
        )
        runner.install(
            slab_specs=[slab],
            scratch_max_tokens=dim,
            scratch_max_intermediate_per_half=0,
            bucket_for_fallback=lambda _n: 0,
        )

    # Warmup so the first iteration's runtime cost (executor thread
    # spawn for python; first cudaLaunchHostFunc compile/cache for
    # native) doesn't pollute medians.
    for _ in range(20):
        runner.submit_with_d2h(x_gpu, x_pinned, y_pinned, op_descriptor)
        runner.wait_and_uva(y_pinned, y_gpu, dummy_anchor_a, dummy_anchor_b)
    torch.cuda.synchronize()

    cycle_us: list[float] = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        runner.submit_with_d2h(x_gpu, x_pinned, y_pinned, op_descriptor)
        runner.wait_and_uva(y_pinned, y_gpu, dummy_anchor_a, dummy_anchor_b)
        torch.cuda.synchronize()
        cycle_us.append((time.perf_counter_ns() - t0) / 1e3)

    runner.close()

    return {
        "runner": runner_kind,
        "n_iters": n_iters,
        "dim": dim,
        "median_us": statistics.median(cycle_us),
        "p90_us": (
            statistics.quantiles(cycle_us, n=10)[8]
            if len(cycle_us) >= 10
            else max(cycle_us)
        ),
        "min_us": min(cycle_us),
        "max_us": max(cycle_us),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-iters",
        type=int,
        default=200,
        help="Cycles per runner. Higher = tighter median, slower bench.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="GEMM dim — small enough that orch dominates real GEMM time.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=1.05,
        help=(
            "Acceptance: native median <= python median * threshold_ratio. "
            "Default 1.05 (5% slack on tiny absolute deltas)."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — skipping substrate bench", file=sys.stderr)
        return 0

    results = {
        "python": _bench_runner("python", n_iters=args.n_iters, dim=args.dim),
        "native": _bench_runner("native", n_iters=args.n_iters, dim=args.dim),
    }

    py_med = results["python"]["median_us"]
    nat_med = results["native"]["median_us"]
    ratio = nat_med / py_med if py_med > 0 else float("inf")

    print()
    print("=" * 72)
    print("Stage 2 substrate gate — orch_eager_native vs orch_python")
    print("=" * 72)
    print(f"  workload: dim={args.dim}, n_iters={args.n_iters}")
    print(f"  python:   median={py_med:.2f}μs   p90={results['python']['p90_us']:.2f}μs")
    print(f"  native:   median={nat_med:.2f}μs   p90={results['native']['p90_us']:.2f}μs")
    print(f"  ratio (native/python):   {ratio:.3f}")
    verdict = "PASS" if ratio <= args.threshold_ratio else "FAIL"
    print(f"  threshold: {args.threshold_ratio:.3f}  →  {verdict}")
    print("=" * 72)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "bench_dryrun_vs_real_native_eager.json"
    out_path.write_text(
        json.dumps(
            {
                "args": vars(args) | {"results_dir": str(args.results_dir)},
                "results": results,
                "ratio_native_over_python": ratio,
                "verdict": verdict,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n  results written to {out_path}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
