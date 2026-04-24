#!/usr/bin/env python3
"""Phase 0.1 — num_tokens axis unification

The Planner's dispatch table is keyed on a single scalar — the forward call's
`num_tokens` (thesis_proposal.md §5.1, planner_design.md §4.5). This requires
that GEMM arithmetic intensity depend **only on num_tokens** and not on how
those tokens are distributed across requests / prefill / decode.

Tested empirically by running the same sub-module GEMM at multiple
pre-flatten compositions that all collapse to the same [num_tokens, hidden]
matmul input:

  flat        — [N, H]                 (already flat; reference)
  1x_N        — [1, N, H]              (1 request × N-token prefill)
  Nx_1        — [N, 1, H]              (N decodes × 1 token each)
  halfx_2     — [N/2, 2, H]            (N/2 requests × 2 tokens; chunked prefill)
  quarterx_4  — [N/4, 4, H]            (N/4 requests × 4 tokens; mixed-depth)

Tested on BOTH:
  - GPU F.linear (cuBLAS) — the primary matmul path
  - CPU F.linear (oneDNN BF16) — the CPU-compute dispatch path

The claim holds per device if the spread across compositions is within noise
(≤5%). A failure would mean the Planner needs a second dispatch axis.

Usage:
    python bench_num_tokens_axis.py --model qwen7b
    python bench_num_tokens_axis.py --model prm1p5b --output-json out.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


MODEL_CONFIGS = {
    "qwen7b": {
        "display_name": "Qwen2.5-7B-Instruct",
        "hidden": 3584,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "intermediate": 18944,
    },
    "prm1p5b": {
        "display_name": "Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "hidden": 1536,
        "num_heads": 12,
        "num_kv_heads": 2,
        "head_dim": 128,
        "intermediate": 8960,
    },
}

NUM_TOKENS_VALUES = [16, 64, 256]
SPREAD_THRESHOLD = 0.05  # 5% = within noise

WARMUP = 20
ITERS_GPU = 50
ITERS_CPU = 50
# On CPU, OpenMP thread scheduling introduces ~20-30% transient variance on
# isolated GEMM calls. We tighten by: (a) high iter count with median, (b) a
# per-(submodule, N) prewarm shared across all compositions, (c) multiple
# trials per composition so a single disturbance doesn't skew the result.
CPU_TRIALS = 3


def submodule_shapes(cfg):
    """(name, in_dim, out_dim) for the four matmul sub-modules."""
    hidden = cfg["hidden"]
    q_dim = cfg["num_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_kv_heads"] * cfg["head_dim"]
    qkv_dim = q_dim + 2 * kv_dim
    assert q_dim == hidden
    return [
        ("WQKV", hidden, qkv_dim),
        ("WO",   hidden, hidden),
        ("MLP1", hidden, 2 * cfg["intermediate"]),
        ("MLP2", cfg["intermediate"], hidden),
    ]


def compositions(N):
    """Return list of (label, prefix_shape) tuples; prefix_shape sums to N tokens."""
    comps = [("flat", (N,)), ("1x_N", (1, N)), ("Nx_1", (N, 1))]
    if N % 2 == 0 and N >= 2:
        comps.append(("halfx_2", (N // 2, 2)))
    if N % 4 == 0 and N >= 4:
        comps.append(("quarterx_4", (N // 4, 4)))
    return comps


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _median(xs):
    ys = sorted(xs)
    return ys[len(ys) // 2]


def time_gpu_linear(x, W, iters, warmup):
    """Time F.linear on CUDA; returns median ms."""
    in_dim = W.shape[1]
    x_flat = x.view(-1, in_dim)
    for _ in range(warmup):
        F.linear(x_flat, W)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        F.linear(x_flat, W)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return _median(times)


def time_cpu_linear(x, W, iters, warmup):
    """Time F.linear on CPU (oneDNN BF16). Returns median ms."""
    in_dim = W.shape[1]
    x_flat = x.view(-1, in_dim)
    for _ in range(warmup):
        F.linear(x_flat, W)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        F.linear(x_flat, W)
        times.append((time.perf_counter() - t0) * 1000)
    return _median(times)


def prewarm_kernel(device, in_dim, out_dim, N):
    """One-time warmup of the [N, in_dim] × [in_dim, out_dim] GEMM on `device`.

    oneDNN JIT-compiles kernels on first use; if we don't pre-warm, the
    first-timed composition absorbs compilation overhead and comes out
    artificially slow (30%+ on CPU). Running the target shape once before
    any per-composition timing fixes this.
    """
    W = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device=device)
    x = torch.randn(N, in_dim, dtype=torch.bfloat16, device=device)
    for _ in range(10):
        F.linear(x, W)
    if device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Per-device sweep
# ---------------------------------------------------------------------------
def run_device(cfg, device, num_tokens_values):
    """Sweep all (sub_module, N, composition) on a single device.

    Returns: dict keyed by N → sub_module → {compositions, comp_spread,
    total_spread, passed}.
    """
    is_gpu = device == "cuda"
    iters = ITERS_GPU if is_gpu else ITERS_CPU
    time_fn = time_gpu_linear if is_gpu else time_cpu_linear

    print(f"\n[{device.upper()}] F.linear — "
          f"{'cuBLAS' if is_gpu else 'oneDNN BF16'}")
    results = {}

    for N in num_tokens_values:
        print(f"\n  num_tokens={N}")
        per_submodule = {}
        comps = compositions(N)

        for name, in_dim, out_dim in submodule_shapes(cfg):
            # Pre-warm the kernel for this (in_dim, out_dim, N) once before
            # timing any composition — this absorbs oneDNN/cuBLAS JIT cost so
            # it doesn't bias whichever composition runs first.
            prewarm_kernel(device, in_dim, out_dim, N)
            W = torch.randn(out_dim, in_dim, dtype=torch.bfloat16,
                            device=device)
            comp_times = {}
            # CPU: run multiple trials to dampen OpenMP scheduling variance.
            trials = CPU_TRIALS if not is_gpu else 1
            for label, prefix_shape in comps:
                x = torch.randn(*prefix_shape, in_dim,
                                dtype=torch.bfloat16, device=device)
                trial_ms = [time_fn(x, W, iters=iters, warmup=WARMUP)
                            for _ in range(trials)]
                comp_times[label] = round(min(trial_ms), 4)

            # Load-bearing metric: spread across reshape compositions only.
            # The "flat" case is a separate allocation pattern, not a composition.
            reshape_times = {k: v for k, v in comp_times.items() if k != "flat"}
            t_max_all = max(comp_times.values())
            t_min_all = min(comp_times.values())
            r_max = max(reshape_times.values())
            r_min = min(reshape_times.values())
            comp_spread = (r_max - r_min) / r_max if r_max > 0 else 0.0
            total_spread = ((t_max_all - t_min_all) / t_max_all
                            if t_max_all > 0 else 0.0)
            passed = comp_spread < SPREAD_THRESHOLD

            tag = "✓" if passed else "✗"
            cells = " ".join(f"{l}={t:.3f}" for l, t in comp_times.items())
            print(f"    {name:<6} {cells}  "
                  f"comp={comp_spread:.1%} total={total_spread:.1%} {tag}")
            per_submodule[name] = {
                "compositions":  comp_times,
                "comp_spread":   round(comp_spread, 4),
                "total_spread":  round(total_spread, 4),
                "passed":        passed,
            }

        results[N] = {
            "per_submodule": per_submodule,
            "all_passed":    all(r["passed"] for r in per_submodule.values()),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="qwen7b")
    p.add_argument("--num-tokens", type=int, nargs="+", default=NUM_TOKENS_VALUES)
    p.add_argument("--skip-cpu", action="store_true",
                   help="Skip CPU sweep (GPU only)")
    p.add_argument("--skip-gpu", action="store_true",
                   help="Skip GPU sweep (CPU only)")
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    print(f"Phase 0.1 — num_tokens axis unification  ({cfg['display_name']})")
    print(f"Claim: same num_tokens → same GEMM time, regardless of "
          f"(B, q_len) composition.")
    print(f"Threshold: spread across reshape compositions <{SPREAD_THRESHOLD:.0%}.")

    all_results = {"model": args.model, "model_cfg": cfg,
                   "num_tokens_values": args.num_tokens,
                   "spread_threshold": SPREAD_THRESHOLD}

    if not args.skip_gpu:
        all_results["gpu"] = run_device(cfg, "cuda", args.num_tokens)
    if not args.skip_cpu:
        all_results["cpu"] = run_device(cfg, "cpu", args.num_tokens)

    # Summary
    print("\n" + "=" * 62)
    print(f"  Summary — {cfg['display_name']}")
    print("=" * 62)
    for dev in ["gpu", "cpu"]:
        if dev not in all_results:
            continue
        all_ok = all(v["all_passed"] for v in all_results[dev].values())
        print(f"  {dev.upper()}: {'PASSED' if all_ok else 'FAILED'}")
        for N, d in all_results[dev].items():
            failures = [n for n, r in d["per_submodule"].items()
                        if not r["passed"]]
            if failures:
                print(f"    N={N} failed on: {failures}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
