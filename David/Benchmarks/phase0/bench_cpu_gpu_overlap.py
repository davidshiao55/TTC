#!/usr/bin/env python3
"""Phase 0.2 — GPU / CPU Sub-Module Profiler

Produces the first iteration of the Profiler's output for the generator
(Qwen2.5-7B-Instruct) and verifier (Skywork-o1-Open-PRM-Qwen-2.5-1.5B).

Schema (see `David/Docs/profiler_design.md` §1):

  gpu_layer_timing[sub_module][num_tokens_bucket] = ms
  cpu_gemm_curve[sub_module][batch_size][slice_frac] = ms
  gpu_reduced_timing[sub_module][num_tokens_bucket][slice_frac] = ms

`num_tokens_bucket` is sampled from vLLM's default `cudagraph_capture_sizes`
pattern `[1, 2, 4] + range(8, 256, 8) + range(256, max, 16)`.
`slice_frac` spans the range the Planner interpolates for CPU slice sizing.

The proposal §5.1 claims that arithmetic intensity is governed by `num_tokens`
alone, collapsing uniform-decode and mixed-prefill-decode onto one axis. For
GEMM sub-modules this is trivially true (matmul shape = [num_tokens, in_dim] ×
[in_dim, out_dim]), but we include a small empirical spot-check that confirms
identical timing at the same `num_tokens` regardless of how it's interpreted.

Usage:
    python bench_cpu_gpu_overlap.py --model qwen7b
    python bench_cpu_gpu_overlap.py --model prm1p5b
    python bench_cpu_gpu_overlap.py --model qwen7b --output-json out.json
    python bench_cpu_gpu_overlap.py --model qwen7b --diagnostic-tables
"""

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model configs (source of truth: config.json in each model's HF snapshot)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "qwen7b": {
        "display_name": "Qwen2.5-7B-Instruct",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "hidden": 3584,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "intermediate": 18944,
        "num_layers": 28,
    },
    "prm1p5b": {
        "display_name": "Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "hf_id": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "hidden": 1536,
        "num_heads": 12,
        "num_kv_heads": 2,
        "head_dim": 128,
        "intermediate": 8960,
        "num_layers": 28,
    },
}


Axis = Literal["col", "row"]


@dataclass(frozen=True)
class SubModule:
    """A split-aware matmul sub-module.

    The (in_dim, out_dim) shape is fixed by the model; the `axis` determines
    how CPU and GPU share the work under tensor-granularity offload:

    - `col` (WQKV, WO, MLP1) — CPU slices the output dim. CPU matmul is
      `[B, in_dim] × [in_dim, f·out]`, assembly is concat on GPU.
    - `row` (MLP2)          — CPU slices the input dim. CPU matmul is
      `[B, f·in] × [f·in, out_dim]`, assembly is add-reduce on GPU.

    All shape math for the Profiler's tables goes through `cpu_shape` and
    `gpu_reduced_shape` so the axis lives in exactly one place per sub-module.
    Design reference: `weight_offload_design.md §Per-Sub-Module Split Axis`.
    """
    name: str
    in_dim: int
    out_dim: int
    axis: Axis

    def cpu_shape(self, frac: float) -> tuple[int, int]:
        """(in, out) of the CPU matmul at slice fraction `frac`."""
        if self.axis == "col":
            return self.in_dim, max(1, int(self.out_dim * frac))
        return max(1, int(self.in_dim * frac)), self.out_dim

    def gpu_reduced_shape(self, frac: float) -> tuple[int, int]:
        """(in, out) of the GPU matmul with the CPU slice removed.

        Zero in either dim means GPU has nothing to compute at this frac
        (CPU owns the whole sub-module); callers should short-circuit.
        """
        if self.axis == "col":
            return self.in_dim, self.out_dim - max(1, int(self.out_dim * frac))
        return self.in_dim - max(1, int(self.in_dim * frac)), self.out_dim


def submodules(cfg) -> list[SubModule]:
    """The four per-layer matmul sub-modules with their split axes.

    Invariant `num_heads * head_dim == hidden` is asserted — Qwen-family
    models place q_dim at hidden-size, which all downstream sizing assumes.
    """
    hidden = cfg["hidden"]
    q_dim = cfg["num_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_kv_heads"] * cfg["head_dim"]
    qkv_dim = q_dim + 2 * kv_dim
    assert q_dim == hidden, (
        f"Expected num_heads*head_dim == hidden for {cfg['display_name']}, "
        f"got q_dim={q_dim}, hidden={hidden}")
    return [
        SubModule("WQKV", hidden, qkv_dim,                  axis="col"),
        # WO is col under Alt A (weight offloaded to CPU) and has no CPU path
        # under Alt B (WO stays GPU-resident). We profile it as col here so
        # the Planner has the data if Alt A is chosen in §0.10c; if Alt B
        # wins, this entry is unused at runtime but costs nothing to keep.
        SubModule("WO",   hidden, hidden,                   axis="col"),
        SubModule("MLP1", hidden, 2 * cfg["intermediate"],  axis="col"),
        SubModule("MLP2", cfg["intermediate"], hidden,      axis="row"),
    ]


# Sampled from vLLM's default cudagraph_capture_sizes pattern:
#   [1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)
# Full set is ~77 entries at max=512; we sample ~10 for interpolatable curves.
NUM_TOKENS_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Slice fractions the Planner interpolates over. Covers the range from
# elective offload (near 0) to full CPU path (1.0), with extra density at
# the points most likely to be chosen (9-22%).
SLICE_FRACS = [0.01, 0.03, 0.05, 0.09, 0.15, 0.22, 0.30, 0.50, 0.70, 1.00]

# Batch sizes for the CPU GEMM curve. We deliberately re-use NUM_TOKENS_BUCKETS
# here: in uniform-decode, batch_size == num_tokens, and the CPU GEMM cost only
# depends on the matmul shape, so a single sweep covers both.
BATCH_SIZES = NUM_TOKENS_BUCKETS

WARMUP = 20
ITERS = 100
# Smaller iter count for the dense curve sweeps, which have many points
CURVE_WARMUP = 10
CURVE_ITERS = 50


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------
def time_gpu(B, in_dim, out_dim, warmup=CURVE_WARMUP, iters=CURVE_ITERS):
    """F.linear on CUDA, BF16. Returns mean ms."""
    W = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(B, in_dim, dtype=torch.bfloat16, device="cuda")

    for _ in range(warmup):
        F.linear(x, W)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        F.linear(x, W)
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def time_cpu(B, in_dim, out_dim, warmup=CURVE_WARMUP, iters=CURVE_ITERS):
    """F.linear on CPU, BF16 (oneDNN path). Returns mean ms.

    `F.linear` is REQUIRED — `torch.mm` with BF16 on non-AMX CPUs falls back to
    a scalar loop (100-250× slower). See `phase0_findings.md` §"Critical Discovery".
    """
    W = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device="cpu")
    x = torch.randn(B, in_dim, dtype=torch.bfloat16, device="cpu")

    for _ in range(warmup):
        F.linear(x, W)
    start = time.perf_counter()
    for _ in range(iters):
        F.linear(x, W)
    return (time.perf_counter() - start) / iters * 1000


# ---------------------------------------------------------------------------
# Profile table builders
#
# Every table is keyed by sub-module name at the outer level and wraps its
# measurements in a `{axis, times: {...}}` envelope so the axis lives with
# the data. Downstream Planner code reads `axis` to know whether the slice
# dim was the output (col) or input (row).
# ---------------------------------------------------------------------------
def measure_gpu_layer_timing(cfg, num_tokens_buckets):
    """`gpu_layer_timing[sub_module][num_tokens] = ms` — full sub-module.

    Baseline GPU compute at each bucket; no slice applied. The Planner uses
    this as the GPU idle-budget ceiling: CPU-compute at the same bucket must
    fit within (gpu_layer_timing − gpu_reduced_timing) for free overlap.
    """
    submods = submodules(cfg)
    print(f"\n[gpu_layer_timing] {len(submods)} sub-modules × "
          f"{len(num_tokens_buckets)} buckets")
    table = {}
    for sm in submods:
        row = {n: round(time_gpu(n, sm.in_dim, sm.out_dim,
                                 warmup=WARMUP, iters=ITERS), 4)
               for n in num_tokens_buckets}
        table[sm.name] = row
        print(f"  {sm.name:<6} " + " ".join(
            f"n={n}:{row[n]:.3f}" for n in num_tokens_buckets))
    return table


def measure_cpu_gemm_curve(cfg, batch_sizes, slice_fracs):
    """`cpu_gemm_curve[sub_module] = {axis, times: {B: {frac: ms}}}`.

    Each sub-module is measured only on its assigned split axis — col-parallel
    (WQKV/WO/MLP1) or row-parallel (MLP2) — so the Planner's CPU-path cost
    model matches what `CpuComputeDispatcher` actually dispatches. See
    `profiler_design.md §1.2`.
    """
    submods = submodules(cfg)
    print(f"\n[cpu_gemm_curve] {len(submods)} sub-modules × "
          f"{len(batch_sizes)} batches × {len(slice_fracs)} slices")
    table = {}
    for sm in submods:
        times = {}
        for B in batch_sizes:
            times[B] = {frac: round(time_cpu(B, *sm.cpu_shape(frac)), 4)
                        for frac in slice_fracs}
            print(f"  {sm.name:<6} [{sm.axis}] B={B:>3}  " + " ".join(
                f"f={frac:.2f}:{times[B][frac]:6.2f}ms" for frac in slice_fracs))
        table[sm.name] = {"axis": sm.axis, "times": times}
    return table


def measure_gpu_reduced_timing(cfg, num_tokens_buckets, slice_fracs):
    """`gpu_reduced_timing[sub_module] = {axis, times: {n: {frac: ms}}}`.

    GPU compute on the (1 − frac) portion that stays on GPU after removing
    CPU's slice along the sub-module's axis. Used by the Planner:
        idle(bucket, f) = gpu_layer_timing(bucket) − gpu_reduced_timing(bucket, f)
    """
    submods = submodules(cfg)
    print(f"\n[gpu_reduced_timing] {len(submods)} sub-modules × "
          f"{len(num_tokens_buckets)} buckets × {len(slice_fracs)} slices")
    table = {}
    for sm in submods:
        times = {}
        for n in num_tokens_buckets:
            times[n] = {}
            for frac in slice_fracs:
                in_, out_ = sm.gpu_reduced_shape(frac)
                ms = time_gpu(n, in_, out_) if (in_ > 0 and out_ > 0) else 0.0
                times[n][frac] = round(ms, 4)
        table[sm.name] = {"axis": sm.axis, "times": times}
    return table


# ---------------------------------------------------------------------------
# Note: num_tokens axis unification validation lives in `bench_num_tokens_axis.py`
# (Phase 0.1) and covers both GPU and CPU F.linear paths.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Diagnostic (human-readable) tables — preserved from the original script
# ---------------------------------------------------------------------------
def run_tp_overlap_analysis(cfg, batch_sizes, f_cpu_values):
    """TP-style overlap at a few (B, f_cpu) points for human inspection.

    `overhead = max(0, CPU - GPU_reduced)` — the wall-clock cost the CPU path
    adds beyond the GPU's already-reduced compute. Each sub-module uses its
    assigned split axis for both CPU and GPU-reduced shapes.
    """
    submods = submodules(cfg)
    print(f"\nMeasuring GPU baseline ({cfg['display_name']}, BF16 F.linear)...")
    gpu_full_ms = {B: {sm.name: time_gpu(B, sm.in_dim, sm.out_dim,
                                         warmup=WARMUP, iters=ITERS)
                       for sm in submods}
                   for B in batch_sizes}

    print(f"\n{'='*90}")
    print(f"  GPU baseline — full sub-module times (BF16, {cfg['display_name']})")
    print(f"{'='*90}")
    print(f"  {'Sub-module':<8} {'Axis':<5} {'Weight':>10}", end="")
    for B in batch_sizes:
        print(f" {'B='+str(B):>9}", end="")
    print()
    print(f"  {'-'*8} {'-'*5} {'-'*10}" + " ".join(["-"*9] * len(batch_sizes)))
    for sm in submods:
        size_mb = sm.in_dim * sm.out_dim * 2 / 1e6
        print(f"  {sm.name:<8} {sm.axis:<5} {size_mb:>8.1f}MB", end="")
        for B in batch_sizes:
            print(f" {gpu_full_ms[B][sm.name]:>8.3f}ms", end="")
        print()

    for B in batch_sizes:
        print(f"\n  B={B}: overhead = max(0, CPU - GPU_reduced) per sub-module")
        print(f"  {'Sub-mod':<8} {'Axis':<5} {'GPU(ms)':>8}", end="")
        for f in f_cpu_values:
            print(f" {'f='+f'{f:.0%}':>8}", end="")
        print()
        print(f"  {'-'*8} {'-'*5} {'-'*8}" + " ".join(["-"*8] * len(f_cpu_values)))
        for sm in submods:
            print(f"  {sm.name:<8} {sm.axis:<5} {gpu_full_ms[B][sm.name]:>7.3f}ms", end="")
            for f in f_cpu_values:
                cpu_ms = time_cpu(B, *sm.cpu_shape(f))
                red_in, red_out = sm.gpu_reduced_shape(f)
                gpu_red = (time_gpu(B, red_in, red_out)
                           if (red_in > 0 and red_out > 0) else 0.0)
                overhead = max(0.0, cpu_ms - gpu_red)
                tag = "✓" if cpu_ms <= gpu_red else ("~" if cpu_ms <= gpu_red*1.3 else "✗")
                print(f" {overhead:>6.3f}{tag}", end="")
            print()


def run_qkv_split_analysis(cfg, batch_sizes):
    """Diagnostic: WQKV Q|K|V split (CPU holds K+V, GPU holds Q).

    Note: this is the f_cpu_store_WQKV = (num_kv × 2 × head_dim) / qkv_dim
    *observation point*, not a fixed operating point. The design invariant is
    K/V-biased priority ordering at arbitrary f (see bench_split_correctness.py).
    """
    q_dim = cfg["num_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_kv_heads"] * cfg["head_dim"]
    qkv_dim = q_dim + 2 * kv_dim
    f_kv_boundary = 2 * kv_dim / qkv_dim
    print(f"\n{'='*90}")
    print(f"  WQKV Q|K|V boundary ({cfg['display_name']}) — diagnostic point")
    print(f"  Q={q_dim} → GPU, K+V={2*kv_dim} → CPU. f = {f_kv_boundary:.1%}")
    print(f"{'='*90}")
    for B in batch_sizes:
        gpu_q_ms = time_gpu(B, cfg["hidden"], q_dim)
        cpu_kv_ms = time_cpu(B, cfg["hidden"], 2 * kv_dim)
        fits = "✓" if cpu_kv_ms <= gpu_q_ms else "✗"
        print(f"  B={B}: GPU(Q)={gpu_q_ms:.3f}ms  CPU(KV)={cpu_kv_ms:.3f}ms  [{fits}]")


# ---------------------------------------------------------------------------
# Hardware identification for cache keying
# ---------------------------------------------------------------------------
def hardware_id():
    gpu = torch.cuda.get_device_name(0).replace(" ", "").replace("/", "")
    cpu = platform.processor() or "unknown-cpu"
    return f"{gpu}_{cpu}".lower().replace(" ", "")[:64]


def env_info():
    return {
        "gpu": torch.cuda.get_device_name(0),
        "cpu": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "mkl_available": torch.backends.mkl.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="qwen7b",
                   help="Model config to profile (default: qwen7b)")
    p.add_argument("--num-tokens", type=int, nargs="+", default=NUM_TOKENS_BUCKETS,
                   help="num_tokens buckets for gpu_layer_timing")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES,
                   help="batch sizes for cpu_gemm_curve")
    p.add_argument("--slice-fracs", type=float, nargs="+", default=SLICE_FRACS,
                   help="slice fractions for cpu_gemm_curve and gpu_reduced_timing")
    p.add_argument("--output-json", type=str, default=None,
                   help="Write profile tables to this JSON path")
    p.add_argument("--diagnostic-tables", action="store_true",
                   help="Also print the legacy TP-overlap and QKV-split tables")
    p.add_argument("--skip-reduced-timing", action="store_true",
                   help="Skip gpu_reduced_timing (saves ~half the runtime)")
    args = p.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    print(f"Phase 0.1+0.2 Profiler — {cfg['display_name']}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  CPU threads: {torch.get_num_threads()}")
    print(f"MKL: {torch.backends.mkl.is_available()}  "
          f"oneDNN: {torch.backends.mkldnn.is_available()}")
    print(f"num_tokens_buckets: {args.num_tokens}")
    print(f"slice_fracs:        {args.slice_fracs}")

    profile = {
        "schema_version": 1,
        "hardware_id": hardware_id(),
        "model_key": args.model,
        "model": cfg,
        "env": env_info(),
        "config": {
            "num_tokens_buckets": args.num_tokens,
            "batch_sizes": args.batch_sizes,
            "slice_fracs": args.slice_fracs,
            "warmup": WARMUP,
            "iters": ITERS,
            "curve_warmup": CURVE_WARMUP,
            "curve_iters": CURVE_ITERS,
        },
    }

    profile["gpu_layer_timing"] = measure_gpu_layer_timing(cfg, args.num_tokens)
    profile["cpu_gemm_curve"] = measure_cpu_gemm_curve(
        cfg, args.batch_sizes, args.slice_fracs)

    if not args.skip_reduced_timing:
        profile["gpu_reduced_timing"] = measure_gpu_reduced_timing(
            cfg, args.num_tokens, args.slice_fracs)

    if args.diagnostic_tables:
        diag_batches = [b for b in [1, 4, 8, 16, 32] if b in args.batch_sizes] or [1, 4, 8]
        diag_fcpu = [f for f in [0.03, 0.05, 0.09, 0.15, 0.30] if f in args.slice_fracs] or args.slice_fracs[:5]
        run_tp_overlap_analysis(cfg, diag_batches, diag_fcpu)
        run_qkv_split_analysis(cfg, diag_batches)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"\nProfile saved to {out_path}")


if __name__ == "__main__":
    main()
