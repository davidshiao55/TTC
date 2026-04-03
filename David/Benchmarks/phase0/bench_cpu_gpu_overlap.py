#!/usr/bin/env python3
"""Phase 0.1+0.2 — GPU-CPU Tensor Parallel Overlap Feasibility

Simulates tensor-parallel-style column split between GPU and CPU for each
sub-module in a Qwen2.5-7B layer. Same split dimension as standard TP
(output columns), but the "second device" is the CPU instead of a second GPU.

For each sub-module, measures:
  - GPU compute time for its (1-f_cpu) portion (BF16 F.linear)
  - CPU compute time for its f_cpu portion (BF16 F.linear → oneDNN)

The split is free when CPU finishes before GPU (like balanced TP).
When CPU is slower, the excess is added latency per layer.

F.linear with BF16 on CPU routes through oneDNN, achieving ~2x memory-BW
advantage at small batch. torch.mm does NOT use this path and is 100x slower.

Usage:
    python bench_cpu_gpu_overlap.py
    python bench_cpu_gpu_overlap.py --f-cpu 0.03 0.09 0.30 1.0
    python bench_cpu_gpu_overlap.py --batch-sizes 1 4 8 16
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Qwen2.5-7B layer dimensions
# ---------------------------------------------------------------------------
HIDDEN = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE = 18944

Q_DIM = NUM_HEADS * HEAD_DIM        # 3584
K_DIM = NUM_KV_HEADS * HEAD_DIM     # 512
V_DIM = NUM_KV_HEADS * HEAD_DIM     # 512
QKV_DIM = Q_DIM + K_DIM + V_DIM     # 4608
GATE_UP_DIM = 2 * INTERMEDIATE      # 37888

# Sub-modules: (name, input_dim, full_output_dim)
# The output dimension is the TP split dimension (same as multi-GPU TP).
SUBMODULES = [
    ("WQKV", HIDDEN,       QKV_DIM),      # Q+K+V fused
    ("WO",   HIDDEN,       HIDDEN),        # attention output projection
    ("MLP1", HIDDEN,       GATE_UP_DIM),   # gate+up fused
    ("MLP2", INTERMEDIATE, HIDDEN),        # down projection
]

BATCH_SIZES = [1, 4, 8, 16, 32, 64]
F_CPU_VALUES = [0.03, 0.05, 0.09, 0.15, 0.30, 0.50, 0.70, 0.90, 1.00]
WARMUP = 20
ITERS = 100


# ---------------------------------------------------------------------------
# GPU timing (BF16, CUDA events)
# ---------------------------------------------------------------------------
def time_gpu(B, in_dim, out_dim, warmup=WARMUP, iters=ITERS):
    """Time F.linear on GPU in BF16. Returns mean ms."""
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


# ---------------------------------------------------------------------------
# CPU timing (BF16 via F.linear → oneDNN)
# ---------------------------------------------------------------------------
def time_cpu(B, in_dim, out_dim, warmup=WARMUP, iters=ITERS):
    """Time F.linear on CPU in BF16 (oneDNN kernel). Returns mean ms."""
    W = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device="cpu")
    x = torch.randn(B, in_dim, dtype=torch.bfloat16, device="cpu")

    for _ in range(warmup):
        F.linear(x, W)

    start = time.perf_counter()
    for _ in range(iters):
        F.linear(x, W)
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000


# ---------------------------------------------------------------------------
# Analysis 1: TP-style overlap (all sub-modules, uniform f_cpu)
# ---------------------------------------------------------------------------
def run_tp_overlap_analysis(batch_sizes, f_cpu_values):
    """Simulate TP split between GPU and CPU for each sub-module.

    Like TP=2 where device 0 is GPU and device 1 is CPU.
    GPU gets (1-f_cpu) output columns, CPU gets f_cpu output columns.
    Both compute in parallel; the slower device determines layer time.
    """
    results = []

    # Measure GPU baseline (full sub-module, no split)
    print("Measuring GPU baseline (BF16 F.linear, no split)...")
    gpu_times = {}
    for B in batch_sizes:
        gpu_times[B] = {}
        for name, in_dim, out_dim in SUBMODULES:
            gpu_times[B][name] = time_gpu(B, in_dim, out_dim)

    # Print GPU baseline
    print(f"\n{'='*90}")
    print(f"  GPU baseline — full sub-module times (BF16, RTX 4090)")
    print(f"  This is the TP=1 (no split) reference.")
    print(f"{'='*90}")
    print(f"  {'Sub-module':<8} {'Weight':>10}", end="")
    for B in batch_sizes:
        print(f" {'B='+str(B):>9}", end="")
    print()
    print(f"  {'-'*8} {'-'*10}" + " ".join(["-"*9] * len(batch_sizes)))

    gpu_layer_total = {B: 0 for B in batch_sizes}
    for name, in_dim, out_dim in SUBMODULES:
        size_mb = in_dim * out_dim * 2 / 1e6
        print(f"  {name:<8} {size_mb:>8.1f}MB", end="")
        for B in batch_sizes:
            ms = gpu_times[B][name]
            gpu_layer_total[B] += ms
            print(f" {ms:>8.3f}ms", end="")
        print()

    print(f"  {'TOTAL':<8} {'':>10}", end="")
    for B in batch_sizes:
        print(f" {gpu_layer_total[B]:>8.3f}ms", end="")
    print()

    # Per-batch overlap analysis
    for B in batch_sizes:
        print(f"\n{'='*90}")
        print(f"  B={B}: GPU-CPU TP overlap (split along output dim, like TP=2)")
        print(f"  GPU computes (1-f_cpu) columns, CPU computes f_cpu columns in parallel.")
        print(f"  ✓ = CPU ≤ GPU (free, like balanced TP)  ✗ = CPU slower (adds latency)")
        print(f"{'='*90}")

        print(f"  {'Sub-mod':<8} {'GPU(ms)':>8}", end="")
        for f_cpu in f_cpu_values:
            print(f" {'f='+f'{f_cpu:.0%}':>8}", end="")
        print()
        print(f"  {'-'*8} {'-'*8}" + " ".join(["-"*8] * len(f_cpu_values)))

        layer_overhead = {f: 0.0 for f in f_cpu_values}

        for name, in_dim, full_out_dim in SUBMODULES:
            gpu_ms = gpu_times[B][name]
            print(f"  {name:<8} {gpu_ms:>7.3f}ms", end="")

            for f_cpu in f_cpu_values:
                cpu_out = max(1, int(full_out_dim * f_cpu))
                cpu_ms = time_cpu(B, in_dim, cpu_out,
                                  warmup=10, iters=50)

                # GPU computes its reduced portion
                gpu_reduced_out = full_out_dim - cpu_out
                if gpu_reduced_out > 0:
                    gpu_reduced_ms = time_gpu(B, in_dim, gpu_reduced_out,
                                              warmup=10, iters=50)
                else:
                    gpu_reduced_ms = 0.0

                # Overhead = max(0, CPU - GPU). Like TP imbalance.
                overhead = max(0.0, cpu_ms - gpu_reduced_ms)
                layer_overhead[f_cpu] += overhead

                if cpu_ms <= gpu_reduced_ms:
                    tag = "✓"
                elif cpu_ms <= gpu_reduced_ms * 1.3:
                    tag = "~"
                else:
                    tag = "✗"

                print(f" {cpu_ms:>6.3f}{tag}", end="")

                results.append({
                    "batch": B, "name": name, "f_cpu": f_cpu,
                    "cpu_ms": round(cpu_ms, 4),
                    "gpu_full_ms": round(gpu_ms, 4),
                    "gpu_reduced_ms": round(gpu_reduced_ms, 4),
                    "overhead_ms": round(overhead, 4),
                    "fits": cpu_ms <= gpu_reduced_ms,
                })

            print()

        # Layer summary
        print(f"  {'-'*8} {'-'*8}" + " ".join(["-"*8] * len(f_cpu_values)))
        print(f"  {'EXTRA':<8} {'':>8}", end="")
        for f_cpu in f_cpu_values:
            oh = layer_overhead[f_cpu]
            if oh < 0.001:
                print(f" {'FREE':>7} ", end="")
            else:
                print(f" {'+'+f'{oh:.3f}':>7} ", end="")
        print("ms added per layer")

        print(f"  {'FREED':<8} {'':>8}", end="")
        for f_cpu in f_cpu_values:
            freed_mb = sum(
                i * max(1, int(o * f_cpu)) * 2 / 1e6
                for _, i, o in SUBMODULES
            )
            print(f" {freed_mb:>6.1f}MB", end="")
        print(" per layer (BF16)")

        print(f"  {'×28 ly':<8} {'':>8}", end="")
        for f_cpu in f_cpu_values:
            freed_mb = sum(
                i * max(1, int(o * f_cpu)) * 2 / 1e6
                for _, i, o in SUBMODULES
            ) * 28
            print(f" {freed_mb/1000:>5.2f}GB", end="")
        print(" total freed")

    return results


# ---------------------------------------------------------------------------
# Analysis 2: Q|K|V split (offloaded WQKV only)
# ---------------------------------------------------------------------------
def run_qkv_split_analysis(batch_sizes, f_cpu_values):
    """WQKV split at Q|K|V semantic boundary instead of arbitrary columns.

    Q → GPU, K+V → CPU (K+V results go to CPU suffix KV cache).
    WO/MLP1/MLP2 at variable f_cpu.
    """
    print(f"\n{'='*90}")
    print(f"  WQKV Q|K|V Split — Semantic Boundary")
    print(f"  Q({Q_DIM}) → GPU, K+V({K_DIM+V_DIM}) → CPU")
    print(f"  WO/MLP1/MLP2 at variable f_cpu")
    print(f"{'='*90}")

    for B in batch_sizes:
        gpu_q_ms = time_gpu(B, HIDDEN, Q_DIM, warmup=10, iters=50)
        cpu_kv_ms = time_cpu(B, HIDDEN, K_DIM + V_DIM, warmup=10, iters=50)
        fits = "✓" if cpu_kv_ms <= gpu_q_ms else "✗"

        print(f"\n  B={B}: GPU Q={gpu_q_ms:.3f}ms, CPU K+V={cpu_kv_ms:.3f}ms [{fits}]")

        print(f"  {'f_cpu(others)':>14} {'WQKV oh':>8} {'WO oh':>8} "
              f"{'MLP1 oh':>8} {'MLP2 oh':>8} {'Total oh':>9} {'Freed':>7}")
        print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*7}")

        for f_cpu in f_cpu_values:
            wqkv_oh = max(0, cpu_kv_ms - gpu_q_ms)

            total_oh = wqkv_oh
            parts = [f"{wqkv_oh:>7.3f}ms"]

            for name, in_dim, full_out_dim in SUBMODULES[1:]:
                cpu_out = max(1, int(full_out_dim * f_cpu))
                gpu_reduced_out = full_out_dim - cpu_out
                cpu_ms = time_cpu(B, in_dim, cpu_out, warmup=5, iters=20)
                gpu_ms = time_gpu(B, in_dim, gpu_reduced_out,
                                  warmup=5, iters=20) if gpu_reduced_out > 0 else 0
                oh = max(0, cpu_ms - gpu_ms)
                total_oh += oh
                parts.append(f"{oh:>7.3f}ms")

            freed_mb = (HIDDEN * (K_DIM + V_DIM) * 2 / 1e6 +
                        sum(i * max(1, int(o * f_cpu)) * 2 / 1e6
                            for _, i, o in SUBMODULES[1:])) * 28 / 1000

            tag = "FREE" if total_oh < 0.001 else f"+{total_oh:.3f}ms"
            print(f"  {f_cpu:>13.0%} {' '.join(parts)} {tag:>9} {freed_mb:>5.2f}GB")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0.1+0.2: GPU-CPU tensor parallel overlap feasibility")
    parser.add_argument("--f-cpu", type=float, nargs="+", default=F_CPU_VALUES)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    print("GPU-CPU Tensor Parallel Overlap Feasibility Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"MKL: {torch.backends.mkl.is_available()}")
    print(f"oneDNN: {torch.backends.mkldnn.is_available()}")
    print(f"Split dimension: output columns (same as TP)")
    print(f"GPU: BF16 F.linear | CPU: BF16 F.linear (oneDNN)")

    results = run_tp_overlap_analysis(args.batch_sizes, args.f_cpu)
    run_qkv_split_analysis(args.batch_sizes, args.f_cpu)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
