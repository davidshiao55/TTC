#!/usr/bin/env python3
"""Phase 0.3 — PCIe Bandwidth Sweep

Measures effective PCIe bandwidth at various transfer sizes.
Extends the existing pcie_bandwidth_test.py with a size sweep.

Run from anywhere:
    python David/Benchmarks/phase0/bench_pcie_sweep.py
"""

import argparse
import json
from pathlib import Path
import time

import torch

SIZES_MB = [0.25, 1, 4, 10, 50, 100, 500]
RUNS = 10
WARMUP = 3


def measure_transfer(size_bytes, direction="h2d", runs=RUNS, warmup=WARMUP):
    """Measure transfer bandwidth. Returns (bandwidth_gbps, latency_us)."""
    n_elements = size_bytes // 2  # BF16 = 2 bytes
    if n_elements < 1:
        n_elements = 1

    buf_cpu = torch.empty(n_elements, dtype=torch.bfloat16, pin_memory=True)
    buf_gpu = torch.empty(n_elements, dtype=torch.bfloat16, device="cuda")

    # Determine copy direction
    if direction == "h2d":
        src, dst = buf_cpu, buf_gpu
    else:
        src, dst = buf_gpu, buf_cpu

    # Warmup
    for _ in range(warmup):
        dst.copy_(src)
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        dst.copy_(src)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    size_gb = (n_elements * 2) / 1e9
    bw_gbps = size_gb * runs / elapsed
    latency_us = (elapsed / runs) * 1e6
    return bw_gbps, latency_us


def main():
    parser = argparse.ArgumentParser(description="Phase 0.3: PCIe bandwidth sweep")
    parser.add_argument("--sizes-mb", type=float, nargs="+", default=SIZES_MB)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    print("PCIe Bandwidth Sweep")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pinned memory: Yes")
    print(f"Runs per size: {args.runs}")
    print()

    results = []

    print(f"{'Size':>10} {'H2D (GB/s)':>12} {'H2D lat (us)':>14} "
          f"{'D2H (GB/s)':>12} {'D2H lat (us)':>14}")
    print("-" * 65)

    for size_mb in args.sizes_mb:
        size_bytes = int(size_mb * 1e6)
        h2d_bw, h2d_lat = measure_transfer(size_bytes, "h2d", args.runs)
        d2h_bw, d2h_lat = measure_transfer(size_bytes, "d2h", args.runs)

        label = f"{size_mb:.2f} MB" if size_mb < 1 else f"{size_mb:.0f} MB"
        print(f"{label:>10} {h2d_bw:>12.2f} {h2d_lat:>14.1f} "
              f"{d2h_bw:>12.2f} {d2h_lat:>14.1f}")

        results.append({
            "size_mb": size_mb,
            "h2d_gbps": round(h2d_bw, 2), "h2d_latency_us": round(h2d_lat, 1),
            "d2h_gbps": round(d2h_bw, 2), "d2h_latency_us": round(d2h_lat, 1),
        })

    # Highlight key transfer sizes for our use case
    print(f"\n--- Key sizes for thesis ---")
    key_sizes = [
        ("Activation result (B=16, hidden=3584, bf16)", 16 * 3584 * 2 / 1e6),
        ("CPU K+V output (B=16, 1024, bf16)", 16 * 1024 * 2 / 1e6),
        ("9% of MLP1 weight", SIZES_MB[2]),  # ~24 MB
        ("Full layer weight (7B)", 466),
    ]
    for desc, size_mb in key_sizes:
        size_bytes = int(size_mb * 1e6)
        h2d_bw, h2d_lat = measure_transfer(size_bytes, "h2d", args.runs)
        d2h_bw, d2h_lat = measure_transfer(size_bytes, "d2h", args.runs)
        print(f"  {desc}")
        print(f"    Size: {size_mb:.3f} MB | H2D: {h2d_bw:.2f} GB/s ({h2d_lat:.1f} us) "
              f"| D2H: {d2h_bw:.2f} GB/s ({d2h_lat:.1f} us)")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
