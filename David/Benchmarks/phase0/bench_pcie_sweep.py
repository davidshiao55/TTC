#!/usr/bin/env python3
"""Phase 0.3 — PCIe Bandwidth Sweep (Explicit Copy vs UVA)

Measures effective PCIe bandwidth for two mechanisms by which GPU code can
access CPU-resident data:

  1. Explicit copy via `cudaMemcpyAsync` (PyTorch `tensor.copy_()`) on pinned
     memory. Data lands in GDDR6X; subsequent GPU reads hit GDDR6X at 1 TB/s.
  2. UVA mapping via `get_cuda_view_from_cpu_tensor`. No explicit copy — the
     GPU reads pinned CPU memory directly through PCIe during kernel execution.

Both methods physically traverse the same PCIe link. What differs is:
  - Where the data ends up (GDDR6X vs. stays on CPU)
  - Launch overhead (one cudaMemcpyAsync vs. zero, folded into kernel launches)
  - Whether subsequent repeated reads hit cache (GDDR6X yes; UVA no — L2 is
    tied to device memory, not BAR)

For the UVA variant we time a full-tensor sum on GPU, since UVA has no
standalone "transfer" step — the work IS the PCIe read. For explicit copy
we time the pure `copy_()`.

Feeds:
  - `pcie_h2d_bw[transfer_bytes]` / `pcie_d2h_bw[transfer_bytes]` (explicit copy)
  - `uva_read_bw[transfer_bytes]` (UVA read via sum kernel)

The Planner picks the right curve for the right purpose:
  - Weight prefetch (MB scale, reused many times) → explicit copy
  - CPU-produced data that is consumed once (activation return) → explicit
    copy is still the right choice because UVA access patterns that re-read
    (e.g. matmul) cannot use L2 cache; see `phase0_findings.md §0.3`.

Usage:
    python bench_pcie_sweep.py
    python bench_pcie_sweep.py --output-json out.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


SIZES_MB = [0.25, 1, 4, 10, 50, 100, 500]
RUNS = 10
WARMUP = 3


def _pinned_cpu_bf16(nelts):
    return torch.empty(nelts, dtype=torch.bfloat16, pin_memory=True)


def _gpu_bf16(nelts):
    return torch.empty(nelts, dtype=torch.bfloat16, device="cuda")


# ---------------------------------------------------------------------------
# Method A: explicit cudaMemcpyAsync (via tensor.copy_)
# ---------------------------------------------------------------------------
def measure_explicit_copy(nelts, direction, runs=RUNS, warmup=WARMUP):
    """Explicit copy: pinned ↔ GDDR6X via the copy engine. Returns (GB/s, us)."""
    cpu_buf = _pinned_cpu_bf16(nelts)
    gpu_buf = _gpu_bf16(nelts)
    src, dst = (cpu_buf, gpu_buf) if direction == "h2d" else (gpu_buf, cpu_buf)

    for _ in range(warmup):
        dst.copy_(src)
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        dst.copy_(src)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    size_gb = (nelts * 2) / 1e9
    return size_gb * runs / elapsed, (elapsed / runs) * 1e6


# ---------------------------------------------------------------------------
# Method B: UVA read (no copy; GPU kernel reads pinned memory directly)
# ---------------------------------------------------------------------------
def measure_uva_read(nelts, runs=RUNS, warmup=WARMUP):
    """UVA: GPU sums a pinned tensor via UVA mapping. Pure PCIe read, no copy
    engine. Bytes read / time → effective PCIe read BW seen by a compute
    kernel. Returns (GB/s, us)."""
    cpu_buf = _pinned_cpu_bf16(nelts)
    uva_view = get_accelerator_view_from_cpu_tensor(cpu_buf)
    # Persistent output to avoid per-iteration allocation.
    out = torch.empty(1, dtype=torch.float32, device="cuda")

    for _ in range(warmup):
        out.copy_(uva_view.sum(dtype=torch.float32))
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        out.copy_(uva_view.sum(dtype=torch.float32))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    size_gb = (nelts * 2) / 1e9
    return size_gb * runs / elapsed, (elapsed / runs) * 1e6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes-mb", type=float, nargs="+", default=SIZES_MB)
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    print("PCIe Bandwidth Sweep — Explicit Copy vs UVA Read")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Pinned memory: Yes | Runs per size: {args.runs}")
    print()

    print(f"{'Size':>9} | {'H2D copy':>22} | {'D2H copy':>22} | {'UVA read':>22}")
    print(f"{'':>9} | {'GB/s':>10} {'lat us':>10} | {'GB/s':>10} {'lat us':>10} | "
          f"{'GB/s':>10} {'lat us':>10}")
    print("-" * 85)

    results = []
    for size_mb in args.sizes_mb:
        nelts = max(1, int(size_mb * 1024 * 1024) // 2)   # bf16 = 2 bytes
        h2d_bw, h2d_lat = measure_explicit_copy(nelts, "h2d", args.runs)
        d2h_bw, d2h_lat = measure_explicit_copy(nelts, "d2h", args.runs)
        uva_bw, uva_lat = measure_uva_read(nelts, args.runs)

        label = f"{size_mb:.2f} MB" if size_mb < 1 else f"{size_mb:.0f} MB"
        print(f"{label:>9} | {h2d_bw:>10.2f} {h2d_lat:>10.1f} | "
              f"{d2h_bw:>10.2f} {d2h_lat:>10.1f} | "
              f"{uva_bw:>10.2f} {uva_lat:>10.1f}")

        results.append({
            "size_mb": size_mb,
            "bytes": nelts * 2,
            "h2d_copy_gbps":   round(h2d_bw, 2),
            "h2d_copy_lat_us": round(h2d_lat, 1),
            "d2h_copy_gbps":   round(d2h_bw, 2),
            "d2h_copy_lat_us": round(d2h_lat, 1),
            "uva_read_gbps":   round(uva_bw, 2),
            "uva_read_lat_us": round(uva_lat, 1),
        })

    # Key thesis sizes at both methods
    print(f"\n--- Key transfer sizes for the thesis ---")
    key_sizes = [
        ("CPU K+V output (B=16, bf16)",                       16 * 1024 * 2 / 1e6),
        ("Activation result (B=16, hidden=3584, bf16)",       16 * 3584 * 2 / 1e6),
        ("9% MLP1 weight slice",                              4.0),
        ("Full layer weight (7B)",                            466.0),
    ]
    key_results = []
    for desc, size_mb in key_sizes:
        nelts = max(1, int(size_mb * 1024 * 1024) // 2)
        h2d_bw, h2d_lat = measure_explicit_copy(nelts, "h2d", args.runs)
        uva_bw, uva_lat = measure_uva_read(nelts, args.runs)
        print(f"  {desc}   ({size_mb:.3f} MB)")
        print(f"    copy H2D: {h2d_bw:6.2f} GB/s ({h2d_lat:6.1f} us) | "
              f"UVA read: {uva_bw:6.2f} GB/s ({uva_lat:6.1f} us)")
        key_results.append({
            "desc": desc, "size_mb": size_mb,
            "h2d_copy_gbps": round(h2d_bw, 2),
            "uva_read_gbps": round(uva_bw, 2),
        })

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "schema_version": 2,
                "gpu": torch.cuda.get_device_name(0),
                "size_sweep": results,
                "key_sizes": key_results,
            }, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
