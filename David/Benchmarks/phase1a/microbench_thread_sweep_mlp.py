#!/usr/bin/env python3
"""Cold-cache CPU MLP across thread counts × num_tokens.

Independent verification of the agent's claim that isolated cold-cache
CPU MLP always prefers 24 threads. Cycles a 28-layer ring of pinned-host
BF16 (gate_up, down) weights so timing is DRAM-streaming, not L3-resident.
"""
import argparse
import json
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F

HIDDEN = 3584
INTERMEDIATE = 18944
F_CPU = 0.09
N_CPU_PER_HALF = round(F_CPU * INTERMEDIATE)
MLP1_N_CPU = 2 * N_CPU_PER_HALF
DOWN_N_CPU = N_CPU_PER_HALF

N_LAYERS = 28
WARMUP = 30
ITERS = 200


def make_layer():
    gu = torch.randn(MLP1_N_CPU, HIDDEN, dtype=torch.bfloat16, pin_memory=True)
    dn = torch.randn(HIDDEN, DOWN_N_CPU, dtype=torch.bfloat16, pin_memory=True)
    return gu, dn


def cpu_mlp(x, gu, dn, y):
    y1 = F.linear(x, gu)
    z = F.silu(y1[:, :N_CPU_PER_HALF]) * y1[:, N_CPU_PER_HALF:]
    y.copy_(F.linear(z, dn))


def time_threads(threads, ring, x, y):
    torch.set_num_threads(threads)
    idx = 0
    for _ in range(WARMUP):
        gu, dn = ring[idx]
        cpu_mlp(x, gu, dn, y)
        idx = (idx + 1) % N_LAYERS
    t0 = time.perf_counter_ns()
    for _ in range(ITERS):
        gu, dn = ring[idx]
        cpu_mlp(x, gu, dn, y)
        idx = (idx + 1) % N_LAYERS
    return (time.perf_counter_ns() - t0) / ITERS / 1000  # µs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+",
                    default=[1, 4, 8, 16, 24, 32])
    ap.add_argument("--tokens", type=int, nargs="+",
                    default=[1, 4, 16, 64, 256])
    args = ap.parse_args()

    print(f"L3 (i9-14900KF): 36 MiB  |  Ring: 28 layers, {N_LAYERS * (MLP1_N_CPU * HIDDEN * 2 + HIDDEN * DOWN_N_CPU * 2) / 2**20:.1f} MiB total")
    print(f"OMP_NUM_THREADS env: {os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"warmup={WARMUP} iters={ITERS}")
    print()
    print(f"{'tokens':>8} | " + " | ".join(f"{f'omp{t}':>10}" for t in args.threads) + " | best")
    print("-" * (10 + 13 * len(args.threads) + 8))

    # Pre-allocate ring once (shared across all configurations).
    ring = [make_layer() for _ in range(N_LAYERS)]
    summary = {}

    for nt in args.tokens:
        x = torch.randn(nt, HIDDEN, dtype=torch.bfloat16, pin_memory=True)
        y = torch.empty(nt, HIDDEN, dtype=torch.bfloat16, pin_memory=True)
        row = {}
        for t in args.threads:
            row[t] = time_threads(t, ring, x, y)
        best_t = min(row, key=row.get)
        summary[nt] = {"by_thread_us": row, "best_threads": best_t,
                       "best_us": row[best_t]}
        cells = " | ".join(
            f"{row[t]:>9.1f}{'*' if t == best_t else ' '}" for t in args.threads
        )
        print(f"{nt:>8} | {cells} | omp{best_t}")

    print()
    out = Path(__file__).resolve().parent / "results" / "thread_sweep_mlp.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2,
              default=lambda x: x if not isinstance(x, dict) else {str(k): v for k, v in x.items()})
    print(f"[wrote] {out}")


if __name__ == "__main__":
    main()
