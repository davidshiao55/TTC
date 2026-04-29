#!/usr/bin/env python3
"""Phase 0.10.3 — vLLM native offloader overlap probe (nsys-driven).

Captures an nsys timeline of `vllm bench latency` under each native offloader
so we can visually confirm whether the offloader's H2D actually overlaps
with compute, or serializes behind it.

Three probe arms (each one ~30 s of timeline). G=4, N=1 for prefetch is
1/4 coverage with uniform single-layer offloads spaced every 4 layers
across Qwen2.5-7B's 28 decoder layers — the cleanest pattern for visualizing
H2D overlap with compute:

    none           : vanilla vLLM, no offload                  (reference)
    prefetch_4x1   : PrefetchOffloader G=4 N=1 step=1          (overlap expected;
                                                                 7 layers offloaded
                                                                 at 25% coverage)
    uva_4          : UVAOffloader 4 GB, UVA enabled            (no explicit H2D —
                                                                 PCIe reads via UVA)

For each arm we shell out to::

    nsys profile -o <arm>.nsys-rep --trace=cuda,nvtx,osrt --force-overwrite=true \\
        vllm bench latency --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \\
            --input-len 256 --output-len 16 --batch-size 16 \\
            --num-iters-warmup 1 --num-iters 2 --enforce-eager \\
            [arm-specific offloader flags]

What to look for in the GUI (`nsys-ui <arm>.nsys-rep`):

    none           : compute kernels on the default CUDA stream; no async H2D
                     inside the decode region.
    prefetch_4x1   : PrefetchOffloader's `copy_stream` (separate from default)
                     shows pinned→device memcpy events occurring *during* the
                     layer compute kernels of preceding layers — 7 evenly-spaced
                     H2D events per decode step.
    uva_4          : no explicit H2D events, but the cuBLAS/SDPA kernels reading
                     UVA-mapped weights take longer (PCIe-bound) — visible as
                     inflated kernel durations.

Usage::

    conda activate thesis
    cd /TTC/FastTTS-thesis
    python /TTC/David/Benchmarks/phase0/probe_native_offload_overlap.py
    python /TTC/David/Benchmarks/phase0/probe_native_offload_overlap.py --arm prefetch_4x1

Outputs go to ``results/0.10_overlap_probe/<arm>.nsys-rep``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PHASE0_DIR = Path(__file__).resolve().parent
OUT_DIR = PHASE0_DIR / "results" / "0.10_overlap_probe"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 256
OUTPUT_LEN = 16
BATCH = 16
WARMUP = 1
ITERS = 2

ARMS = {
    "none": {
        "flags": [],
        "env": {},
    },
    "prefetch_4x1": {
        "flags": [
            "--offload-group-size", "4",
            "--offload-num-in-group", "1",
            "--offload-prefetch-step", "1",
        ],
        "env": {},
    },
    "uva_4": {
        "flags": ["--cpu-offload-gb", "4"],
        "env": {},
    },
}


def run_arm(arm: str, force: bool = False) -> Path:
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choices: {list(ARMS)}")
    spec = ARMS[arm]
    out = OUT_DIR / f"{arm}.nsys-rep"
    if out.exists() and not force:
        print(f"[skip] {out.name} already exists (use --force to overwrite)")
        return out

    if not shutil.which("nsys"):
        raise SystemExit("nsys is not on PATH; install Nsight Systems first")

    # vLLM's V1 engine spawns a worker subprocess; without --trace-fork-before-exec
    # nsys misses CUDA activity in the child, and without --wait=primary nsys hangs
    # waiting for the child to terminate. See vllm profiling docs (CLAUDE.md).
    cmd = [
        "nsys", "profile",
        "-o", str(out.with_suffix("")),  # nsys appends .nsys-rep
        "--trace=cuda,nvtx,osrt",
        "--trace-fork-before-exec=true",
        "--force-overwrite=true",
        "vllm", "bench", "latency",
        "--model", MODEL,
        "--dtype", DTYPE,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--batch-size", str(BATCH),
        "--num-iters-warmup", str(WARMUP),
        "--num-iters", str(ITERS),
        "--enforce-eager",
        *spec["flags"],
    ]
    env = {**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn", **spec["env"]}

    print(f"\n[probe] arm={arm}")
    print(f"        env: {spec['env']}")
    print(f"        flags: {' '.join(spec['flags']) or '(none)'}")
    print(f"        out: {out}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, check=False)
    dur = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"[FAIL] arm={arm} rc={proc.returncode} ({dur:.1f}s)")
    else:
        print(f"[ok]  arm={arm} ({dur:.1f}s) → {out.name}")
        # Dump short stats for cross-reference
        try:
            stats = subprocess.run(
                ["nsys", "stats", "--report", "cuda_gpu_mem_time_sum", str(out)],
                capture_output=True, text=True, timeout=60,
            )
            if stats.returncode == 0:
                print("        memcpy summary:")
                for line in stats.stdout.splitlines()[:15]:
                    print("        " + line)
        except Exception:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=[*ARMS, "all"], default="all")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing nsys-rep files")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for a in arms:
        run_arm(a, force=args.force)

    print(f"\nDone. Reports in {OUT_DIR}")
    print("Open in GUI:  nsys-ui <arm>.nsys-rep")


if __name__ == "__main__":
    main()
