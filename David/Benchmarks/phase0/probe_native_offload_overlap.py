#!/usr/bin/env python3
"""Phase 0.10 — vLLM native offloader overlap probe (nsys-driven).

Captures one nsys timeline of `vllm bench latency` per invocation. All
offloader knobs and workload parameters are CLI flags so the same script
serves any (offloader, knob, workload) point we want to inspect — for
either the §0.10.4 head-to-head visualization or the §0.10.3 prefetch
hiding-cap mechanism.

Trace inspection (sync stats, H2D timeline, kernel idle gaps, etc.) is
not part of this script — use `nsys stats`, `nsys-ui`, or direct
`sqlite3` queries on the exported `<arm>.sqlite`.

Usage::

    conda activate thesis
    cd /TTC/FastTTS-thesis

    # Head-to-head overlap visualization (the original §0.10.4 set):
    python probe_native_offload_overlap.py --offloader none -o none
    python probe_native_offload_overlap.py --offloader prefetch -G 4 -o prefetch_4x1
    python probe_native_offload_overlap.py --offloader uva --cpu-offload-gb 4 -o uva_4

    # Prefetch hiding-cap probe at G=28 (§0.10.3 hypothesis verification):
    python probe_native_offload_overlap.py --offloader prefetch -G 28 \\
        --input-len 8 --output-len 32 --batch-size 1 --num-iters-warmup 0 \\
        -o prefetch_g28_decode

    # Sweep G ∈ {1,2,4,7,14,28} at decode_heavy:
    for G in 1 2 4 7 14 28; do
        python probe_native_offload_overlap.py --offloader prefetch -G $G \\
            --input-len 8 --output-len 32 --batch-size 1 --num-iters-warmup 0 \\
            -o prefetch_g${G}_decode
    done

Outputs go to ``results/0.10_overlap_probe/<output-name>.nsys-rep``.
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


def offloader_flags(args: argparse.Namespace) -> list[str]:
    if args.offloader == "none":
        return []
    if args.offloader == "prefetch":
        return [
            "--offload-group-size", str(args.group_size),
            "--offload-num-in-group", str(args.num_in_group),
            "--offload-prefetch-step", str(args.prefetch_step),
        ]
    if args.offloader == "uva":
        return ["--cpu-offload-gb", str(args.cpu_offload_gb)]
    raise SystemExit(f"unknown offloader {args.offloader!r}")


def default_output_name(args: argparse.Namespace) -> str:
    if args.offloader == "none":
        return "none"
    if args.offloader == "prefetch":
        return f"prefetch_g{args.group_size}_n{args.num_in_group}_k{args.prefetch_step}"
    if args.offloader == "uva":
        return f"uva_{args.cpu_offload_gb}"
    raise SystemExit(f"unknown offloader {args.offloader!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Offloader selection + knobs
    ap.add_argument("--offloader", choices=["none", "prefetch", "uva"],
                    default="none", help="Which native offloader to probe")
    ap.add_argument("-G", "--group-size", type=int, default=4,
                    help="PrefetchOffloader group_size (divides num_layers)")
    ap.add_argument("-N", "--num-in-group", type=int, default=1,
                    help="PrefetchOffloader num_in_group")
    ap.add_argument("-K", "--prefetch-step", type=int, default=1,
                    help="PrefetchOffloader prefetch_step (layers ahead)")
    ap.add_argument("--cpu-offload-gb", type=float, default=4.0,
                    help="UVAOffloader cpu_offload_gb")
    # Workload
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-iters-warmup", type=int, default=1)
    ap.add_argument("--num-iters", type=int, default=2)
    # Output
    ap.add_argument("-o", "--output-name", default=None,
                    help="Output filename stem (default: derived from offloader+knobs)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing nsys-rep file")
    args = ap.parse_args()

    if not shutil.which("nsys"):
        sys.exit("nsys is not on PATH; install Nsight Systems first")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = args.output_name or default_output_name(args)
    out = OUT_DIR / f"{name}.nsys-rep"
    if out.exists() and not args.force:
        print(f"[skip] {out.name} already exists (use --force to overwrite)")
        return

    flags = offloader_flags(args)

    # vLLM's V1 engine spawns a worker subprocess; without --trace-fork-before-exec
    # nsys misses CUDA activity in the child.
    cmd = [
        "nsys", "profile",
        "-o", str(out.with_suffix("")),  # nsys appends .nsys-rep
        "--trace=cuda,nvtx,osrt",
        "--trace-fork-before-exec=true",
        "--force-overwrite=true",
        "vllm", "bench", "latency",
        "--model", MODEL,
        "--dtype", DTYPE,
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--batch-size", str(args.batch_size),
        "--num-iters-warmup", str(args.num_iters_warmup),
        "--num-iters", str(args.num_iters),
        "--enforce-eager",
        *flags,
    ]
    env = {**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}

    print(f"[probe] {name}")
    print(f"        workload: input={args.input_len} output={args.output_len} "
          f"B={args.batch_size} warmup={args.num_iters_warmup} iters={args.num_iters}")
    print(f"        flags: {' '.join(flags) or '(none)'}")
    print(f"        out: {out}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, check=False)
    dur = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.exit(f"[FAIL] rc={proc.returncode} ({dur:.1f}s)")
    print(f"[ok]  ({dur:.1f}s) → {out}")


if __name__ == "__main__":
    main()
