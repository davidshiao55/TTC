#!/usr/bin/env python3
"""Sweep COTS CPU thread count on prefill-heavy vLLM latency workloads.

This complements ``results/thread_sweep_decode``.  The goal is to test
whether one fixed OpenMP/PyTorch CPU thread count is close to best across
decode-heavy and prefill-heavy COTS workloads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PHASE1_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE1_DIR.parents[2]
VLLM_ROOT = REPO_ROOT / "vllm"
RESULTS_DIR = PHASE1_DIR / "results" / "thread_sweep_prefill"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
F_CPU_STORE = 0.09


def run_cell(
    threads: int,
    batch_size: int,
    input_len: int,
    output_len: int,
    warmup_iters: int,
    bench_iters: int,
    max_model_len: int | None,
    max_num_batched_tokens: int | None,
    force: bool,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"omp{threads}_b{batch_size}_in{input_len}_out{output_len}"
    out_json = RESULTS_DIR / f"{stem}.json"
    out_log = RESULTS_DIR / f"{stem}.log"
    if out_json.exists() and not force:
        print(f"[skip] {stem}")
        return out_json

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model",
        MODEL,
        "--dtype",
        DTYPE,
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--batch-size",
        str(batch_size),
        "--num-iters-warmup",
        str(warmup_iters),
        "--num-iters",
        str(bench_iters),
        "--enforce-eager",
        "--output-json",
        str(out_json),
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        str(F_CPU_STORE),
    ]
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(max_num_batched_tokens)])

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    # Avoid tokenizer pool noise in latency runs.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    existing_pythonpath = env.get("PYTHONPATH")
    local_paths = [str(VLLM_ROOT), str(REPO_ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(
        local_paths + ([existing_pythonpath] if existing_pythonpath else [])
    )

    t0 = time.perf_counter()
    with out_log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(out_log.read_text(errors="replace").splitlines()[-20:])
        raise RuntimeError(f"{stem} failed rc={proc.returncode} after {elapsed:.1f}s\n{tail}")

    avg = json.loads(out_json.read_text())["avg_latency"]
    print(f"[ok]   {stem} avg={avg:.4f}s elapsed={elapsed:.1f}s")
    return out_json


def parse_avg(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["avg_latency"])
    except (OSError, KeyError, json.JSONDecodeError, TypeError, ValueError):
        return None


def summarize(threads: list[int], batches: list[int], input_len: int, output_len: int) -> dict:
    by_batch: dict[str, dict[str, object]] = {}
    for batch in batches:
        vals: dict[int, float] = {}
        for th in threads:
            path = RESULTS_DIR / f"omp{th}_b{batch}_in{input_len}_out{output_len}.json"
            avg = parse_avg(path)
            if avg is not None:
                vals[th] = avg
        best = min(vals.values()) if vals else None
        by_thread = {
            str(th): None if th not in vals else {
                "avg_latency_s": round(vals[th], 4),
                "slowdown_vs_best": None if best is None else round(vals[th] / best, 3),
            }
            for th in threads
        }
        by_batch[str(batch)] = {
            "best_avg_latency_s": None if best is None else round(best, 4),
            "by_thread": by_thread,
        }

    out = {
        "model": MODEL,
        "dtype": DTYPE,
        "offload_backend": "cots",
        "cots_f_cpu_store": F_CPU_STORE,
        "input_len": input_len,
        "output_len": output_len,
        "threads": threads,
        "batches": batches,
        "summary": by_batch,
    }
    (RESULTS_DIR / f"summary_in{input_len}_out{output_len}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, nargs="+", default=[4, 8, 16, 24])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--bench-iters", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=384)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for batch in args.batches:
        for th in args.threads:
            run_cell(
                threads=th,
                batch_size=batch,
                input_len=args.input_len,
                output_len=args.output_len,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                force=args.force,
            )

    summary = summarize(args.threads, args.batches, args.input_len, args.output_len)
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
