#!/usr/bin/env python3
"""Sweep cots arms × thread counts at decode-heavy (input=8/output=128).

Companion to ``bench_cots_thread_sweep_prefill.py``. Sweeps THREADS × ARMS
× BATCHES with skip-if-cached. Outputs ``results/thread_sweep_decode/<arm>
_b<B>_t<T>.json``. See `phase1a_findings.md §1.13b` for the synthesis.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
REPO = Path("/TTC")
VLLM_ROOT = REPO / "vllm"
OUT = PHASE1_DIR / "results" / "thread_sweep_decode"
OUT.mkdir(exist_ok=True)
MODEL = "Qwen/Qwen2.5-7B-Instruct"

ARMS = [("cots_005", 0.05), ("cots_009", 0.09),
        ("cots_022", 0.22), ("cots_050", 0.50)]
BATCHES = [1, 4, 16]
THREADS = [4, 8, 16, 24]


def run(arm, f, batch, threads):
    out = OUT / f"{arm}_b{batch}_t{threads}.json"
    log = OUT / f"{arm}_b{batch}_t{threads}.log"
    if out.exists():
        return json.loads(out.read_text()).get("avg_latency")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", MODEL, "--dtype", "bfloat16",
        "--input-len", "8", "--output-len", "128",
        "--batch-size", str(batch),
        "--num-iters-warmup", "2", "--num-iters", "3",
        "--enforce-eager",
        "--output-json", str(out),
        "--offload-backend", "cots",
        "--cots-f-cpu-store", str(f),
        "--cots-cpu-num-threads", str(threads),
    ]
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = f"{VLLM_ROOT}:{REPO}:" + env.get("PYTHONPATH", "")
    t0 = time.perf_counter()
    with log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-8:])
        print(f"[FAIL] {arm} b{batch} t{threads} rc={proc.returncode} t={elapsed:.0f}s\n{tail}")
        return None
    avg = json.loads(out.read_text()).get("avg_latency")
    print(f"[ok]  {arm} b{batch} t{threads}: avg={avg:.4f}s elapsed={elapsed:.0f}s")
    return avg


for arm, f in ARMS:
    for b in BATCHES:
        for t in THREADS:
            run(arm, f, b, t)

print("\n=== sweep summary ===")
print(f"{'arm':<10} {'B':>3}  " + "  ".join(f"{f't={t}':>9}" for t in THREADS))
for arm, _ in ARMS:
    for b in BATCHES:
        cells = []
        for t in THREADS:
            p = OUT / f"{arm}_b{b}_t{t}.json"
            if p.exists():
                v = json.loads(p.read_text()).get("avg_latency")
                cells.append(f"{v:>9.3f}" if v is not None else f"{'—':>9}")
            else:
                cells.append(f"{'—':>9}")
        print(f"{arm:<10} {b:>3}  " + "  ".join(cells))
