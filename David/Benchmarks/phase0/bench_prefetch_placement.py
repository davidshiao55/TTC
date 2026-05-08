#!/usr/bin/env python3
"""Phase 0 §0.10.3 — placement-sensitivity sweep (Codex finding 2).

Tests whether shifting the offload picker so the LAST model layer is
GPU-resident (and the last *offloaded* layer has tail compute behind it)
gives the unfixed prefetch path partial relief from the wraparound H2D.

Workload: G=4 N=1 (7 offloaded layers, 3.04 GiB), Qwen2.5-7B BF16,
input=8 output=32 B=1, --enforce-eager. Five cells:

    offset=0 unfixed   last offloaded = layer 27 (last model layer; 0 tail)
    offset=1 unfixed   last offloaded = layer 24 (3 tail layers: 25,26,27)
    offset=2 unfixed   last offloaded = layer 25 (2 tail layers: 26,27)
    offset=3 unfixed   last offloaded = layer 26 (1 tail layer: 27)
    offset=0 defer     validation cell — confirms defer fix delivers full
                       hiding from the canonical picker, replicating §0.10.3
                       G=4 fixed numbers within run-to-run noise.

REQUIRES the temporary `--offload-offset` CLI flag from the Experiment B
instrumentation (added to vllm/config/offload.py + arg_utils.py + base.py +
prefetch.py). After the experiment writeup lands, the vllm-side patch is
reverted; re-running this script then needs the patch re-applied. See
`David/Docs/phase0_findings.md` §0.10.3 placement-sensitivity sub-finding
for the original numbers.

Outputs to ``David/Benchmarks/phase0/results/0.10_placement_sweep/``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PHASE0_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE0_DIR / "results" / "0.10_placement_sweep"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 32
BATCH = 1
WARMUP_ITERS = 2
BENCH_ITERS = 3

CELLS = [
    # (label,            offset, defer, last_offloaded_layer, tail_layers)
    ("offset0_unfixed",  0, False, 27, 0),
    ("offset1_unfixed",  1, False, 24, 3),
    ("offset2_unfixed",  2, False, 25, 2),
    ("offset3_unfixed",  3, False, 26, 1),
    ("offset0_fixed",    0, True,  27, 0),  # validation cell
]


def run_cell(label: str, offset: int, defer: bool) -> Path:
    out_json = RESULTS_DIR / f"{label}.json"
    out_log = RESULTS_DIR / f"{label}.log"
    if out_json.exists():
        print(f"  [skip] {label} (cached)")
        return out_json
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", MODEL, "--dtype", DTYPE,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--batch-size", str(BATCH),
        "--num-iters-warmup", str(WARMUP_ITERS),
        "--num-iters", str(BENCH_ITERS),
        "--enforce-eager",
        "--offload-group-size", "4",
        "--offload-num-in-group", "1",
        "--offload-prefetch-step", "1",
        "--offload-offset", str(offset),
        "--prefetch-defer-wraparound" if defer else "--no-prefetch-defer-wraparound",
        "--output-json", str(out_json),
    ]
    t0 = time.perf_counter()
    with open(out_log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    dur = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-15:])
        print(f"  [FAIL] {label} rc={proc.returncode} ({dur:.1f}s)\n        {tail}")
    else:
        avg = json.loads(out_json.read_text()).get("avg_latency")
        print(f"  [ok]  {label}: avg={avg:.4f}s ({dur:.1f}s)")
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[setup] G=4 N=1 K=1, input={INPUT_LEN} output={OUTPUT_LEN} B={BATCH}, "
          f"{len(CELLS)} cells")
    for label, offset, defer, *_ in CELLS:
        run_cell(label, offset, defer)

    print("\n" + "=" * 70)
    print(f"PLACEMENT SWEEP — Qwen2.5-7B BF16, input={INPUT_LEN} output={OUTPUT_LEN} B={BATCH}")
    print("=" * 70)
    print(f"{'cell':<22} {'offset':>6} {'defer':>5} {'last_off':>9} "
          f"{'tail':>4} {'avg_lat (s)':>12}")
    print("-" * 70)
    none_path = RESULTS_DIR / "none_baseline.json"
    summary: dict = {"model": MODEL, "config": {
        "input_len": INPUT_LEN, "output_len": OUTPUT_LEN, "batch_size": BATCH,
        "warmup_iters": WARMUP_ITERS, "bench_iters": BENCH_ITERS,
        "G": 4, "N": 1, "K": 1,
    }, "cells": {}}
    for label, offset, defer, last_off, tail in CELLS:
        p = RESULTS_DIR / f"{label}.json"
        avg = json.loads(p.read_text()).get("avg_latency") if p.exists() else None
        avg_s = f"{avg:.4f}" if avg is not None else "—"
        defer_s = "True" if defer else "False"
        print(f"{label:<22} {offset:>6} {defer_s:>5} {last_off:>9} {tail:>4} {avg_s:>12}")
        summary["cells"][label] = {
            "offset": offset, "defer_wraparound": defer,
            "last_offloaded_layer": last_off, "tail_layers": tail,
            "avg_latency_s": avg,
        }
    print("=" * 70)

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
