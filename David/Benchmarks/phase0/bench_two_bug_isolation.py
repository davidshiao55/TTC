#!/usr/bin/env python3
"""Phase 0 §0.10.3 — confirm CE0 contention and sync_prev_onload are
TWO SEPARATE bugs, not one bug seen via two surfaces.

Strategy: structurally isolate each mechanism and observe whether the
expected sync exposure persists.

  Mechanism 1 — eager-mode CE0 contention with input prep H2Ds.
    Killed structurally by: VLLM_INPUT_PREP_UVA=1 (input prep H2Ds
    bypass CE0 via SM-issued reads on UVA-mapped pinned memory).
  Mechanism 2 — graph-mode sync_prev_onload drain of copy_stream.
    Killed structurally by: --enforce-eager (sync_prev_onload code
    path is never invoked in eager mode).

The 2x2 cells per G:

  α (mech 1 only):  --enforce-eager,    no UVA  → only mech 1 can fire
  β (mech 2 only):  no --enforce-eager, UVA on  → only mech 2 can fire
  γ (both active):  no --enforce-eager, no UVA  → both can fire
  δ (neither):      --enforce-eager,    UVA on  → control / baseline

If γ ≈ α + β: independent additive bugs.
If γ ≈ max(α, β): same bug seen through two surfaces.

REQUIRES the temporary `VLLM_INPUT_PREP_UVA` shim in `vllm/v1/utils.py`
(re-applied for this experiment, reverted afterwards). All cells use the
factory `prefetch` backend so the bug is present.

Outputs to David/Benchmarks/phase0/results/0.10_two_bug_isolation/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

PHASE0_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE0_DIR / "results" / "0.10_two_bug_isolation"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 32
BATCH = 1
WARMUP_ITERS = 3
BENCH_ITERS = 2

# (cell label, eager?, uva_input_prep?)
CELLS = [
    ("alpha_eager_noUVA", True,  False),  # mech 1 only
    ("beta_graph_UVA",    False, True),   # mech 2 only
    ("gamma_graph_noUVA", False, False),  # both
    ("delta_eager_UVA",   True,  True),   # neither
]


def run_cell(G: int, label: str, eager: bool, uva: bool) -> Path:
    stem = f"g{G}_{label}"
    out_rep = RESULTS_DIR / f"{stem}.nsys-rep"
    out_log = RESULTS_DIR / f"{stem}.log"
    if out_rep.exists():
        print(f"  [skip] {stem} (cached)")
        return out_rep

    cmd = [
        "nsys", "profile",
        "-o", str(out_rep.with_suffix("")),
        "--trace=cuda,nvtx,osrt",
        "--trace-fork-before-exec=true",
        "--force-overwrite=true",
        "vllm", "bench", "latency",
        "--model", MODEL, "--dtype", DTYPE,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--batch-size", str(BATCH),
        "--num-iters-warmup", str(WARMUP_ITERS),
        "--num-iters", str(BENCH_ITERS),
        "--offload-group-size", str(G),
        "--offload-num-in-group", "1",
        "--offload-prefetch-step", "1",
    ]
    if eager:
        cmd.append("--enforce-eager")

    env = {**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    if uva:
        env["VLLM_INPUT_PREP_UVA"] = "1"

    t0 = time.perf_counter()
    with open(out_log, "w") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
            print(f"  [TIMEOUT] {stem}")
            return out_rep
    dur = time.perf_counter() - t0
    if rc != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-10:])
        print(f"  [FAIL] {stem} rc={rc} ({dur:.1f}s)\n        {tail}")
    else:
        print(f"  [ok]  {stem} ({dur:.1f}s)")
    return out_rep


def export_sqlite(rep: Path) -> Path:
    sqlite = rep.with_suffix(".sqlite")
    subprocess.run(
        ["nsys", "export", "--type", "sqlite", "--force-overwrite=true", str(rep)],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sqlite


def parse_avg_lat(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    m = re.search(r"Avg latency:\s*([\d.]+)\s*seconds", log_path.read_text())
    return float(m.group(1)) if m else None


def sync_breakdown(sql: Path, threshold_ms: float = 1.0) -> dict:
    if not sql.exists():
        return {}
    out: dict = {}
    con = sqlite3.connect(sql)
    cur = con.cursor()
    for st, n, avg, sm, mx in cur.execute(
        "SELECT syncType, COUNT(*), AVG(end-start), SUM(end-start), MAX(end-start) "
        "FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
        "WHERE (end-start) > ? GROUP BY syncType",
        (int(threshold_ms * 1e6),),
    ).fetchall():
        out[f"syncType{st}"] = {
            "count": n, "avg_ms": avg / 1e6,
            "sum_s": sm / 1e9, "max_ms": mx / 1e6,
        }
    con.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gs", type=int, nargs="+", default=[28],
                    help="G values to sweep (default: 28 only)")
    ap.add_argument("--threshold-ms", type=float, default=1.0,
                    help="Min sync duration to count (default: 1ms)")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip running cells; only re-parse existing traces")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] G ∈ {args.gs}, {len(CELLS)} cells per G, "
          f"workload input={INPUT_LEN} output={OUTPUT_LEN} B={BATCH}")
    print(f"[setup] sync threshold = {args.threshold_ms} ms")

    if not args.report_only:
        for G in args.gs:
            print(f"\n=== G={G} ===")
            for label, eager, uva in CELLS:
                run_cell(G, label, eager, uva)

    summary: dict = {"model": MODEL, "config": {
        "input_len": INPUT_LEN, "output_len": OUTPUT_LEN, "batch_size": BATCH,
        "warmup_iters": WARMUP_ITERS, "bench_iters": BENCH_ITERS,
        "threshold_ms": args.threshold_ms,
    }, "cells": []}

    print("\n" + "=" * 110)
    print(f"{'cell':<28} {'avg_lat (s)':>11} "
          f"{'syncT1 ms (n)':>16} {'syncT1 sum_s':>13} "
          f"{'syncT2 ms (n)':>16} {'syncT2 sum_s':>13}")
    print("-" * 110)
    for G in args.gs:
        for label, eager, uva in CELLS:
            stem = f"g{G}_{label}"
            rep = RESULTS_DIR / f"{stem}.nsys-rep"
            if not rep.exists():
                print(f"{stem:<28}  (missing trace)")
                continue
            sql = rep.with_suffix(".sqlite")
            export_sqlite(rep)  # always force re-export to refresh
            sb = sync_breakdown(sql, args.threshold_ms)
            avg_lat = parse_avg_lat(RESULTS_DIR / f"{stem}.log")
            avg_s = f"{avg_lat:.4f}" if avg_lat else "—"
            t1 = sb.get("syncType1", {})
            t2 = sb.get("syncType2", {})
            t1_avg = (f"{t1['avg_ms']:.2f} ({t1['count']})"
                      if t1 else "—")
            t1_sum = f"{t1['sum_s']:.3f}" if t1 else "—"
            t2_avg = (f"{t2['avg_ms']:.2f} ({t2['count']})"
                      if t2 else "—")
            t2_sum = f"{t2['sum_s']:.3f}" if t2 else "—"
            print(f"{stem:<28} {avg_s:>11} {t1_avg:>16} {t1_sum:>13} "
                  f"{t2_avg:>16} {t2_sum:>13}")
            summary["cells"].append({
                "G": G, "label": label, "eager": eager, "uva_input_prep": uva,
                "avg_latency_s": avg_lat, "sync_breakdown": sb,
            })
    print("=" * 110)
    print("\nKey: syncType1 = cudaEventSynchronize (host-side wait, used by")
    print("       prepare_inputs_event.synchronize() — mech 1's surface)")
    print("     syncType2 = cudaStreamWaitEvent (GPU stream wait, used by")
    print("       sync_prev_onload's wait_stream — mech 2's surface)")

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
