#!/usr/bin/env python3
"""Run a small Phase 1c capture-gap workload grid for Qwen2.5-7B."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
F_CPU_STORE = 0.05
CPU_THREADS = 16
DEFAULT_ARMS = [
    "native_eager_real",
    "capture_wait_kernel_real",
    "cots_default_real",
    "piecewise_cots_split_wait_kernel_real",
]


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1c_capture_gap/grid") / stamp


def cell_dir(root: Path, batch: int, input_len: int, output_len: int) -> Path:
    return root / f"b{batch}_in{input_len}_out{output_len}"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--f-cpu-store", type=float, default=F_CPU_STORE)
    parser.add_argument("--cpu-threads", type=int, default=CPU_THREADS)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--input-lens", type=int, nargs="+", default=[8, 128, 512])
    parser.add_argument("--output-lens", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--num-iters-warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    root = args.results_dir
    root.mkdir(parents=True, exist_ok=True)
    harness = Path(__file__).with_name("bench_capture_gap_qwen.py")

    grid_records = []
    t0 = time.perf_counter()
    for batch in args.batch_sizes:
        for input_len in args.input_lens:
            for output_len in args.output_lens:
                out_dir = cell_dir(root, batch, input_len, output_len)
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(harness),
                    "--results-dir",
                    str(out_dir),
                    "--model",
                    args.model,
                    "--dtype",
                    args.dtype,
                    "--input-len",
                    str(input_len),
                    "--output-len",
                    str(output_len),
                    "--batch-size",
                    str(batch),
                    "--f-cpu-store",
                    str(args.f_cpu_store),
                    "--cpu-threads",
                    str(args.cpu_threads),
                    "--repeat",
                    str(args.repeat),
                    "--num-iters",
                    str(args.num_iters),
                    "--num-iters-warmup",
                    str(args.num_iters_warmup),
                    "--only-arms",
                    *args.arms,
                ]
                if args.force:
                    cmd.append("--force")

                print(
                    f"\n[cell] batch={batch} input={input_len} "
                    f"output={output_len} -> {out_dir}"
                )
                cell_t0 = time.perf_counter()
                proc = subprocess.run(cmd, check=False)
                elapsed = time.perf_counter() - cell_t0
                summary_path = out_dir / "summary.json"
                if proc.returncode != 0 or not summary_path.exists():
                    record = {
                        "batch_size": batch,
                        "input_len": input_len,
                        "output_len": output_len,
                        "elapsed_s": elapsed,
                        "returncode": proc.returncode,
                        "error": "cell failed or summary missing",
                    }
                    grid_records.append(record)
                    (root / "grid_summary.json").write_text(
                        json.dumps({"records": grid_records}, indent=2)
                    )
                    if not args.keep_going:
                        return proc.returncode or 1
                    continue

                summary = load_summary(summary_path)
                rows = summary["rows"]
                means = {name: rows[name]["mean"] for name in rows}
                eager = means.get("native_eager_real")
                default = means.get("cots_default_real")
                split = means.get("piecewise_cots_split_wait_kernel_real")
                capture = means.get("capture_wait_kernel_real")
                record = {
                    "batch_size": batch,
                    "input_len": input_len,
                    "output_len": output_len,
                    "elapsed_s": elapsed,
                    "summary_path": str(summary_path),
                    "means": means,
                    "delta_split_vs_eager_ms": (
                        None if eager is None or split is None else (split - eager) * 1000
                    ),
                    "delta_default_vs_eager_ms": (
                        None
                        if eager is None or default is None
                        else (default - eager) * 1000
                    ),
                    "delta_capture_vs_eager_ms": (
                        None
                        if eager is None or capture is None
                        else (capture - eager) * 1000
                    ),
                    "delta_default_vs_capture_ms": (
                        None
                        if default is None or capture is None
                        else (default - capture) * 1000
                    ),
                    "delta_split_vs_capture_ms": (
                        None
                        if split is None or capture is None
                        else (split - capture) * 1000
                    ),
                    "delta_split_vs_default_ms": (
                        None
                        if split is None or default is None
                        else (split - default) * 1000
                    ),
                }
                grid_records.append(record)
                (root / "grid_summary.json").write_text(
                    json.dumps({"records": grid_records}, indent=2)
                )

    total_elapsed = time.perf_counter() - t0
    split_wins_eager = 0
    split_wins_capture = 0
    default_wins_eager = 0
    default_wins_capture = 0
    valid = 0
    for record in grid_records:
        means = record.get("means")
        if not means:
            continue
        valid += 1
        eager = means.get("native_eager_real")
        default = means.get("cots_default_real")
        split = means.get("piecewise_cots_split_wait_kernel_real")
        capture = means.get("capture_wait_kernel_real")
        if eager is not None and default is not None and default < eager:
            default_wins_eager += 1
        if capture is not None and default is not None and default < capture:
            default_wins_capture += 1
        if eager is not None and split is not None and split < eager:
            split_wins_eager += 1
        if capture is not None and split is not None and split < capture:
            split_wins_capture += 1

    grid_summary = {
        "model": args.model,
        "dtype": args.dtype,
        "f_cpu_store": args.f_cpu_store,
        "cpu_threads": args.cpu_threads,
        "num_iters": args.num_iters,
        "num_iters_warmup": args.num_iters_warmup,
        "repeat": args.repeat,
        "arms": args.arms,
        "elapsed_s": total_elapsed,
        "num_cells": len(grid_records),
        "num_valid_cells": valid,
        "default_wins_eager_cells": default_wins_eager,
        "default_wins_capture_cells": default_wins_capture,
        "split_wins_eager_cells": split_wins_eager,
        "split_wins_capture_cells": split_wins_capture,
        "records": grid_records,
    }
    summary_path = root / "grid_summary.json"
    summary_path.write_text(json.dumps(grid_summary, indent=2))

    print("\n" + "=" * 110)
    print(
        f"{'B':>2} {'in':>5} {'out':>5} "
        f"{'eager':>9} {'full_cap':>9} {'default':>9} {'split':>9} "
        f"{'def-eager(ms)':>14} {'def-full(ms)':>13} "
        f"{'split-def(ms)':>13}"
    )
    print("-" * 110)
    for record in grid_records:
        means = record.get("means") or {}
        print(
            f"{record['batch_size']:>2} {record['input_len']:>5} "
            f"{record['output_len']:>5} "
            f"{means.get('native_eager_real', float('nan')):>9.4f} "
            f"{means.get('capture_wait_kernel_real', float('nan')):>9.4f} "
            f"{means.get('cots_default_real', float('nan')):>9.4f} "
            f"{means.get('piecewise_cots_split_wait_kernel_real', float('nan')):>9.4f} "
            f"{record.get('delta_default_vs_eager_ms', float('nan')):>14.1f} "
            f"{record.get('delta_default_vs_capture_ms', float('nan')):>13.1f} "
            f"{record.get('delta_split_vs_default_ms', float('nan')):>13.1f}"
        )
    print("=" * 110)
    print(
        f"default wins vs eager: {default_wins_eager}/{valid}; "
        f"default wins vs full capture: {default_wins_capture}/{valid}; "
        f"explicit split wins vs eager: {split_wins_eager}/{valid}; "
        f"explicit split wins vs full capture: {split_wins_capture}/{valid}"
    )
    print(f"[summary] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
