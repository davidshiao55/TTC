#!/usr/bin/env python3
"""Evaluate Phase 1 weight-offload CPU thread policies.

This benchmark compares scalar `--cots-cpu-num-threads` policies against
simple per-bucket policies. It is intentionally policy-level: each cell is a
fresh vLLM latency process so the result includes the same load/runtime path
that the Planner will model.

Run from `/TTC/FastTTS-thesis` in the thesis environment, for example:

    python /TTC/David/Benchmarks/phase1_analysis/bench_weight_thread_policy.py \
        --exp --batches 1 16 --f-values 0.02 0.05 \
        --policies scalar4 scalar16 scalar24 workscore
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.75
WARMUP_ITERS = 2
BENCH_ITERS = 3

# Default graph capture buckets observed in current vLLM logs.
CAPTURE_BUCKETS = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
    216,
    224,
    232,
    240,
    248,
    256,
    272,
    288,
    304,
    320,
    336,
    352,
    368,
    384,
    400,
    416,
    432,
    448,
    464,
    480,
    496,
    512,
)


@dataclass(frozen=True)
class Cell:
    batch: int
    f_cpu_store: float | None
    policy: str


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1_analysis/thread_policy_weight") / stamp


def float_tag(value: float | None) -> str:
    if value is None:
        return "none"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def policy_flags(policy: str, f_cpu_store: float) -> list[str]:
    if policy.startswith("scalar"):
        threads = int(policy.removeprefix("scalar"))
        return ["--cots-cpu-num-threads", str(threads)]

    if policy == "bucket_ramp":
        mapping = {
            bucket: 4 if bucket <= 4 else 8 if bucket <= 16 else 16 if bucket <= 64 else 24
            for bucket in CAPTURE_BUCKETS
        }
        return [
            "--cots-cpu-num-threads",
            "24",
            "--cots-cpu-num-threads-by-bucket",
            json.dumps(mapping),
        ]

    if policy == "workscore":
        mapping: dict[int, int] = {}
        for bucket in CAPTURE_BUCKETS:
            score = bucket * f_cpu_store
            if score <= 0.08:
                threads = 4
            elif score <= 0.24:
                threads = 16
            else:
                threads = 24
            mapping[bucket] = threads
        return [
            "--cots-cpu-num-threads",
            "24",
            "--cots-cpu-num-threads-by-bucket",
            json.dumps(mapping),
        ]

    raise ValueError(f"unknown policy: {policy}")


def cell_paths(results_dir: Path, cell: Cell) -> tuple[Path, Path]:
    ftag = float_tag(cell.f_cpu_store)
    stem = f"b{cell.batch}_f{ftag}_{cell.policy}"
    return results_dir / f"{stem}.json", results_dir / f"{stem}.log"


def build_command(args: argparse.Namespace, cell: Cell, out_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--batch-size",
        str(cell.batch),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--num-iters-warmup",
        str(args.num_iters_warmup),
        "--num-iters",
        str(args.num_iters),
        "--output-json",
        str(out_json),
    ]
    if args.mode == "eager":
        cmd.append("--enforce-eager")
    if cell.f_cpu_store is not None:
        cmd += [
            "--offload-backend",
            "cots",
            "--cots-f-cpu-store",
            str(cell.f_cpu_store),
            "--cots-f-prefetch",
            "0.0",
            "--cots-cpu-runner",
            "native",
        ]
        cmd += policy_flags(cell.policy, cell.f_cpu_store)
    return cmd


def run_cell(args: argparse.Namespace, cell: Cell) -> int:
    out_json, log_path = cell_paths(args.results_dir, cell)
    if out_json.exists() and not args.force:
        print(f"[skip] {out_json}")
        return 0
    cmd = build_command(args, cell, out_json)
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        print(
            f"[fail] b={cell.batch} f={cell.f_cpu_store} policy={cell.policy} "
            f"rc={proc.returncode} elapsed={elapsed:.0f}s\n{tail}"
        )
        return proc.returncode
    latency = json.loads(out_json.read_text()).get("avg_latency")
    print(
        f"[ok] b={cell.batch} f={cell.f_cpu_store} policy={cell.policy} "
        f"latency={latency:.4f}s elapsed={elapsed:.0f}s"
    )
    return 0


def collect_result(results_dir: Path, cell: Cell) -> dict[str, Any] | None:
    out_json, _ = cell_paths(results_dir, cell)
    if not out_json.exists():
        return None
    data = json.loads(out_json.read_text())
    latencies = data.get("latencies") or []
    return {
        "batch": cell.batch,
        "f_cpu_store": cell.f_cpu_store,
        "policy": cell.policy,
        "avg_latency_s": data.get("avg_latency"),
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "latencies": latencies,
    }


def summarize(args: argparse.Namespace, cells: list[Cell]) -> dict[str, Any]:
    rows = [r for cell in cells if (r := collect_result(args.results_dir, cell))]
    baseline_by_batch = {
        r["batch"]: r["avg_latency_s"]
        for r in rows
        if r["f_cpu_store"] is None and r["avg_latency_s"] is not None
    }
    for r in rows:
        base = baseline_by_batch.get(r["batch"])
        if base and r["avg_latency_s"]:
            r["slowdown_vs_none"] = r["avg_latency_s"] / base
    best_by_case: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["f_cpu_store"] is None:
            continue
        key = f"b{r['batch']}_f{float_tag(r['f_cpu_store'])}"
        cur = best_by_case.get(key)
        if cur is None or r["avg_latency_s"] < cur["avg_latency_s"]:
            best_by_case[key] = r
    return {
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "mode": args.mode,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batches": args.batches,
            "f_values": args.f_values,
            "policies": args.policies,
            "num_iters_warmup": args.num_iters_warmup,
            "num_iters": args.num_iters,
        },
        "rows": rows,
        "best_by_case": best_by_case,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Weight Thread Policy Results",
        "",
        f"Model: `{summary['config']['model']}`; mode: `{summary['config']['mode']}`; "
        f"input/output: `{summary['config']['input_len']}/{summary['config']['output_len']}`.",
        "",
        "## Rows",
        "",
        "| B | f | policy | avg s | slowdown |",
        "|---:|---:|---|---:|---:|",
    ]
    for row in summary["rows"]:
        fval = "-" if row["f_cpu_store"] is None else f"{row['f_cpu_store']:.4f}"
        slow = row.get("slowdown_vs_none")
        slow_text = f"{slow:.3f}" if slow else "-"
        lines.append(
            f"| {row['batch']} | {fval} | `{row['policy']}` | "
            f"{row['avg_latency_s']:.4f} | {slow_text} |"
        )
    lines += [
        "",
        "## Best By Case",
        "",
        "| case | policy | avg s | slowdown |",
        "|---|---|---:|---:|",
    ]
    for key, row in summary["best_by_case"].items():
        slow = row.get("slowdown_vs_none")
        slow_text = f"{slow:.3f}" if slow else "-"
        lines.append(
            f"| `{key}` | `{row['policy']}` | {row['avg_latency_s']:.4f} | "
            f"{slow_text} |"
        )
    path.write_text("\n".join(lines) + "\n")


def build_cells(args: argparse.Namespace) -> list[Cell]:
    cells = [Cell(batch=batch, f_cpu_store=None, policy="none") for batch in args.batches]
    for batch in args.batches:
        for f_cpu_store in args.f_values:
            for policy in args.policies:
                cells.append(Cell(batch=batch, f_cpu_store=f_cpu_store, policy=policy))
    return cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--input-len", type=int, default=INPUT_LEN)
    parser.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--f-values", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["scalar4", "scalar16", "scalar24", "workscore"],
    )
    parser.add_argument("--num-iters-warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--num-iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--exp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cells = build_cells(args)
    print(
        f"[setup] results={args.results_dir} batches={args.batches} "
        f"f={args.f_values} policies={args.policies} exp={args.exp}"
    )
    exit_code = 0
    if args.exp:
        for cell in cells:
            rc = run_cell(args, cell)
            if rc != 0:
                exit_code = rc
                if not args.keep_going:
                    break

    summary = summarize(args, cells)
    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, args.results_dir / "summary.md")
    print(f"[summary] wrote {summary_path}")
    print(f"[summary] wrote {args.results_dir / 'summary.md'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
