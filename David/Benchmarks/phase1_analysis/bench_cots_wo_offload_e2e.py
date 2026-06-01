#!/usr/bin/env python3
"""E2E COTS WO module-selection benchmark.

Compares the production COTS weight path with its default module set
(`qkv,mlp`) against an opt-in `qkv,mlp,wo` run. This isolates the policy
question for the Planner default: does adding WO to the COTS CPU-compute path
pay for itself, or should WO remain GPU-resident unless forced by memory?

Run from `/TTC/FastTTS-thesis` in the thesis environment, for example:

    python /TTC/David/Benchmarks/phase1_analysis/bench_cots_wo_offload_e2e.py \
        --exp --batches 1 16 --output-len 64 --f-cpu-store 0.05
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
OUTPUT_LEN = 64
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.75
WARMUP_ITERS = 1
BENCH_ITERS = 2
REPEATS = 1

# Qwen2.5-7B BF16 per-layer arithmetic.
HIDDEN = 3584
HEAD_DIM = 128
QKV_OUT = (28 + 2 * 4) * 128
INTERMEDIATE = 18944
NUM_LAYERS = 28
BF16_BYTES = 2
NON_WO_MODULE_BYTES_PER_LAYER = (
    HIDDEN * QKV_OUT * BF16_BYTES
    + HIDDEN * 2 * INTERMEDIATE * BF16_BYTES
    + INTERMEDIATE * HIDDEN * BF16_BYTES
)
WO_BYTES_PER_LAYER = HIDDEN * HIDDEN * BF16_BYTES


@dataclass(frozen=True)
class Arm:
    name: str
    modules: tuple[str, ...]
    includes_wo: bool


ARMS = (
    Arm("no_wo", ("qkv", "mlp"), includes_wo=False),
    Arm("with_wo", ("qkv", "mlp", "wo"), includes_wo=True),
)


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1_analysis/cots_wo_offload_e2e") / stamp


def gib(nbytes: int | float) -> float:
    return float(nbytes) / (1024**3)


def snap_channels(requested: float, limit: int, granularity: int) -> int:
    if requested <= 0:
        return 0
    if requested >= limit:
        return limit
    snapped = int(((requested + granularity / 2) // granularity) * granularity)
    return min(snapped, limit)


def snapped_wo_bytes(f_cpu_store: float) -> int:
    wo_rows = snap_channels(f_cpu_store * HIDDEN, HIDDEN, HEAD_DIM)
    return NUM_LAYERS * wo_rows * HIDDEN * BF16_BYTES


def cell_stem(arm: Arm, batch: int, repeat: int) -> str:
    modules = "-".join(arm.modules)
    return f"r{repeat:02d}_{arm.name}_{modules}_b{batch}"


def cell_paths(
    results_dir: Path,
    arm: Arm,
    batch: int,
    repeat: int,
) -> tuple[Path, Path]:
    stem = cell_stem(arm, batch, repeat)
    return results_dir / f"{stem}.json", results_dir / f"{stem}.log"


def build_command(
    args: argparse.Namespace,
    *,
    arm: Arm,
    batch: int,
    out_json: Path,
) -> list[str]:
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
        str(batch),
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
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        str(args.f_cpu_store),
        "--cots-f-prefetch",
        str(args.f_prefetch),
        "--cots-cpu-num-threads",
        str(args.cpu_num_threads),
        "--cots-weight-modules",
        *arm.modules,
    ]
    if args.mode == "eager":
        cmd.append("--enforce-eager")
    if args.cots_weight_capture_sync_mode is not None:
        cmd.extend(
            [
                "--cots-weight-capture-sync-mode",
                args.cots_weight_capture_sync_mode,
            ]
        )
    cmd.extend(args.extra_vllm_args)
    return cmd


def run_cell(
    args: argparse.Namespace,
    *,
    arm: Arm,
    batch: int,
    repeat: int,
) -> int:
    out_json, log_path = cell_paths(args.results_dir, arm, batch, repeat)
    if out_json.exists() and not args.force:
        print(f"[skip] {out_json}")
        return 0

    cmd = build_command(args, arm=arm, batch=batch, out_json=out_json)
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        print(
            f"[fail] arm={arm.name} b={batch} r={repeat} "
            f"rc={proc.returncode} elapsed={elapsed:.0f}s\n{tail}"
        )
        return proc.returncode

    latency = json.loads(out_json.read_text()).get("avg_latency")
    print(
        f"[ok] arm={arm.name} b={batch} r={repeat} "
        f"latency={latency:.4f}s elapsed={elapsed:.0f}s"
    )
    return 0


def collect_result(
    args: argparse.Namespace,
    arm: Arm,
    batch: int,
    repeat: int,
) -> dict[str, Any] | None:
    out_json, _ = cell_paths(args.results_dir, arm, batch, repeat)
    if not out_json.exists():
        return None
    data = json.loads(out_json.read_text())
    latencies = data.get("latencies") or []
    incremental_wo_bytes = 0
    wo_output_rows = 0
    if arm.includes_wo:
        wo_output_rows = snap_channels(args.f_cpu_store * HIDDEN, HIDDEN, HEAD_DIM)
        incremental_wo_bytes = snapped_wo_bytes(args.f_cpu_store)
    return {
        "batch": batch,
        "arm": arm.name,
        "modules": list(arm.modules),
        "includes_wo": arm.includes_wo,
        "repeat": repeat,
        "avg_latency_s": data.get("avg_latency"),
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "latencies": latencies,
        "cpu_stored_gib": gib(
            NUM_LAYERS * NON_WO_MODULE_BYTES_PER_LAYER * args.f_cpu_store
            + incremental_wo_bytes
        ),
        "incremental_wo_gib": gib(incremental_wo_bytes) if arm.includes_wo else 0.0,
        "wo_output_rows": wo_output_rows,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        row
        for arm in ARMS
        for batch in args.batches
        for repeat in range(args.repeats)
        if (row := collect_result(args, arm, batch, repeat))
    ]
    by_case: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault((row["batch"], row["arm"]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (batch, arm_name), items in sorted(by_case.items()):
        avgs = [r["avg_latency_s"] for r in items if r["avg_latency_s"] is not None]
        if not avgs:
            continue
        first = items[0]
        aggregates.append(
            {
                "batch": batch,
                "arm": arm_name,
                "modules": first["modules"],
                "includes_wo": first["includes_wo"],
                "repeats": len(avgs),
                "mean_latency_s": statistics.mean(avgs),
                "stdev_latency_s": statistics.stdev(avgs) if len(avgs) > 1 else 0.0,
                "cpu_stored_gib": first["cpu_stored_gib"],
                "incremental_wo_gib": first["incremental_wo_gib"],
                "wo_output_rows": first["wo_output_rows"],
            }
        )

    no_wo_by_batch = {
        row["batch"]: row for row in aggregates if row["arm"] == "no_wo"
    }
    comparisons: list[dict[str, Any]] = []
    for row in aggregates:
        if row["arm"] != "with_wo":
            continue
        base = no_wo_by_batch.get(row["batch"])
        if base is None:
            continue
        delta = row["mean_latency_s"] - base["mean_latency_s"]
        comparisons.append(
            {
                "batch": row["batch"],
                "no_wo_latency_s": base["mean_latency_s"],
                "with_wo_latency_s": row["mean_latency_s"],
                "delta_s": delta,
                "delta_pct": delta / base["mean_latency_s"] * 100.0,
                "incremental_wo_gib": row["incremental_wo_gib"],
            }
        )

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
            "f_cpu_store": args.f_cpu_store,
            "f_prefetch": args.f_prefetch,
            "cpu_num_threads": args.cpu_num_threads,
            "cots_weight_capture_sync_mode": args.cots_weight_capture_sync_mode,
            "num_iters_warmup": args.num_iters_warmup,
            "num_iters": args.num_iters,
            "repeats": args.repeats,
        },
        "rows": rows,
        "aggregates": aggregates,
        "comparisons": comparisons,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# COTS WO Offload E2E Results",
        "",
        f"Model: `{summary['config']['model']}`; "
        f"mode: `{summary['config']['mode']}`; "
        f"f_cpu_store/f_prefetch: `{summary['config']['f_cpu_store']}/"
        f"{summary['config']['f_prefetch']}`; "
        f"input/output: `{summary['config']['input_len']}/"
        f"{summary['config']['output_len']}`.",
        "",
        "## Comparison",
        "",
        "| B | no WO s | with WO s | delta | delta % | extra WO CPU GiB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['batch']} | "
            f"{row['no_wo_latency_s']:.4f} | {row['with_wo_latency_s']:.4f} | "
            f"{row['delta_s']:+.4f} | {row['delta_pct']:+.2f}% | "
            f"{row['incremental_wo_gib']:.3f} |"
        )

    lines += [
        "",
        "## Aggregate Rows",
        "",
        "| B | arm | modules | mean s | stdev s | CPU-stored GiB | WO rows |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        modules = ",".join(row["modules"])
        lines.append(
            f"| {row['batch']} | `{row['arm']}` | `{modules}` | "
            f"{row['mean_latency_s']:.4f} | {row['stdev_latency_s']:.4f} | "
            f"{row['cpu_stored_gib']:.3f} | {row['wo_output_rows']} |"
        )
    path.write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--f-cpu-store", type=float, default=0.05)
    parser.add_argument("--f-prefetch", type=float, default=0.0)
    parser.add_argument("--cpu-num-threads", type=int, default=24)
    parser.add_argument(
        "--cots-weight-capture-sync-mode",
        choices=("host_callback", "wait_kernel"),
        default=None,
    )
    parser.add_argument("--num-iters-warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--num-iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--exp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def write_outputs(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    summary_json = args.results_dir / "summary.json"
    summary_md = args.results_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, summary_md)
    print(f"[summary] wrote {summary_json}")
    print(f"[summary] wrote {summary_md}")


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] results={args.results_dir} batches={args.batches} "
        f"f_cpu_store={args.f_cpu_store} f_prefetch={args.f_prefetch} exp={args.exp}"
    )

    exit_code = 0
    if args.exp:
        for repeat in range(args.repeats):
            for batch in args.batches:
                for arm in ARMS:
                    rc = run_cell(args, arm=arm, batch=batch, repeat=repeat)
                    if rc != 0:
                        exit_code = rc
                        if not args.keep_going:
                            summary = summarize(args)
                            write_outputs(args, summary)
                            return exit_code

    summary = summarize(args)
    write_outputs(args, summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
