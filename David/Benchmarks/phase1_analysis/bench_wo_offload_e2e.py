#!/usr/bin/env python3
"""E2E WO-offload inclusion test.

This benchmark compares the same native prefetch placement with and without
`self_attn.o_proj` in the parameter whitelist. It is meant to answer the
policy question before Planner work: does making WO participate in weight
offload have negligible E2E cost, or is the Phase 0 no-WO decision still
visible once measured through the full vLLM path?

Run from `/TTC/FastTTS-thesis` in the thesis environment, for example:

    python /TTC/David/Benchmarks/phase1_analysis/bench_wo_offload_e2e.py \
        --exp --depths 01L 07L 14L --batches 1 16 64
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
REPEATS = 1

# Qwen2.5-7B BF16 per-layer arithmetic.
HIDDEN = 3584
QKV_OUT = (28 + 2 * 4) * 128
INTERMEDIATE = 18944
NUM_LAYERS = 28
BF16_BYTES = 2
NON_WO_PARAMS = ("qkv_proj", "gate_up_proj", "down_proj")
WITH_WO_PARAMS = (*NON_WO_PARAMS, "o_proj")
WO_BYTES_PER_LAYER = HIDDEN * HIDDEN * BF16_BYTES
NON_WO_BYTES_PER_LAYER = (
    HIDDEN * QKV_OUT * BF16_BYTES
    + HIDDEN * 2 * INTERMEDIATE * BF16_BYTES
    + INTERMEDIATE * HIDDEN * BF16_BYTES
)


@dataclass(frozen=True)
class Depth:
    label: str
    n_layers: int
    group_size: int


@dataclass(frozen=True)
class Arm:
    name: str
    params: tuple[str, ...]
    includes_wo: bool


DEPTHS = {
    "01L": Depth("01L", 1, 28),
    "07L": Depth("07L", 7, 4),
    "14L": Depth("14L", 14, 2),
}
ARMS = (
    Arm("no_wo", NON_WO_PARAMS, includes_wo=False),
    Arm("with_wo", WITH_WO_PARAMS, includes_wo=True),
)


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1_analysis/wo_offload_e2e") / stamp


def gib(nbytes: int | float) -> float:
    return float(nbytes) / (1024**3)


def cell_stem(depth: Depth, arm: Arm, batch: int, repeat: int) -> str:
    return f"r{repeat:02d}_{depth.label}_{arm.name}_b{batch}"


def cell_paths(
    results_dir: Path,
    depth: Depth,
    arm: Arm,
    batch: int,
    repeat: int,
) -> tuple[Path, Path]:
    stem = cell_stem(depth, arm, batch, repeat)
    return results_dir / f"{stem}.json", results_dir / f"{stem}.log"


def build_command(
    args: argparse.Namespace,
    *,
    depth: Depth,
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
        args.offload_backend,
        "--offload-group-size",
        str(depth.group_size),
        "--offload-num-in-group",
        "1",
        "--offload-prefetch-step",
        str(args.prefetch_step),
        "--offload-params",
        *arm.params,
    ]
    if args.mode == "eager":
        cmd.append("--enforce-eager")
    cmd.extend(args.extra_vllm_args)
    return cmd


def run_cell(
    args: argparse.Namespace,
    *,
    depth: Depth,
    arm: Arm,
    batch: int,
    repeat: int,
) -> int:
    out_json, log_path = cell_paths(args.results_dir, depth, arm, batch, repeat)
    if out_json.exists() and not args.force:
        print(f"[skip] {out_json}")
        return 0

    cmd = build_command(
        args,
        depth=depth,
        arm=arm,
        batch=batch,
        out_json=out_json,
    )
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        print(
            f"[fail] depth={depth.label} arm={arm.name} b={batch} "
            f"r={repeat} rc={proc.returncode} elapsed={elapsed:.0f}s\n{tail}"
        )
        return proc.returncode

    latency = json.loads(out_json.read_text()).get("avg_latency")
    print(
        f"[ok] depth={depth.label} arm={arm.name} b={batch} r={repeat} "
        f"latency={latency:.4f}s elapsed={elapsed:.0f}s"
    )
    return 0


def collect_result(
    results_dir: Path,
    depth: Depth,
    arm: Arm,
    batch: int,
    repeat: int,
) -> dict[str, Any] | None:
    out_json, _ = cell_paths(results_dir, depth, arm, batch, repeat)
    if not out_json.exists():
        return None
    data = json.loads(out_json.read_text())
    latencies = data.get("latencies") or []
    offloaded_bytes = depth.n_layers * NON_WO_BYTES_PER_LAYER
    if arm.includes_wo:
        offloaded_bytes += depth.n_layers * WO_BYTES_PER_LAYER
    return {
        "depth": depth.label,
        "n_layers": depth.n_layers,
        "batch": batch,
        "arm": arm.name,
        "includes_wo": arm.includes_wo,
        "repeat": repeat,
        "avg_latency_s": data.get("avg_latency"),
        "median_latency_s": statistics.median(latencies) if latencies else None,
        "latencies": latencies,
        "offloaded_gib": gib(offloaded_bytes),
        "incremental_wo_gib": (
            gib(depth.n_layers * WO_BYTES_PER_LAYER) if arm.includes_wo else 0.0
        ),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    depths = [DEPTHS[label] for label in args.depths]
    rows = [
        row
        for depth in depths
        for arm in ARMS
        for batch in args.batches
        for repeat in range(args.repeats)
        if (row := collect_result(args.results_dir, depth, arm, batch, repeat))
    ]
    by_case: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault((row["depth"], row["batch"], row["arm"]), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (depth_label, batch, arm_name), items in sorted(by_case.items()):
        avgs = [r["avg_latency_s"] for r in items if r["avg_latency_s"] is not None]
        if not avgs:
            continue
        first = items[0]
        aggregates.append(
            {
                "depth": depth_label,
                "batch": batch,
                "arm": arm_name,
                "includes_wo": first["includes_wo"],
                "repeats": len(avgs),
                "mean_latency_s": statistics.mean(avgs),
                "stdev_latency_s": statistics.stdev(avgs) if len(avgs) > 1 else 0.0,
                "offloaded_gib": first["offloaded_gib"],
                "incremental_wo_gib": first["incremental_wo_gib"],
            }
        )

    no_wo_by_case = {
        (row["depth"], row["batch"]): row
        for row in aggregates
        if row["arm"] == "no_wo"
    }
    comparisons: list[dict[str, Any]] = []
    for row in aggregates:
        if row["arm"] != "with_wo":
            continue
        base = no_wo_by_case.get((row["depth"], row["batch"]))
        if base is None:
            continue
        delta = row["mean_latency_s"] - base["mean_latency_s"]
        comparisons.append(
            {
                "depth": row["depth"],
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
            "offload_backend": args.offload_backend,
            "prefetch_step": args.prefetch_step,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batches": args.batches,
            "depths": args.depths,
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
        "# WO Offload E2E Results",
        "",
        f"Model: `{summary['config']['model']}`; "
        f"backend: `{summary['config']['offload_backend']}`; "
        f"mode: `{summary['config']['mode']}`; "
        f"input/output: `{summary['config']['input_len']}/"
        f"{summary['config']['output_len']}`.",
        "",
        "## Comparison",
        "",
        "| depth | B | no WO s | with WO s | delta | delta % | extra WO GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparisons"]:
        lines.append(
            f"| `{row['depth']}` | {row['batch']} | "
            f"{row['no_wo_latency_s']:.4f} | {row['with_wo_latency_s']:.4f} | "
            f"{row['delta_s']:+.4f} | {row['delta_pct']:+.2f}% | "
            f"{row['incremental_wo_gib']:.3f} |"
        )

    lines += [
        "",
        "## Aggregate Rows",
        "",
        "| depth | B | arm | mean s | stdev s | offloaded GiB |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        lines.append(
            f"| `{row['depth']}` | {row['batch']} | `{row['arm']}` | "
            f"{row['mean_latency_s']:.4f} | {row['stdev_latency_s']:.4f} | "
            f"{row['offloaded_gib']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument("--offload-backend", default="prefetch_defer")
    parser.add_argument("--prefetch-step", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=INPUT_LEN)
    parser.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION
    )
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument(
        "--depths",
        nargs="+",
        choices=sorted(DEPTHS),
        default=["01L", "07L", "14L"],
    )
    parser.add_argument("--num-iters-warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--num-iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--exp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] results={args.results_dir} depths={args.depths} "
        f"batches={args.batches} exp={args.exp}"
    )

    exit_code = 0
    if args.exp:
        for repeat in range(args.repeats):
            for depth_label in args.depths:
                depth = DEPTHS[depth_label]
                for batch in args.batches:
                    for arm in ARMS:
                        rc = run_cell(
                            args,
                            depth=depth,
                            arm=arm,
                            batch=batch,
                            repeat=repeat,
                        )
                        if rc != 0:
                            exit_code = rc
                            if not args.keep_going:
                                summary = summarize(args)
                                write_outputs(args, summary)
                                return exit_code

    summary = summarize(args)
    write_outputs(args, summary)
    return exit_code


def write_outputs(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    summary_json = args.results_dir / "summary.json"
    summary_md = args.results_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, summary_md)
    print(f"[summary] wrote {summary_json}")
    print(f"[summary] wrote {summary_md}")


if __name__ == "__main__":
    raise SystemExit(main())
