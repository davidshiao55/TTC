#!/usr/bin/env python3
"""Validate the Planner's per-bucket COTS dispatch model against runtime.

This harness measures the inner Planner primitive before we trust it inside the
global placement search. For a fixed `f_cpu_store`, it sweeps
`f_cpu_compute`, sets `f_prefetch_compute = f_cpu_store - f_cpu_compute`, emits
a forced `cots_dispatch_table`, and records real vLLM latency. The default
layout applies the same split across all dispatch buckets; `decode-only` keeps
non-decode buckets pure prefetch and varies only the measured decode bucket.

The first validation question is qualitative ranking:

* small decode buckets can use CPU compute,
* prefill-heavy buckets may need prefetch-heavy dispatch,
* slowdowns should expose whether the model is missing phase or overhead terms.

Run from `/TTC/FastTTS-thesis` in the thesis environment, for example:

    python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
        --exp --smoke

Full-ish first pass:

    python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
        --exp --batches 1 4 16 64 --f-cpu-store-values 0.02 0.05 0.09

Forced-offload pass:

    python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
        --exp --batches 1 16 64 \
        --f-cpu-store-values 0.01 0.05 0.15 0.25 0.40 \
        --f-cpu-ratios 0 0.25 0.5 0.75 1 \
        --cell-timeout-s 300 --keep-going

Decode-only dispatch isolation:

    python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
        --exp --dispatch-layout decode-only --batches 16 64 \
        --f-cpu-store-values 0.15 \
        --f-cpu-ratios 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TTC_ROOT = Path(__file__).resolve().parents[3]
FASTTTS_ROOT = TTC_ROOT / "FastTTS-thesis"
if str(FASTTTS_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTTTS_ROOT))

from planner import derive_weight_thread_policy  # noqa: E402


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.75
WARMUP_ITERS = 2
BENCH_ITERS = 3
REPEATS = 1
CELL_TIMEOUT_S: float | None = None
DEFAULT_BATCHES = (1, 4, 16, 64)
DEFAULT_F_CPU_STORE_VALUES = (0.02, 0.05, 0.09)
DEFAULT_F_CPU_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Default COTS dispatch buckets. This intentionally mirrors the current vLLM V1
# graph bucket grid plus larger fallback buckets, but the harness uses it as the
# dispatch-table key set, not as CUDA graph capture policy.
DISPATCH_BUCKETS = (
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
    768,
    1024,
    1536,
    2048,
    4096,
    8192,
)
THREAD_POLICY_BUCKETS = tuple(bucket for bucket in DISPATCH_BUCKETS if bucket <= 512)


@dataclass(frozen=True)
class Cell:
    batch: int
    f_cpu_store: float | None
    f_cpu: float | None
    f_prefetch: float | None
    repeat: int

    @property
    def is_baseline(self) -> bool:
        return self.f_cpu_store is None

    @property
    def split_name(self) -> str:
        if self.is_baseline:
            return "none"
        return f"cpu{float_tag(self.f_cpu)}_pf{float_tag(self.f_prefetch)}"


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/planner/dispatch_model_validation") / stamp


def float_tag(value: float | None) -> str:
    if value is None:
        return "none"
    text = f"{float(value):.5f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def parse_float_list(values: list[str]) -> list[float]:
    result: list[float] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                result.append(float(part))
    return result


def make_dispatch_table(
    *,
    dispatch_buckets: list[int],
    f_cpu: float,
    f_prefetch: float,
    f_cpu_store: float | None = None,
    batch: int | None = None,
    layout: str = "uniform",
) -> dict[int, tuple[float, float]]:
    if layout == "uniform":
        return {
            int(bucket): (float(f_cpu), float(f_prefetch))
            for bucket in dispatch_buckets
        }
    if layout == "decode-only":
        if f_cpu_store is None:
            raise ValueError("decode-only dispatch requires f_cpu_store")
        if batch is None:
            raise ValueError("decode-only dispatch requires batch")
        table = {
            int(bucket): (0.0, float(f_cpu_store))
            for bucket in dispatch_buckets
        }
        table[bucket_for(batch, dispatch_buckets)] = (
            float(f_cpu),
            float(f_prefetch),
        )
        return table
    raise ValueError(f"unknown dispatch layout: {layout}")


def bucket_for(num_tokens: int, dispatch_buckets: list[int]) -> int:
    for bucket in sorted(dispatch_buckets):
        if int(num_tokens) <= int(bucket):
            return int(bucket)
    return int(max(dispatch_buckets))


def jsonable_dispatch_table(
    table: dict[int, tuple[float, float]],
) -> dict[str, list[float]]:
    return {str(bucket): [entry[0], entry[1]] for bucket, entry in table.items()}


def cots_flags(args: argparse.Namespace, cell: Cell) -> list[str]:
    assert cell.f_cpu_store is not None
    assert cell.f_cpu is not None
    assert cell.f_prefetch is not None

    dispatch_table = make_dispatch_table(
        dispatch_buckets=args.dispatch_buckets,
        f_cpu=cell.f_cpu,
        f_prefetch=cell.f_prefetch,
        f_cpu_store=cell.f_cpu_store,
        batch=cell.batch,
        layout=args.dispatch_layout,
    )
    flags = [
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        f"{cell.f_cpu_store:.12g}",
        "--cots-f-prefetch",
        "0.0",
        "--cots-dispatch-table",
        json.dumps(jsonable_dispatch_table(dispatch_table), separators=(",", ":")),
        "--cots-cpu-runner",
        args.cots_cpu_runner,
    ]
    if args.cots_weight_modules:
        flags += ["--cots-weight-modules", *args.cots_weight_modules]
    if args.thread_policy == "workscore":
        thread_map = derive_weight_thread_policy(
            make_dispatch_table(
                dispatch_buckets=args.thread_buckets,
                f_cpu=cell.f_cpu,
                f_prefetch=cell.f_prefetch,
                f_cpu_store=cell.f_cpu_store,
                batch=cell.batch,
                layout=args.dispatch_layout,
            )
        )
        flags += [
            "--cots-cpu-num-threads",
            str(args.cots_cpu_num_threads),
            "--cots-cpu-num-threads-by-bucket",
            json.dumps({str(k): v for k, v in thread_map.items()}, separators=(",", ":")),
        ]
    elif args.thread_policy == "scalar":
        flags += ["--cots-cpu-num-threads", str(args.cots_cpu_num_threads)]
    else:
        raise ValueError(f"unknown thread policy: {args.thread_policy}")
    return flags


def cell_paths(results_dir: Path, mode: str, cell: Cell) -> tuple[Path, Path]:
    if cell.is_baseline:
        stem = f"r{cell.repeat:02d}_{mode}_b{cell.batch}_none"
    else:
        stem = (
            f"r{cell.repeat:02d}_{mode}_b{cell.batch}_"
            f"store{float_tag(cell.f_cpu_store)}_{cell.split_name}"
        )
    return results_dir / f"{stem}.json", results_dir / f"{stem}.log"


def build_command(args: argparse.Namespace, mode: str, cell: Cell, out_json: Path) -> list[str]:
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
    if mode == "eager":
        cmd.append("--enforce-eager")
    if not cell.is_baseline:
        cmd += cots_flags(args, cell)
    cmd += args.extra_vllm_args
    return cmd


def run_cell(args: argparse.Namespace, mode: str, cell: Cell) -> int:
    out_json, out_log = cell_paths(args.results_dir, mode, cell)
    if out_json.exists() and not args.force:
        print(f"[skip] {out_json}", flush=True)
        return 0
    cmd = build_command(args, mode, cell, out_json)
    t0 = time.perf_counter()
    try:
        with out_log.open("w") as log:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=args.cell_timeout_s,
            )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        with out_log.open("a") as log:
            log.write(f"\n[timeout] exceeded {args.cell_timeout_s}s\n")
        print(
            f"[timeout] mode={mode} b={cell.batch} store={cell.f_cpu_store} "
            f"split={cell.split_name} elapsed={elapsed:.0f}s",
            flush=True,
        )
        return 124
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(out_log.read_text(errors="replace").splitlines()[-20:])
        print(
            f"[fail] mode={mode} b={cell.batch} store={cell.f_cpu_store} "
            f"split={cell.split_name} rc={proc.returncode} "
            f"elapsed={elapsed:.0f}s\n{tail}",
            flush=True,
        )
        return proc.returncode
    latency = json.loads(out_json.read_text()).get("avg_latency")
    print(
        f"[ok] mode={mode} b={cell.batch} store={cell.f_cpu_store} "
        f"split={cell.split_name} latency={latency:.4f}s elapsed={elapsed:.0f}s",
        flush=True,
    )
    return 0


def read_latency(path: Path) -> float:
    return float(json.loads(path.read_text())["avg_latency"])


def stat(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "cv": None}
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = stdev / mean if mean else 0.0
    return {"n": len(values), "mean": mean, "stdev": stdev, "cv": cv}


def collect_rows(args: argparse.Namespace, modes: list[str], cells: list[Cell]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_case: dict[tuple[str, int, float | None, float | None, float | None], list[float]] = {}
    for mode in modes:
        for cell in cells:
            out_json, _ = cell_paths(args.results_dir, mode, cell)
            if not out_json.exists():
                continue
            key = (mode, cell.batch, cell.f_cpu_store, cell.f_cpu, cell.f_prefetch)
            by_case.setdefault(key, []).append(read_latency(out_json))

    for (mode, batch, f_cpu_store, f_cpu, f_prefetch), values in sorted(
        by_case.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or -1, item[0][3] or -1)
    ):
        s = stat(values)
        rows.append(
            {
                "mode": mode,
                "batch": batch,
                "f_cpu_store": f_cpu_store,
                "f_cpu": f_cpu,
                "f_prefetch": f_prefetch,
                "num_repeats": s["n"],
                "mean_latency_s": s["mean"],
                "stdev_latency_s": s["stdev"],
                "cv": s["cv"],
            }
        )
    return rows


def failure_reason(path: Path) -> str:
    if not path.exists():
        return "not_started"
    text = path.read_text(errors="replace")
    if not text.strip():
        return "empty_log"
    if "[timeout]" in text:
        return "timeout"
    if "Segfault encountered" in text:
        return "segfault"
    if "KeyboardInterrupt" in text:
        return "interrupted"
    if "Engine core initialization failed" in text:
        return "engine_init_failed"
    if "Traceback" in text:
        return "traceback"
    return "missing_json"


def collect_failed_cells(
    args: argparse.Namespace,
    modes: list[str],
    cells: list[Cell],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for mode in modes:
        for cell in cells:
            out_json, out_log = cell_paths(args.results_dir, mode, cell)
            if out_json.exists() or not out_log.exists():
                continue
            failures.append(
                {
                    "mode": mode,
                    "batch": cell.batch,
                    "f_cpu_store": cell.f_cpu_store,
                    "f_cpu": cell.f_cpu,
                    "f_prefetch": cell.f_prefetch,
                    "reason": failure_reason(out_log),
                    "log": str(out_log),
                }
            )
    return failures


def summarize(args: argparse.Namespace, modes: list[str], cells: list[Cell]) -> dict[str, Any]:
    rows = collect_rows(args, modes, cells)
    baseline: dict[tuple[str, int], float] = {}
    for row in rows:
        if row["f_cpu_store"] is None and row["mean_latency_s"] is not None:
            baseline[(row["mode"], row["batch"])] = float(row["mean_latency_s"])

    for row in rows:
        base = baseline.get((row["mode"], row["batch"]))
        if base and row["mean_latency_s"] is not None:
            row["slowdown_vs_none"] = float(row["mean_latency_s"]) / base

    best_by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["f_cpu_store"] is None or row["mean_latency_s"] is None:
            continue
        key = f"{row['mode']}_b{row['batch']}_store{float_tag(row['f_cpu_store'])}"
        cur = best_by_case.get(key)
        if cur is None or row["mean_latency_s"] < cur["mean_latency_s"]:
            best_by_case[key] = row

    qualitative_checks: list[dict[str, Any]] = []
    for mode in modes:
        for f_cpu_store in args.f_cpu_store_values:
            best_for_store = [
                row for key, row in best_by_case.items()
                if key.startswith(f"{mode}_")
                and key.endswith(f"store{float_tag(f_cpu_store)}")
            ]
            if len(best_for_store) < 2:
                continue
            ordered = sorted(best_for_store, key=lambda row: row["batch"])
            qualitative_checks.append(
                {
                    "mode": mode,
                    "f_cpu_store": f_cpu_store,
                    "smallest_batch": ordered[0]["batch"],
                    "smallest_batch_best_f_cpu": ordered[0]["f_cpu"],
                    "largest_batch": ordered[-1]["batch"],
                    "largest_batch_best_f_cpu": ordered[-1]["f_cpu"],
                    "cpu_heavier_at_small_batch": (
                        ordered[0]["f_cpu"] >= ordered[-1]["f_cpu"]
                    ),
                }
            )

    return {
        "env": env_info(),
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "modes": modes,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batches": args.batches,
            "f_cpu_store_values": args.f_cpu_store_values,
            "f_cpu_ratios": args.f_cpu_ratios,
            "dispatch_buckets": args.dispatch_buckets,
            "thread_buckets": args.thread_buckets,
            "thread_policy": args.thread_policy,
            "cots_cpu_num_threads": args.cots_cpu_num_threads,
            "cots_cpu_runner": args.cots_cpu_runner,
            "cots_weight_modules": args.cots_weight_modules,
            "dispatch_layout": args.dispatch_layout,
            "cell_timeout_s": args.cell_timeout_s,
            "num_iters_warmup": args.num_iters_warmup,
            "num_iters": args.num_iters,
            "repeat": args.repeat,
            "extra_vllm_args": args.extra_vllm_args,
        },
        "rows": rows,
        "best_by_case": best_by_case,
        "qualitative_checks": qualitative_checks,
        "failed_cells": collect_failed_cells(args, modes, cells),
    }


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic only.
        info["torch_error"] = repr(exc)
    return info


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    cfg = summary["config"]
    lines = [
        "# Dispatch Model Validation",
        "",
        f"Model: `{cfg['model']}`; dtype: `{cfg['dtype']}`; "
        f"workload input/output: `{cfg['input_len']}/{cfg['output_len']}`.",
        "",
        "## Measured Best Split",
        "",
        "| case | f_cpu | f_prefetch | mean s | slowdown | CV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary["best_by_case"].items():
        lines.append(
            f"| `{key}` | {fmt(row['f_cpu'], 5)} | "
            f"{fmt(row['f_prefetch'], 5)} | {fmt(row['mean_latency_s'])} | "
            f"{fmt(row.get('slowdown_vs_none'), 3)} | {fmt(row['cv'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Qualitative Check",
            "",
            "| mode | f_cpu_store | small B best f_cpu | large B best f_cpu | pass |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for check in summary["qualitative_checks"]:
        lines.append(
            f"| `{check['mode']}` | {fmt(check['f_cpu_store'], 5)} | "
            f"B={check['smallest_batch']} / {fmt(check['smallest_batch_best_f_cpu'], 5)} | "
            f"B={check['largest_batch']} / {fmt(check['largest_batch_best_f_cpu'], 5)} | "
            f"{check['cpu_heavier_at_small_batch']} |"
        )

    if summary["failed_cells"]:
        lines.extend(
            [
                "",
                "## Failed Or Incomplete Cells",
                "",
                "| mode | B | f_store | f_cpu | f_prefetch | reason |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in summary["failed_cells"]:
            lines.append(
                f"| `{row['mode']}` | {row['batch']} | "
                f"{fmt(row['f_cpu_store'], 5)} | {fmt(row['f_cpu'], 5)} | "
                f"{fmt(row['f_prefetch'], 5)} | `{row['reason']}` |"
            )

    lines.extend(
        [
            "",
            "## All Rows",
            "",
            "| mode | B | f_store | f_cpu | f_prefetch | mean s | slowdown | CV |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| `{row['mode']}` | {row['batch']} | "
            f"{fmt(row['f_cpu_store'], 5)} | {fmt(row['f_cpu'], 5)} | "
            f"{fmt(row['f_prefetch'], 5)} | {fmt(row['mean_latency_s'])} | "
            f"{fmt(row.get('slowdown_vs_none'), 3)} | {fmt(row['cv'], 3)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def unique_splits_for_store(f_cpu_store: float, ratios: list[float]) -> list[tuple[float, float]]:
    splits: dict[tuple[float, float], tuple[float, float]] = {}
    for ratio in ratios:
        if ratio < -1e-12 or ratio > 1.0 + 1e-12:
            raise ValueError(f"f_cpu ratio must be in [0, 1], got {ratio}")
        f_cpu = round(f_cpu_store * min(1.0, max(0.0, ratio)), 12)
        f_prefetch = round(f_cpu_store - f_cpu, 12)
        splits[(f_cpu, f_prefetch)] = (f_cpu, f_prefetch)
    splits[(0.0, round(f_cpu_store, 12))] = (0.0, round(f_cpu_store, 12))
    splits[(round(f_cpu_store, 12), 0.0)] = (round(f_cpu_store, 12), 0.0)
    return list(splits.values())


def build_cells(args: argparse.Namespace, modes: list[str]) -> list[Cell]:
    del modes
    cells: list[Cell] = []
    for repeat in range(args.repeat):
        for batch in args.batches:
            cells.append(
                Cell(
                    batch=batch,
                    f_cpu_store=None,
                    f_cpu=None,
                    f_prefetch=None,
                    repeat=repeat,
                )
            )
            for f_cpu_store in args.f_cpu_store_values:
                splits = unique_splits_for_store(
                    f_cpu_store, args.f_cpu_ratios
                )
                for f_cpu, f_prefetch in splits:
                    cells.append(
                        Cell(
                            batch=batch,
                            f_cpu_store=f_cpu_store,
                            f_cpu=f_cpu,
                            f_prefetch=f_prefetch,
                            repeat=repeat,
                        )
                    )
    return cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--modes", choices=("graph", "eager"), nargs="+", default=["graph"])
    parser.add_argument("--input-len", type=int, default=INPUT_LEN)
    parser.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION
    )
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument(
        "--f-cpu-store-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_F_CPU_STORE_VALUES),
    )
    parser.add_argument(
        "--f-cpu-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_F_CPU_RATIOS),
        help="Candidate f_cpu values as ratios of f_cpu_store.",
    )
    parser.add_argument(
        "--dispatch-buckets",
        type=int,
        nargs="+",
        default=list(DISPATCH_BUCKETS),
    )
    parser.add_argument(
        "--capture-buckets",
        dest="dispatch_buckets",
        type=int,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--thread-buckets",
        type=int,
        nargs="+",
        default=list(THREAD_POLICY_BUCKETS),
        help=(
            "Buckets covered by cots_cpu_num_threads_by_bucket in graph mode. "
            "Must be a subset of the COTS dispatch buckets."
        ),
    )
    parser.add_argument(
        "--thread-policy",
        choices=("workscore", "scalar"),
        default="workscore",
    )
    parser.add_argument(
        "--dispatch-layout",
        choices=("uniform", "decode-only"),
        default="uniform",
        help=(
            "uniform applies each candidate split to every dispatch bucket. "
            "decode-only keeps all buckets pure prefetch and overrides only "
            "the benchmark batch's decode bucket with the candidate split."
        ),
    )
    parser.add_argument("--cots-cpu-num-threads", type=int, default=24)
    parser.add_argument("--cots-cpu-runner", choices=("native", "python"), default="native")
    parser.add_argument(
        "--cots-weight-modules",
        nargs="+",
        default=None,
        help="Optional COTS module subset, e.g. qkv mlp or qkv mlp wo.",
    )
    parser.add_argument("--repeat", type=int, default=REPEATS)
    parser.add_argument("--num-iters-warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--num-iters", type=int, default=BENCH_ITERS)
    parser.add_argument(
        "--cell-timeout-s",
        type=float,
        default=CELL_TIMEOUT_S,
        help="Optional timeout for each vLLM latency subprocess.",
    )
    parser.add_argument("--exp", action="store_true", help="Run missing cells.")
    parser.add_argument("--force", action="store_true", help="Overwrite cached cells.")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny command-plumbing check.")
    parser.add_argument(
        "--extra-vllm-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments appended after harness-controlled vLLM args.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.modes = ["eager"]
        args.batches = [1]
        args.f_cpu_store_values = [0.01]
        args.f_cpu_ratios = [0.0, 1.0]
        args.output_len = 16
        args.repeat = 1
        args.num_iters_warmup = 0
        args.num_iters = 1
    return args


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    modes = list(dict.fromkeys(args.modes))
    cells = build_cells(args, modes)
    print(
        f"[setup] results={args.results_dir} modes={modes} "
        f"batches={args.batches} stores={args.f_cpu_store_values} "
        f"ratios={args.f_cpu_ratios} repeat={args.repeat} exp={args.exp}",
        flush=True,
    )

    exit_code = 0
    try:
        if args.exp:
            for mode in modes:
                for cell in cells:
                    rc = run_cell(args, mode, cell)
                    if rc != 0:
                        exit_code = rc
                        if not args.keep_going:
                            break
                if exit_code != 0 and not args.keep_going:
                    break
    except KeyboardInterrupt:
        exit_code = 130
        print("[interrupt] writing summary for completed cells", flush=True)

    summary = summarize(args, modes, cells)
    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, args.results_dir / "summary.md")
    print(f"[summary] wrote {summary_path}", flush=True)
    print(f"[summary] wrote {args.results_dir / 'summary.md'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
