#!/usr/bin/env python3
"""Phase 1 analysis: COTS pure-prefetch vs native vLLM prefetch.

This is the current-code rerun of the Phase 1b exact-match experiment. It
compares COTS tensor-granularity pure prefetch
(`f_cpu_store == f_prefetch`) against native layer-granularity prefetch at
matched offloaded layer counts. It can run both production graph mode and
diagnostic eager mode.

Run from the thesis package directory:

    cd /TTC/FastTTS-thesis
    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1_analysis/bench_cots_vs_native_prefetch.py \
        --exp --smoke
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


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.75
WARMUP_ITERS = 2
BENCH_ITERS = 3
REPEATS = 1
N_LAYERS = 28
PER_LAYER_GIB = 12.15 / N_LAYERS
DEFAULT_BATCHES = (1, 64)
DEFAULT_MODES = ("graph", "eager")
DEFAULT_NATIVE_BACKENDS = ("prefetch_defer",)
DEFAULT_NATIVE_K_VALUES = (1, 2)


@dataclass(frozen=True)
class DepthPair:
    label: str
    n_layers: int
    native_group_size: int
    cots_f: float
    depth_gib: float


@dataclass(frozen=True)
class Arm:
    name: str
    family: str
    flags: tuple[str, ...]
    depth_label: str | None = None
    n_layers: int | None = None
    depth_gib: float | None = None
    native_backend: str | None = None
    native_k: int | None = None


DEPTH_PAIRS = (
    DepthPair("01L", 1, 28, 1 / N_LAYERS, 1 * PER_LAYER_GIB),
    DepthPair("02L", 2, 14, 2 / N_LAYERS, 2 * PER_LAYER_GIB),
    DepthPair("04L", 4, 7, 4 / N_LAYERS, 4 * PER_LAYER_GIB),
    DepthPair("07L", 7, 4, 7 / N_LAYERS, 7 * PER_LAYER_GIB),
    DepthPair("14L", 14, 2, 14 / N_LAYERS, 14 * PER_LAYER_GIB),
)


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1_analysis/cots_vs_native_prefetch") / stamp


def cots_flags(f: float) -> tuple[str, ...]:
    value = f"{f:.6f}"
    return (
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        value,
        "--cots-f-prefetch",
        value,
        "--cots-cpu-runner",
        "native",
    )


def native_flags(backend: str, group_size: int, k: int) -> tuple[str, ...]:
    return (
        "--offload-backend",
        backend,
        "--offload-group-size",
        str(group_size),
        "--offload-num-in-group",
        "1",
        "--offload-prefetch-step",
        str(k),
    )


def build_arms(
    *,
    depth_pairs: list[DepthPair],
    native_backends: list[str],
    native_k_values: list[int],
) -> list[Arm]:
    arms = [
        Arm(name="none", family="none", flags=()),
    ]
    for pair in depth_pairs:
        arms.append(
            Arm(
                name=f"cots_{pair.label}",
                family="cots_prefetch_only",
                flags=cots_flags(pair.cots_f),
                depth_label=pair.label,
                n_layers=pair.n_layers,
                depth_gib=pair.depth_gib,
            )
        )
        for backend in native_backends:
            for k in native_k_values:
                arms.append(
                    Arm(
                        name=f"native_{backend}_k{k}_{pair.label}",
                        family=backend,
                        flags=native_flags(backend, pair.native_group_size, k),
                        depth_label=pair.label,
                        n_layers=pair.n_layers,
                        depth_gib=pair.depth_gib,
                        native_backend=backend,
                        native_k=k,
                    )
                )
    return arms


def filter_depth_pairs(labels: list[str]) -> list[DepthPair]:
    by_label = {pair.label: pair for pair in DEPTH_PAIRS}
    unknown = [label for label in labels if label not in by_label]
    if unknown:
        raise ValueError(f"unknown depth labels: {unknown}")
    return [by_label[label] for label in labels]


def filter_arms_in_requested_order(arms: list[Arm], requested: list[str] | None) -> list[Arm]:
    if not requested:
        return arms
    order = {name: idx for idx, name in enumerate(requested)}

    def key(arm: Arm) -> int:
        return min(
            order.get(arm.name, len(order)),
            order.get(arm.family, len(order)),
        )

    selected = [
        arm for arm in arms if arm.name in order or arm.family in order
    ]
    return sorted(selected, key=key)


def cell_stem(mode: str, arm: str, batch: int, repeat: int) -> str:
    return f"r{repeat:02d}_{mode}_{arm}_b{batch}"


def cell_json(results_dir: Path, mode: str, arm: str, batch: int, repeat: int) -> Path:
    return results_dir / f"{cell_stem(mode, arm, batch, repeat)}.json"


def build_vllm_command(
    *,
    mode: str,
    arm: Arm,
    batch: int,
    out_json: Path,
    args: argparse.Namespace,
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
    ]
    if mode == "eager":
        cmd.append("--enforce-eager")
    cmd.extend(arm.flags)
    cmd.extend(args.extra_vllm_args)
    return cmd


def run_cell(
    *,
    mode: str,
    arm: Arm,
    batch: int,
    repeat: int,
    results_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, int]:
    out_json = cell_json(results_dir, mode, arm.name, batch, repeat)
    out_log = out_json.with_suffix(".log")
    if out_json.exists() and not args.force:
        print(f"  [skip] r={repeat} {mode} {arm.name} B={batch} (cached)")
        return out_json, 0

    cmd = build_vllm_command(
        mode=mode,
        arm=arm,
        batch=batch,
        out_json=out_json,
        args=args,
    )
    t0 = time.perf_counter()
    with out_log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0:
        avg = parse_avg_latency(out_json)
        print(
            f"  [ok] r={repeat} {mode} {arm.name} B={batch}: "
            f"avg={avg:.4f}s ({elapsed:.1f}s)"
        )
    else:
        tail = "\n        ".join(out_log.read_text(errors="replace").splitlines()[-20:])
        print(
            f"  [FAIL] r={repeat} {mode} {arm.name} B={batch} "
            f"rc={proc.returncode} ({elapsed:.1f}s)\n        {tail}"
        )
    return out_json, proc.returncode


def parse_avg_latency(path: Path) -> float:
    return float(json.loads(path.read_text())["avg_latency"])


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "cv": None}
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = stdev / mean if mean else 0.0
    return {"n": len(values), "mean": mean, "stdev": stdev, "cv": cv}


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


def summarize(
    args: argparse.Namespace,
    arms: list[Arm],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    mean_by_key: dict[tuple[str, str, int], float | None] = {}

    for mode in args.modes:
        for arm in arms:
            for batch in args.batches:
                values = []
                for repeat in range(args.repeat):
                    path = cell_json(args.results_dir, mode, arm.name, batch, repeat)
                    if path.exists():
                        values.append(parse_avg_latency(path))
                stat = stats(values)
                mean = stat["mean"]
                mean_by_key[(mode, arm.name, batch)] = (
                    float(mean) if mean is not None else None
                )

    baseline_by_mode_batch = {
        (mode, batch): mean_by_key.get((mode, "none", batch))
        for mode in args.modes
        for batch in args.batches
    }
    cots_by_depth_mode_batch = {
        (arm.depth_label, mode, batch): mean_by_key.get((mode, arm.name, batch))
        for arm in arms
        if arm.family == "cots_prefetch_only"
        for mode in args.modes
        for batch in args.batches
    }

    for mode in args.modes:
        for arm in arms:
            for batch in args.batches:
                values = []
                for repeat in range(args.repeat):
                    path = cell_json(args.results_dir, mode, arm.name, batch, repeat)
                    if path.exists():
                        values.append(parse_avg_latency(path))
                stat = stats(values)
                mean = stat["mean"]
                base = baseline_by_mode_batch[(mode, batch)]
                slowdown = None if mean is None or base is None else float(mean) / base
                cots_mean = (
                    cots_by_depth_mode_batch.get((arm.depth_label, mode, batch))
                    if arm.depth_label
                    else None
                )
                cots_vs_native = (
                    None
                    if mean is None
                    or cots_mean is None
                    or arm.family == "cots_prefetch_only"
                    else float(cots_mean) / float(mean)
                )
                cells.append(
                    {
                        "mode": mode,
                        "arm": arm.name,
                        "family": arm.family,
                        "batch_size": batch,
                        "depth_label": arm.depth_label,
                        "n_layers": arm.n_layers,
                        "depth_gib": arm.depth_gib,
                        "native_backend": arm.native_backend,
                        "native_k": arm.native_k,
                        "num_repeats": stat["n"],
                        "mean_latency_s": mean,
                        "stdev_latency_s": stat["stdev"],
                        "cv": stat["cv"],
                        "slowdown_vs_none": slowdown,
                        "tokens_per_s": (
                            None
                            if mean is None
                            else float(batch * args.output_len) / float(mean)
                        ),
                        "cots_latency_div_native_latency": cots_vs_native,
                    }
                )

    return {
        "env": env_info(),
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batches": args.batches,
            "modes": args.modes,
            "repeat": args.repeat,
            "num_iters": args.num_iters,
            "num_iters_warmup": args.num_iters_warmup,
            "native_backends": args.native_backends,
            "native_k_values": args.native_k_values,
            "depth_labels": args.depth_labels,
            "extra_vllm_args": args.extra_vllm_args,
        },
        "cells": cells,
    }


def fmt_optional(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    cfg = summary["config"]
    lines = [
        "# COTS Pure-Prefetch vs Native Prefetch",
        "",
        f"Model: `{cfg['model']}`; dtype: `{cfg['dtype']}`; "
        f"workload: input={cfg['input_len']}, output={cfg['output_len']}; "
        f"`max_model_len={cfg['max_model_len']}`; "
        f"`gpu_memory_utilization={cfg['gpu_memory_utilization']}`.",
        "",
        "| mode | B | depth | arm | mean s | vs none | tok/s | COTS/native |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        lines.append(
            f"| `{cell['mode']}` | {cell['batch_size']} | "
            f"{fmt_optional(cell['depth_gib'], 2)} | `{cell['arm']}` | "
            f"{fmt_optional(cell['mean_latency_s'], 4)} | "
            f"{fmt_optional(cell['slowdown_vs_none'], 3)} | "
            f"{fmt_optional(cell['tokens_per_s'], 1)} | "
            f"{fmt_optional(cell['cots_latency_div_native_latency'], 3)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.depth_labels = ["01L"]
    args.batches = [1]
    args.modes = ["graph", "eager"]
    args.native_backends = ["prefetch_defer"]
    args.native_k_values = [1]
    args.repeat = 1
    args.num_iters_warmup = 0
    args.num_iters = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--input-len", type=int, default=INPUT_LEN)
    parser.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION
    )
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["graph", "eager"],
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--depth-labels",
        nargs="+",
        choices=[pair.label for pair in DEPTH_PAIRS],
        default=[pair.label for pair in DEPTH_PAIRS],
    )
    parser.add_argument(
        "--native-backends",
        nargs="+",
        choices=["prefetch", "prefetch_defer"],
        default=list(DEFAULT_NATIVE_BACKENDS),
    )
    parser.add_argument(
        "--native-k-values",
        type=int,
        nargs="+",
        choices=[1, 2],
        default=list(DEFAULT_NATIVE_K_VALUES),
    )
    parser.add_argument("--repeat", type=int, default=REPEATS)
    parser.add_argument("--num-iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--num-iters-warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--only-arms", nargs="*", default=None)
    parser.add_argument("--exp", action="store_true", help="Run missing cells.")
    parser.add_argument("--force", action="store_true", help="Overwrite cached cells.")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny validation grid.")
    parser.add_argument(
        "--extra-vllm-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments appended after all harness-controlled vLLM args.",
    )
    args = parser.parse_args()
    if args.smoke:
        apply_smoke_overrides(args)
    return args


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    depth_pairs = filter_depth_pairs(args.depth_labels)
    arms = filter_arms_in_requested_order(
        build_arms(
            depth_pairs=depth_pairs,
            native_backends=args.native_backends,
            native_k_values=args.native_k_values,
        ),
        args.only_arms,
    )

    print(
        f"[setup] results={args.results_dir} modes={args.modes} "
        f"batches={args.batches} arms={len(arms)} repeats={args.repeat} "
        f"exp={args.exp}"
    )
    exit_code = 0
    if args.exp:
        for mode in args.modes:
            for arm in arms:
                for batch in args.batches:
                    for repeat in range(args.repeat):
                        _, rc = run_cell(
                            mode=mode,
                            arm=arm,
                            batch=batch,
                            repeat=repeat,
                            results_dir=args.results_dir,
                            args=args,
                        )
                        if rc != 0:
                            exit_code = rc
                            if not args.keep_going:
                                summary = summarize(args, arms)
                                (args.results_dir / "summary.json").write_text(
                                    json.dumps(summary, indent=2)
                                )
                                write_markdown_summary(
                                    summary, args.results_dir / "summary.md"
                                )
                                return exit_code

    summary = summarize(args, arms)
    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown_summary(summary, args.results_dir / "summary.md")
    print(f"[summary] wrote {summary_path}")
    print(f"[summary] wrote {args.results_dir / 'summary.md'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
