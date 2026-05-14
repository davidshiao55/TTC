#!/usr/bin/env python3
"""Phase 1 analysis: COTS free-regime latency sweep.

This harness runs fresh vLLM latency cells in separate Python processes so each
repeat gets a clean engine load. By default it only summarizes cached results;
pass ``--exp`` to launch benchmarks.

Run from the thesis package directory to avoid the /TTC Python path gotcha:

    cd /TTC/FastTTS-thesis
    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1_analysis/bench_cots_free_regime.py --exp

Smoke subset:

    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1_analysis/bench_cots_free_regime.py \
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
FREE_MARGIN = 0.05
MAX_CV = 0.03
WARMUP_ITERS = 2
BENCH_ITERS = 3
REPEATS = 3
DEFAULT_BATCHES = (1, 4, 16, 64)
DEFAULT_F_VALUES = (0.005, 0.01, 0.02, 0.0357, 0.05, 0.0714, 0.09, 0.15)


@dataclass(frozen=True)
class Arm:
    name: str
    strategy: str
    f_cpu_store: float
    f_prefetch: float
    flags: tuple[str, ...]


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1_analysis/free_regime") / stamp


def float_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def cots_flags(f_cpu_store: float, f_prefetch: float) -> tuple[str, ...]:
    return (
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        str(f_cpu_store),
        "--cots-f-prefetch",
        str(f_prefetch),
        "--cots-cpu-runner",
        "native",
    )


def build_arms(f_values: list[float]) -> list[Arm]:
    arms = [
        Arm(
            name="none",
            strategy="none",
            f_cpu_store=0.0,
            f_prefetch=0.0,
            flags=(),
        )
    ]
    for f_cpu_store in f_values:
        tag = float_tag(f_cpu_store)
        arms.extend(
            [
                Arm(
                    name=f"cots_prefetch_only_f{tag}",
                    strategy="cots_prefetch_only",
                    f_cpu_store=f_cpu_store,
                    f_prefetch=f_cpu_store,
                    flags=cots_flags(f_cpu_store, f_cpu_store),
                ),
                Arm(
                    name=f"cots_cpu_only_f{tag}",
                    strategy="cots_cpu_only",
                    f_cpu_store=f_cpu_store,
                    f_prefetch=0.0,
                    flags=cots_flags(f_cpu_store, 0.0),
                ),
                Arm(
                    name=f"cots_collab_50_f{tag}",
                    strategy="cots_collab_50",
                    f_cpu_store=f_cpu_store,
                    f_prefetch=f_cpu_store * 0.5,
                    flags=cots_flags(f_cpu_store, f_cpu_store * 0.5),
                ),
            ]
        )
    return arms


def filter_arms_in_requested_order(arms: list[Arm], requested: list[str] | None) -> list[Arm]:
    if not requested:
        return arms
    order = {name: idx for idx, name in enumerate(requested)}

    def key(arm: Arm) -> int:
        return min(
            order.get(arm.name, len(order)),
            order.get(arm.strategy, len(order)),
        )

    selected = [
        arm for arm in arms if arm.name in order or arm.strategy in order
    ]
    return sorted(selected, key=key)


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


def cell_json(results_dir: Path, arm: str, batch: int, repeat: int) -> Path:
    return results_dir / f"r{repeat:02d}_{arm}_b{batch}.json"


def should_run_cell(arm: Arm, batch: int, args: argparse.Namespace) -> bool:
    if not args.focused_grid:
        return True
    if arm.strategy == "none":
        return True
    strategy_batches = {
        "cots_cpu_only": args.focused_cpu_batches,
        "cots_prefetch_only": args.focused_prefetch_batches,
        "cots_collab_50": args.focused_collab_batches,
    }
    return batch in strategy_batches.get(arm.strategy, args.batch_sizes)


def run_cell(
    *,
    arm: Arm,
    batch: int,
    repeat: int,
    results_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, int]:
    out_json = cell_json(results_dir, arm.name, batch, repeat)
    out_log = out_json.with_suffix(".log")
    if out_json.exists() and not args.force:
        print(f"  [skip] r={repeat} {arm.name} B={batch} (cached)")
        return out_json, 0

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
        *arm.flags,
        *args.extra_vllm_args,
    ]

    t0 = time.perf_counter()
    with out_log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0:
        avg = parse_avg_latency(out_json)
        print(
            f"  [ok] r={repeat} {arm.name} B={batch}: "
            f"avg={avg:.4f}s ({elapsed:.1f}s)"
        )
    else:
        tail = "\n        ".join(out_log.read_text(errors="replace").splitlines()[-20:])
        print(
            f"  [FAIL] r={repeat} {arm.name} B={batch} "
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


def classify_latency(
    *,
    slowdown: float | None,
    cv: float | None,
    free_margin: float,
    max_cv: float,
    is_baseline: bool,
) -> str:
    if is_baseline:
        return "baseline"
    if slowdown is None:
        return "missing"
    cv_ok = cv is not None and cv <= max_cv
    if slowdown <= 1.0 - free_margin and cv_ok:
        return "faster"
    if slowdown <= 1.0 + free_margin and cv_ok:
        return "free"
    if slowdown <= 1.0 + free_margin:
        return "free_cv_high"
    return "lose"


def summarize(args: argparse.Namespace, arms: list[Arm]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    baseline_by_batch: dict[int, float] = {}

    for batch in args.batch_sizes:
        values = []
        for repeat in range(args.repeat):
            path = cell_json(args.results_dir, "none", batch, repeat)
            if path.exists():
                values.append(parse_avg_latency(path))
        stat = stats(values)
        if stat["mean"] is not None:
            baseline_by_batch[batch] = float(stat["mean"])

    for arm in arms:
        for batch in args.batch_sizes:
            values = []
            for repeat in range(args.repeat):
                path = cell_json(args.results_dir, arm.name, batch, repeat)
                if path.exists():
                    values.append(parse_avg_latency(path))
            stat = stats(values)
            mean = stat["mean"]
            base = baseline_by_batch.get(batch)
            slowdown = None if mean is None or base is None else float(mean) / base
            tokens_per_s = (
                None
                if mean is None
                else float(batch * args.output_len) / float(mean)
            )
            classification = classify_latency(
                slowdown=slowdown,
                cv=stat["cv"],
                free_margin=args.free_margin,
                max_cv=args.max_cv,
                is_baseline=arm.strategy == "none",
            )
            cells.append(
                {
                    "arm": arm.name,
                    "strategy": arm.strategy,
                    "batch_size": batch,
                    "f_cpu_store": arm.f_cpu_store,
                    "f_prefetch": arm.f_prefetch,
                    "num_repeats": stat["n"],
                    "mean_latency_s": mean,
                    "stdev_latency_s": stat["stdev"],
                    "cv": stat["cv"],
                    "slowdown_vs_none": slowdown,
                    "output_tokens_per_s": tokens_per_s,
                    "classification": classification,
                }
            )

    max_free: dict[str, dict[str, float | None]] = {}
    strategies = sorted({arm.strategy for arm in arms if arm.strategy != "none"})
    for strategy in strategies:
        max_free[strategy] = {}
        for batch in args.batch_sizes:
            free_fs = [
                cell["f_cpu_store"]
                for cell in cells
                if cell["strategy"] == strategy
                and cell["batch_size"] == batch
                and cell["classification"] in {"free", "faster"}
            ]
            max_free[strategy][str(batch)] = max(free_fs) if free_fs else None

    return {
        "env": env_info(),
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batch_sizes": args.batch_sizes,
            "f_values": args.f_values,
            "repeat": args.repeat,
            "num_iters": args.num_iters,
            "num_iters_warmup": args.num_iters_warmup,
            "free_margin": args.free_margin,
            "max_cv": args.max_cv,
            "extra_vllm_args": args.extra_vllm_args,
        },
        "cells": cells,
        "max_free_f_cpu_store": max_free,
    }


def fmt_optional(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    cfg = summary["config"]
    batches = cfg["batch_sizes"]
    lines = [
        "# COTS Free-Regime Latency Sweep",
        "",
        f"Model: `{cfg['model']}`; dtype: `{cfg['dtype']}`; "
        f"workload: input={cfg['input_len']}, output={cfg['output_len']}; "
        f"`gpu_memory_utilization={cfg['gpu_memory_utilization']}`.",
        "",
        f"Free threshold: latency `<= {1.0 + cfg['free_margin']:.2f}x` "
        f"no-offload with CV `<= {cfg['max_cv']:.0%}`.",
        "",
        "## Max Free `f_cpu_store`",
        "",
        "| strategy | " + " | ".join(f"B={b}" for b in batches) + " |",
        "|---|" + "|".join("---:" for _ in batches) + "|",
    ]
    for strategy, by_batch in summary["max_free_f_cpu_store"].items():
        lines.append(
            f"| `{strategy}` | "
            + " | ".join(fmt_optional(by_batch[str(b)]) for b in batches)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Cell Summary",
            "",
            "| arm | B | mean s | slowdown | CV | tok/s | verdict |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cell in summary["cells"]:
        if cell["strategy"] == "none":
            continue
        lines.append(
            f"| `{cell['arm']}` | {cell['batch_size']} | "
            f"{fmt_optional(cell['mean_latency_s'], 4)} | "
            f"{fmt_optional(cell['slowdown_vs_none'], 3)} | "
            f"{fmt_optional(cell['cv'], 3)} | "
            f"{fmt_optional(cell['output_tokens_per_s'], 1)} | "
            f"{cell['classification']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.batch_sizes = [1]
    args.f_values = [0.01]
    args.output_len = 16
    args.repeat = 1
    args.num_iters_warmup = 0
    args.num_iters = 1
    tag = float_tag(0.01)
    args.only_arms = [
        "none",
        f"cots_cpu_only_f{tag}",
        f"cots_prefetch_only_f{tag}",
    ]


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
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--f-values", type=float, nargs="+", default=list(DEFAULT_F_VALUES))
    parser.add_argument("--repeat", type=int, default=REPEATS)
    parser.add_argument("--num-iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--num-iters-warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--free-margin", type=float, default=FREE_MARGIN)
    parser.add_argument("--max-cv", type=float, default=MAX_CV)
    parser.add_argument("--only-arms", nargs="*", default=None)
    parser.add_argument(
        "--focused-grid",
        action="store_true",
        help=(
            "Run a smaller shape-aware grid: CPU/collab at low batch and "
            "prefetch at high batch, with baselines for all requested batches."
        ),
    )
    parser.add_argument(
        "--focused-cpu-batches",
        type=int,
        nargs="+",
        default=[1],
        help="Batches used for cots_cpu_only when --focused-grid is set.",
    )
    parser.add_argument(
        "--focused-prefetch-batches",
        type=int,
        nargs="+",
        default=[64],
        help="Batches used for cots_prefetch_only when --focused-grid is set.",
    )
    parser.add_argument(
        "--focused-collab-batches",
        type=int,
        nargs="+",
        default=[1],
        help="Batches used for cots_collab_50 when --focused-grid is set.",
    )
    parser.add_argument("--exp", action="store_true", help="Run missing cells.")
    parser.add_argument("--force", action="store_true", help="Overwrite cached cells.")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run the tiny plan smoke.")
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
    arms = filter_arms_in_requested_order(build_arms(args.f_values), args.only_arms)

    print(
        f"[setup] results={args.results_dir} arms={len(arms)} "
        f"batches={args.batch_sizes} repeats={args.repeat} exp={args.exp}"
    )
    exit_code = 0
    if args.exp:
        for arm in arms:
            for batch in args.batch_sizes:
                if not should_run_cell(arm, batch, args):
                    print(f"  [skip-grid] {arm.name} B={batch}")
                    continue
                for repeat in range(args.repeat):
                    _, rc = run_cell(
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
