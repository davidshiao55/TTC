#!/usr/bin/env python3
"""Phase 1 analysis: COTS KV-capacity throughput sweep.

This harness uses ``vllm bench throughput`` with random synthetic requests and
extracts both throughput and resolved GPU KV-cache capacity from the benchmark
logs. By default it only summarizes cached results; pass ``--exp`` to launch
benchmarks.

Run from the thesis package directory:

    cd /TTC/FastTTS-thesis
    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1_analysis/bench_cots_kv_throughput.py --exp
"""

from __future__ import annotations

import argparse
import json
import platform
import re
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
GPU_MEMORY_UTILIZATION = 0.75
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 8192
NUM_PROMPTS = 512
REPEATS = 3
MAX_CV = 0.03
WIN_MARGIN = 0.05
DEFAULT_PREFETCH_F_VALUES = (0.02, 0.05, 0.09, 0.15, 0.30)
DEFAULT_COLLAB_F_VALUES = (0.05, 0.09, 0.15, 0.30)
DEFAULT_COLLAB_RATIOS = (0.75, 0.90)
FOCUSED_PREFETCH_F_VALUES = (0.02, 0.05, 0.09)
FOCUSED_SHORT_PREFETCH_F_VALUES = (0.02,)
FOCUSED_COLLAB_F_VALUES = (0.09,)
FOCUSED_COLLAB_RATIOS = (0.90,)
DEFAULT_WORKLOADS = {
    "short": (8, 128),
    "medium": (32, 512),
    "long": (32, 1024),
}

NUMBER = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
KV_CACHE_RE = re.compile(r"GPU KV cache size:\s*" + NUMBER + r"\s*tokens")
MAX_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s*" + NUMBER + r"\s*tokens per request:\s*"
    + NUMBER
    + r"x"
)
THROUGHPUT_RE = re.compile(
    r"Throughput:\s*"
    + NUMBER
    + r"\s*requests/s,\s*"
    + NUMBER
    + r"\s*total tokens/s,\s*"
    + NUMBER
    + r"\s*output tokens/s"
)
PROMPT_TOKENS_RE = re.compile(r"Total num prompt tokens:\s*" + NUMBER)
OUTPUT_TOKENS_RE = re.compile(r"Total num output tokens:\s*" + NUMBER)


@dataclass(frozen=True)
class Arm:
    name: str
    strategy: str
    f_cpu_store: float
    f_prefetch: float
    f_prefetch_ratio: float | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class Workload:
    name: str
    input_len: int
    output_len: int


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1_analysis/kv_throughput") / stamp


def parse_number(text: str) -> float:
    return float(text.replace(",", ""))


def parse_int_number(text: str) -> int:
    return int(parse_number(text))


def parse_log_text(text: str) -> dict[str, float | int | None]:
    """Extract stable benchmark metrics from vLLM log text."""
    kv_match = KV_CACHE_RE.search(text)
    concurrency_match = MAX_CONCURRENCY_RE.search(text)
    throughput_match = THROUGHPUT_RE.search(text)
    prompt_match = PROMPT_TOKENS_RE.search(text)
    output_match = OUTPUT_TOKENS_RE.search(text)
    metrics: dict[str, float | int | None] = {
        "kv_cache_tokens": (
            parse_int_number(kv_match.group(1)) if kv_match else None
        ),
        "max_concurrency_request_tokens": (
            parse_int_number(concurrency_match.group(1))
            if concurrency_match
            else None
        ),
        "max_concurrency": (
            parse_number(concurrency_match.group(2)) if concurrency_match else None
        ),
        "requests_per_s_log": (
            parse_number(throughput_match.group(1)) if throughput_match else None
        ),
        "total_tokens_per_s_log": (
            parse_number(throughput_match.group(2)) if throughput_match else None
        ),
        "output_tokens_per_s_log": (
            parse_number(throughput_match.group(3)) if throughput_match else None
        ),
        "total_prompt_tokens_log": (
            parse_int_number(prompt_match.group(1)) if prompt_match else None
        ),
        "total_output_tokens_log": (
            parse_int_number(output_match.group(1)) if output_match else None
        ),
    }
    return metrics


def parse_log_file(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return parse_log_text("")
    return parse_log_text(path.read_text(errors="replace"))


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


def build_arms(
    *,
    prefetch_f_values: list[float],
    collab_f_values: list[float],
    collab_ratios: list[float],
) -> list[Arm]:
    arms = [
        Arm(
            name="none",
            strategy="none",
            f_cpu_store=0.0,
            f_prefetch=0.0,
            f_prefetch_ratio=None,
            flags=(),
        )
    ]
    for f_cpu_store in prefetch_f_values:
        tag = float_tag(f_cpu_store)
        arms.append(
            Arm(
                name=f"cots_prefetch_only_f{tag}",
                strategy="cots_prefetch_only",
                f_cpu_store=f_cpu_store,
                f_prefetch=f_cpu_store,
                f_prefetch_ratio=1.0,
                flags=cots_flags(f_cpu_store, f_cpu_store),
            )
        )
    ratios_by_f = {f: list(collab_ratios) for f in collab_f_values}
    if 0.30 in ratios_by_f and 0.50 not in ratios_by_f[0.30]:
        ratios_by_f[0.30].append(0.50)
    for f_cpu_store in collab_f_values:
        for ratio in sorted(set(ratios_by_f[f_cpu_store])):
            f_prefetch = f_cpu_store * ratio
            arms.append(
                Arm(
                    name=(
                        f"cots_collab_f{float_tag(f_cpu_store)}"
                        f"_r{float_tag(ratio)}"
                    ),
                    strategy="cots_collab",
                    f_cpu_store=f_cpu_store,
                    f_prefetch=f_prefetch,
                    f_prefetch_ratio=ratio,
                    flags=cots_flags(f_cpu_store, f_prefetch),
                )
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


def build_workloads(names: list[str]) -> list[Workload]:
    workloads = []
    for name in names:
        if name not in DEFAULT_WORKLOADS:
            raise ValueError(f"unknown workload {name!r}")
        input_len, output_len = DEFAULT_WORKLOADS[name]
        workloads.append(Workload(name=name, input_len=input_len, output_len=output_len))
    return workloads


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


def cell_stem(workload: str, arm: str, repeat: int) -> str:
    return f"r{repeat:02d}_{workload}_{arm}"


def cell_json(results_dir: Path, workload: str, arm: str, repeat: int) -> Path:
    return results_dir / f"{cell_stem(workload, arm, repeat)}.json"


def build_vllm_command(
    *,
    workload: Workload,
    arm: Arm,
    out_json: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build the vLLM throughput command for one benchmark cell."""
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "throughput",
        "--backend",
        "vllm",
        "--dataset-name",
        "random",
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--random-input-len",
        str(workload.input_len),
        "--random-output-len",
        str(workload.output_len),
        "--random-prefix-len",
        "0",
        "--random-range-ratio",
        "0.0",
        "--num-prompts",
        str(args.num_prompts),
        "--max-model-len",
        str(workload.input_len + workload.output_len + args.max_model_len_slack),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--disable-detokenize",
        "--output-json",
        str(out_json),
        *arm.flags,
        *args.extra_vllm_args,
    ]


def should_run_cell(workload: Workload, arm: Arm, args: argparse.Namespace) -> bool:
    if not args.focused_grid:
        return True
    if arm.strategy == "none":
        return True
    if arm.strategy == "cots_prefetch_only":
        values = (
            args.focused_short_prefetch_f_values
            if workload.name == "short"
            else args.focused_prefetch_f_values
        )
        return arm.f_cpu_store in set(values)
    if arm.strategy == "cots_collab":
        return (
            workload.name == "long"
            and arm.f_cpu_store in set(args.focused_collab_f_values)
            and arm.f_prefetch_ratio in set(args.focused_collab_ratios)
        )
    return False


def run_cell(
    *,
    workload: Workload,
    arm: Arm,
    repeat: int,
    results_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, int]:
    out_json = cell_json(results_dir, workload.name, arm.name, repeat)
    out_log = out_json.with_suffix(".log")
    if out_json.exists() and not args.force:
        print(f"  [skip] r={repeat} {workload.name} {arm.name} (cached)")
        return out_json, 0

    cmd = build_vllm_command(
        workload=workload,
        arm=arm,
        out_json=out_json,
        args=args,
    )
    t0 = time.perf_counter()
    with out_log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0:
        metrics = parse_cell_metrics(out_json, out_log, workload, args.num_prompts)
        print(
            f"  [ok] r={repeat} {workload.name} {arm.name}: "
            f"out_tps={metrics.get('output_tokens_per_s'):.2f} ({elapsed:.1f}s)"
        )
    else:
        tail = "\n        ".join(out_log.read_text(errors="replace").splitlines()[-20:])
        print(
            f"  [FAIL] r={repeat} {workload.name} {arm.name} "
            f"rc={proc.returncode} ({elapsed:.1f}s)\n        {tail}"
        )
    return out_json, proc.returncode


def parse_cell_metrics(
    out_json: Path,
    out_log: Path,
    workload: Workload,
    num_prompts: int,
) -> dict[str, float | int | None]:
    payload = json.loads(out_json.read_text()) if out_json.exists() else {}
    log_metrics = parse_log_file(out_log)
    elapsed = float(payload["elapsed_time"]) if "elapsed_time" in payload else None
    fallback_output_tokens = num_prompts * workload.output_len
    output_tokens_per_s = log_metrics["output_tokens_per_s_log"]
    if output_tokens_per_s is None and elapsed:
        output_tokens_per_s = fallback_output_tokens / elapsed
    total_output_tokens = log_metrics["total_output_tokens_log"]
    if total_output_tokens is None:
        total_output_tokens = fallback_output_tokens

    return {
        "elapsed_time_s": elapsed,
        "requests_per_s": payload.get("requests_per_second"),
        "total_tokens_per_s": payload.get("tokens_per_second"),
        "output_tokens_per_s": output_tokens_per_s,
        "total_num_tokens": payload.get("total_num_tokens"),
        "total_output_tokens": total_output_tokens,
        "kv_cache_tokens": log_metrics["kv_cache_tokens"],
        "max_concurrency_request_tokens": log_metrics[
            "max_concurrency_request_tokens"
        ],
        "max_concurrency": log_metrics["max_concurrency"],
    }


def mean_optional(values: list[float | int | None]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return statistics.mean(nums) if nums else None


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "cv": None}
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = stdev / mean if mean else 0.0
    return {"n": len(values), "mean": mean, "stdev": stdev, "cv": cv}


def classify_throughput(
    *,
    gain: float | None,
    cv: float | None,
    win_margin: float,
    max_cv: float,
    is_baseline: bool,
) -> str:
    if is_baseline:
        return "baseline"
    if gain is None:
        return "missing"
    cv_ok = cv is not None and cv <= max_cv
    if gain >= 1.0 + win_margin and cv_ok:
        return "win"
    if gain >= 1.0 + win_margin:
        return "win_cv_high"
    if gain >= 1.0 - win_margin and cv_ok:
        return "tie"
    if gain >= 1.0 - win_margin:
        return "tie_cv_high"
    return "lose"


def summarize(
    args: argparse.Namespace,
    arms: list[Arm],
    workloads: list[Workload],
) -> dict[str, Any]:
    baseline_by_workload: dict[str, dict[str, float | None]] = {}
    cells: list[dict[str, Any]] = []

    raw_metrics: dict[tuple[str, str], list[dict[str, float | int | None]]] = {}
    for workload in workloads:
        for arm in arms:
            if not should_run_cell(workload, arm, args):
                continue
            records = []
            for repeat in range(args.repeat):
                out_json = cell_json(args.results_dir, workload.name, arm.name, repeat)
                out_log = out_json.with_suffix(".log")
                if out_json.exists():
                    records.append(
                        parse_cell_metrics(
                            out_json, out_log, workload, args.num_prompts
                        )
                    )
            raw_metrics[(workload.name, arm.name)] = records

    for workload in workloads:
        records = raw_metrics.get((workload.name, "none"), [])
        output_values = [
            float(r["output_tokens_per_s"])
            for r in records
            if r["output_tokens_per_s"] is not None
        ]
        baseline_by_workload[workload.name] = {
            "output_tokens_per_s": stats(output_values)["mean"],
            "kv_cache_tokens": mean_optional([r["kv_cache_tokens"] for r in records]),
            "max_concurrency": mean_optional([r["max_concurrency"] for r in records]),
        }

    for workload in workloads:
        base = baseline_by_workload[workload.name]
        for arm in arms:
            if not should_run_cell(workload, arm, args):
                continue
            records = raw_metrics.get((workload.name, arm.name), [])
            output_values = [
                float(r["output_tokens_per_s"])
                for r in records
                if r["output_tokens_per_s"] is not None
            ]
            output_stat = stats(output_values)
            out_mean = output_stat["mean"]
            base_out = base["output_tokens_per_s"]
            gain = None if out_mean is None or base_out is None else out_mean / base_out
            kv_tokens = mean_optional([r["kv_cache_tokens"] for r in records])
            base_kv = base["kv_cache_tokens"]
            kv_gain = None if kv_tokens is None or base_kv is None else kv_tokens / base_kv
            verdict = classify_throughput(
                gain=gain,
                cv=output_stat["cv"],
                win_margin=args.win_margin,
                max_cv=args.max_cv,
                is_baseline=arm.strategy == "none",
            )
            cells.append(
                {
                    "workload": workload.name,
                    "input_len": workload.input_len,
                    "output_len": workload.output_len,
                    "arm": arm.name,
                    "strategy": arm.strategy,
                    "f_cpu_store": arm.f_cpu_store,
                    "f_prefetch": arm.f_prefetch,
                    "f_prefetch_ratio": arm.f_prefetch_ratio,
                    "num_repeats": output_stat["n"],
                    "mean_output_tokens_per_s": out_mean,
                    "stdev_output_tokens_per_s": output_stat["stdev"],
                    "cv_output_tokens_per_s": output_stat["cv"],
                    "throughput_gain_vs_none": gain,
                    "kv_cache_tokens": kv_tokens,
                    "kv_capacity_gain_vs_none": kv_gain,
                    "max_concurrency": mean_optional(
                        [r["max_concurrency"] for r in records]
                    ),
                    "max_concurrency_request_tokens": mean_optional(
                        [r["max_concurrency_request_tokens"] for r in records]
                    ),
                    "classification": verdict,
                }
            )

    winning_cells = [
        cell
        for cell in cells
        if cell["classification"] == "win" and cell["strategy"] != "none"
    ]
    return {
        "env": env_info(),
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "num_prompts": args.num_prompts,
            "max_model_len_slack": args.max_model_len_slack,
            "repeat": args.repeat,
            "max_cv": args.max_cv,
            "win_margin": args.win_margin,
            "focused_grid": args.focused_grid,
            "focused_prefetch_f_values": args.focused_prefetch_f_values,
            "focused_short_prefetch_f_values": args.focused_short_prefetch_f_values,
            "focused_collab_f_values": args.focused_collab_f_values,
            "focused_collab_ratios": args.focused_collab_ratios,
            "workloads": [
                {
                    "name": w.name,
                    "input_len": w.input_len,
                    "output_len": w.output_len,
                }
                for w in workloads
            ],
            "extra_vllm_args": args.extra_vllm_args,
        },
        "cells": cells,
        "winning_cells": winning_cells,
    }


def fmt_optional(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    cfg = summary["config"]
    lines = [
        "# COTS KV-Capacity Throughput Sweep",
        "",
        f"Model: `{cfg['model']}`; dtype: `{cfg['dtype']}`; "
        f"`gpu_memory_utilization={cfg['gpu_memory_utilization']}`; "
        f"`max_num_seqs={cfg['max_num_seqs']}`; "
        f"`max_num_batched_tokens={cfg['max_num_batched_tokens']}`.",
        "",
        f"Win threshold: output-token throughput `>= {1.0 + cfg['win_margin']:.2f}x` "
        f"no-offload with CV `<= {cfg['max_cv']:.0%}`.",
        "",
        "## Winning Cells",
        "",
        "| workload | arm | output tok/s | gain | KV tokens | KV gain | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    if summary["winning_cells"]:
        for cell in summary["winning_cells"]:
            lines.append(
                f"| `{cell['workload']}` | `{cell['arm']}` | "
                f"{fmt_optional(cell['mean_output_tokens_per_s'], 2)} | "
                f"{fmt_optional(cell['throughput_gain_vs_none'], 3)} | "
                f"{fmt_optional(cell['kv_cache_tokens'], 0)} | "
                f"{fmt_optional(cell['kv_capacity_gain_vs_none'], 3)} | "
                f"{cell['classification']} |"
            )
    else:
        lines.append("| — | — | — | — | — | — | no winning cells in cached data |")

    lines.extend(
        [
            "",
            "## Cell Summary",
            "",
            "| workload | arm | output tok/s | gain | CV | KV tokens | KV gain | max conc | verdict |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cell in summary["cells"]:
        lines.append(
            f"| `{cell['workload']}` | `{cell['arm']}` | "
            f"{fmt_optional(cell['mean_output_tokens_per_s'], 2)} | "
            f"{fmt_optional(cell['throughput_gain_vs_none'], 3)} | "
            f"{fmt_optional(cell['cv_output_tokens_per_s'], 3)} | "
            f"{fmt_optional(cell['kv_cache_tokens'], 0)} | "
            f"{fmt_optional(cell['kv_capacity_gain_vs_none'], 3)} | "
            f"{fmt_optional(cell['max_concurrency'], 2)} | "
            f"{cell['classification']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION
    )
    parser.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    parser.add_argument("--max-num-batched-tokens", type=int, default=MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument(
        "--max-model-len-slack",
        type=int,
        default=1,
        help=(
            "Extra tokens added to input_len + output_len for --max-model-len. "
            "vLLM throughput asserts this limit is strictly greater than the "
            "synthetic request length."
        ),
    )
    parser.add_argument("--repeat", type=int, default=REPEATS)
    parser.add_argument("--max-cv", type=float, default=MAX_CV)
    parser.add_argument("--win-margin", type=float, default=WIN_MARGIN)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=sorted(DEFAULT_WORKLOADS),
        default=list(DEFAULT_WORKLOADS),
    )
    parser.add_argument(
        "--prefetch-f-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_PREFETCH_F_VALUES),
    )
    parser.add_argument(
        "--collab-f-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_COLLAB_F_VALUES),
    )
    parser.add_argument(
        "--collab-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_COLLAB_RATIOS),
    )
    parser.add_argument(
        "--focused-grid",
        action="store_true",
        help=(
            "Use the post-free-regime focused KV grid: all baselines; short "
            "negative control at tiny prefetch only; medium/long prefetch-only "
            "at plausible fractions; long-only prefetch-heavy collaborative "
            "diagnostic."
        ),
    )
    parser.add_argument(
        "--focused-prefetch-f-values",
        type=float,
        nargs="+",
        default=list(FOCUSED_PREFETCH_F_VALUES),
        help="Prefetch-only fractions used for medium/long when --focused-grid is set.",
    )
    parser.add_argument(
        "--focused-short-prefetch-f-values",
        type=float,
        nargs="+",
        default=list(FOCUSED_SHORT_PREFETCH_F_VALUES),
        help="Prefetch-only fractions used for short when --focused-grid is set.",
    )
    parser.add_argument(
        "--focused-collab-f-values",
        type=float,
        nargs="+",
        default=list(FOCUSED_COLLAB_F_VALUES),
        help="Collaborative f_cpu_store values used on long only with --focused-grid.",
    )
    parser.add_argument(
        "--focused-collab-ratios",
        type=float,
        nargs="+",
        default=list(FOCUSED_COLLAB_RATIOS),
        help="Collaborative prefetch ratios used on long only with --focused-grid.",
    )
    parser.add_argument("--only-arms", nargs="*", default=None)
    parser.add_argument("--exp", action="store_true", help="Run missing cells.")
    parser.add_argument("--force", action="store_true", help="Overwrite cached cells.")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--extra-vllm-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments appended after all harness-controlled vLLM args.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    workloads = build_workloads(args.workloads)
    arms = filter_arms_in_requested_order(
        build_arms(
            prefetch_f_values=args.prefetch_f_values,
            collab_f_values=args.collab_f_values,
            collab_ratios=args.collab_ratios,
        ),
        args.only_arms,
    )

    print(
        f"[setup] results={args.results_dir} workloads={len(workloads)} "
        f"arms={len(arms)} repeats={args.repeat} exp={args.exp}"
    )
    exit_code = 0
    if args.exp:
        for workload in workloads:
            for arm in arms:
                if not should_run_cell(workload, arm, args):
                    continue
                for repeat in range(args.repeat):
                    _, rc = run_cell(
                        workload=workload,
                        arm=arm,
                        repeat=repeat,
                        results_dir=args.results_dir,
                        args=args,
                    )
                    if rc != 0:
                        exit_code = rc
                        if not args.keep_going:
                            summary = summarize(args, arms, workloads)
                            (args.results_dir / "summary.json").write_text(
                                json.dumps(summary, indent=2)
                            )
                            write_markdown_summary(
                                summary, args.results_dir / "summary.md"
                            )
                            return exit_code

    summary = summarize(args, arms, workloads)
    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown_summary(summary, args.results_dir / "summary.md")
    print(f"[summary] wrote {summary_path}")
    print(f"[summary] wrote {args.results_dir / 'summary.md'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
