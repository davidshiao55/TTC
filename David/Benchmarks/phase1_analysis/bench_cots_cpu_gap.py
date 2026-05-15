#!/usr/bin/env python3
"""Phase 1 analysis: decompose the COTS CPU-compute gap vs no offload.

This is the CPU-compute sibling of ``bench_cots_prefetch_gap.py``.  It runs
fresh vLLM latency processes for:

* no offload;
* COTS CPU-only real work: ``f_prefetch=0``;
* COTS CPU-only dry-run: same storage/control path, but active CPU GEMM is
  skipped by ``--cots-dry-run``.

The real-vs-dry delta estimates exposed CPU-compute cost in the real vLLM
forward path.  Dry-run still uses the reduced GPU-resident split shape, so
dry-vs-none is not just control overhead: it is the COTS control floor plus
the GPU work removed by CPU placement.  The summary also reports the
snap64/head-aligned effective fraction, which is the right quantity to compare
against the Phase 0 microbench overlap bound.

Run from the thesis package directory:

    cd /TTC/FastTTS-thesis
    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1_analysis/bench_cots_cpu_gap.py --exp
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
DEFAULT_BATCHES = (1,)
DEFAULT_F_VALUES = (0.002, 0.005, 0.0068, 0.01, 0.0135, 0.02, 0.0357, 0.05)
DEFAULT_THREAD_COUNTS = (16,)
DEFAULT_MODES = ("graph",)
MLP_CHANNEL_GRANULARITY = 64

FALLBACK_SHAPE = {
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
}


@dataclass(frozen=True)
class Arm:
    name: str
    family: str
    f: float | None
    threads: int | None
    dry_run: bool
    flags: tuple[str, ...]


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1_analysis/cpu_gap") / stamp


def float_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def cots_cpu_flags(
    f: float,
    *,
    threads: int,
    dry_run: bool,
    affinity: list[int] | None,
) -> tuple[str, ...]:
    flags = [
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        str(f),
        "--cots-f-prefetch",
        "0.0",
        "--cots-cpu-runner",
        "native",
        "--cots-cpu-num-threads",
        str(threads),
    ]
    if affinity:
        flags.append("--cots-cpu-worker-affinity")
        flags.extend(str(cpu) for cpu in affinity)
    if dry_run:
        flags.append("--cots-dry-run")
    return tuple(flags)


def build_arms(args: argparse.Namespace) -> list[Arm]:
    arms = [Arm("none", "none", None, None, False, ())]
    for threads in args.thread_counts:
        for f in args.f_values:
            tag = float_tag(f)
            arms.append(
                Arm(
                    f"cots_cpu_real_t{threads}_f{tag}",
                    "cpu_real",
                    f,
                    threads,
                    False,
                    cots_cpu_flags(
                        f,
                        threads=threads,
                        dry_run=False,
                        affinity=args.cpu_worker_affinity,
                    ),
                )
            )
            arms.append(
                Arm(
                    f"cots_cpu_dry_t{threads}_f{tag}",
                    "cpu_dry",
                    f,
                    threads,
                    True,
                    cots_cpu_flags(
                        f,
                        threads=threads,
                        dry_run=True,
                        affinity=args.cpu_worker_affinity,
                    ),
                )
            )
    return arms


def cell_json(results_dir: Path, mode: str, arm: str, batch: int, repeat: int) -> Path:
    return results_dir / f"r{repeat:02d}_{mode}_{arm}_b{batch}.json"


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
        mode=mode, arm=arm, batch=batch, out_json=out_json, args=args
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


def load_model_shape(model: str) -> dict[str, int]:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=False)
        hidden = int(cfg.hidden_size)
        heads = int(cfg.num_attention_heads)
        head_dim = int(getattr(cfg, "head_dim", hidden // heads))
        return {
            "hidden_size": hidden,
            "intermediate_size": int(cfg.intermediate_size),
            "num_hidden_layers": int(cfg.num_hidden_layers),
            "num_attention_heads": heads,
            "num_key_value_heads": int(cfg.num_key_value_heads),
            "head_dim": head_dim,
        }
    except Exception:
        return dict(FALLBACK_SHAPE)


def snap_mlp_channels(requested: float, limit: int) -> int:
    if requested <= 0:
        return 0
    if requested >= limit:
        return limit
    gran = MLP_CHANNEL_GRANULARITY
    return min(int(((requested + gran / 2) // gran) * gran), limit)


def snapped_qkv_rows(
    q_size: int,
    kv_size: int,
    head_dim: int,
    requested: float,
) -> int:
    """Mirror COTS' KV-biased QKV snapping for benchmark theory metadata."""
    total = q_size + 2 * kv_size
    n_cpu_cols = round(requested)
    if n_cpu_cols <= 0:
        return 0
    if n_cpu_cols >= total:
        return total
    kv_total = 2 * kv_size
    if n_cpu_cols <= kv_total:
        n_kv_heads = kv_size // head_dim
        n_pairs = min(round(n_cpu_cols / (2 * head_dim)), n_kv_heads)
        return 2 * n_pairs * head_dim
    n_q_tail_raw = n_cpu_cols - kv_total
    n_q_heads = min(round(n_q_tail_raw / head_dim), q_size // head_dim)
    return n_q_heads * head_dim + kv_total


def full_target_bytes_per_layer(shape: dict[str, int], dtype_bytes: int) -> int:
    hidden = shape["hidden_size"]
    intermediate = shape["intermediate_size"]
    kv_heads = shape["num_key_value_heads"]
    head_dim = shape["head_dim"]
    qkv_out = hidden + 2 * kv_heads * head_dim
    elems = qkv_out * hidden + 3 * intermediate * hidden
    return int(elems * dtype_bytes)


def cpu_bytes_per_layer(shape: dict[str, int], f: float, dtype_bytes: int) -> dict[str, Any]:
    hidden = shape["hidden_size"]
    intermediate = shape["intermediate_size"]
    kv_heads = shape["num_key_value_heads"]
    head_dim = shape["head_dim"]
    qkv_out = hidden + 2 * kv_heads * head_dim
    qkv_rows = snapped_qkv_rows(hidden, kv_heads * head_dim, head_dim, f * qkv_out)
    mlp_half = snap_mlp_channels(f * intermediate, intermediate)
    elems = qkv_rows * hidden + 3 * mlp_half * hidden
    bytes_layer = int(elems * dtype_bytes)
    full_bytes = full_target_bytes_per_layer(shape, dtype_bytes)
    return {
        "qkv_rows": qkv_rows,
        "mlp_half_channels": mlp_half,
        "bytes_per_layer": bytes_layer,
        "actual_f_by_target_bytes": bytes_layer / full_bytes if full_bytes else None,
    }


def phase0_microbench_bound(batch: int) -> float:
    if batch == 1:
        return 0.05
    if batch == 4:
        return 0.03
    return 0.0


def summarize(
    args: argparse.Namespace,
    arms: list[Arm],
    modes: list[str],
    shape: dict[str, int],
) -> dict[str, Any]:
    dtype_bytes = 2 if args.dtype in ("bfloat16", "float16", "half") else 4
    layers = shape["num_hidden_layers"]
    cells: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str], list[float]] = {}

    for mode in modes:
        for batch in args.batch_sizes:
            for arm in arms:
                values = []
                for repeat in range(args.repeat):
                    out_json = cell_json(args.results_dir, mode, arm.name, batch, repeat)
                    if out_json.exists():
                        values.append(parse_avg_latency(out_json))
                grouped[(mode, batch, arm.name)] = values

    for mode in modes:
        for batch in args.batch_sizes:
            base_stat = stats(grouped.get((mode, batch, "none"), []))
            base = base_stat["mean"]
            for arm in arms:
                values = grouped.get((mode, batch, arm.name), [])
                st = stats(values)
                mean = st["mean"]
                slowdown = None if mean is None or base is None else mean / base
                extra_s = None if mean is None or base is None else mean - base
                geom = (
                    {
                        "qkv_rows": None,
                        "mlp_half_channels": None,
                        "bytes_per_layer": None,
                        "actual_f_by_target_bytes": None,
                    }
                    if arm.f is None
                    else cpu_bytes_per_layer(shape, arm.f, dtype_bytes)
                )
                cells.append(
                    {
                        "mode": mode,
                        "batch": batch,
                        "arm": arm.name,
                        "family": arm.family,
                        "requested_f": arm.f,
                        "threads": arm.threads,
                        "dry_run": arm.dry_run,
                        "n": st["n"],
                        "mean_latency_s": mean,
                        "cv": st["cv"],
                        "slowdown_vs_none": slowdown,
                        "extra_s_vs_none": extra_s,
                        "extra_ms_per_token_layer": (
                            None
                            if extra_s is None
                            else extra_s * 1000.0 / args.output_len / layers
                        ),
                        "phase0_microbench_bound_f": phase0_microbench_bound(batch),
                        **geom,
                    }
                )

    return {
        "env": env_info(),
        "shape": shape,
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batch_sizes": args.batch_sizes,
            "f_values": args.f_values,
            "thread_counts": args.thread_counts,
            "cpu_worker_affinity": args.cpu_worker_affinity,
            "modes": modes,
            "repeat": args.repeat,
            "num_iters": args.num_iters,
            "num_iters_warmup": args.num_iters_warmup,
            "extra_vllm_args": args.extra_vllm_args,
        },
        "cells": cells,
        "pairs": build_pairs(cells, output_len=args.output_len, layers=layers),
    }


def build_pairs(
    cells: list[dict[str, Any]],
    *,
    output_len: int,
    layers: int,
) -> list[dict[str, Any]]:
    by_key = {
        (
            cell["mode"],
            cell["batch"],
            cell["family"],
            cell["threads"],
            cell["requested_f"],
        ): cell
        for cell in cells
    }
    pairs: list[dict[str, Any]] = []
    for real in cells:
        if real["family"] != "cpu_real" or real["requested_f"] is None:
            continue
        dry = by_key.get(
            (
                real["mode"],
                real["batch"],
                "cpu_dry",
                real["threads"],
                real["requested_f"],
            )
        )
        if dry is None:
            continue
        real_extra = real["extra_s_vs_none"]
        dry_extra = dry["extra_s_vs_none"]
        active_s = (
            None
            if real_extra is None or dry_extra is None
            else real_extra - dry_extra
        )
        base_s = (
            None
            if real["mean_latency_s"] is None or real_extra is None
            else real["mean_latency_s"] - real_extra
        )
        actual_f = real["actual_f_by_target_bytes"]
        active_ms_per_token_layer = (
            None
            if active_s is None
            else active_s * 1000.0 / output_len / layers
        )
        active_ms_per_token_layer_per_f = (
            None
            if active_ms_per_token_layer is None or not actual_f
            else active_ms_per_token_layer / actual_f
        )
        budget_with_dry = (
            None
            if base_s is None or dry["mean_latency_s"] is None
            else 1.05 * base_s - dry["mean_latency_s"]
        )
        free_f_5pct_with_dry = (
            None
            if active_s is None
            or active_s <= 0
            or actual_f is None
            or budget_with_dry is None
            else max(0.0, actual_f * budget_with_dry / active_s)
        )
        free_f_5pct_active_only = (
            None
            if active_s is None or active_s <= 0 or actual_f is None or base_s is None
            else max(0.0, actual_f * (0.05 * base_s) / active_s)
        )
        pairs.append(
            {
                "mode": real["mode"],
                "batch": real["batch"],
                "threads": real["threads"],
                "requested_f": real["requested_f"],
                "actual_f_by_target_bytes": actual_f,
                "qkv_rows": real["qkv_rows"],
                "mlp_half_channels": real["mlp_half_channels"],
                "real_slowdown": real["slowdown_vs_none"],
                "dry_slowdown": dry["slowdown_vs_none"],
                "active_s_vs_dry": active_s,
                "active_ms_per_token_layer": active_ms_per_token_layer,
                "active_ms_per_token_layer_per_f": active_ms_per_token_layer_per_f,
                "phase0_microbench_bound_f": real["phase0_microbench_bound_f"],
                "estimated_5pct_free_f_with_dry_floor": free_f_5pct_with_dry,
                "estimated_5pct_free_f_active_only": free_f_5pct_active_only,
            }
        )
    return pairs


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    cfg = summary["config"]
    lines = [
        "# COTS CPU-Compute Gap Decomposition",
        "",
        f"Model: `{cfg['model']}`; dtype: `{cfg['dtype']}`; "
        f"input/output `{cfg['input_len']}/{cfg['output_len']}`; "
        f"`gpu_memory_utilization={cfg['gpu_memory_utilization']}`.",
        "",
        "Theory metadata mirrors current COTS geometry: KV-head-aligned QKV "
        "assignment plus snap64 MLP channels. `actual_f` is the effective "
        "fraction of Phase-1 target bytes (`qkv_proj`, `gate_up_proj`, "
        "`down_proj`) that are CPU-computed.",
        "",
        "## Cells",
        "",
        "| mode | B | t | arm | req f | actual f | qkv rows | mlp half | mean s | slowdown | extra ms/(tok*layer) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        lines.append(
            f"| `{cell['mode']}` | {cell['batch']} | "
            f"{fmt(cell['threads'], 0)} | `{cell['arm']}` | "
            f"{fmt(cell['requested_f'], 4)} | "
            f"{fmt(cell['actual_f_by_target_bytes'], 4)} | "
            f"{fmt(cell['qkv_rows'], 0)} | "
            f"{fmt(cell['mlp_half_channels'], 0)} | "
            f"{fmt(cell['mean_latency_s'], 4)} | "
            f"{fmt(cell['slowdown_vs_none'], 3)} | "
            f"{fmt(cell['extra_ms_per_token_layer'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Real Minus Control Dry-Run",
            "",
            "| mode | B | t | req f | actual f | qkv rows | mlp half | real slow | dry slow | active ms/(tok*layer) | active ms/(tok*layer*f) | est 5% free f w/ dry | est 5% free f active-only | Phase0 bound f |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in summary["pairs"]:
        lines.append(
            f"| `{pair['mode']}` | {pair['batch']} | "
            f"{fmt(pair['threads'], 0)} | "
            f"{fmt(pair['requested_f'], 4)} | "
            f"{fmt(pair['actual_f_by_target_bytes'], 4)} | "
            f"{fmt(pair['qkv_rows'], 0)} | "
            f"{fmt(pair['mlp_half_channels'], 0)} | "
            f"{fmt(pair['real_slowdown'], 3)} | "
            f"{fmt(pair['dry_slowdown'], 3)} | "
            f"{fmt(pair['active_ms_per_token_layer'], 3)} | "
            f"{fmt(pair['active_ms_per_token_layer_per_f'], 1)} | "
            f"{fmt(pair['estimated_5pct_free_f_with_dry_floor'], 4)} | "
            f"{fmt(pair['estimated_5pct_free_f_active_only'], 4)} | "
            f"{fmt(pair['phase0_microbench_bound_f'], 3)} |"
        )

    path.write_text("\n".join(lines) + "\n")


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
    parser.add_argument(
        "--thread-counts", type=int, nargs="+", default=list(DEFAULT_THREAD_COUNTS)
    )
    parser.add_argument("--modes", nargs="+", choices=("graph", "eager"), default=list(DEFAULT_MODES))
    parser.add_argument("--num-iters-warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--num-iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--repeat", type=int, default=REPEATS)
    parser.add_argument("--cpu-worker-affinity", type=int, nargs="+", default=None)
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
    modes = list(args.modes)
    arms = build_arms(args)
    shape = load_model_shape(args.model)

    print(
        f"[setup] results={args.results_dir} modes={modes} "
        f"batches={args.batch_sizes} threads={args.thread_counts} "
        f"arms={len(arms)} repeat={args.repeat} exp={args.exp}"
    )
    exit_code = 0
    if args.exp:
        for mode in modes:
            for batch in args.batch_sizes:
                for arm in arms:
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
                                summary = summarize(args, arms, modes, shape)
                                (args.results_dir / "summary.json").write_text(
                                    json.dumps(summary, indent=2)
                                )
                                write_markdown_summary(
                                    summary, args.results_dir / "summary.md"
                                )
                                return exit_code

    summary = summarize(args, arms, modes, shape)
    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown_summary(summary, args.results_dir / "summary.md")
    print(f"[summary] wrote {summary_path}")
    print(f"[summary] wrote {args.results_dir / 'summary.md'}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
