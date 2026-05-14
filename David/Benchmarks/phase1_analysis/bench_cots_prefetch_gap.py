#!/usr/bin/env python3
"""Phase 1 analysis: decompose the COTS pure-prefetch "not free" gap.

This harness compares no-offload latency against COTS pure-prefetch with and
without ``--cots-dry-run``.  The public COTS dry-run is a control-plane
diagnostic: it keeps COTS wrappers, bucket lookup, slot bookkeeping, events,
and graph behavior, but skips active offloaded work from both paths.  For pure
prefetch, that means no H2D prefetch copy and no prefetched-slice GPU compute
contribution.  The real-vs-dry delta estimates active offloaded-work cost; the
dry-vs-none delta estimates the remaining COTS control-plane/shape cost.

Run from the thesis package directory:

    cd /TTC/FastTTS-thesis
    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1_analysis/bench_cots_prefetch_gap.py --exp
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
DEFAULT_BATCHES = (64,)
DEFAULT_F_VALUES = (0.005, 0.01, 0.02, 0.0357)
DEFAULT_MODES = ("graph",)
DEFAULT_H2D_BW_GBPS = 28.0
MLP_CHANNEL_GRANULARITY = 64

# Fallback for Qwen/Qwen2.5-7B-Instruct if transformers config loading is not
# available in a dry summary-only environment.
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
    dry_run: bool
    flags: tuple[str, ...]


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1_analysis/prefetch_gap") / stamp


def float_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def cots_flags(
    f: float,
    *,
    f_prefetch: float | None = None,
    dry_run: bool,
) -> tuple[str, ...]:
    if f_prefetch is None:
        f_prefetch = f
    flags = [
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        str(f),
        "--cots-f-prefetch",
        str(f_prefetch),
        "--cots-cpu-runner",
        "native",
    ]
    if dry_run:
        flags.append("--cots-dry-run")
    return tuple(flags)


def build_arms(
    f_values: list[float],
    *,
    include_empty_cots: bool,
) -> list[Arm]:
    arms = [Arm("none", "none", None, False, ())]
    if include_empty_cots:
        arms.append(
            Arm(
                "cots_empty",
                "cots_empty",
                0.0,
                False,
                cots_flags(0.0, dry_run=False),
            )
        )
    for f in f_values:
        tag = float_tag(f)
        arms.append(
            Arm(
                f"cots_real_f{tag}",
                "cots_real",
                f,
                False,
                cots_flags(f, dry_run=False),
            )
        )
        arms.append(
            Arm(
                f"cots_dry_f{tag}",
                "cots_dry",
                f,
                True,
                cots_flags(f, dry_run=True),
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


def prefetch_bytes_per_layer(shape: dict[str, int], f: float, dtype_bytes: int) -> int:
    hidden = shape["hidden_size"]
    intermediate = shape["intermediate_size"]
    kv_heads = shape["num_key_value_heads"]
    head_dim = shape["head_dim"]
    qkv_out = hidden + 2 * kv_heads * head_dim

    qkv_rows = snapped_qkv_rows(hidden, kv_heads * head_dim, head_dim, f * qkv_out)
    mlp_half = snap_mlp_channels(f * intermediate, intermediate)
    gate_up_rows = 2 * mlp_half
    down_rows = mlp_half
    elems = qkv_rows * hidden + gate_up_rows * hidden + down_rows * hidden
    return int(elems * dtype_bytes)


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


def summarize(
    args: argparse.Namespace,
    arms: list[Arm],
    modes: list[str],
    shape: dict[str, int],
) -> dict[str, Any]:
    dtype_bytes = 2 if args.dtype in ("bfloat16", "float16", "half") else 4
    layers = shape["num_hidden_layers"]
    full_layer_bytes = full_target_bytes_per_layer(shape, dtype_bytes)
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
            layer_window_s = (
                None
                if base is None
                else float(base) / args.output_len / layers
            )
            ideal_f_free = (
                None
                if layer_window_s is None
                else layer_window_s
                / (full_layer_bytes / (args.h2d_bw_gbps * 1e9))
            )
            for arm in arms:
                values = grouped.get((mode, batch, arm.name), [])
                st = stats(values)
                mean = st["mean"]
                slowdown = None if mean is None or base is None else mean / base
                extra_s = None if mean is None or base is None else mean - base
                theory = None
                theory_total = None
                if arm.f is not None:
                    bytes_layer = prefetch_bytes_per_layer(
                        shape, arm.f, dtype_bytes
                    )
                    theory = bytes_layer / (args.h2d_bw_gbps * 1e9)
                    theory_total = theory * layers * args.output_len
                cells.append(
                    {
                        "mode": mode,
                        "batch": batch,
                        "arm": arm.name,
                        "family": arm.family,
                        "f": arm.f,
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
                        "theory_h2d_ms_per_layer": (
                            None if theory is None else theory * 1000.0
                        ),
                        "theory_h2d_total_s_unoverlapped": theory_total,
                        "ideal_f_free_from_layer_window": ideal_f_free,
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
            "modes": modes,
            "repeat": args.repeat,
            "num_iters": args.num_iters,
            "num_iters_warmup": args.num_iters_warmup,
            "h2d_bw_gbps": args.h2d_bw_gbps,
            "full_target_bytes_per_layer": full_layer_bytes,
            "extra_vllm_args": args.extra_vllm_args,
        },
        "cells": cells,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    cfg = summary["config"]
    lines = [
        "# COTS Prefetch Gap Decomposition",
        "",
        f"Model: `{cfg['model']}`; dtype: `{cfg['dtype']}`; "
        f"input/output `{cfg['input_len']}/{cfg['output_len']}`; "
        f"`gpu_memory_utilization={cfg['gpu_memory_utilization']}`.",
        "",
        f"Theory uses H2D bandwidth `{cfg['h2d_bw_gbps']:.1f} GB/s` and COTS "
        "Phase-1 target tensors only: `qkv_proj`, `gate_up_proj`, `down_proj`.",
        "",
        "## Cells",
        "",
        "| mode | B | arm | mean s | slowdown | extra ms/(tok*layer) | theory H2D ms/layer | theory serial s | ideal free f |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        lines.append(
            f"| `{cell['mode']}` | {cell['batch']} | `{cell['arm']}` | "
            f"{fmt(cell['mean_latency_s'], 4)} | "
            f"{fmt(cell['slowdown_vs_none'], 3)} | "
            f"{fmt(cell['extra_ms_per_token_layer'], 3)} | "
            f"{fmt(cell['theory_h2d_ms_per_layer'], 3)} | "
            f"{fmt(cell['theory_h2d_total_s_unoverlapped'], 3)} | "
            f"{fmt(cell['ideal_f_free_from_layer_window'], 3)} |"
        )

    lines.extend(["", "## Real Minus Control Dry-Run", ""])
    lines.extend(
        [
            "| mode | B | f | real extra ms/(tok*layer) | dry extra ms/(tok*layer) | active offloaded-work ms/(tok*layer) | active/theory-H2D |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_key = {
        (cell["mode"], cell["batch"], cell["family"], cell["f"]): cell
        for cell in summary["cells"]
    }
    for cell in summary["cells"]:
        if cell["family"] != "cots_real" or cell["f"] is None:
            continue
        dry = by_key.get((cell["mode"], cell["batch"], "cots_dry", cell["f"]))
        real_extra = cell["extra_ms_per_token_layer"]
        dry_extra = None if dry is None else dry["extra_ms_per_token_layer"]
        active = (
            None
            if real_extra is None or dry_extra is None
            else real_extra - dry_extra
        )
        theory = cell["theory_h2d_ms_per_layer"]
        active_ratio = None if active is None or not theory else active / theory
        lines.append(
            f"| `{cell['mode']}` | {cell['batch']} | {cell['f']:.4f} | "
            f"{fmt(real_extra, 3)} | {fmt(dry_extra, 3)} | "
            f"{fmt(active, 3)} | {fmt(active_ratio, 3)} |"
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
    parser.add_argument("--modes", nargs="+", choices=("graph", "eager"), default=list(DEFAULT_MODES))
    parser.add_argument("--num-iters-warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--num-iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--repeat", type=int, default=REPEATS)
    parser.add_argument("--h2d-bw-gbps", type=float, default=DEFAULT_H2D_BW_GBPS)
    parser.add_argument("--include-empty-cots", action="store_true")
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
    arms = build_arms(
        args.f_values,
        include_empty_cots=args.include_empty_cots,
    )
    shape = load_model_shape(args.model)

    print(
        f"[setup] results={args.results_dir} modes={modes} "
        f"batches={args.batch_sizes} arms={len(arms)} repeat={args.repeat} "
        f"exp={args.exp}"
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
    raise SystemExit(main())
