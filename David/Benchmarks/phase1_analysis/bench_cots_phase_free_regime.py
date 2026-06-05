#!/usr/bin/env python3
"""Phase-aware COTS free-regime latency sweep and oracle dispatch builder.

The original Phase 1 free-regime sweep used one decode-heavy request shape
(`input_len=8`, `output_len=128`) and classified a storage fraction as "free"
from whole-request latency. That hides the fact that the COTS route is selected
per vLLM dispatch bucket: a CPU-compute route can be free for decode, but the
same global route may make the request look non-free when prefill exposes CPU
work.

This harness keeps the old fresh-process measurement style, but adds:

* multiple workload shapes, so prefill-heavy and decode-heavy cells are
  summarized independently;
* explicit Planner-style COTS dispatch tables;
* phase-aware arms such as "decode CPU, non-decode prefetch".
* an empirical oracle mode that sweeps candidate bucket rows, chooses the best
  measured split per bucket, and emits a composed dispatch table.

Run from the thesis package directory to avoid the /TTC Python path gotcha:

    cd /TTC/FastTTS-thesis
    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1_analysis/bench_cots_phase_free_regime.py \
        --exp --smoke

Focused redo:

    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1_analysis/bench_cots_phase_free_regime.py \
        --exp --keep-going \
        --workloads decode8x128:8:128 prefill128x1:128:1 mixed128x128:128:128 \
        --batch-sizes 1 16 64 \
        --f-values 0.005 0.01 0.02 0.0357 0.05 \
        --only-strategies none cots_cpu_all cots_prefetch_all \
            cots_decode_cpu_prefill_prefetch \
        --repeat 3

Empirical oracle pass:

    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1_analysis/bench_cots_phase_free_regime.py \
        --oracle --exp --keep-going \
        --oracle-decode-buckets 1 16 64 \
        --oracle-prefill-buckets 128 512 2048 \
        --f-values 0.005 0.01 0.02 0.0357 0.05 \
        --oracle-split-ratios 0 0.25 0.5 0.75 1 \
        --oracle-validate-e2e \
        --workloads decode8x128:8:128 prefill128x1:128:1 mixed128x128:128:128 \
        --batch-sizes 1 16 64 \
        --repeat 3
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
MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.75
FREE_MARGIN = 0.05
MAX_CV = 0.03
WARMUP_ITERS = 2
BENCH_ITERS = 3
REPEATS = 3
DEFAULT_BATCHES = (1, 4, 16, 64)
DEFAULT_F_VALUES = (0.005, 0.01, 0.02, 0.0357, 0.05, 0.0714, 0.09, 0.15)
DEFAULT_WORKLOADS = (
    "decode8x128:8:128",
    "prefill128x1:128:1",
    "mixed128x128:128:128",
)
DEFAULT_ORACLE_DECODE_BUCKETS = (1, 4, 16, 64)
DEFAULT_ORACLE_PREFILL_BUCKETS = (128, 512, 2048)
DEFAULT_ORACLE_SPLIT_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
ORACLE_DEFAULT_OTHER_POLICY = "prefetch"

# Mirrors the current COTS dispatch grid used by the planner validation
# harness, with the larger vLLM fallback buckets included as explicit rows.
DISPATCH_BUCKETS = tuple(
    sorted(
        set(
            (1, 2, 4)
            + tuple(range(8, 256, 8))
            + tuple(range(256, 513, 16))
            + (768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
        )
    )
)


@dataclass(frozen=True)
class Workload:
    name: str
    input_len: int
    output_len: int

    @property
    def phase_hint(self) -> str:
        if self.output_len <= 1:
            return "prefill"
        if self.input_len <= 16 and self.output_len >= 64:
            return "decode"
        return "mixed"


@dataclass(frozen=True)
class Arm:
    name: str
    strategy: str
    f_cpu_store: float
    decode_split: tuple[float, float]
    other_split: tuple[float, float]
    dispatch_table: dict[int, tuple[float, float]] | None = None
    module_variant: str | None = None
    weight_modules: tuple[str, ...] | None = None

    @property
    def is_baseline(self) -> bool:
        return self.strategy == "none"


@dataclass(frozen=True)
class OracleRow:
    phase: str
    bucket: int
    workload: Workload
    batch: int
    arm: Arm


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1_analysis/phase_free_regime") / stamp


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "workload"


def float_tag(value: float) -> str:
    text = f"{value:.5f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def parse_workload(text: str) -> Workload:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "workloads must use NAME:INPUT_LEN:OUTPUT_LEN"
        )
    name, input_text, output_text = parts
    try:
        input_len = int(input_text)
        output_len = int(output_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid workload token lengths in {text!r}"
        ) from exc
    if input_len <= 0 or output_len <= 0:
        raise argparse.ArgumentTypeError(
            f"workload lengths must be positive in {text!r}"
        )
    return Workload(sanitize_name(name), input_len, output_len)


def parse_workloads(values: list[str]) -> list[Workload]:
    workloads: list[Workload] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                workloads.append(parse_workload(part))
    seen: set[str] = set()
    for workload in workloads:
        if workload.name in seen:
            raise argparse.ArgumentTypeError(
                f"duplicate workload name {workload.name!r}"
            )
        seen.add(workload.name)
    return workloads


def parse_module_variants(
    values: list[str] | None,
) -> list[tuple[str, tuple[str, ...]]]:
    if not values:
        return []
    variants: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(
                "module variants must use LABEL:MODULE[,MODULE...]"
            )
        label_text, modules_text = value.split(":", 1)
        label = sanitize_name(label_text)
        modules = tuple(
            part.strip()
            for part in re.split(r"[,+]", modules_text)
            if part.strip()
        )
        if not modules:
            raise argparse.ArgumentTypeError(
                f"module variant {value!r} does not list any modules"
            )
        if label in seen:
            raise argparse.ArgumentTypeError(
                f"duplicate module variant label {label!r}"
            )
        seen.add(label)
        variants.append((label, modules))
    return variants


def bucket_for(num_tokens: int, dispatch_buckets: list[int]) -> int:
    for bucket in sorted(dispatch_buckets):
        if int(num_tokens) <= int(bucket):
            return int(bucket)
    return int(max(dispatch_buckets))


def jsonable_dispatch_table(
    table: dict[int, tuple[float, float]],
) -> dict[str, list[float]]:
    return {
        str(bucket): [round(float(pair[0]), 12), round(float(pair[1]), 12)]
        for bucket, pair in table.items()
    }


def make_dispatch_table(
    *,
    arm: Arm,
    batch: int,
    dispatch_buckets: list[int],
) -> dict[int, tuple[float, float]]:
    if arm.dispatch_table is not None:
        missing = sorted(set(dispatch_buckets) - set(arm.dispatch_table))
        if missing:
            raise ValueError(f"{arm.name} dispatch table missing buckets: {missing}")
        return {
            int(bucket): arm.dispatch_table[int(bucket)]
            for bucket in dispatch_buckets
        }
    table = {int(bucket): arm.other_split for bucket in dispatch_buckets}
    table[bucket_for(batch, dispatch_buckets)] = arm.decode_split
    return table


def weight_thread_count_for_score(score: float) -> int:
    if score <= 0.08:
        return 4
    if score <= 0.24:
        return 16
    return 24


def derive_weight_thread_policy(
    dispatch_table: dict[int, tuple[float, float]],
) -> dict[int, int]:
    return {
        int(bucket): weight_thread_count_for_score(int(bucket) * f_cpu_compute)
        for bucket, (f_cpu_compute, _) in dispatch_table.items()
    }


def build_arms(f_values: list[float]) -> list[Arm]:
    arms = [
        Arm(
            name="none",
            strategy="none",
            f_cpu_store=0.0,
            decode_split=(0.0, 0.0),
            other_split=(0.0, 0.0),
        )
    ]
    for f_cpu_store in f_values:
        tag = float_tag(f_cpu_store)
        half = round(f_cpu_store * 0.5, 12)
        f_cpu_store = round(float(f_cpu_store), 12)
        arms.extend(
            [
                Arm(
                    name=f"cots_prefetch_all_f{tag}",
                    strategy="cots_prefetch_all",
                    f_cpu_store=f_cpu_store,
                    decode_split=(0.0, f_cpu_store),
                    other_split=(0.0, f_cpu_store),
                ),
                Arm(
                    name=f"cots_cpu_all_f{tag}",
                    strategy="cots_cpu_all",
                    f_cpu_store=f_cpu_store,
                    decode_split=(f_cpu_store, 0.0),
                    other_split=(f_cpu_store, 0.0),
                ),
                Arm(
                    name=f"cots_collab50_all_f{tag}",
                    strategy="cots_collab50_all",
                    f_cpu_store=f_cpu_store,
                    decode_split=(half, half),
                    other_split=(half, half),
                ),
                Arm(
                    name=f"cots_decode_cpu_prefill_prefetch_f{tag}",
                    strategy="cots_decode_cpu_prefill_prefetch",
                    f_cpu_store=f_cpu_store,
                    decode_split=(f_cpu_store, 0.0),
                    other_split=(0.0, f_cpu_store),
                ),
                Arm(
                    name=f"cots_decode_collab50_prefill_prefetch_f{tag}",
                    strategy="cots_decode_collab50_prefill_prefetch",
                    f_cpu_store=f_cpu_store,
                    decode_split=(half, half),
                    other_split=(0.0, f_cpu_store),
                ),
            ]
        )
    return arms


def filter_arms_in_requested_order(
    arms: list[Arm],
    requested: list[str] | None,
) -> list[Arm]:
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


def expand_arms_by_module_variants(
    arms: list[Arm],
    variants: list[tuple[str, tuple[str, ...]]],
) -> list[Arm]:
    if not variants:
        return arms
    expanded: list[Arm] = [arm for arm in arms if arm.is_baseline]
    for arm in arms:
        if arm.is_baseline:
            continue
        for label, modules in variants:
            expanded.append(
                Arm(
                    name=f"{label}_{arm.name}",
                    strategy=f"{label}_{arm.strategy}",
                    f_cpu_store=arm.f_cpu_store,
                    decode_split=arm.decode_split,
                    other_split=arm.other_split,
                    dispatch_table=arm.dispatch_table,
                    module_variant=label,
                    weight_modules=modules,
                )
            )
    return expanded


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


def cell_stem(workload: Workload, arm: Arm, batch: int, repeat: int) -> str:
    return f"r{repeat:02d}_{workload.name}_{arm.name}_b{batch}"


def cell_json(
    results_dir: Path,
    workload: Workload,
    arm: Arm,
    batch: int,
    repeat: int,
) -> Path:
    return results_dir / f"{cell_stem(workload, arm, batch, repeat)}.json"


def cots_flags(args: argparse.Namespace, arm: Arm, batch: int) -> list[str]:
    dispatch_table = make_dispatch_table(
        arm=arm,
        batch=batch,
        dispatch_buckets=args.dispatch_buckets,
    )
    flags = [
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        f"{arm.f_cpu_store:.12g}",
        "--cots-f-prefetch",
        "0.0",
        "--cots-dispatch-table",
        json.dumps(jsonable_dispatch_table(dispatch_table), separators=(",", ":")),
        "--cots-cpu-runner",
        args.cots_cpu_runner,
    ]
    if args.thread_policy == "workscore":
        thread_map = derive_weight_thread_policy(dispatch_table)
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
    weight_modules = arm.weight_modules or args.cots_weight_modules
    if weight_modules:
        flags += ["--cots-weight-modules", *weight_modules]
    if args.cots_weight_capture_sync_mode is not None:
        flags += [
            "--cots-weight-capture-sync-mode",
            args.cots_weight_capture_sync_mode,
        ]
    return flags


def run_cell(
    *,
    workload: Workload,
    arm: Arm,
    batch: int,
    repeat: int,
    results_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, int]:
    out_json = cell_json(results_dir, workload, arm, batch, repeat)
    out_log = out_json.with_suffix(".log")
    if out_json.exists() and not args.force:
        print(
            f"  [skip] r={repeat} {workload.name} {arm.name} B={batch} (cached)",
            flush=True,
        )
        return out_json, 0

    max_model_len = max(args.max_model_len, workload.input_len + workload.output_len)
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
        str(workload.input_len),
        "--output-len",
        str(workload.output_len),
        "--batch-size",
        str(batch),
        "--max-model-len",
        str(max_model_len),
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
    if not arm.is_baseline:
        cmd += cots_flags(args, arm, batch)
    cmd += args.extra_vllm_args

    t0 = time.perf_counter()
    with out_log.open("w") as fh:
        try:
            proc = subprocess.run(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=args.cell_timeout_s,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            fh.write(f"\n[timeout] exceeded {args.cell_timeout_s}s\n")
            print(
                f"  [timeout] r={repeat} {workload.name} {arm.name} "
                f"B={batch} ({elapsed:.0f}s)",
                flush=True,
            )
            return out_json, 124
    elapsed = time.perf_counter() - t0
    if proc.returncode == 0:
        avg = parse_avg_latency(out_json)
        print(
            f"  [ok] r={repeat} {workload.name} {arm.name} B={batch}: "
            f"avg={avg:.4f}s ({elapsed:.1f}s)",
            flush=True,
        )
    else:
        tail = "\n        ".join(out_log.read_text(errors="replace").splitlines()[-20:])
        print(
            f"  [FAIL] r={repeat} {workload.name} {arm.name} B={batch} "
            f"rc={proc.returncode} ({elapsed:.1f}s)\n        {tail}",
            flush=True,
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


def failure_reason(path: Path) -> str:
    if not path.exists():
        return "not_started"
    text = path.read_text(errors="replace")
    if not text.strip():
        return "empty_log"
    if "[timeout]" in text:
        return "timeout"
    if "Engine core initialization failed" in text:
        return "engine_init_failed"
    if "CUDA out of memory" in text or "OutOfMemoryError" in text:
        return "oom"
    if "Traceback" in text:
        return "traceback"
    return "missing_json"


def collect_failures(
    args: argparse.Namespace,
    workloads: list[Workload],
    arms: list[Arm],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for workload in workloads:
        for arm in arms:
            for batch in args.batch_sizes:
                for repeat in range(args.repeat):
                    out_json = cell_json(args.results_dir, workload, arm, batch, repeat)
                    out_log = out_json.with_suffix(".log")
                    if out_json.exists() or not out_log.exists():
                        continue
                    failures.append(
                        {
                            "workload": workload.name,
                            "arm": arm.name,
                            "strategy": arm.strategy,
                            "module_variant": arm.module_variant,
                            "weight_modules": (
                                list(arm.weight_modules)
                                if arm.weight_modules is not None
                                else None
                            ),
                            "batch_size": batch,
                            "repeat": repeat,
                            "reason": failure_reason(out_log),
                            "log": str(out_log),
                        }
                    )
    return failures


def summarize(
    args: argparse.Namespace,
    workloads: list[Workload],
    arms: list[Arm],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    baseline_by_case: dict[tuple[str, int], float] = {}

    for workload in workloads:
        for batch in args.batch_sizes:
            values = []
            baseline_arm = next(arm for arm in arms if arm.is_baseline)
            for repeat in range(args.repeat):
                path = cell_json(args.results_dir, workload, baseline_arm, batch, repeat)
                if path.exists():
                    values.append(parse_avg_latency(path))
            stat = stats(values)
            if stat["mean"] is not None:
                baseline_by_case[(workload.name, batch)] = float(stat["mean"])

    for workload in workloads:
        for arm in arms:
            for batch in args.batch_sizes:
                values = []
                for repeat in range(args.repeat):
                    path = cell_json(args.results_dir, workload, arm, batch, repeat)
                    if path.exists():
                        values.append(parse_avg_latency(path))
                stat = stats(values)
                mean = stat["mean"]
                base = baseline_by_case.get((workload.name, batch))
                slowdown = None if mean is None or base is None else float(mean) / base
                classification = classify_latency(
                    slowdown=slowdown,
                    cv=stat["cv"],
                    free_margin=args.free_margin,
                    max_cv=args.max_cv,
                    is_baseline=arm.is_baseline,
                )
                prompt_tokens_per_s = (
                    None
                    if mean is None
                    else float(batch * workload.input_len) / float(mean)
                )
                output_tokens_per_s = (
                    None
                    if mean is None
                    else float(batch * workload.output_len) / float(mean)
                )
                total_tokens_per_s = (
                    None
                    if mean is None
                    else float(batch * (workload.input_len + workload.output_len))
                    / float(mean)
                )
                decode_bucket = bucket_for(batch, args.dispatch_buckets)
                prefill_num_tokens = batch * workload.input_len
                prefill_bucket = bucket_for(prefill_num_tokens, args.dispatch_buckets)
                cells.append(
                    {
                        "workload": workload.name,
                        "phase_hint": workload.phase_hint,
                        "input_len": workload.input_len,
                        "output_len": workload.output_len,
                        "arm": arm.name,
                        "strategy": arm.strategy,
                        "module_variant": arm.module_variant,
                        "weight_modules": (
                            list(arm.weight_modules)
                            if arm.weight_modules is not None
                            else None
                        ),
                        "batch_size": batch,
                        "prefill_num_tokens": prefill_num_tokens,
                        "prefill_bucket": prefill_bucket,
                        "decode_bucket": decode_bucket,
                        "f_cpu_store": arm.f_cpu_store,
                        "decode_f_cpu": arm.decode_split[0],
                        "decode_f_prefetch": arm.decode_split[1],
                        "other_f_cpu": arm.other_split[0],
                        "other_f_prefetch": arm.other_split[1],
                        "num_repeats": stat["n"],
                        "mean_latency_s": mean,
                        "stdev_latency_s": stat["stdev"],
                        "cv": stat["cv"],
                        "slowdown_vs_none": slowdown,
                        "prompt_tokens_per_s": prompt_tokens_per_s,
                        "output_tokens_per_s": output_tokens_per_s,
                        "total_tokens_per_s": total_tokens_per_s,
                        "classification": classification,
                    }
                )

    max_free: dict[str, dict[str, dict[str, float | None]]] = {}
    strategies = sorted({arm.strategy for arm in arms if not arm.is_baseline})
    for workload in workloads:
        max_free[workload.name] = {}
        for strategy in strategies:
            max_free[workload.name][strategy] = {}
            for batch in args.batch_sizes:
                free_fs = [
                    cell["f_cpu_store"]
                    for cell in cells
                    if cell["workload"] == workload.name
                    and cell["strategy"] == strategy
                    and cell["batch_size"] == batch
                    and cell["classification"] in {"free", "faster"}
                ]
                max_free[workload.name][strategy][str(batch)] = (
                    max(free_fs) if free_fs else None
                )

    return {
        "env": env_info(),
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "mode": args.mode,
            "workloads": [
                {
                    "name": workload.name,
                    "input_len": workload.input_len,
                    "output_len": workload.output_len,
                    "phase_hint": workload.phase_hint,
                }
                for workload in workloads
            ],
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batch_sizes": args.batch_sizes,
            "f_values": args.f_values,
            "repeat": args.repeat,
            "num_iters": args.num_iters,
            "num_iters_warmup": args.num_iters_warmup,
            "free_margin": args.free_margin,
            "max_cv": args.max_cv,
            "dispatch_buckets": args.dispatch_buckets,
            "thread_policy": args.thread_policy,
            "cots_cpu_num_threads": args.cots_cpu_num_threads,
            "cots_cpu_runner": args.cots_cpu_runner,
            "cots_weight_capture_sync_mode": args.cots_weight_capture_sync_mode,
            "cots_weight_modules": args.cots_weight_modules,
            "module_variants": [
                {"label": label, "weight_modules": list(modules)}
                for label, modules in args.module_variants
            ],
            "cell_timeout_s": args.cell_timeout_s,
            "extra_vllm_args": args.extra_vllm_args,
        },
        "cells": cells,
        "max_free_f_cpu_store": max_free,
        "failed_cells": collect_failures(args, workloads, arms),
    }


def fmt_optional(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    cfg = summary["config"]
    batches = cfg["batch_sizes"]
    lines = [
        "# Phase-Aware COTS Free-Regime Sweep",
        "",
        f"Model: `{cfg['model']}`; dtype: `{cfg['dtype']}`; mode: `{cfg['mode']}`; "
        f"`gpu_memory_utilization={cfg['gpu_memory_utilization']}`.",
        "",
        f"Free threshold: latency `<= {1.0 + cfg['free_margin']:.2f}x` "
        f"no-offload with CV `<= {cfg['max_cv']:.0%}`.",
        "",
        "## Max Free `f_cpu_store`",
    ]
    for workload in cfg["workloads"]:
        workload_name = workload["name"]
        lines.extend(
            [
                "",
                f"### `{workload_name}` ({workload['input_len']}/{workload['output_len']}, "
                f"{workload['phase_hint']})",
                "",
                "| strategy | " + " | ".join(f"B={b}" for b in batches) + " |",
                "|---|" + "|".join("---:" for _ in batches) + "|",
            ]
        )
        for strategy, by_batch in summary["max_free_f_cpu_store"][workload_name].items():
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
            "| workload | module | arm | B | D/P bucket | mean s | slowdown | CV | "
            "prompt tok/s | output tok/s | verdict |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for cell in summary["cells"]:
        if cell["strategy"] == "none":
            continue
        lines.append(
            f"| `{cell['workload']}` | `{cell['module_variant'] or '-'}` | "
            f"`{cell['arm']}` | {cell['batch_size']} | "
            f"{cell['decode_bucket']}/{cell['prefill_bucket']} | "
            f"{fmt_optional(cell['mean_latency_s'], 4)} | "
            f"{fmt_optional(cell['slowdown_vs_none'], 3)} | "
            f"{fmt_optional(cell['cv'], 3)} | "
            f"{fmt_optional(cell['prompt_tokens_per_s'], 1)} | "
            f"{fmt_optional(cell['output_tokens_per_s'], 1)} | "
            f"{cell['classification']} |"
        )

    if summary["failed_cells"]:
        lines.extend(
            [
                "",
                "## Failed Or Incomplete Cells",
                "",
                "| workload | arm | B | repeat | reason |",
                "|---|---|---:|---:|---|",
            ]
        )
        for failure in summary["failed_cells"]:
            lines.append(
                f"| `{failure['workload']}` | `{failure['arm']}` | "
                f"{failure['batch_size']} | {failure['repeat']} | "
                f"`{failure['reason']}` |"
            )
    path.write_text("\n".join(lines) + "\n")


def unique_oracle_ratios(ratios: list[float]) -> list[float]:
    values = {0.0, 1.0}
    for ratio in ratios:
        if ratio < -1e-12 or ratio > 1.0 + 1e-12:
            raise ValueError(f"oracle split ratio must be in [0, 1], got {ratio}")
        values.add(round(min(1.0, max(0.0, ratio)), 12))
    return sorted(values)


def split_for_ratio(f_cpu_store: float, ratio: float) -> tuple[float, float]:
    f_cpu = round(float(f_cpu_store) * float(ratio), 12)
    f_prefetch = round(float(f_cpu_store) - f_cpu, 12)
    return f_cpu, f_prefetch


def oracle_probe_workload(args: argparse.Namespace, phase: str, bucket: int) -> Workload:
    if phase == "decode":
        return Workload(
            name=f"oracle_decode_b{bucket}",
            input_len=args.oracle_decode_input_len,
            output_len=args.oracle_decode_output_len,
        )
    if phase == "prefill":
        return Workload(
            name=f"oracle_prefill_b{bucket}",
            input_len=int(bucket),
            output_len=args.oracle_prefill_output_len,
        )
    raise ValueError(f"unknown oracle phase: {phase}")


def oracle_probe_batch(phase: str, bucket: int) -> int:
    if phase == "decode":
        return int(bucket)
    if phase == "prefill":
        return 1
    raise ValueError(f"unknown oracle phase: {phase}")


def oracle_arm(
    *,
    phase: str,
    bucket: int,
    f_cpu_store: float,
    ratio: float,
) -> Arm:
    f_cpu, f_prefetch = split_for_ratio(f_cpu_store, ratio)
    name = (
        f"oracle_{phase}_b{bucket}_f{float_tag(f_cpu_store)}_"
        f"cpu{float_tag(f_cpu)}_pf{float_tag(f_prefetch)}"
    )
    if phase == "decode":
        decode_split = (f_cpu, f_prefetch)
        other_split = (0.0, round(float(f_cpu_store), 12))
    elif phase == "prefill":
        decode_split = (0.0, round(float(f_cpu_store), 12))
        other_split = (f_cpu, f_prefetch)
    else:
        raise ValueError(f"unknown oracle phase: {phase}")
    return Arm(
        name=name,
        strategy=f"oracle_{phase}",
        f_cpu_store=round(float(f_cpu_store), 12),
        decode_split=decode_split,
        other_split=other_split,
    )


def build_oracle_rows(args: argparse.Namespace) -> list[OracleRow]:
    rows: list[OracleRow] = []
    none = Arm(
        name="none",
        strategy="none",
        f_cpu_store=0.0,
        decode_split=(0.0, 0.0),
        other_split=(0.0, 0.0),
    )
    phases = (
        ("decode", args.oracle_decode_buckets),
        ("prefill", args.oracle_prefill_buckets),
    )
    for phase, buckets in phases:
        for bucket in buckets:
            workload = oracle_probe_workload(args, phase, int(bucket))
            batch = oracle_probe_batch(phase, int(bucket))
            rows.append(
                OracleRow(
                    phase=phase,
                    bucket=int(bucket),
                    workload=workload,
                    batch=batch,
                    arm=none,
                )
            )
            for f_cpu_store in args.f_values:
                for ratio in unique_oracle_ratios(args.oracle_split_ratios):
                    rows.append(
                        OracleRow(
                            phase=phase,
                            bucket=int(bucket),
                            workload=workload,
                            batch=batch,
                            arm=oracle_arm(
                                phase=phase,
                                bucket=int(bucket),
                                f_cpu_store=float(f_cpu_store),
                                ratio=ratio,
                            ),
                        )
                    )
    return rows


def oracle_row_values(args: argparse.Namespace, row: OracleRow) -> list[float]:
    values: list[float] = []
    for repeat in range(args.repeat):
        path = cell_json(args.results_dir, row.workload, row.arm, row.batch, repeat)
        if path.exists():
            values.append(parse_avg_latency(path))
    return values


def oracle_failures(
    args: argparse.Namespace,
    rows: list[OracleRow],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        for repeat in range(args.repeat):
            out_json = cell_json(args.results_dir, row.workload, row.arm, row.batch, repeat)
            out_log = out_json.with_suffix(".log")
            if out_json.exists() or not out_log.exists():
                continue
            failures.append(
                {
                    "phase": row.phase,
                    "bucket": row.bucket,
                    "workload": row.workload.name,
                    "arm": row.arm.name,
                    "repeat": repeat,
                    "reason": failure_reason(out_log),
                    "log": str(out_log),
                }
            )
    return failures


def oracle_split_from_row(row: OracleRow) -> tuple[float, float]:
    if row.phase == "decode":
        return row.arm.decode_split
    if row.phase == "prefill":
        return row.arm.other_split
    raise ValueError(f"unknown oracle phase: {row.phase}")


def summarize_oracle(args: argparse.Namespace, rows: list[OracleRow]) -> dict[str, Any]:
    baselines: dict[tuple[str, int], float] = {}
    probe_rows: list[dict[str, Any]] = []

    for row in rows:
        if not row.arm.is_baseline:
            continue
        stat = stats(oracle_row_values(args, row))
        if stat["mean"] is not None:
            baselines[(row.phase, row.bucket)] = float(stat["mean"])
        probe_rows.append(
            {
                "phase": row.phase,
                "bucket": row.bucket,
                "workload": row.workload.name,
                "batch_size": row.batch,
                "arm": row.arm.name,
                "strategy": row.arm.strategy,
                "f_cpu_store": None,
                "f_cpu": None,
                "f_prefetch": None,
                "num_repeats": stat["n"],
                "mean_latency_s": stat["mean"],
                "stdev_latency_s": stat["stdev"],
                "cv": stat["cv"],
                "slowdown_vs_none": 1.0 if stat["mean"] is not None else None,
                "classification": "baseline",
            }
        )

    for row in rows:
        if row.arm.is_baseline:
            continue
        stat = stats(oracle_row_values(args, row))
        mean = stat["mean"]
        base = baselines.get((row.phase, row.bucket))
        slowdown = None if mean is None or base is None else float(mean) / base
        f_cpu, f_prefetch = oracle_split_from_row(row)
        probe_rows.append(
            {
                "phase": row.phase,
                "bucket": row.bucket,
                "workload": row.workload.name,
                "batch_size": row.batch,
                "arm": row.arm.name,
                "strategy": row.arm.strategy,
                "f_cpu_store": row.arm.f_cpu_store,
                "f_cpu": f_cpu,
                "f_prefetch": f_prefetch,
                "num_repeats": stat["n"],
                "mean_latency_s": mean,
                "stdev_latency_s": stat["stdev"],
                "cv": stat["cv"],
                "slowdown_vs_none": slowdown,
                "classification": classify_latency(
                    slowdown=slowdown,
                    cv=stat["cv"],
                    free_margin=args.free_margin,
                    max_cv=args.max_cv,
                    is_baseline=False,
                ),
            }
        )

    split_groups: dict[tuple[float, int, float, float], list[dict[str, Any]]] = {}
    for row in probe_rows:
        if row["f_cpu_store"] is None or row["mean_latency_s"] is None:
            continue
        key = (
            round(float(row["f_cpu_store"]), 12),
            int(row["bucket"]),
            round(float(row["f_cpu"]), 12),
            round(float(row["f_prefetch"]), 12),
        )
        split_groups.setdefault(key, []).append(row)

    selected_rows: list[dict[str, Any]] = []
    grouped_by_bucket: dict[
        tuple[float, int],
        list[tuple[tuple[float, int, float, float], list[dict[str, Any]]]],
    ] = {}
    for key, group in split_groups.items():
        f_cpu_store, bucket, _, _ = key
        grouped_by_bucket.setdefault((f_cpu_store, bucket), []).append((key, group))

    for (f_cpu_store, bucket), candidates in sorted(grouped_by_bucket.items()):
        scored: list[tuple[tuple[bool, float, float, float], dict[str, Any]]] = []
        for (_, _, f_cpu, f_prefetch), group in candidates:
            slowdowns = [
                float(row["slowdown_vs_none"])
                for row in group
                if row["slowdown_vs_none"] is not None
            ]
            if not slowdowns:
                continue
            all_free = all(
                row["classification"] in {"free", "faster"} for row in group
            )
            worst = max(slowdowns)
            mean_slowdown = statistics.mean(slowdowns)
            scored.append(
                (
                    (
                        not all_free,
                        worst,
                        mean_slowdown,
                        -float(f_cpu),
                    ),
                    {
                        "f_cpu_store": f_cpu_store,
                        "bucket": bucket,
                        "f_cpu": f_cpu,
                        "f_prefetch": f_prefetch,
                        "classification": "free" if all_free else "non_free",
                        "worst_slowdown_vs_none": worst,
                        "mean_slowdown_vs_none": mean_slowdown,
                        "phases": sorted({str(row["phase"]) for row in group}),
                        "num_phase_rows": len(group),
                        "source_rows": group,
                    },
                )
            )
        if scored:
            selected_rows.append(sorted(scored, key=lambda item: item[0])[0][1])

    tables: dict[str, dict[str, Any]] = {}
    for f_cpu_store in sorted({float(value) for value in args.f_values}):
        table = {
            int(bucket): (0.0, round(float(f_cpu_store), 12))
            for bucket in args.dispatch_buckets
        }
        selected_for_store = [
            row for row in selected_rows
            if abs(float(row["f_cpu_store"]) - f_cpu_store) < 1e-12
        ]
        for selected in selected_for_store:
            if int(selected["bucket"]) in table:
                table[int(selected["bucket"])] = (
                    round(float(selected["f_cpu"]), 12),
                    round(float(selected["f_prefetch"]), 12),
                )
        tables[float_tag(f_cpu_store)] = {
            "f_cpu_store": f_cpu_store,
            "default_unmeasured_policy": ORACLE_DEFAULT_OTHER_POLICY,
            "dispatch_table": jsonable_dispatch_table(table),
            "selected_buckets": selected_for_store,
        }

    return {
        "env": env_info(),
        "config": {
            "model": args.model,
            "dtype": args.dtype,
            "mode": args.mode,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "f_values": args.f_values,
            "oracle_decode_buckets": args.oracle_decode_buckets,
            "oracle_prefill_buckets": args.oracle_prefill_buckets,
            "oracle_split_ratios": args.oracle_split_ratios,
            "oracle_decode_input_len": args.oracle_decode_input_len,
            "oracle_decode_output_len": args.oracle_decode_output_len,
            "oracle_prefill_output_len": args.oracle_prefill_output_len,
            "repeat": args.repeat,
            "num_iters": args.num_iters,
            "num_iters_warmup": args.num_iters_warmup,
            "free_margin": args.free_margin,
            "max_cv": args.max_cv,
            "dispatch_buckets": args.dispatch_buckets,
            "thread_policy": args.thread_policy,
            "cots_cpu_num_threads": args.cots_cpu_num_threads,
            "cots_cpu_runner": args.cots_cpu_runner,
            "cots_weight_capture_sync_mode": args.cots_weight_capture_sync_mode,
            "cots_weight_modules": args.cots_weight_modules,
            "cell_timeout_s": args.cell_timeout_s,
            "extra_vllm_args": args.extra_vllm_args,
        },
        "probe_rows": probe_rows,
        "selected_rows": selected_rows,
        "oracle_dispatch_tables": tables,
        "failed_cells": oracle_failures(args, rows),
    }


def write_oracle_markdown(summary: dict[str, Any], path: Path) -> None:
    cfg = summary["config"]
    lines = [
        "# COTS Empirical Oracle Dispatch",
        "",
        f"Model: `{cfg['model']}`; dtype: `{cfg['dtype']}`; mode: `{cfg['mode']}`.",
        "",
        f"Free threshold: latency `<= {1.0 + cfg['free_margin']:.2f}x` "
        f"no-offload with CV `<= {cfg['max_cv']:.0%}`.",
        "",
        "## Selected Oracle Rows",
        "",
        "| f_store | bucket | split cpu/pf | phases | worst slowdown | verdict |",
        "|---:|---:|---:|---|---:|---|",
    ]
    for row in summary["selected_rows"]:
        lines.append(
            f"| {row['f_cpu_store']:.5f} | {row['bucket']} | "
            f"{row['f_cpu']:.5f}/{row['f_prefetch']:.5f} | "
            f"{','.join(row['phases'])} | "
            f"{fmt_optional(row['worst_slowdown_vs_none'], 3)} | "
            f"{row['classification']} |"
        )

    lines.extend(
        [
            "",
            "## Probe Rows",
            "",
            "| phase | bucket | f_store | split cpu/pf | mean s | slowdown | CV | verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["probe_rows"]:
        if row["strategy"] == "none":
            continue
        lines.append(
            f"| `{row['phase']}` | {row['bucket']} | "
            f"{fmt_optional(row['f_cpu_store'], 5)} | "
            f"{fmt_optional(row['f_cpu'], 5)}/"
            f"{fmt_optional(row['f_prefetch'], 5)} | "
            f"{fmt_optional(row['mean_latency_s'], 4)} | "
            f"{fmt_optional(row['slowdown_vs_none'], 3)} | "
            f"{fmt_optional(row['cv'], 3)} | {row['classification']} |"
        )

    if summary["failed_cells"]:
        lines.extend(
            [
                "",
                "## Failed Or Incomplete Cells",
                "",
                "| phase | bucket | arm | repeat | reason |",
                "|---|---:|---|---:|---|",
            ]
        )
        for failure in summary["failed_cells"]:
            lines.append(
                f"| `{failure['phase']}` | {failure['bucket']} | "
                f"`{failure['arm']}` | {failure['repeat']} | "
                f"`{failure['reason']}` |"
            )
    path.write_text("\n".join(lines) + "\n")


def oracle_composed_arm(table_entry: dict[str, Any]) -> Arm:
    f_cpu_store = float(table_entry["f_cpu_store"])
    dispatch_table = {
        int(bucket): (float(pair[0]), float(pair[1]))
        for bucket, pair in table_entry["dispatch_table"].items()
    }
    return Arm(
        name=f"oracle_composed_f{float_tag(f_cpu_store)}",
        strategy="oracle_composed",
        f_cpu_store=f_cpu_store,
        decode_split=(0.0, f_cpu_store),
        other_split=(0.0, f_cpu_store),
        dispatch_table=dispatch_table,
    )


def run_oracle_validation(
    args: argparse.Namespace,
    oracle_summary: dict[str, Any],
) -> int:
    none = Arm(
        name="none",
        strategy="none",
        f_cpu_store=0.0,
        decode_split=(0.0, 0.0),
        other_split=(0.0, 0.0),
    )
    arms = [none] + [
        oracle_composed_arm(table)
        for _, table in sorted(oracle_summary["oracle_dispatch_tables"].items())
    ]
    exit_code = 0
    if args.exp:
        for workload in args.workloads:
            for arm in arms:
                for batch in args.batch_sizes:
                    for repeat in range(args.repeat):
                        _, rc = run_cell(
                            workload=workload,
                            arm=arm,
                            batch=batch,
                            repeat=repeat,
                            results_dir=args.results_dir,
                            args=args,
                        )
                        if rc != 0:
                            exit_code = rc
                            if not args.keep_going:
                                return exit_code
    summary = summarize(args, args.workloads, arms)
    summary_path = args.results_dir / "oracle_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown_summary(summary, args.results_dir / "oracle_validation_summary.md")
    print(f"[summary] wrote {summary_path}", flush=True)
    print(
        f"[summary] wrote {args.results_dir / 'oracle_validation_summary.md'}",
        flush=True,
    )
    return exit_code


def run_oracle(args: argparse.Namespace) -> int:
    rows = build_oracle_rows(args)
    print(
        f"[setup] oracle results={args.results_dir} mode={args.mode} "
        f"decode_buckets={args.oracle_decode_buckets} "
        f"prefill_buckets={args.oracle_prefill_buckets} "
        f"stores={args.f_values} ratios={args.oracle_split_ratios} "
        f"rows={len(rows)} repeats={args.repeat} exp={args.exp}",
        flush=True,
    )
    exit_code = 0
    try:
        if args.exp:
            for row in rows:
                for repeat in range(args.repeat):
                    _, rc = run_cell(
                        workload=row.workload,
                        arm=row.arm,
                        batch=row.batch,
                        repeat=repeat,
                        results_dir=args.results_dir,
                        args=args,
                    )
                    if rc != 0:
                        exit_code = rc
                        if not args.keep_going:
                            raise RuntimeError("oracle cell failed")
    except KeyboardInterrupt:
        exit_code = 130
        print("[interrupt] writing oracle summary for completed cells", flush=True)
    except RuntimeError:
        pass

    summary = summarize_oracle(args, rows)
    summary_path = args.results_dir / "oracle_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_oracle_markdown(summary, args.results_dir / "oracle_summary.md")
    dispatch_path = args.results_dir / "oracle_dispatch_tables.json"
    dispatch_path.write_text(json.dumps(summary["oracle_dispatch_tables"], indent=2))
    print(f"[summary] wrote {summary_path}", flush=True)
    print(f"[summary] wrote {args.results_dir / 'oracle_summary.md'}", flush=True)
    print(f"[summary] wrote {dispatch_path}", flush=True)

    if args.oracle_validate_e2e:
        validation_rc = run_oracle_validation(args, summary)
        exit_code = exit_code or validation_rc
    return exit_code


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.mode = "eager"
    args.workloads = [parse_workload("decode4x4:4:4")]
    args.batch_sizes = [1]
    args.f_values = [0.01]
    args.repeat = 1
    args.num_iters_warmup = 0
    args.num_iters = 1
    tag = float_tag(0.01)
    args.only_strategies = [
        "none",
        f"cots_cpu_all_f{tag}",
        f"cots_prefetch_all_f{tag}",
        f"cots_decode_cpu_prefill_prefetch_f{tag}",
    ]
    args.oracle_decode_buckets = [1]
    args.oracle_prefill_buckets = [4]
    args.oracle_split_ratios = [0.0, 1.0]
    args.oracle_decode_input_len = 4
    args.oracle_decode_output_len = 4
    args.oracle_prefill_output_len = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--mode", choices=("graph", "eager"), default="graph")
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=list(DEFAULT_WORKLOADS),
        help="Workload specs as NAME:INPUT_LEN:OUTPUT_LEN.",
    )
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--f-values", type=float, nargs="+", default=list(DEFAULT_F_VALUES))
    parser.add_argument(
        "--dispatch-buckets",
        type=int,
        nargs="+",
        default=list(DISPATCH_BUCKETS),
    )
    parser.add_argument(
        "--thread-policy",
        choices=("scalar", "workscore"),
        default="workscore",
        help="CPU thread policy for native COTS CPU-compute buckets.",
    )
    parser.add_argument("--cots-cpu-num-threads", type=int, default=16)
    parser.add_argument("--cots-cpu-runner", choices=("native", "python"), default="native")
    parser.add_argument(
        "--cots-weight-capture-sync-mode",
        choices=("host_callback", "wait_kernel"),
        default=None,
        help="Optional explicit sync mode. None lets vLLM apply its graph default.",
    )
    parser.add_argument(
        "--cots-weight-modules",
        nargs="+",
        default=None,
        help="Optional COTS module subset, e.g. qkv mlp or qkv mlp wo.",
    )
    parser.add_argument(
        "--module-variants",
        nargs="+",
        default=None,
        help=(
            "Clone every non-baseline arm for module subset variants. "
            "Use LABEL:MODULE[,MODULE...] tokens, e.g. "
            "current:qkv,mlp mlp_only:mlp qkv_only:qkv."
        ),
    )
    parser.add_argument("--repeat", type=int, default=REPEATS)
    parser.add_argument("--num-iters", type=int, default=BENCH_ITERS)
    parser.add_argument("--num-iters-warmup", type=int, default=WARMUP_ITERS)
    parser.add_argument("--free-margin", type=float, default=FREE_MARGIN)
    parser.add_argument("--max-cv", type=float, default=MAX_CV)
    parser.add_argument("--only-strategies", nargs="*", default=None)
    parser.add_argument(
        "--oracle",
        action="store_true",
        help=(
            "Run empirical oracle mode: sweep candidate dispatch rows, select "
            "best measured splits, and emit oracle_dispatch_tables.json."
        ),
    )
    parser.add_argument(
        "--oracle-decode-buckets",
        type=int,
        nargs="+",
        default=list(DEFAULT_ORACLE_DECODE_BUCKETS),
        help="Decode dispatch buckets to probe with decode-heavy workloads.",
    )
    parser.add_argument(
        "--oracle-prefill-buckets",
        type=int,
        nargs="+",
        default=list(DEFAULT_ORACLE_PREFILL_BUCKETS),
        help="Prefill dispatch buckets to probe with prefill-heavy workloads.",
    )
    parser.add_argument(
        "--oracle-split-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_ORACLE_SPLIT_RATIOS),
        help=(
            "Candidate f_cpu_compute values as ratios of f_cpu_store. Pure "
            "prefetch and pure CPU are always included."
        ),
    )
    parser.add_argument("--oracle-decode-input-len", type=int, default=8)
    parser.add_argument("--oracle-decode-output-len", type=int, default=128)
    parser.add_argument("--oracle-prefill-output-len", type=int, default=1)
    parser.add_argument(
        "--oracle-validate-e2e",
        action="store_true",
        help="After synthesis, run E2E validation workloads with composed tables.",
    )
    parser.add_argument(
        "--cell-timeout-s",
        type=float,
        default=None,
        help="Optional timeout for each vLLM latency subprocess.",
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
    args.workloads = parse_workloads(args.workloads)
    args.module_variants = parse_module_variants(args.module_variants)
    if args.module_variants and args.cots_weight_modules:
        raise SystemExit("--module-variants and --cots-weight-modules are mutually exclusive")
    if args.smoke:
        apply_smoke_overrides(args)
    args.oracle_split_ratios = unique_oracle_ratios(args.oracle_split_ratios)
    if not any(bucket >= max(args.batch_sizes) for bucket in args.dispatch_buckets):
        raise SystemExit("--dispatch-buckets must cover the largest batch size")
    max_oracle_bucket = max(
        [*args.oracle_decode_buckets, *args.oracle_prefill_buckets, 0]
    )
    if args.oracle and not any(
        bucket >= max_oracle_bucket for bucket in args.dispatch_buckets
    ):
        raise SystemExit("--dispatch-buckets must cover the largest oracle bucket")
    return args


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.oracle:
        return run_oracle(args)
    arms = filter_arms_in_requested_order(
        build_arms(args.f_values),
        args.only_strategies,
    )
    arms = expand_arms_by_module_variants(arms, args.module_variants)
    if not any(arm.is_baseline for arm in arms):
        raise SystemExit("--only-strategies must include none for slowdown baselines")

    print(
        f"[setup] results={args.results_dir} mode={args.mode} "
        f"workloads={[w.name for w in args.workloads]} arms={len(arms)} "
        f"batches={args.batch_sizes} module_variants={args.module_variants} "
        f"repeats={args.repeat} exp={args.exp}",
        flush=True,
    )
    exit_code = 0
    try:
        if args.exp:
            for workload in args.workloads:
                for arm in arms:
                    for batch in args.batch_sizes:
                        for repeat in range(args.repeat):
                            _, rc = run_cell(
                                workload=workload,
                                arm=arm,
                                batch=batch,
                                repeat=repeat,
                                results_dir=args.results_dir,
                                args=args,
                            )
                            if rc != 0:
                                exit_code = rc
                                if not args.keep_going:
                                    raise RuntimeError("cell failed")
    except KeyboardInterrupt:
        exit_code = 130
        print("[interrupt] writing summary for completed cells", flush=True)
    except RuntimeError:
        pass

    summary = summarize(args, args.workloads, arms)
    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown_summary(summary, args.results_dir / "summary.md")
    print(f"[summary] wrote {summary_path}", flush=True)
    print(f"[summary] wrote {args.results_dir / 'summary.md'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
