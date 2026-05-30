#!/usr/bin/env python3
"""Run a resumable Phase 2 KV capacity-window sweep.

The sweep keeps the synthetic benchmark settings fixed while varying GPU memory
and hybrid split. It records one JSONL row per run plus a full text log so we can
compare the GPU-only capacity curve against the best hybrid split frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CAPACITY_RE = re.compile(
    r"Maximum concurrency for (?P<tokens>\d+) tokens per request: "
    r"(?P<capacity>[0-9.]+)x"
)
GPU_KV_RE = re.compile(r"GPU KV cache size: (?P<tokens>[0-9,]+) tokens")
HYBRID_CAPACITY_RE = re.compile(
    r"COTS hybrid KV capacity for (?P<tokens>\d+) tokens per request: "
    r"GPU prefix (?P<gpu_prefix_blocks>\d+) blocks/request -> "
    r"(?P<gpu_prefix_capacity>[0-9.]+)x; "
    r"CPU suffix (?P<cpu_suffix_blocks>\d+) blocks/request over "
    r"(?P<cpu_blocks>\d+) CPU blocks -> (?P<cpu_capacity>[0-9.]+)x; "
    r"effective (?P<capacity>[0-9.]+)x"
)


def _parse_csv_tail(output: str) -> dict[str, str] | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if line.startswith("mode,batch,total_tokens,"):
            if idx + 1 >= len(lines):
                return None
            return dict(zip(next(csv.reader([line])), next(csv.reader([lines[idx + 1]]))))
    return None


def _parse_capacity(output: str, *, mode: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    gpu_match = GPU_KV_RE.search(output)
    if gpu_match:
        parsed["gpu_kv_tokens"] = int(gpu_match.group("tokens").replace(",", ""))

    if mode == "hybrid":
        match = HYBRID_CAPACITY_RE.search(output)
        if match:
            parsed.update(
                total_tokens_per_request=int(match.group("tokens")),
                effective_capacity=float(match.group("capacity")),
                gpu_prefix_blocks_per_request=int(match.group("gpu_prefix_blocks")),
                gpu_prefix_capacity=float(match.group("gpu_prefix_capacity")),
                cpu_suffix_blocks_per_request=int(match.group("cpu_suffix_blocks")),
                cpu_blocks=int(match.group("cpu_blocks")),
                cpu_capacity=float(match.group("cpu_capacity")),
            )
            return parsed

    match = CAPACITY_RE.search(output)
    if match:
        parsed.update(
            total_tokens_per_request=int(match.group("tokens")),
            effective_capacity=float(match.group("capacity")),
        )
    return parsed


def _run_key(*, mode: str, mem: float, split: int | None) -> str:
    split_part = "gpu" if split is None else f"split{split}"
    return f"{mode}-mem{mem:.3f}-{split_part}"


def _existing_successes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("returncode") == 0 and row.get("run_key"):
            keys.add(str(row["run_key"]))
    return keys


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--label", default="llama")
    parser.add_argument("--modes", default="gpu,hybrid")
    parser.add_argument("--mems", default="0.72,0.74,0.75,0.76,0.77,0.78,0.79,0.80")
    parser.add_argument("--splits", default="640,672,704,736,752")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--prompt-tokens", type=int, default=608)
    parser.add_argument("--total-tokens", type=int, default=768)
    parser.add_argument("--cpu-pool-gb", type=float, default=12.0)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--async-scheduling", choices=["true", "false"], default="false")
    parser.add_argument("--enforce-eager", choices=["true", "false"], default="true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/TTC/results/phase2_capacity_window_llama.jsonl"),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/TTC/results/phase2_capacity_window_logs"),
    )
    parser.add_argument("--benchmark", type=Path, default=Path(__file__).with_name("benchmark_ratio_e2e.py"))
    parser.add_argument("--workdir", type=Path, default=Path("/TTC/FastTTS-thesis"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    mems = _float_list(args.mems)
    splits = _int_list(args.splits)
    modes = {item for item in args.modes.split(",") if item}
    unknown_modes = modes - {"gpu", "hybrid"}
    if unknown_modes:
        raise ValueError(f"unknown modes: {sorted(unknown_modes)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    successes = set() if args.no_resume else _existing_successes(args.output)

    tasks: list[tuple[str, float, int | None]] = []
    for mem in mems:
        if "gpu" in modes:
            tasks.append(("gpu", mem, None))
        if "hybrid" in modes:
            for split in splits:
                if split >= args.prompt_tokens and split < args.total_tokens:
                    tasks.append(("hybrid", mem, split))

    for mode, mem, split in tasks:
        run_key = _run_key(mode=mode, mem=mem, split=split)
        if run_key in successes:
            print(f"skip {run_key}", flush=True)
            continue

        split_tokens = split if split is not None else max(splits)
        cmd = [
            sys.executable,
            str(args.benchmark),
            "--mode",
            mode,
            "--model",
            args.model,
            "--batch",
            str(args.batch),
            "--prompt-tokens",
            str(args.prompt_tokens),
            "--total-tokens",
            str(args.total_tokens),
            "--split-tokens",
            str(split_tokens),
            "--gpu-memory-utilization",
            f"{mem:.3f}",
            "--cpu-pool-gb",
            f"{args.cpu_pool_gb:g}",
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--async-scheduling",
            args.async_scheduling,
            "--enforce-eager",
            args.enforce_eager,
        ]
        print("run " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue

        start = time.time()
        proc = subprocess.run(
            cmd,
            cwd=args.workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
        elapsed = time.time() - start
        output = proc.stdout
        log_path = args.log_dir / f"{args.label}-{run_key}.log"
        log_path.write_text(output)

        csv_row = _parse_csv_tail(output)
        row: dict[str, Any] = {
            "run_key": run_key,
            "label": args.label,
            "model": args.model,
            "mode": mode,
            "gpu_memory_utilization": mem,
            "split_tokens": split,
            "batch": args.batch,
            "prompt_tokens": args.prompt_tokens,
            "total_tokens": args.total_tokens,
            "cpu_pool_gb": args.cpu_pool_gb,
            "max_num_seqs": args.max_num_seqs,
            "async_scheduling": args.async_scheduling == "true",
            "enforce_eager": args.enforce_eager == "true",
            "returncode": proc.returncode,
            "wall_time_s": round(elapsed, 3),
            "log_path": str(log_path),
            "command": cmd,
        }
        row.update(_parse_capacity(output, mode=mode))
        if csv_row is not None:
            row["benchmark_row"] = csv_row
            if csv_row.get("out_tok_s") is not None:
                row["out_tok_s"] = float(csv_row["out_tok_s"])
            if csv_row.get("elapsed_s") is not None:
                row["benchmark_elapsed_s"] = float(csv_row["elapsed_s"])
        if proc.returncode != 0:
            row["output_tail"] = "\n".join(output.splitlines()[-40:])
        _write_jsonl(args.output, row)
        print(
            f"done {run_key} rc={proc.returncode} "
            f"cap={row.get('effective_capacity')} out={row.get('out_tok_s')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
