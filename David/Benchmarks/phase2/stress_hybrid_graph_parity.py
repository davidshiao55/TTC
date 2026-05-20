#!/usr/bin/env python3
"""Repeat Phase 2 hybrid graph forced-context parity probes.

This is a stability harness around `check_hybrid_forced_context_parity.py`.
It intentionally runs each probe in a fresh Python process so CUDA illegal
accesses or EngineCore deaths are counted as failures without poisoning the
next run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
SINGLE_PROBE = THIS_DIR / "check_hybrid_forced_context_parity.py"


def parse_int_list(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batches", type=parse_int_list, default="4")
    parser.add_argument("--prompt-tokens", type=int, default=608)
    parser.add_argument("--split-tokens", type=int, default=608)
    parser.add_argument("--total-tokens", type=int, default=640)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--cpu-pool-gb", type=float, default=1.0)
    parser.add_argument("--cots-f-cpu-store", type=float, default=0.0)
    parser.add_argument("--cots-f-prefetch", type=float, default=0.0)
    parser.add_argument("--hybrid-enforce-eager", choices=["true", "false"], default="false")
    parser.add_argument("--out-dir", default="/tmp/phase2_graph_stress")
    parser.add_argument("--summary-json", default="/tmp/phase2_graph_stress_summary.json")
    parser.add_argument("--disable-compile-cache", action="store_true")
    parser.add_argument("--cuda-launch-blocking", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.split_tokens % 16:
        raise SystemExit("split must be block aligned")
    if args.prompt_tokens > args.split_tokens:
        raise SystemExit("prompt must be <= split")
    if args.prompt_tokens + args.decode_tokens > args.total_tokens:
        raise SystemExit("prompt + decode must fit total tokens")
    return args


def run_one(args: argparse.Namespace, *, batch: int, repeat_idx: int) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"b{batch}_r{repeat_idx:03d}"
    out_jsonl = out_dir / f"{stem}.jsonl"
    summary_json = out_dir / f"{stem}.summary.json"
    log_path = out_dir / f"{stem}.log"

    cmd = [
        sys.executable,
        str(SINGLE_PROBE),
        "--model",
        args.model,
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--split-tokens",
        str(args.split_tokens),
        "--total-tokens",
        str(args.total_tokens),
        "--decode-tokens",
        str(args.decode_tokens),
        "--batch",
        str(batch),
        "--topk",
        str(args.topk),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cpu-pool-gb",
        str(args.cpu_pool_gb),
        "--cots-f-cpu-store",
        str(args.cots_f_cpu_store),
        "--cots-f-prefetch",
        str(args.cots_f_prefetch),
        "--hybrid-enforce-eager",
        args.hybrid_enforce_eager,
        "--out-jsonl",
        str(out_jsonl),
        "--summary-json",
        str(summary_json),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(THIS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    if args.disable_compile_cache:
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    if args.cuda_launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd="/TTC/FastTTS-thesis",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed_s = time.perf_counter() - start
    log_path.write_text(proc.stdout)

    result: dict[str, Any] = {
        "batch": batch,
        "repeat": repeat_idx,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "log": str(log_path),
        "summary_json": str(summary_json),
        "out_jsonl": str(out_jsonl),
        "ok": False,
    }
    if proc.returncode == 0 and summary_json.exists():
        summary = json.loads(summary_json.read_text())
        result.update(
            positions_compared=summary.get("positions_compared"),
            post_split_positions=summary.get("post_split_positions"),
            top1_same=summary.get("top1_same"),
            post_split_top1_same=summary.get("post_split_top1_same"),
            forced_output_failures=summary.get("forced_output_failures", []),
            forced_delta=summary.get("forced_token_logprob_delta"),
            post_forced_delta=summary.get("post_split_forced_token_logprob_delta"),
            top1_mismatches=summary.get("top1_mismatches", []),
        )
        result["ok"] = not result["forced_output_failures"]
    else:
        tail = proc.stdout[-4000:]
        result["error_tail"] = tail
    return result


def format_delta(delta: Any) -> str:
    if not isinstance(delta, dict) or delta.get("max") is None:
        return "n/a"
    return (
        f"max={delta['max']:.6g},mean={delta['mean']:.6g},"
        f"p95={delta['p95']:.6g}"
    )


def main() -> None:
    args = parse_args()
    results: list[dict[str, Any]] = []
    for batch in args.batches:
        for repeat_idx in range(args.repeats):
            print(f"stress_run_start batch={batch} repeat={repeat_idx}", flush=True)
            result = run_one(args, batch=batch, repeat_idx=repeat_idx)
            results.append(result)
            status = "ok" if result.get("ok") else "FAIL"
            print(
                "stress_run_result "
                f"batch={batch} repeat={repeat_idx} status={status} "
                f"returncode={result['returncode']} elapsed_s={result['elapsed_s']:.2f} "
                f"top1={result.get('top1_same')}/{result.get('positions_compared')} "
                f"post_top1={result.get('post_split_top1_same')}/{result.get('post_split_positions')} "
                f"forced_delta={format_delta(result.get('forced_delta'))} "
                f"log={result['log']}",
                flush=True,
            )
            if args.stop_on_failure and not result.get("ok"):
                break
        if args.stop_on_failure and results and not results[-1].get("ok"):
            break

    failures = [r for r in results if not r.get("ok")]
    aggregate = {
        "total_runs": len(results),
        "failed_runs": len(failures),
        "args": vars(args),
        "results": results,
    }
    out_path = Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2))
    print(
        f"stress_summary total_runs={len(results)} failed_runs={len(failures)} "
        f"summary_json={out_path}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
