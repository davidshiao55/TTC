#!/usr/bin/env python3
"""Greedy parity check for Planner COTS dispatch cells.

Run from /TTC/FastTTS-thesis in the thesis environment.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
CAPTURE_BUCKETS = (
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


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/planner/dispatch_parity") / stamp


def bucket_for(n: int) -> int:
    for bucket in CAPTURE_BUCKETS:
        if n <= bucket:
            return bucket
    return CAPTURE_BUCKETS[-1]


def decode_only_dispatch_table(
    *,
    batch: int,
    f_cpu_store: float,
    f_cpu: float,
    f_prefetch: float,
) -> dict[int, tuple[float, float]]:
    table = {int(bucket): (0.0, float(f_cpu_store)) for bucket in CAPTURE_BUCKETS}
    table[bucket_for(batch)] = (float(f_cpu), float(f_prefetch))
    return table


def thread_table(batch: int, cpu_threads: int) -> dict[int, int]:
    # vLLM's graph capture buckets top out at 512 for these Qwen latency cells.
    # Extra thread-table keys are rejected, so keep this to the known graph set.
    return {int(bucket): 4 for bucket in CAPTURE_BUCKETS if bucket <= 512} | {
        bucket_for(batch): int(cpu_threads)
    }


def prompts(batch: int) -> list[str]:
    base = "Question: What is {a} + {b}? Answer:"
    return [base.format(a=17 + i % 7, b=25 + i % 5) for i in range(batch)]


def arm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": 0,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
    }
    if args.run_arm in ("none", "none_graph"):
        return base
    if args.run_arm == "none_eager":
        return {**base, "enforce_eager": True}
    cots_arms = {
        "cots_decode_cpu",
        "cots_decode_cpu_graph",
        "cots_decode_cpu_eager",
    }
    if args.run_arm not in cots_arms:
        raise ValueError(f"unknown arm: {args.run_arm}")
    table = decode_only_dispatch_table(
        batch=args.batch,
        f_cpu_store=args.f_cpu_store,
        f_cpu=args.f_cpu,
        f_prefetch=args.f_prefetch,
    )
    kwargs = {
        **base,
        "offload_backend": "cots",
        "cots_f_cpu_store": args.f_cpu_store,
        "cots_f_prefetch": 0.0,
        "cots_dispatch_table": table,
        "cots_cpu_runner": "native",
        "cots_cpu_num_threads": args.cpu_threads,
    }
    if args.run_arm != "cots_decode_cpu_eager":
        kwargs["cots_cpu_num_threads_by_bucket"] = thread_table(
            args.batch, args.cpu_threads
        )
    else:
        kwargs["enforce_eager"] = True
    return kwargs


def run_one(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams

    llm = LLM(**arm_kwargs(args))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    outputs = llm.generate(prompts(args.batch), sampling, use_tqdm=False)
    records = []
    for idx, output in enumerate(outputs):
        completion = output.outputs[0]
        records.append(
            {
                "index": idx,
                "prompt": output.prompt,
                "text": completion.text,
                "token_ids": list(completion.token_ids),
            }
        )
    args.output_json.write_text(
        json.dumps(
            {
                "arm": args.run_arm,
                "batch": args.batch,
                "max_tokens": args.max_tokens,
                "records": records,
            },
            indent=2,
        )
    )
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def run_parent(args: argparse.Namespace) -> int:
    if os.path.abspath(os.getcwd()) == "/TTC":
        raise RuntimeError("Run from /TTC/FastTTS-thesis, not /TTC")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    arm_names = [args.reference_arm, *args.candidate_arms]
    records_by_arm = {}
    for arm in arm_names:
        out_json = args.results_dir / f"{arm}.json"
        out_log = args.results_dir / f"{arm}.log"
        if out_json.exists() and not args.force:
            print(f"[skip] {arm}", flush=True)
        else:
            cmd = [
                sys.executable,
                __file__,
                "--run-arm",
                arm,
                "--output-json",
                str(out_json),
                "--batch",
                str(args.batch),
                "--max-tokens",
                str(args.max_tokens),
                "--max-model-len",
                str(args.max_model_len),
                "--f-cpu-store",
                str(args.f_cpu_store),
                "--f-cpu",
                str(args.f_cpu),
                "--f-prefetch",
                str(args.f_prefetch),
                "--cpu-threads",
                str(args.cpu_threads),
            ]
            print(f"[run] {arm}", flush=True)
            with out_log.open("w") as log:
                proc = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            if proc.returncode != 0:
                tail = "\n".join(out_log.read_text(errors="replace").splitlines()[-40:])
                raise RuntimeError(f"{arm} failed rc={proc.returncode}\n{tail}")
        records_by_arm[arm] = json.loads(out_json.read_text())["records"]

    ref = records_by_arm[args.reference_arm]
    comparisons = {}
    for arm in args.candidate_arms:
        got = records_by_arm[arm]
        mismatches = []
        for lhs, rhs in zip(ref, got):
            if lhs["token_ids"] != rhs["token_ids"]:
                mismatches.append(
                    {
                        "index": lhs["index"],
                        "reference_token_ids": lhs["token_ids"],
                        "candidate_token_ids": rhs["token_ids"],
                        "reference_text": lhs["text"],
                        "candidate_text": rhs["text"],
                    }
                )
        comparisons[arm] = {
            "num_mismatches": len(mismatches),
            "all_match": not mismatches,
            "mismatches": mismatches[:10],
        }
    summary = {
        "reference_arm": args.reference_arm,
        "candidate_arms": args.candidate_arms,
        "batch": args.batch,
        "max_tokens": args.max_tokens,
        "f_cpu_store": args.f_cpu_store,
        "f_cpu": args.f_cpu,
        "f_prefetch": args.f_prefetch,
        "comparisons": comparisons,
    }
    summary["all_match"] = all(c["all_match"] for c in comparisons.values())
    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["all_match"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--f-cpu-store", type=float, default=0.30)
    parser.add_argument("--f-cpu", type=float, default=0.30)
    parser.add_argument("--f-prefetch", type=float, default=0.0)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--reference-arm", default="none")
    parser.add_argument(
        "--candidate-arms",
        nargs="+",
        default=["cots_decode_cpu"],
    )
    parser.add_argument("--run-arm", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.run_arm is not None:
        if args.output_json is None:
            parser.error("--run-arm requires --output-json")
        return run_one(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
