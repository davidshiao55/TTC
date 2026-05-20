#!/usr/bin/env python3
"""Fixed-context Phase 2 parity probe.

Free generation is too sensitive to tiny logit drift for a clean hybrid-KV
guard. This script first records a GPU-only reference continuation, then runs
GPU-only and hybrid KV while forcing both modes through that exact continuation.
It compares raw logprobs and top-k sets before and after the split.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams, TokensPrompt

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
os.environ["PYTHONPATH"] = (
    str(THIS_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")
)


def make_prompts(batch: int, prompt_len: int) -> list[TokensPrompt]:
    shared = [100] * (prompt_len - 1)
    return [
        TokensPrompt(prompt_token_ids=shared + [200 + idx])
        for idx in range(batch)
    ]


def llm_kwargs(
    args: argparse.Namespace,
    *,
    hybrid: bool,
    forced: bool,
    enforce_eager: bool | None = None,
) -> dict[str, Any]:
    if enforce_eager is None:
        enforce_eager = args.hybrid_enforce_eager == "true" if hybrid else True
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.total_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "trust_remote_code": True,
        "disable_log_stats": True,
        "disable_hybrid_kv_cache_manager": True,
        "async_scheduling": False,
        "max_num_seqs": args.max_num_seqs or args.batch,
    }
    if forced:
        kwargs["logits_processors"] = [
            "phase2_forced_logits_proc:CaptureForceLogitsProcessor"
        ]
    if args.cots_f_cpu_store > 0.0 or args.cots_f_prefetch > 0.0:
        kwargs.update(
            offload_backend="cots",
            cots_f_cpu_store=args.cots_f_cpu_store,
            cots_f_prefetch=args.cots_f_prefetch,
        )
    if hybrid:
        kwargs.update(
            offload_backend="cots",
            cots_kv_split_blocks=args.split_tokens // 16,
            cots_kv_cpu_pool_bytes=int(args.cpu_pool_gb * (1 << 30)),
        )
    return kwargs


def run_reference(
    args: argparse.Namespace,
    prompts: list[TokensPrompt],
) -> dict[str, list[int]]:
    llm = LLM(**llm_kwargs(args, hybrid=False, forced=False, enforce_eager=True))
    sampling = SamplingParams(
        max_tokens=args.decode_tokens,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    forced: dict[str, list[int]] = {}
    for prompt, output in zip(prompts, outputs):
        tail = str(prompt["prompt_token_ids"][-1])
        forced[tail] = [int(token) for token in output.outputs[0].token_ids]
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return forced


def run_forced(
    args: argparse.Namespace,
    prompts: list[TokensPrompt],
    forced: dict[str, list[int]],
    *,
    mode: str,
    hybrid: bool,
    enforce_eager: bool | None = None,
) -> list[list[int]]:
    os.environ["COTS_FORCE_LOGITS_MODE"] = mode
    os.environ["COTS_FORCE_LOGITS_OUT"] = args.out_jsonl
    os.environ["COTS_FORCE_LOGITS_TOPK"] = str(args.topk)
    llm = LLM(
        **llm_kwargs(
            args,
            hybrid=hybrid,
            forced=True,
            enforce_eager=enforce_eager,
        )
    )
    sampling = SamplingParams(
        max_tokens=args.decode_tokens,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
        extra_args={"forced_by_prompt_tail": forced},
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    token_ids = [
        [int(token) for token in output.outputs[0].token_ids]
        for output in outputs
    ]
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return token_ids


def load_records(path: str) -> dict[tuple[str, int, int], dict[str, Any]]:
    records = {}
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        records[(rec["mode"], int(rec["prompt_tail"]), int(rec["step"]))] = rec
    return records


def stats(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"max": None, "mean": None, "p95": None}
    ordered = sorted(xs)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {"max": max(xs), "mean": sum(xs) / len(xs), "p95": p95}


def format_stats(name: str, values: dict[str, float | None]) -> str:
    if values["max"] is None:
        return f"{name}=n/a"
    return (
        f"{name}=max={values['max']:.6g}, "
        f"mean={values['mean']:.6g}, p95={values['p95']:.6g}"
    )


def compare(
    args: argparse.Namespace,
    forced: dict[str, list[int]],
) -> dict[str, Any]:
    records = load_records(args.out_jsonl)
    split_step = args.split_tokens - args.prompt_tokens
    total = 0
    post = 0
    top1_same = 0
    post_top1_same = 0
    forced_deltas: list[float] = []
    post_forced_deltas: list[float] = []
    top1_deltas: list[float] = []
    post_top1_deltas: list[float] = []
    jaccards: list[float] = []
    post_jaccards: list[float] = []
    top1_mismatches: list[tuple[int, int, int, int, float, float]] = []

    for tail_s, tokens in forced.items():
        tail = int(tail_s)
        for step in range(len(tokens)):
            gpu = records.get(("gpu", tail, step))
            other = records.get((args.compare_mode, tail, step))
            if gpu is None or other is None:
                continue
            total += 1
            is_post = step >= split_step
            if is_post:
                post += 1
            forced_delta = abs(
                float(gpu["forced_logprob"])
                - float(other["forced_logprob"])
            )
            forced_deltas.append(forced_delta)
            if is_post:
                post_forced_deltas.append(forced_delta)

            gpu_top = int(gpu["top_ids"][0])
            other_top = int(other["top_ids"][0])
            if gpu_top == other_top:
                top1_same += 1
                top1_delta = abs(
                    float(gpu["top_logprobs"][0])
                    - float(other["top_logprobs"][0])
                )
                top1_deltas.append(top1_delta)
                if is_post:
                    post_top1_same += 1
                    post_top1_deltas.append(top1_delta)
            elif len(top1_mismatches) < 12:
                top1_mismatches.append(
                    (
                        tail,
                        step,
                        gpu_top,
                        other_top,
                        float(gpu["top_logprobs"][0]),
                        float(other["top_logprobs"][0]),
                    )
                )

            gpu_set = set(int(token) for token in gpu["top_ids"])
            other_set = set(int(token) for token in other["top_ids"])
            jaccard = len(gpu_set & other_set) / len(gpu_set | other_set)
            jaccards.append(jaccard)
            if is_post:
                post_jaccards.append(jaccard)

    return {
        "batch": args.batch,
        "prompt_tokens": args.prompt_tokens,
        "split_tokens": args.split_tokens,
        "decode_tokens": args.decode_tokens,
        "split_step": split_step,
        "positions_compared": total,
        "post_split_positions": post,
        "top1_same": top1_same,
        "post_split_top1_same": post_top1_same,
        "forced_token_logprob_delta": stats(forced_deltas),
        "post_split_forced_token_logprob_delta": stats(post_forced_deltas),
        "top1_logprob_delta_when_same": stats(top1_deltas),
        "post_split_top1_logprob_delta_when_same": stats(post_top1_deltas),
        f"top{args.topk}_jaccard": stats(jaccards),
        f"post_split_top{args.topk}_jaccard": stats(post_jaccards),
        "top1_mismatches": top1_mismatches,
    }


def print_summary(summary: dict[str, Any], topk: int) -> None:
    total = summary["positions_compared"]
    post = summary["post_split_positions"]
    print("phase2_forced_context_parity_summary")
    print(
        f"batch={summary['batch']} prompt={summary['prompt_tokens']} "
        f"split={summary['split_tokens']} decode={summary['decode_tokens']} "
        f"split_step={summary['split_step']}"
    )
    print(f"positions_compared={total} post_split_positions={post}")
    print(
        f"top1_same={summary['top1_same']}/{total} "
        f"post_split_top1_same={summary['post_split_top1_same']}/{post}"
    )
    print(
        format_stats(
            "forced_token_logprob_delta",
            summary["forced_token_logprob_delta"],
        )
    )
    print(
        format_stats(
            "post_split_forced_token_logprob_delta",
            summary["post_split_forced_token_logprob_delta"],
        )
    )
    print(
        format_stats(
            "top1_logprob_delta_when_same",
            summary["top1_logprob_delta_when_same"],
        )
    )
    print(
        format_stats(
            "post_split_top1_logprob_delta_when_same",
            summary["post_split_top1_logprob_delta_when_same"],
        )
    )
    print(format_stats(f"top{topk}_jaccard", summary[f"top{topk}_jaccard"]))
    print(
        format_stats(
            f"post_split_top{topk}_jaccard",
            summary[f"post_split_top{topk}_jaccard"],
        )
    )
    print(f"top1_mismatches={summary['top1_mismatches']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--prompt-tokens", type=int, default=608)
    parser.add_argument("--split-tokens", type=int, default=672)
    parser.add_argument("--total-tokens", type=int, default=768)
    parser.add_argument("--decode-tokens", type=int, default=160)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.68)
    parser.add_argument("--cpu-pool-gb", type=float, default=4.0)
    parser.add_argument("--cots-f-cpu-store", type=float, default=0.0)
    parser.add_argument("--cots-f-prefetch", type=float, default=0.0)
    parser.add_argument(
        "--out-jsonl",
        default="/tmp/phase2_forced_logits.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        default="/tmp/phase2_forced_context_parity_summary.json",
    )
    parser.add_argument("--second-gpu-control", action="store_true")
    parser.add_argument(
        "--gpu-control-enforce-eager",
        choices=["true", "false"],
        default="true",
        help="Use eager execution for the second GPU control leg.",
    )
    parser.add_argument(
        "--hybrid-enforce-eager",
        choices=["true", "false"],
        default="true",
        help="Use eager execution for the hybrid comparison leg.",
    )
    args = parser.parse_args()
    if args.split_tokens % 16:
        raise SystemExit("split must be block aligned")
    if args.prompt_tokens > args.split_tokens:
        raise SystemExit("prompt must be <= split")
    if args.prompt_tokens + args.decode_tokens > args.total_tokens:
        raise SystemExit("prompt + decode must fit total tokens")
    if args.second_gpu_control:
        args.compare_mode = "gpu_again"
    else:
        args.compare_mode = (
            "hybrid" if args.hybrid_enforce_eager == "true" else "hybrid_graph"
        )
    return args


def main() -> None:
    args = parse_args()
    Path(args.out_jsonl).unlink(missing_ok=True)
    prompts = make_prompts(args.batch, args.prompt_tokens)
    print("reference_gpu")
    forced = run_reference(args, prompts)
    print("forced_gpu")
    gpu_tokens = run_forced(
        args,
        prompts,
        forced,
        mode="gpu",
        hybrid=False,
        enforce_eager=True,
    )
    print("forced_gpu_again" if args.second_gpu_control else "forced_hybrid")
    other_tokens = run_forced(
        args,
        prompts,
        forced,
        mode=args.compare_mode,
        hybrid=not args.second_gpu_control,
        enforce_eager=(
            args.gpu_control_enforce_eager == "true"
            if args.second_gpu_control
            else None
        ),
    )
    failures = []
    for idx, prompt in enumerate(prompts):
        tail = str(prompt["prompt_token_ids"][-1])
        if gpu_tokens[idx] != forced[tail] or other_tokens[idx] != forced[tail]:
            failures.append(
                (
                    idx,
                    tail,
                    gpu_tokens[idx][:5],
                    other_tokens[idx][:5],
                    forced[tail][:5],
                )
            )
    print(f"forced_output_failures={failures}")
    summary = compare(args, forced)
    summary["forced_output_failures"] = failures
    summary["second_gpu_control"] = args.second_gpu_control
    summary["hybrid_enforce_eager"] = args.hybrid_enforce_eager
    summary["gpu_control_enforce_eager"] = args.gpu_control_enforce_eager
    summary["cots_f_cpu_store"] = args.cots_f_cpu_store
    summary["cots_f_prefetch"] = args.cots_f_prefetch
    print_summary(summary, args.topk)
    out_path = Path(args.summary_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
