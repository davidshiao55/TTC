#!/usr/bin/env python3
"""Synthetic Phase 2 E2E split-ratio benchmark.

This is the reusable version of the split/prompt/total sweep harness used for
Phase 2 measurements. It creates a large shared prefix with one perturbed token
per request, then measures decode throughput for GPU-only or COTS hybrid KV.
Run it from `/TTC/FastTTS-thesis` so the thesis vLLM install is resolved.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from vllm import LLM, SamplingParams, TokensPrompt


def make_prompts(batch: int, prompt_len: int) -> list[TokensPrompt]:
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    shared = [100] * max(prompt_len - 1, 0)
    return [
        TokensPrompt(prompt_token_ids=shared + [200 + idx])
        for idx in range(batch)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gpu", "hybrid"], required=True)
    parser.add_argument("--split-tokens", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--total-tokens", type=int, default=640)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--cpu-pool-gb", type=float, default=4.0)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--log-iterations", action="store_true")
    parser.add_argument("--enable-log-stats", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--async-scheduling",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument("--cots-f-cpu-store", type=float, default=0.0)
    parser.add_argument("--cots-f-prefetch", type=float, default=0.0)
    parser.add_argument("--cuda-profile-range", action="store_true")
    parser.add_argument(
        "--enforce-eager",
        choices=["true", "false"],
        default="true",
        help="Use eager execution. Hybrid KV currently expects true.",
    )
    args = parser.parse_args()

    prompt_tokens_per_req = (
        args.split_tokens if args.prompt_tokens is None else args.prompt_tokens
    )
    if args.total_tokens <= prompt_tokens_per_req:
        raise ValueError("total_tokens must be > prompt_tokens")
    if args.split_tokens < prompt_tokens_per_req:
        raise ValueError("split_tokens must be >= prompt_tokens")
    if args.split_tokens % 16 != 0:
        raise ValueError("split_tokens must be block aligned")

    max_tokens = args.total_tokens - prompt_tokens_per_req
    engine_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.total_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager == "true",
        "trust_remote_code": True,
        "disable_log_stats": not args.enable_log_stats,
        "disable_hybrid_kv_cache_manager": True,
    }
    if args.async_scheduling != "auto":
        engine_kwargs["async_scheduling"] = args.async_scheduling == "true"
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.log_iterations:
        engine_kwargs["enable_logging_iteration_details"] = True
    if args.cots_f_cpu_store > 0.0 or args.cots_f_prefetch > 0.0:
        engine_kwargs.update(
            offload_backend="cots",
            cots_f_cpu_store=args.cots_f_cpu_store,
            cots_f_prefetch=args.cots_f_prefetch,
        )
    if args.mode == "hybrid":
        engine_kwargs.update(
            offload_backend="cots",
            cots_kv_split_blocks=args.split_tokens // 16,
            cots_kv_cpu_pool_bytes=int(args.cpu_pool_gb * (1 << 30)),
        )

    llm = LLM(**engine_kwargs)
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    warm_sampling = SamplingParams(
        max_tokens=min(max_tokens, 8),
        temperature=0.0,
        ignore_eos=True,
    )
    llm.generate(
        make_prompts(min(args.batch, 4), prompt_tokens_per_req),
        warm_sampling,
        use_tqdm=False,
    )
    prompts = make_prompts(args.batch, prompt_tokens_per_req)

    if args.cuda_profile_range and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    if args.cuda_profile_range and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    elapsed = time.perf_counter() - t0

    out_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    prompt_tokens = args.batch * prompt_tokens_per_req
    ratio = args.split_tokens / max_tokens
    print(
        "mode,batch,total_tokens,prompt_tokens_per_req,split_tokens,"
        "suffix_tokens,ratio,elapsed_s,prompt_tokens,out_tokens,"
        "out_tok_s,total_tok_s"
    )
    print(
        f"{args.mode},{args.batch},{args.total_tokens},"
        f"{prompt_tokens_per_req},{args.split_tokens},{max_tokens},"
        f"{ratio:.3f},{elapsed:.6f},{prompt_tokens},{out_tokens},"
        f"{out_tokens / elapsed:.3f},"
        f"{(prompt_tokens + out_tokens) / elapsed:.3f}"
    )

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
