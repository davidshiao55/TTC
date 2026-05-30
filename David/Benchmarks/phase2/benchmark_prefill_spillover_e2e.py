#!/usr/bin/env python3
"""Profile Phase 2 split-crossing prefill spillover.

Unlike benchmark_ratio_e2e.py, this intentionally allows prompt_tokens >
split_tokens so the prefill itself creates CPU suffix rows. It reports coarse
end-to-end timing and relies on vLLM COTS stats logging for behavior details.
Run from /TTC/FastTTS-thesis.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch
from vllm import LLM, SamplingParams, TokensPrompt


def make_prompts(batch: int, prompt_len: int, *, prompt_mode: str) -> list[TokensPrompt]:
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    if prompt_mode == "shared":
        shared = [100] * max(prompt_len - 1, 0)
        return [TokensPrompt(prompt_token_ids=shared + [200 + idx]) for idx in range(batch)]
    if prompt_mode == "unique":
        return [
            TokensPrompt(
                prompt_token_ids=[
                    1000 + ((idx * 15485863 + pos * 32452843) % 30000)
                    for pos in range(prompt_len)
                ]
            )
            for idx in range(batch)
        ]
    raise ValueError(f"unknown prompt_mode: {prompt_mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gpu", "hybrid"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=704)
    parser.add_argument("--total-tokens", type=int, default=705)
    parser.add_argument("--split-tokens", type=int, default=672)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.67)
    parser.add_argument("--cpu-pool-gb", type=float, default=12.0)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--prompt-mode", choices=["shared", "unique"], default="unique")
    parser.add_argument("--enable-prefix-caching", choices=["true", "false"], default="false")
    parser.add_argument("--async-scheduling", choices=["true", "false"], default="false")
    parser.add_argument("--enforce-eager", choices=["true", "false"], default="true")
    parser.add_argument("--enable-log-stats", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    if args.total_tokens <= args.prompt_tokens:
        raise ValueError("total_tokens must be > prompt_tokens")
    if args.split_tokens % 16 != 0:
        raise ValueError("split_tokens must be block aligned")
    if args.split_tokens >= args.total_tokens and args.mode == "hybrid":
        raise ValueError("hybrid split must leave at least one CPU suffix token")

    max_tokens = args.total_tokens - args.prompt_tokens
    engine_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.total_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager == "true",
        "trust_remote_code": True,
        "disable_log_stats": not args.enable_log_stats,
        "disable_hybrid_kv_cache_manager": True,
        "max_num_seqs": args.max_num_seqs,
        "async_scheduling": args.async_scheduling == "true",
        "enable_prefix_caching": args.enable_prefix_caching == "true",
    }
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.mode == "hybrid":
        engine_kwargs.update(
            offload_backend="cots",
            cots_kv_split_blocks=args.split_tokens // 16,
            cots_kv_cpu_pool_bytes=int(args.cpu_pool_gb * (1 << 30)),
        )

    llm = LLM(**engine_kwargs)
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    if args.warmup:
        warm_prompt_len = min(args.prompt_tokens, args.split_tokens)
        llm.generate(
            make_prompts(min(args.batch, 2), warm_prompt_len, prompt_mode=args.prompt_mode),
            SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
            use_tqdm=False,
        )
    prompts = make_prompts(args.batch, args.prompt_tokens, prompt_mode=args.prompt_mode)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    out_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    prompt_tokens_total = args.batch * args.prompt_tokens
    spill_prompt_tokens_per_req = max(args.prompt_tokens - args.split_tokens, 0)
    active_suffix_tokens_per_req = max(args.total_tokens - args.split_tokens, 0)

    print(
        "mode,model,batch,prompt_tokens,total_tokens,split_tokens,"
        "spill_prompt_tokens_per_req,active_suffix_tokens_per_req,"
        "prompt_mode,prefix_caching,enforce_eager,elapsed_s,prompt_tok_s,"
        "out_tokens,out_tok_s,total_tok_s"
    )
    print(
        f"{args.mode},{args.model},{args.batch},{args.prompt_tokens},"
        f"{args.total_tokens},{args.split_tokens},{spill_prompt_tokens_per_req},"
        f"{active_suffix_tokens_per_req},{args.prompt_mode},"
        f"{args.enable_prefix_caching},{args.enforce_eager},{elapsed:.6f},"
        f"{prompt_tokens_total / elapsed:.3f},{out_tokens},"
        f"{out_tokens / elapsed:.3f},"
        f"{(prompt_tokens_total + out_tokens) / elapsed:.3f}"
    )
    if args.enable_log_stats:
        llm.llm_engine.do_log_stats()

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
