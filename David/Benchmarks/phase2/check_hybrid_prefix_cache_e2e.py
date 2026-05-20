#!/usr/bin/env python3
"""End-to-end COTS hybrid KV prefix-cache probe.

The probe verifies the transparent CPU suffix prefix-cache path in a full vLLM
engine run:

1. Request A has a prompt ending exactly at the COTS split and decodes enough
   tokens to materialize one or more CPU suffix blocks. vLLM samples a token
   before that token has KV, so this needs one extra decode step beyond the
   suffix tokens reused by Request B.
2. Request B uses A's prompt plus those generated suffix tokens as its prompt,
   then adds one tail token. vLLM should hit GPU prefix blocks and CPU suffix
   blocks, while recomputing only the uncached tail token for logits.

Success requires prefix-cache hit tokens to exceed split_tokens. A hit count
at or below split_tokens only proves GPU prefix cache reuse.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from vllm import LLM, SamplingParams, TokensPrompt


BLOCK_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--split-tokens", type=int, default=64)
    parser.add_argument("--cached-suffix-tokens", type=int, default=16)
    parser.add_argument("--first-decode-tokens", type=int, default=17)
    parser.add_argument("--second-decode-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--cpu-pool-gb", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--enforce-eager", choices=["true", "false"], default="true")
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    if args.split_tokens <= 0 or args.split_tokens % BLOCK_SIZE:
        raise SystemExit("--split-tokens must be positive and block aligned")
    if args.cached_suffix_tokens <= 0 or args.cached_suffix_tokens % BLOCK_SIZE:
        raise SystemExit("--cached-suffix-tokens must be positive and block aligned")
    if args.first_decode_tokens <= args.cached_suffix_tokens:
        raise SystemExit(
            "--first-decode-tokens must exceed --cached-suffix-tokens; "
            "vLLM materializes a generated token KV on the next decode step"
        )
    if args.second_decode_tokens <= 0:
        raise SystemExit("--second-decode-tokens must be positive")
    if args.cpu_pool_gb <= 0.0:
        raise SystemExit("--cpu-pool-gb must be positive")

    min_model_len = (
        args.split_tokens + args.cached_suffix_tokens + 1 + args.second_decode_tokens
    )
    if args.max_model_len == 0:
        args.max_model_len = max(min_model_len, args.split_tokens + args.first_decode_tokens + 2)
    elif args.max_model_len < min_model_len:
        raise SystemExit(
            "--max-model-len must cover split + cached suffix + tail + decode"
        )
    return args


def llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager == "true",
        "trust_remote_code": True,
        "disable_log_stats": False,
        "enable_prefix_caching": True,
        "disable_hybrid_kv_cache_manager": True,
        "async_scheduling": False,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_model_len,
        "block_size": BLOCK_SIZE,
        "offload_backend": "cots",
        "cots_kv_split_blocks": args.split_tokens // BLOCK_SIZE,
        "cots_kv_cpu_pool_bytes": int(args.cpu_pool_gb * (1 << 30)),
    }


def prefix_metric_totals(llm: LLM) -> tuple[int, int] | None:
    manager = getattr(llm.llm_engine, "logger_manager", None)
    if manager is None:
        return None

    queries = 0
    hits = 0
    found = False
    for stat_logger in getattr(manager, "stat_loggers", []):
        leaf_loggers = getattr(stat_logger, "per_engine_stat_loggers", None)
        if leaf_loggers is not None:
            leaves = leaf_loggers.values()
        else:
            leaves = (stat_logger,)
        for leaf in leaves:
            metrics = getattr(leaf, "prefix_caching_metrics", None)
            if metrics is None:
                continue
            found = True
            queries += int(getattr(metrics, "aggregated_query_total", 0))
            hits += int(getattr(metrics, "aggregated_query_hit", 0))
    if not found:
        return None
    return queries, hits


def last_scheduler_stats_dict(llm: LLM) -> dict[str, Any] | None:
    manager = getattr(llm.llm_engine, "logger_manager", None)
    if manager is None:
        return None
    for stat_logger in getattr(manager, "stat_loggers", []):
        leaf_loggers = getattr(stat_logger, "per_engine_stat_loggers", None)
        leaves = leaf_loggers.values() if leaf_loggers is not None else (stat_logger,)
        for leaf in leaves:
            stats = getattr(leaf, "last_scheduler_stats", None)
            if stats is None:
                continue
            out: dict[str, Any] = {}
            prefix_stats = getattr(stats, "prefix_cache_stats", None)
            if prefix_stats is not None:
                out["prefix_cache_stats"] = asdict(prefix_stats)
            cots_stats = getattr(stats, "cots_hybrid_kv_stats", None)
            if cots_stats is not None:
                out["cots_hybrid_kv_stats"] = asdict(cots_stats)
            return out
    return None


def run_generate(
    llm: LLM,
    prompt_token_ids: list[int],
    *,
    max_tokens: int,
) -> list[int]:
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids)],
        sampling,
        use_tqdm=False,
    )
    return [int(token) for token in outputs[0].outputs[0].token_ids]


def main() -> None:
    args = parse_args()

    # Keep token ids deterministic and inside the Qwen tokenizer vocabulary.
    first_prompt = [100] * (args.split_tokens - 1) + [200]
    uncached_tail_token = 201

    llm = LLM(**llm_kwargs(args))
    before = prefix_metric_totals(llm)

    first_generated = run_generate(
        llm,
        first_prompt,
        max_tokens=args.first_decode_tokens,
    )
    after_first = prefix_metric_totals(llm)

    cached_suffix = first_generated[: args.cached_suffix_tokens]
    second_prompt = first_prompt + cached_suffix + [uncached_tail_token]
    second_generated = run_generate(
        llm,
        second_prompt,
        max_tokens=args.second_decode_tokens,
    )
    after_second = prefix_metric_totals(llm)

    if after_first is not None and after_second is not None:
        second_delta_queries = after_second[0] - after_first[0]
        second_delta_hits = after_second[1] - after_first[1]
    else:
        second_delta_queries = None
        second_delta_hits = None

    expected_min_cpu_suffix_hit_tokens = args.split_tokens + args.cached_suffix_tokens
    ok = (
        second_delta_hits is not None
        and second_delta_hits >= expected_min_cpu_suffix_hit_tokens
    )

    summary = {
        "ok": ok,
        "model": args.model,
        "split_tokens": args.split_tokens,
        "cached_suffix_tokens": args.cached_suffix_tokens,
        "first_decode_tokens": args.first_decode_tokens,
        "second_prompt_len": len(second_prompt),
        "expected_min_cpu_suffix_hit_tokens": expected_min_cpu_suffix_hit_tokens,
        "prefix_metric_totals": {
            "before": before,
            "after_first": after_first,
            "after_second": after_second,
            "second_delta_queries": second_delta_queries,
            "second_delta_hits": second_delta_hits,
        },
        "last_scheduler_stats": last_scheduler_stats_dict(llm),
        "first_generated_len": len(first_generated),
        "second_generated": second_generated,
        "kwargs": {
            key: value
            for key, value in llm_kwargs(args).items()
            if key not in {"trust_remote_code"}
        },
    }

    print(json.dumps(summary, indent=2), flush=True)
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not ok:
        raise SystemExit(
            "CPU suffix prefix-cache hit was not observed: "
            f"second_delta_hits={second_delta_hits}, "
            f"expected>={expected_min_cpu_suffix_hit_tokens}"
        )


if __name__ == "__main__":
    # Preserve local benchmark imports if invoked from another working dir.
    os.environ["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent)
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    )
    main()
