#!/usr/bin/env python3
"""End-to-end COTS hybrid KV CPU-pool pressure probe.

This validates the full scheduler/worker path for a tiny CPU suffix KV pool.
The intended pressure pattern is:

1. Populate suffix block A and leave it cached with ref_cnt=0.
2. Populate a different suffix block B. With a two-block CPU pool, this must
   evict cached A because the request also needs one live partial tail block.
3. Reuse B and require GPU+CPU cache hits.
4. Try to reuse A and require only GPU prefix hits, proving A was evicted.

The probe uses prefix-cache hit counters as the observable signal:

    split hits      -> GPU prefix cache only
    split + suffix  -> GPU prefix plus CPU suffix cache
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
    parser.add_argument("--decode-tokens", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument(
        "--cpu-pool-bytes",
        type=int,
        default=2 << 20,
        help=(
            "Defaults to 2 MiB, which is two Qwen2.5-7B BF16 suffix "
            "blocks: one cached suffix block plus one live partial tail block."
        ),
    )
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
    if args.decode_tokens <= 0:
        raise SystemExit("--decode-tokens must be positive")
    if args.cpu_pool_bytes <= 0:
        raise SystemExit("--cpu-pool-bytes must be positive")
    if args.cpu_pool_bytes < (2 << 20):
        raise SystemExit(
            "--cpu-pool-bytes must be at least 2 MiB for this probe. "
            "The cache-hit prompt needs one cached suffix block and one live "
            "partial tail block; a one-block pool cannot make scheduler progress."
        )

    min_model_len = args.split_tokens + args.cached_suffix_tokens + 1 + args.decode_tokens
    if args.max_model_len == 0:
        args.max_model_len = max(
            min_model_len,
            args.split_tokens + args.first_decode_tokens + 2,
        )
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
        "cots_kv_cpu_pool_bytes": args.cpu_pool_bytes,
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
        leaves = leaf_loggers.values() if leaf_loggers is not None else (stat_logger,)
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


def generate(llm: LLM, prompt_token_ids: list[int], *, max_tokens: int) -> list[int]:
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


def delta(before: tuple[int, int] | None, after: tuple[int, int] | None) -> dict[str, int | None]:
    if before is None or after is None:
        return {"queries": None, "hits": None}
    return {"queries": after[0] - before[0], "hits": after[1] - before[1]}


def main() -> None:
    args = parse_args()

    first_prompt = [100] * (args.split_tokens - 1) + [200]
    synthetic_suffix_b = [300 + i for i in range(args.cached_suffix_tokens)]
    tail_a = 201
    tail_b_populate = 202
    tail_b_hit = 203
    tail_a_after_eviction = 204

    expected_gpu_only_hit_tokens = args.split_tokens
    expected_cpu_hit_tokens = args.split_tokens + args.cached_suffix_tokens

    llm = LLM(**llm_kwargs(args))
    before_a = prefix_metric_totals(llm)

    generated_a = generate(
        llm,
        first_prompt,
        max_tokens=args.first_decode_tokens,
    )
    after_a = prefix_metric_totals(llm)
    suffix_a = generated_a[: args.cached_suffix_tokens]

    prompt_b_populate = first_prompt + synthetic_suffix_b + [tail_b_populate]
    before_b_populate = after_a
    _ = generate(llm, prompt_b_populate, max_tokens=args.decode_tokens)
    after_b_populate = prefix_metric_totals(llm)

    prompt_b_hit = first_prompt + synthetic_suffix_b + [tail_b_hit]
    before_b_hit = after_b_populate
    output_b_hit = generate(llm, prompt_b_hit, max_tokens=args.decode_tokens)
    after_b_hit = prefix_metric_totals(llm)

    prompt_a_after_eviction = first_prompt + suffix_a + [tail_a_after_eviction]
    before_a_after_eviction = after_b_hit
    output_a_after_eviction = generate(
        llm,
        prompt_a_after_eviction,
        max_tokens=args.decode_tokens,
    )
    after_a_after_eviction = prefix_metric_totals(llm)

    populate_a_delta = delta(before_a, after_a)
    populate_b_delta = delta(before_b_populate, after_b_populate)
    b_hit_delta = delta(before_b_hit, after_b_hit)
    a_after_eviction_delta = delta(before_a_after_eviction, after_a_after_eviction)

    populate_b_hits = populate_b_delta["hits"]
    b_hit_hits = b_hit_delta["hits"]
    a_after_eviction_hits = a_after_eviction_delta["hits"]
    ok = (
        populate_b_hits == expected_gpu_only_hit_tokens
        and b_hit_hits is not None
        and b_hit_hits >= expected_cpu_hit_tokens
        and a_after_eviction_hits == expected_gpu_only_hit_tokens
    )

    summary = {
        "ok": ok,
        "model": args.model,
        "split_tokens": args.split_tokens,
        "cached_suffix_tokens": args.cached_suffix_tokens,
        "first_decode_tokens": args.first_decode_tokens,
        "cpu_pool_bytes": args.cpu_pool_bytes,
        "expected_gpu_only_hit_tokens": expected_gpu_only_hit_tokens,
        "expected_cpu_hit_tokens": expected_cpu_hit_tokens,
        "prefix_metric_totals": {
            "before_a": before_a,
            "after_a": after_a,
            "after_b_populate": after_b_populate,
            "after_b_hit": after_b_hit,
            "after_a_after_eviction": after_a_after_eviction,
        },
        "deltas": {
            "populate_a": populate_a_delta,
            "populate_b": populate_b_delta,
            "b_hit": b_hit_delta,
            "a_after_eviction": a_after_eviction_delta,
        },
        "last_scheduler_stats": last_scheduler_stats_dict(llm),
        "generated_a_len": len(generated_a),
        "b_hit_output": output_b_hit,
        "a_after_eviction_output": output_a_after_eviction,
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
            "CPU pool pressure expectations failed: "
            f"populate_b_hits={populate_b_hits}, "
            f"b_hit_hits={b_hit_hits}, "
            f"a_after_eviction_hits={a_after_eviction_hits}"
        )


if __name__ == "__main__":
    os.environ["PYTHONPATH"] = (
        str(Path(__file__).resolve().parent)
        + os.pathsep
        + os.environ.get("PYTHONPATH", "")
    )
    main()
