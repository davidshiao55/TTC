#!/usr/bin/env python3
"""Benchmark COTS prepared suffix attention runner variants.

This isolates the Phase 2 graph-substrate boundary:

- direct: blocking C++ CPU suffix attention op
- direct_scatter: blocking C++ scatter + blocking suffix attention
- prepared: stream-ordered native runner, attention only
- prepared_scatter: stream-ordered native runner, QKV scatter + attention
- graph_prepared_scatter: CUDA graph replay of prepared QKV scatter + attention

The benchmark is intentionally small and local to the thesis tree. It measures
wall-clock time including CUDA stream synchronization for prepared modes, because
that is the cost the following GPU merge/kernel would observe.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time


def _worker(args: argparse.Namespace) -> None:
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import torch

    from vllm._custom_ops import (
        cots_gqa_bf16_scatter_suffix_kv,
        cots_gqa_bf16_suffix_attention,
    )
    from vllm.v1.attention.backends.cots_hybrid_attention import (
        CotsPreparedNativeSuffixAttentionRunner,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for prepared suffix runner modes")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.manual_seed(args.seed)
    block_size = args.block_size
    if args.model_shape == "qwen2.5-7b":
        num_q_heads, num_kv_heads, head_dim = 28, 4, 128
        if args.layers is None:
            args.layers = 28
    elif args.model_shape == "llama3-8b":
        num_q_heads, num_kv_heads, head_dim = 32, 8, 128
        if args.layers is None:
            args.layers = 32
    else:
        raise SystemExit(f"unknown model shape: {args.model_shape}")
    max_blocks = math.ceil(args.seq_len / block_size)
    num_blocks = args.batch * max_blocks
    scale = head_dim**-0.5
    pin = bool(args.pin_inputs)
    total_heads = num_q_heads + 2 * num_kv_heads

    qkv = torch.randn(
        args.batch, total_heads, head_dim, dtype=torch.bfloat16, pin_memory=pin
    )
    query = qkv[:, :num_q_heads, :]
    key_src = qkv[:, num_q_heads : num_q_heads + num_kv_heads, :]
    value_src = qkv[:, num_q_heads + num_kv_heads :, :]
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, pin_memory=pin
    ).reshape(args.batch, max_blocks)
    seq_lens = torch.full(
        (args.batch,), args.seq_len, dtype=torch.int32, pin_memory=pin
    )
    scatter_col = (args.seq_len - 1) // block_size
    scatter_offset = (args.seq_len - 1) % block_size
    scatter_block_ids = block_table[:, scatter_col].to(torch.long).contiguous()
    scatter_block_offsets = torch.full(
        (args.batch,), scatter_offset, dtype=torch.long, pin_memory=pin
    )
    anchor = torch.empty(1, device="cuda")

    key_caches = []
    value_caches = []
    outputs = []
    output_lses = []
    for _ in range(args.layers):
        key_cache = torch.randn(
            num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16, pin_memory=pin
        )
        value_cache = torch.randn_like(key_cache)
        key_caches.append(key_cache)
        value_caches.append(value_cache)
        outputs.append(torch.empty_like(query, pin_memory=pin))
        output_lses.append(
            torch.empty(num_q_heads, args.batch, dtype=torch.float32, pin_memory=pin)
        )

    def run_direct(scatter: bool) -> None:
        for layer_idx in range(args.layers):
            if scatter:
                cots_gqa_bf16_scatter_suffix_kv(
                    key_src,
                    value_src,
                    scatter_block_ids,
                    scatter_block_offsets,
                    key_caches[layer_idx],
                    value_caches[layer_idx],
                )
            cots_gqa_bf16_suffix_attention(
                query=query,
                key_cache=key_caches[layer_idx],
                value_cache=value_caches[layer_idx],
                block_table=block_table,
                seq_lens=seq_lens,
                scale=scale,
                output=outputs[layer_idx],
                output_lse=output_lses[layer_idx],
            )

    def make_prepared_runner() -> CotsPreparedNativeSuffixAttentionRunner:
        return CotsPreparedNativeSuffixAttentionRunner(num_tasks=args.layers)

    def run_prepared(
        runner: CotsPreparedNativeSuffixAttentionRunner, scatter: bool, sync: bool = True
    ) -> None:
        for layer_idx in range(args.layers):
            runner.run_gqa_bf16_suffix_attention(
                query=query,
                key_cache=key_caches[layer_idx],
                value_cache=value_caches[layer_idx],
                block_table=block_table,
                seq_lens=seq_lens,
                scale=scale,
                output=outputs[layer_idx],
                output_lse=output_lses[layer_idx],
                cuda_anchor=anchor,
                task_id=layer_idx,
                scatter_block_ids=scatter_block_ids if scatter else None,
                scatter_block_offsets=scatter_block_offsets if scatter else None,
                scatter_from_qkv=scatter,
                snapshot_inputs=not args.no_snapshot_inputs,
            )
        if sync:
            torch.cuda.synchronize()

    def measure(label: str, fn) -> tuple[float, float, float, float]:
        for _ in range(args.warmup):
            fn()
        times_ms: list[float] = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        median_ms = statistics.median(times_ms)
        mean_ms = statistics.mean(times_ms)
        p90_ms = statistics.quantiles(times_ms, n=10)[8] if len(times_ms) >= 10 else max(times_ms)
        print(
            "mode,model_shape,batch,seq_len,layers,threads,pin_inputs,"
            "num_q_heads,num_kv_heads,head_dim,mean_ms,median_ms,p90_ms,"
            "median_us_per_layer,out_tok_s"
        ) if not getattr(measure, "_printed_header", False) else None
        measure._printed_header = True
        print(
            f"{label},{args.model_shape},{args.batch},{args.seq_len},"
            f"{args.layers},{args.threads},{int(pin)},{num_q_heads},"
            f"{num_kv_heads},{head_dim},{mean_ms:.3f},{median_ms:.3f},"
            f"{p90_ms:.3f},{median_ms * 1000.0 / args.layers:.3f},"
            f"{args.batch * args.layers / (median_ms / 1000.0):.3f}"
        )
        return mean_ms, median_ms, p90_ms, min(times_ms)

    requested = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in requested:
        if mode == "direct":
            measure(mode, lambda: run_direct(False))
        elif mode == "direct_scatter":
            measure(mode, lambda: run_direct(True))
        elif mode == "prepared":
            runner = make_prepared_runner()
            try:
                measure(mode, lambda: run_prepared(runner, False))
            finally:
                runner.close()
        elif mode == "prepared_scatter":
            runner = make_prepared_runner()
            try:
                measure(mode, lambda: run_prepared(runner, True))
            finally:
                runner.close()
        elif mode == "graph_prepared_scatter":
            runner = make_prepared_runner()
            try:
                run_prepared(runner, True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run_prepared(runner, True, sync=False)
                torch.cuda.synchronize()
                measure(mode, lambda: (graph.replay(), torch.cuda.synchronize()))
            finally:
                runner.close()
        else:
            raise SystemExit(f"unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=160)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument(
        "--model-shape", choices=["qwen2.5-7b", "llama3-8b"], default="qwen2.5-7b"
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--pin-inputs", action="store_true")
    parser.add_argument("--no-snapshot-inputs", action="store_true")
    parser.add_argument(
        "--modes",
        default="direct,direct_scatter,prepared,prepared_scatter,graph_prepared_scatter",
    )
    parser.add_argument("--seed", type=int, default=2038)
    args = parser.parse_args()
    _worker(args)


if __name__ == "__main__":
    main()
