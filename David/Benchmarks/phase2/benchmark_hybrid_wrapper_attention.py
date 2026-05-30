#!/usr/bin/env python3
"""Benchmark the exact Phase 2 hybrid attention wrapper overhead.

This complements benchmark_flash_attn_lse.py. It compares raw paged
FlashAttention against the Python/tensor wrapper path used by the active hybrid
split diagnostics, including a coalesced-prefix + no-op suffix upper bound.

Run from /TTC/FastTTS-thesis so the editable thesis vLLM install is resolved.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections.abc import Callable

import torch

from vllm.v1.attention.backends.cots_hybrid_attention import (
    CotsHybridDecodeMetadata,
    cots_hybrid_decode_attention,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _make_block_table(batch: int, max_blocks: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch * max_blocks, dtype=torch.int32, device=device).reshape(
        batch, max_blocks
    )


def _stats(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return (
        sum(ordered) / len(ordered),
        statistics.median(ordered),
        ordered[int(0.9 * (len(ordered) - 1))],
    )


class _NoOpSuffixRunner:
    kind = "benchmark_noop"

    def set_runtime_counts(self, num_tokens: int, scatter_count: int) -> None:
        del num_tokens, scatter_count

    def get_counters(self) -> dict[str, int]:
        return {}

    def reset_counters(self) -> None:
        pass

    def run_gqa_bf16_suffix_attention(
        self,
        *,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
        output: torch.Tensor,
        output_lse: torch.Tensor,
        cuda_anchor: torch.Tensor | None = None,
        task_id: int = 0,
        scatter_block_ids: torch.Tensor | None = None,
        scatter_block_offsets: torch.Tensor | None = None,
        scatter_key_cpu: torch.Tensor | None = None,
        scatter_value_cpu: torch.Tensor | None = None,
        scatter_from_qkv: bool = False,
        scatter_from_separate_kv: bool = False,
        snapshot_inputs: bool = True,
        submit_stream: torch.cuda.Stream | None = None,
        submit_ready_event: torch.cuda.Event | None = None,
        submit_done_event: torch.cuda.Event | None = None,
        sync_after_submit: bool = True,
    ) -> None:
        del query, key_cache, value_cache, block_table, seq_lens, scale
        del cuda_anchor, task_id, scatter_block_ids, scatter_block_offsets
        del scatter_key_cpu, scatter_value_cpu, scatter_from_qkv
        del scatter_from_separate_kv, snapshot_inputs, submit_stream
        del submit_ready_event, submit_done_event, sync_after_submit
        output.zero_()
        output_lse.fill_(float("-inf"))

    def sync_gqa_bf16_suffix_attention(
        self, *, cuda_anchor: torch.Tensor, task_id: int = 0
    ) -> None:
        del cuda_anchor, task_id


_NOOP_SUFFIX_RUNNER = _NoOpSuffixRunner()


def _measure(
    fn: Callable[[], object], *, warmup: int, iters: int
) -> tuple[float, float, float, float, float, float]:
    fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    cuda_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000.0)
        cuda_ms.append(start.elapsed_time(end))
    cuda_avg, cuda_median, cuda_p90 = _stats(cuda_ms)
    wall_avg, wall_median, wall_p90 = _stats(wall_ms)
    return cuda_avg, cuda_median, cuda_p90, wall_avg, wall_median, wall_p90


def _make_hybrid_metadata(
    *,
    suffix_rows: int,
    split_blocks: int,
    suffix_blocks: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    source_indices: torch.Tensor | None,
    device: torch.device,
) -> CotsHybridDecodeMetadata:
    cpu_blocks = max(1, suffix_rows * max(1, suffix_blocks))
    block_cols = max(1, suffix_blocks)
    return CotsHybridDecodeMetadata(
        cpu_key_cache=torch.empty(
            cpu_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device="cpu"
        ),
        cpu_value_cache=torch.empty(
            cpu_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device="cpu"
        ),
        cpu_block_table=torch.zeros(suffix_rows, block_cols, dtype=torch.int32),
        cpu_seq_lens=torch.ones(suffix_rows, dtype=torch.int32),
        split_blocks=split_blocks,
        suffix_out_cpu=torch.empty(
            suffix_rows, num_q_heads, head_dim, dtype=dtype, device="cpu", pin_memory=True
        ),
        suffix_lse_cpu=torch.empty(
            num_q_heads, suffix_rows, dtype=torch.float32, device="cpu", pin_memory=True
        ),
        suffix_attention_runner=_NOOP_SUFFIX_RUNNER,
        req_indices_gpu=(
            source_indices.to(device=device, non_blocking=True)
            if source_indices is not None
            else None
        ),
        scatter_source_indices=source_indices,
    )


def _run_one(
    *,
    batch: int,
    split_len: int,
    seq_len: int,
    suffix_fraction: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    fa_version: int | None,
    warmup: int,
    iters: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    if seq_len % block_size or split_len % block_size:
        raise ValueError("seq_len and split_len must be block aligned")
    if split_len > seq_len:
        raise ValueError("split_len must be <= seq_len")
    split_blocks = split_len // block_size
    max_blocks = seq_len // block_size
    suffix_blocks = max(1, (seq_len - split_len + block_size - 1) // block_size)
    suffix_rows = max(1, min(batch, round(batch * suffix_fraction)))

    q = torch.randn(batch, num_q_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        batch * max_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    output = torch.empty_like(q)
    block_table = _make_block_table(batch, max_blocks, device)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    full_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    prefix_lens = torch.full((batch,), split_len, dtype=torch.int32, device=device)
    scale = head_dim**-0.5
    descale_shape = (batch, num_kv_heads)
    q_descale = torch.ones(descale_shape, dtype=torch.float32, device=device)
    k_descale = torch.ones_like(q_descale)
    v_descale = torch.ones_like(q_descale)
    source_indices = torch.arange(suffix_rows, dtype=torch.long)

    def raw_full() -> object:
        return flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            out=output,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=full_lens,
            max_seqlen_k=seq_len,
            softmax_scale=scale,
            causal=True,
            alibi_slopes=None,
            window_size=None,
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=None,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=0,
        )

    def raw_prefix_lse() -> object:
        return flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            out=output,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=prefix_lens,
            max_seqlen_k=split_len,
            softmax_scale=scale,
            causal=False,
            alibi_slopes=None,
            window_size=None,
            block_table=block_table[:, :split_blocks],
            softcap=0.0,
            scheduler_metadata=None,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=True,
            num_splits=1,
        )

    hybrid_all = _make_hybrid_metadata(
        suffix_rows=batch,
        split_blocks=split_blocks,
        suffix_blocks=suffix_blocks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        source_indices=None,
        device=device,
    )
    hybrid_partial = _make_hybrid_metadata(
        suffix_rows=suffix_rows,
        split_blocks=split_blocks,
        suffix_blocks=suffix_blocks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        source_indices=source_indices,
        device=device,
    )

    def hybrid_direct_noop_suffix() -> object:
        return cots_hybrid_decode_attention(
            output=output,
            query=q,
            gpu_key_cache=k_cache,
            gpu_value_cache=v_cache,
            gpu_block_table=block_table,
            hybrid_metadata=hybrid_all,
            softmax_scale=scale,
            fa_version=fa_version,
            q_descale=q_descale[:1],
            k_descale=k_descale[:1],
            v_descale=v_descale[:1],
        )

    def coalesced_noop_suffix() -> object:
        prefix_out, prefix_lse = raw_prefix_lse()
        return cots_hybrid_decode_attention(
            output=output,
            query=q,
            gpu_key_cache=k_cache,
            gpu_value_cache=v_cache,
            gpu_block_table=block_table,
            hybrid_metadata=hybrid_partial,
            softmax_scale=scale,
            fa_version=fa_version,
            q_descale=q_descale[:1],
            k_descale=k_descale[:1],
            v_descale=v_descale[:1],
            precomputed_prefix_out=prefix_out,
            precomputed_prefix_lse=prefix_lse,
        )

    cases: list[tuple[str, Callable[[], object]]] = [
        ("raw_full_fa", raw_full),
        ("raw_prefix_lse_fa", raw_prefix_lse),
        ("hybrid_direct_noop_suffix", hybrid_direct_noop_suffix),
        ("coalesced_noop_suffix", coalesced_noop_suffix),
    ]

    rows: list[dict[str, float | int | str]] = []
    for name, fn in cases:
        cuda_avg, cuda_median, cuda_p90, wall_avg, wall_median, wall_p90 = _measure(
            fn, warmup=warmup, iters=iters
        )
        rows.append(
            {
                "case": name,
                "batch": batch,
                "suffix_rows": suffix_rows,
                "suffix_fraction": suffix_fraction,
                "seq_len": seq_len,
                "split_len": split_len,
                "effective_capacity": seq_len / split_len,
                "cuda_avg_ms": cuda_avg,
                "cuda_median_ms": cuda_median,
                "cuda_p90_ms": cuda_p90,
                "wall_avg_ms": wall_avg,
                "wall_median_ms": wall_median,
                "wall_p90_ms": wall_p90,
                "rows_per_cuda_ms": batch / cuda_avg,
                "rows_per_wall_ms": batch / wall_avg,
                "dtype": str(dtype).replace("torch.", ""),
                "fa_version": fa_version if fa_version is not None else "none",
            }
        )
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="32,64")
    parser.add_argument("--splits", default="752,736,704,672,608")
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--suffix-fraction", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_q_heads % args.num_kv_heads:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if not (0.0 < args.suffix_fraction <= 1.0):
        raise ValueError("suffix_fraction must be in (0, 1]")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device("cuda")
    fa_version = get_flash_attn_version(head_size=args.head_dim)

    fieldnames = [
        "case",
        "batch",
        "suffix_rows",
        "suffix_fraction",
        "seq_len",
        "split_len",
        "effective_capacity",
        "cuda_avg_ms",
        "cuda_median_ms",
        "cuda_p90_ms",
        "wall_avg_ms",
        "wall_median_ms",
        "wall_p90_ms",
        "rows_per_cuda_ms",
        "rows_per_wall_ms",
        "dtype",
        "fa_version",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for batch in _parse_ints(args.batches):
        for split_len in _parse_ints(args.splits):
            rows = _run_one(
                batch=batch,
                split_len=split_len,
                seq_len=args.seq_len,
                suffix_fraction=args.suffix_fraction,
                num_q_heads=args.num_q_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                dtype=dtype,
                fa_version=fa_version,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            for row in rows:
                writer.writerow(row)
                sys.stdout.flush()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
