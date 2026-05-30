#!/usr/bin/env python3
"""Measure FlashAttention decode overhead from returning softmax LSE.

This isolates the Phase 2 active-split suspicion: hybrid prefix attention must
produce mergeable state (output + log-sum-exp), while normal GPU-only decode can
write only the final output. The benchmark uses the same paged KV layout as the
vLLM FlashAttention backend.
"""


from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass

import torch

from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    get_flash_attn_version,
)


@dataclass(frozen=True)
class Case:
    name: str
    kv_len: int
    max_kv_len: int
    causal: bool
    return_lse: bool
    use_out: bool
    num_splits: int


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _make_block_table(batch: int, max_blocks: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        batch * max_blocks,
        dtype=torch.int32,
        device=device,
    ).reshape(batch, max_blocks)


def _run_case(
    *,
    case: Case,
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    fa_version: int | None,
    device: torch.device,
) -> dict[str, float | int | str]:
    if case.max_kv_len % block_size != 0:
        raise ValueError(f"max_kv_len must be block aligned: {case.max_kv_len}")
    max_blocks = case.max_kv_len // block_size
    block_table = _make_block_table(batch, max_blocks, device)
    num_blocks = int(block_table.numel())

    q = torch.randn(batch, num_q_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)
    out = torch.empty_like(q)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), case.kv_len, dtype=torch.int32, device=device)
    scale = head_dim**-0.5
    descale_shape = (batch, num_kv_heads)
    q_descale = torch.ones(descale_shape, dtype=torch.float32, device=device)
    k_descale = torch.ones_like(q_descale)
    v_descale = torch.ones_like(q_descale)

    def call() -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        kwargs = dict(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out if case.use_out else None,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            seqused_k=seq_lens,
            max_seqlen_k=case.max_kv_len,
            softmax_scale=scale,
            causal=case.causal,
            alibi_slopes=None,
            window_size=None,
            block_table=block_table,
            softcap=0.0,
            scheduler_metadata=None,
            fa_version=fa_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=case.num_splits,
        )
        if case.return_lse:
            kwargs["return_softmax_lse"] = True
        return flash_attn_varlen_func(**kwargs)

    # Touch once before timing to catch unsupported argument combinations early.
    result = call()
    if case.return_lse:
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(f"{case.name} did not return (output, lse)")
        output, lse = result
        if output.shape != q.shape:
            raise RuntimeError(f"bad output shape: {tuple(output.shape)}")
        if lse.shape != (num_q_heads, batch):
            raise RuntimeError(f"bad LSE shape: {tuple(lse.shape)}")
    torch.cuda.synchronize()

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end))

    samples_ms.sort()
    avg_ms = sum(samples_ms) / len(samples_ms)
    median_ms = statistics.median(samples_ms)
    p90_ms = samples_ms[int(0.9 * (len(samples_ms) - 1))]
    return {
        "case": case.name,
        "batch": batch,
        "kv_len": case.kv_len,
        "max_kv_len": case.max_kv_len,
        "causal": int(case.causal),
        "return_lse": int(case.return_lse),
        "use_out": int(case.use_out),
        "num_splits": case.num_splits,
        "avg_ms": avg_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "rows_per_ms": batch / avg_ms,
        "dtype": str(dtype).replace("torch.", ""),
        "fa_version": fa_version if fa_version is not None else "none",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="64,128,192,256,512")
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--split-len", type=int, default=752)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_q_heads % args.num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if args.seq_len % args.block_size != 0 or args.split_len % args.block_size != 0:
        raise ValueError("seq_len and split_len must be block aligned")
    if args.split_len > args.seq_len:
        raise ValueError("split_len must be <= seq_len")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device("cuda")
    fa_version = get_flash_attn_version(head_size=args.head_dim)

    cases = [
        Case("full_out_only_s0", args.seq_len, args.seq_len, True, False, True, 0),
        Case("full_lse_out_s0", args.seq_len, args.seq_len, True, True, True, 0),
        Case("prefix_out_only_s0", args.split_len, args.split_len, False, False, True, 0),
        Case("prefix_lse_out_s0", args.split_len, args.split_len, False, True, True, 0),
        Case("prefix_lse_alloc_s0", args.split_len, args.split_len, False, True, False, 0),
        Case("prefix_out_only_s1", args.split_len, args.split_len, False, False, True, 1),
        Case("prefix_lse_out_s1", args.split_len, args.split_len, False, True, True, 1),
        Case("prefix_lse_alloc_s1", args.split_len, args.split_len, False, True, False, 1),
    ]

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "case",
            "batch",
            "kv_len",
            "max_kv_len",
            "causal",
            "return_lse",
            "use_out",
            "num_splits",
            "avg_ms",
            "median_ms",
            "p90_ms",
            "rows_per_ms",
            "dtype",
            "fa_version",
        ],
    )
    writer.writeheader()
    for batch in _parse_ints(args.batches):
        for case in cases:
            row = _run_case(
                case=case,
                batch=batch,
                num_q_heads=args.num_q_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                fa_version=fa_version,
                device=device,
            )
            writer.writerow(row)
            sys.stdout.flush()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
