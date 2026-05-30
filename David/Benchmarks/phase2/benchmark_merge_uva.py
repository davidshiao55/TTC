#!/usr/bin/env python3
"""Measure Phase 2 merge cost with GPU vs UVA suffix artifacts.

This isolates the online-softmax merge after CPU suffix attention. It does not
replace E2E throughput runs; it explains whether the merge gap is caused by the
large BF16 suffix output artifact, the smaller suffix LSE artifact, or an
explicit H2D artifact copy.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states


def _median_cuda_ms(fn: Callable[[], None], *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def _time_case(
    *,
    batch: int,
    heads: int,
    head_dim: int,
    iters: int,
    warmup: int,
    case: str,
) -> float:
    torch.manual_seed(0)
    prefix_out = torch.randn(
        batch, heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    prefix_lse = torch.randn(heads, batch, device="cuda", dtype=torch.float32)
    suffix_out_cpu = torch.randn(
        batch,
        heads,
        head_dim,
        device="cpu",
        dtype=torch.bfloat16,
        pin_memory=True,
    )
    suffix_lse_cpu = torch.randn(
        heads, batch, device="cpu", dtype=torch.float32, pin_memory=True
    )
    suffix_out_gpu = torch.empty_like(prefix_out)
    suffix_lse_gpu = torch.empty_like(prefix_lse)
    suffix_out_gpu.copy_(suffix_out_cpu, non_blocking=True)
    suffix_lse_gpu.copy_(suffix_lse_cpu, non_blocking=True)
    torch.cuda.synchronize()

    suffix_out_uva = get_accelerator_view_from_cpu_tensor(suffix_out_cpu)
    suffix_lse_uva = get_accelerator_view_from_cpu_tensor(suffix_lse_cpu)
    output = torch.empty_like(prefix_out)

    def merge_gpu() -> None:
        merge_attn_states(output, prefix_out, prefix_lse, suffix_out_gpu, suffix_lse_gpu)

    def merge_uva_out_gpu_lse() -> None:
        merge_attn_states(output, prefix_out, prefix_lse, suffix_out_uva, suffix_lse_gpu)

    def merge_gpu_out_uva_lse() -> None:
        merge_attn_states(output, prefix_out, prefix_lse, suffix_out_gpu, suffix_lse_uva)

    def merge_uva() -> None:
        merge_attn_states(output, prefix_out, prefix_lse, suffix_out_uva, suffix_lse_uva)

    def copy_artifacts() -> None:
        suffix_out_gpu.copy_(suffix_out_cpu, non_blocking=True)
        suffix_lse_gpu.copy_(suffix_lse_cpu, non_blocking=True)

    def copy_artifacts_then_merge() -> None:
        suffix_out_gpu.copy_(suffix_out_cpu, non_blocking=True)
        suffix_lse_gpu.copy_(suffix_lse_cpu, non_blocking=True)
        merge_attn_states(output, prefix_out, prefix_lse, suffix_out_gpu, suffix_lse_gpu)

    cases: dict[str, Callable[[], None]] = {
        "gpu_out+gpu_lse_merge": merge_gpu,
        "uva_out+gpu_lse_merge": merge_uva_out_gpu_lse,
        "gpu_out+uva_lse_merge": merge_gpu_out_uva_lse,
        "uva_out+uva_lse_merge": merge_uva,
        "copy_out+copy_lse": copy_artifacts,
        "copy_out+copy_lse+gpu_merge": copy_artifacts_then_merge,
    }
    if case not in cases:
        raise ValueError(f"unknown case: {case}")
    return _median_cuda_ms(cases[case], iters=iters, warmup=warmup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="64,128,256,512")
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    cases = [
        "gpu_out+gpu_lse_merge",
        "uva_out+gpu_lse_merge",
        "gpu_out+uva_lse_merge",
        "uva_out+uva_lse_merge",
        "copy_out+copy_lse",
        "copy_out+copy_lse+gpu_merge",
    ]
    print("batch,heads,head_dim,artifact_mb,case,median_ms")
    for batch in [int(item) for item in args.batches.split(",") if item]:
        artifact_bytes = (
            batch * args.heads * args.head_dim * torch.bfloat16.itemsize
            + args.heads * batch * torch.float32.itemsize
        )
        artifact_mb = artifact_bytes / 1_000_000
        for case in cases:
            ms = _time_case(
                batch=batch,
                heads=args.heads,
                head_dim=args.head_dim,
                iters=args.iters,
                warmup=args.warmup,
                case=case,
            )
            print(
                f"{batch},{args.heads},{args.head_dim},{artifact_mb:.3f},"
                f"{case},{ms:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
