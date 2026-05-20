#!/usr/bin/env python3
"""Measure Phase 2 merge cost with GPU vs UVA suffix artifacts.

This isolates the online-softmax merge after CPU suffix attention. It does not
replace E2E throughput runs; it explains whether the merge gap is caused by the
large BF16 suffix output artifact or the smaller suffix LSE artifact.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states


def _time_merge(
    *,
    batch: int,
    iters: int,
    warmup: int,
    suffix_out_dev: str,
    suffix_lse_dev: str,
) -> float:
    torch.manual_seed(0)
    heads = 28
    head_dim = 128
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
    suffix_out_gpu = suffix_out_cpu.cuda(non_blocking=True)
    suffix_lse_gpu = suffix_lse_cpu.cuda(non_blocking=True)
    torch.cuda.synchronize()

    suffix_out = (
        get_accelerator_view_from_cpu_tensor(suffix_out_cpu)
        if suffix_out_dev == "uva"
        else suffix_out_gpu
    )
    suffix_lse = (
        get_accelerator_view_from_cpu_tensor(suffix_lse_cpu)
        if suffix_lse_dev == "uva"
        else suffix_lse_gpu
    )
    output = torch.empty_like(prefix_out)

    for _ in range(warmup):
        merge_attn_states(output, prefix_out, prefix_lse, suffix_out, suffix_lse)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            merge_attn_states(output, prefix_out, prefix_lse, suffix_out, suffix_lse)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iters)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="64,128,256,512")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    cases = [
        ("gpu_out+gpu_lse", "gpu", "gpu"),
        ("uva_out+gpu_lse", "uva", "gpu"),
        ("gpu_out+uva_lse", "gpu", "uva"),
        ("uva_out+uva_lse", "uva", "uva"),
    ]
    print("batch,case,median_ms")
    for batch in [int(item) for item in args.batches.split(",") if item]:
        for label, out_dev, lse_dev in cases:
            ms = _time_merge(
                batch=batch,
                iters=args.iters,
                warmup=args.warmup,
                suffix_out_dev=out_dev,
                suffix_lse_dev=lse_dev,
            )
            print(f"{batch},{label},{ms:.6f}", flush=True)


if __name__ == "__main__":
    main()
