#!/usr/bin/env python3
"""Focused COTS CPU suffix attention microbenchmark.

This is intentionally local-thesis tooling, not a vLLM benchmark entrypoint.
It isolates the Phase 2 CPU suffix attention op and can compare pageable vs
pinned host tensors, matching the hybrid sidecar's allocation choices.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import statistics
import subprocess
import sys
import time


def _madvise_hugepage(tensor: "torch.Tensor") -> None:
    # Linux THP is configured as "madvise" on the thesis machine. Applying the
    # advice before first touch lets the benchmark test whether large CPU KV
    # regions are TLB-limited without changing the COTS op itself.
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    madv_hugepage = 14
    page_size = os.sysconf("SC_PAGE_SIZE")
    addr = tensor.data_ptr()
    start = addr & ~(page_size - 1)
    end = (addr + tensor.nbytes + page_size - 1) & ~(page_size - 1)
    ret = libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start),
                       ctypes.c_int(madv_hugepage))
    if ret != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _worker(args: argparse.Namespace) -> None:
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import torch

    from vllm._custom_ops import cots_gqa_bf16_suffix_attention

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.manual_seed(2031)
    block_size = args.block_size
    if args.model_shape == "qwen2.5-7b":
        num_q_heads, num_kv_heads, head_dim = 28, 4, 128
    elif args.model_shape == "llama3-8b":
        num_q_heads, num_kv_heads, head_dim = 32, 8, 128
    else:
        raise SystemExit(f"unknown model shape: {args.model_shape}")
    max_blocks = math.ceil(args.seq_len / block_size)
    num_blocks = args.batch * max_blocks

    query = torch.randn(args.batch, num_q_heads, head_dim, dtype=torch.bfloat16)
    if args.pin_inputs:
        query = query.pin_memory()
    key_caches = []
    value_caches = []
    for _ in range(args.layers):
        key_cache = torch.empty(
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
            dtype=torch.bfloat16,
            pin_memory=args.pin_inputs,
        )
        if args.madvise_hugepage:
            _madvise_hugepage(key_cache)
        key_cache.normal_()
        value_cache = torch.empty_like(key_cache)
        if args.madvise_hugepage:
            _madvise_hugepage(value_cache)
        value_cache.normal_()
        key_caches.append(key_cache)
        value_caches.append(value_cache)
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, pin_memory=args.pin_inputs
    ).reshape(args.batch, max_blocks)
    seq_lens = torch.full(
        (args.batch,),
        args.seq_len,
        dtype=torch.int32,
        pin_memory=args.pin_inputs,
    )
    outputs = [
        torch.empty_like(query, pin_memory=args.pin_inputs)
        for _ in range(args.layers)
    ]
    output_lses = [
        torch.empty(num_q_heads, args.batch, dtype=torch.float32, pin_memory=args.pin_inputs)
        for _ in range(args.layers)
    ]
    scale = head_dim**-0.5

    def run_once() -> None:
        for layer_idx in range(args.layers):
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

    for _ in range(args.warmup):
        run_once()

    times_ms: list[float] = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        run_once()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    p90_ms = statistics.quantiles(times_ms, n=10)[8] if len(times_ms) >= 10 else max(
        times_ms
    )
    print(
        f"threads={args.threads},model_shape={args.model_shape},batch={args.batch},"
        f"seq={args.seq_len},layers={args.layers},pin_inputs={int(args.pin_inputs)},"
        f"madvise_hugepage={int(args.madvise_hugepage)},"
        f"mean_ms={mean_ms:.3f},"
        f"median_ms={median_ms:.3f},p90_ms={p90_ms:.3f},"
        f"min_ms={min(times_ms):.3f},max_ms={max(times_ms):.3f},"
        f"median_ms_per_layer={median_ms / args.layers:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=160)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--threads-list", default="4,8,12,16,20,24,28,32")
    parser.add_argument(
        "--model-shape", choices=["qwen2.5-7b", "llama3-8b"], default="qwen2.5-7b"
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument("--pin-inputs", action="store_true")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--madvise-hugepage", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        if args.threads is None:
            raise SystemExit("--worker requires --threads")
        _worker(args)
        return

    for threads in [int(x) for x in args.threads_list.split(",") if x]:
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--threads",
            str(threads),
            "--batch",
            str(args.batch),
            "--seq-len",
            str(args.seq_len),
            "--block-size",
            str(args.block_size),
            "--model-shape",
            args.model_shape,
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--layers",
            str(args.layers),
        ]
        if args.pin_inputs:
            cmd.append("--pin-inputs")
        if args.madvise_hugepage:
            cmd.append("--madvise-hugepage")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        print(subprocess.check_output(cmd, text=True, env=env).strip())


if __name__ == "__main__":
    main()
