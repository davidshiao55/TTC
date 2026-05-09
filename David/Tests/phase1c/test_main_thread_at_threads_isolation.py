# SPDX-License-Identifier: Apache-2.0
"""Stage 4 — main-thread `at::get_num_threads()` isolation.

The C++ worker dispatcher calls `at::set_num_threads(slab.n_threads)`
before each slab's GEMM. The risk-register flagged this (plan §risk
register #3): if `at::set_num_threads` is process-global rather than
thread-local, the worker's setting leaks into the main thread's CUDA
dispatch path and the bucket-aware policy actively HURTS the main
thread's launch latency.

This test confirms PyTorch's at-thread-pool semantics on this build —
specifically, that `at::set_num_threads` called from the worker thread
does NOT change `at::get_num_threads()` observed from the main thread.
If RED, switch the C++ callback to `omp_set_num_threads` inside a
`#pragma omp parallel num_threads(n)` region (per the plan's
contingency).
"""

from __future__ import annotations

import pytest
import torch

from vllm._cots_C import CotsCpuInfer

pytestmark = pytest.mark.needs_cuda


def _populate_one_qkv_slab(
    ci: CotsCpuInfer,
    *,
    n_threads: int,
    in_dim: int = 64,
    out_dim: int = 32,
) -> dict:
    x_pinned = torch.empty(1, in_dim, dtype=torch.bfloat16, pin_memory=True)
    y_pinned = torch.empty(1, out_dim, dtype=torch.bfloat16, pin_memory=True)
    w_cpu = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, pin_memory=True)
    ci.populate_slab_qkv(
        task_id=0,
        n_threads=n_threads,
        x_pinned_ptr=x_pinned.data_ptr(),
        in_dim=in_dim,
        y_pinned_ptr=y_pinned.data_ptr(),
        cpu_out_dim=out_dim,
        w_cpu_ptr=w_cpu.data_ptr(),
        w_cpu_rows=out_dim,
    )
    return {"x": x_pinned, "y": y_pinned, "w": w_cpu}


def test_worker_set_num_threads_does_not_leak_to_main_thread() -> None:
    """The load-bearing isolation check.

    Procedure:
      1. Snapshot main thread's `at::get_num_threads()` BEFORE any
         worker activity.
      2. Drive a slab requesting a DIFFERENT n_threads (so the
         cache-guarded set actually fires on the worker).
      3. Drain the stream so the worker has finished and stored its
         observation.
      4. Sample main thread's `at::get_num_threads()` again.

    Pass: post == pre (worker's setting was thread-local).
    Fail: post == n_threads_worker (worker leaked into main thread).
          → contingency: switch C++ callback to omp pragma form.
    """
    main_before = torch.get_num_threads()

    # Pick an n_threads that is GUARANTEED different from the main
    # thread's current setting so the worker's cache-guarded
    # at::set_num_threads actually trips.
    worker_n = 1 if main_before != 1 else 2
    assert worker_n != main_before

    ci = CotsCpuInfer()
    ci.install(
        n_slabs=1, scratch_max_tokens=1, scratch_max_intermediate_per_half=0
    )
    keepalive = _populate_one_qkv_slab(ci, n_threads=worker_n)

    stream = torch.cuda.current_stream().cuda_stream
    ci.submit_on_stream(task_id=0, num_tokens=1, cuda_stream=stream, x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1)
    ci.sync_on_stream(cuda_stream=stream)
    torch.cuda.current_stream().synchronize()
    assert not ci.has_error()

    # Sanity: the worker actually saw worker_n.
    assert ci.last_observed_num_threads() == worker_n, (
        f"Worker did not observe its requested n_threads={worker_n} "
        f"(saw {ci.last_observed_num_threads()}); the test cannot "
        f"distinguish 'no leak' from 'set never fired'."
    )

    main_after = torch.get_num_threads()
    assert main_after == main_before, (
        f"MAIN-THREAD AT THREADS LEAKED: main thread's "
        f"at::get_num_threads() was {main_before} before the worker's "
        f"at::set_num_threads({worker_n}), but {main_after} after. "
        f"This means PyTorch's at-thread-pool is process-global on "
        f"this build, NOT thread-local. The Stage 4 design assumed "
        f"isolation; the contingency per the plan §risk register #3 "
        f"is to switch the C++ callback to `omp_set_num_threads` "
        f"inside a `#pragma omp parallel num_threads(n)` region. "
        f"Halt Stage 4 and apply that contingency before proceeding."
    )
    del keepalive


def test_repeated_slab_dispatch_keeps_main_thread_isolated() -> None:
    """N back-to-back slab dispatches with different n_threads. Main
    thread's `at::get_num_threads` should be stable throughout."""
    main_before = torch.get_num_threads()

    n_threads_sequence = [1, 4, 2, 8, 1]
    if main_before in n_threads_sequence:
        # Make sure the sequence transitions away from main_before so
        # the cache-guarded set fires at least once.
        n_threads_sequence = [
            n if n != main_before else (n + 1) for n in n_threads_sequence
        ]

    ci = CotsCpuInfer()
    n_slabs = len(n_threads_sequence)
    ci.install(
        n_slabs=n_slabs,
        scratch_max_tokens=1,
        scratch_max_intermediate_per_half=0,
    )
    keepalives = []
    for i, n in enumerate(n_threads_sequence):
        # Re-use _populate but with arbitrary task_id.
        x_pinned = torch.empty(1, 64, dtype=torch.bfloat16, pin_memory=True)
        y_pinned = torch.empty(1, 32, dtype=torch.bfloat16, pin_memory=True)
        w_cpu = torch.randn(32, 64, dtype=torch.bfloat16, pin_memory=True)
        ci.populate_slab_qkv(
            task_id=i,
            n_threads=n,
            x_pinned_ptr=x_pinned.data_ptr(),
            in_dim=64,
            y_pinned_ptr=y_pinned.data_ptr(),
            cpu_out_dim=32,
            w_cpu_ptr=w_cpu.data_ptr(),
            w_cpu_rows=32,
        )
        keepalives.append((x_pinned, y_pinned, w_cpu))

    stream = torch.cuda.current_stream().cuda_stream
    for tid in range(n_slabs):
        ci.submit_on_stream(task_id=tid, num_tokens=1, cuda_stream=stream, x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1)
        ci.sync_on_stream(cuda_stream=stream)
    torch.cuda.current_stream().synchronize()
    assert not ci.has_error()

    main_after = torch.get_num_threads()
    assert main_after == main_before, (
        f"After {n_slabs} per-bucket dispatches with varying "
        f"n_threads={n_threads_sequence}, main thread's "
        f"at::get_num_threads() drifted from {main_before} to "
        f"{main_after}. Worker's set leaked across thread boundary."
    )
    del keepalives
