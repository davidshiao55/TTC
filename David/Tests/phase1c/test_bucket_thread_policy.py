# SPDX-License-Identifier: Apache-2.0
"""Stage 4 — bucket-aware thread policy gate.

Confirms that `slab.n_threads` is wired end-to-end:
  * `CotsOffloadConfig.cpu_num_threads_by_bucket` populates per-bucket
    n_threads on the slab specs (`_build_native_slab_specs`).
  * `NativeCotsRunner.install` populates each slab via
    `populate_slab_qkv` / `_mlp` with that n_threads.
  * The C++ worker dispatcher's cache-guarded
    `at::set_num_threads(slab.n_threads)` runs before the GEMM and
    `last_observed_num_threads_` mirrors the post-set value.
  * Each per-bucket setting takes effect when a slab keyed on that
    bucket is dispatched.

The C++ `last_observed_num_threads()` getter is the side-channel the
test reads to confirm the value the worker actually saw.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm._cots_C import CotsCpuInfer
from vllm.model_executor.offloader import cots

pytestmark = pytest.mark.needs_cuda


def _drive_one_qkv_slab(ci: CotsCpuInfer, num_tokens: int = 1) -> None:
    """Submit one QKV slab on the current CUDA stream and drain the
    stream so the worker has finished by the time the test reads
    `last_observed_num_threads()`."""
    stream = torch.cuda.current_stream().cuda_stream
    ci.submit_on_stream(task_id=0, num_tokens=num_tokens, cuda_stream=stream, x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1)
    ci.sync_on_stream(cuda_stream=stream)
    torch.cuda.current_stream().synchronize()


def _populate_qkv_slab(
    ci: CotsCpuInfer,
    *,
    task_id: int,
    n_threads: int,
    in_dim: int = 64,
    out_dim: int = 32,
) -> dict:
    """Populate one QKV slab. Returns refs to backing tensors so the
    caller can keep them alive."""
    x_pinned = torch.empty(1, in_dim, dtype=torch.bfloat16, pin_memory=True)
    y_pinned = torch.empty(1, out_dim, dtype=torch.bfloat16, pin_memory=True)
    w_cpu = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, pin_memory=True)
    ci.populate_slab_qkv(
        task_id=task_id,
        n_threads=n_threads,
        bucket_capacity_tokens=1,
        x_pinned_ptr=x_pinned.data_ptr(),
        in_dim=in_dim,
        y_pinned_ptr=y_pinned.data_ptr(),
        cpu_out_dim=out_dim,
        w_cpu_ptr=w_cpu.data_ptr(),
        w_cpu_rows=out_dim,
    )
    return {"x": x_pinned, "y": y_pinned, "w": w_cpu}


# --- Direct C++ pybind: per-slab n_threads observed by worker -------------


@pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
def test_worker_observes_slab_n_threads(n_threads: int) -> None:
    """One slab with n_threads=N — drive it, then read
    `last_observed_num_threads()`. Should equal N (the worker called
    at::set_num_threads(N) before the GEMM and the side-channel
    captured `at::get_num_threads()` post-set).
    """
    ci = CotsCpuInfer()
    ci.install(
        n_slabs=1, max_num_tokens=1)
    _keepalive = _populate_qkv_slab(ci, task_id=0, n_threads=n_threads)
    _drive_one_qkv_slab(ci)
    assert not ci.has_error()
    observed = ci.last_observed_num_threads()
    assert observed == n_threads, (
        f"Worker observed at::get_num_threads()={observed} but slab "
        f"requested n_threads={n_threads}"
    )
    del _keepalive


def test_per_bucket_n_threads_takes_effect() -> None:
    """Three slabs with n_threads ∈ {1, 4, 8} — drive each in turn,
    confirm the side-channel reflects the just-dispatched value. The
    cache-guarded `if (slab->n_threads != worker_current_n_threads_)`
    branch must trip on each transition.
    """
    ci = CotsCpuInfer()
    ci.install(
        n_slabs=3, max_num_tokens=1)
    keepalives = [
        _populate_qkv_slab(ci, task_id=i, n_threads=n)
        for i, n in enumerate([1, 4, 8])
    ]
    stream = torch.cuda.current_stream().cuda_stream
    for tid, expected in zip([0, 1, 2], [1, 4, 8]):
        ci.submit_on_stream(task_id=tid, num_tokens=1, cuda_stream=stream, x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1)
        ci.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()
        assert not ci.has_error()
        assert ci.last_observed_num_threads() == expected, (
            f"slab task_id={tid} requested n_threads={expected}, worker "
            f"observed {ci.last_observed_num_threads()}"
        )
    del keepalives


# --- Offloader-side: cpu_num_threads_by_bucket -> slab specs --------------


def test_cpu_num_threads_by_bucket_drives_slab_n_threads() -> None:
    """Smoke that the offloader's `_build_native_slab_specs` reads
    `config.cpu_num_threads_by_bucket` per-bucket. Doesn't need a real
    model — exercise the helper directly with a stubbed
    `_capture_buckets`."""
    from vllm.config.offload import CotsOffloadConfig

    cfg = CotsOffloadConfig(
        f_cpu_store=0.10,
        cpu_runner="native",
        cpu_num_threads=16,
        cpu_num_threads_by_bucket={1: 4, 4: 8, 16: 16},
    )
    off = cots.CotsOffloader(config=cfg)
    # Stub _capture_buckets so the resolver / validator have inputs
    # without going through wrap_modules.
    off._capture_buckets = [1, 4, 16]

    assert off._n_threads_for(1) == 4
    assert off._n_threads_for(4) == 8
    assert off._n_threads_for(16) == 16
    # Bucket not in the dict falls back to scalar (the Planner is
    # allowed to specify only profiled buckets).
    assert off._n_threads_for(64) == 16

    # Validation: a key not in _capture_buckets is a hard error.
    cfg_bad = CotsOffloadConfig(
        f_cpu_store=0.10,
        cpu_runner="native",
        cpu_num_threads=16,
        cpu_num_threads_by_bucket={1: 4, 99: 8},
    )
    off_bad = cots.CotsOffloader(config=cfg_bad)
    off_bad._capture_buckets = [1, 4, 16]
    with pytest.raises(ValueError, match="not in cudagraph_capture_sizes"):
        off_bad._validate_thread_policy()

    if off._runner is not None:
        off._runner.close()
    if off_bad._runner is not None:
        off_bad._runner.close()


def test_cpu_num_threads_by_bucket_default_falls_back_to_scalar() -> None:
    """Phase 1a/1b regression: when `cpu_num_threads_by_bucket` is None
    (default), every bucket resolves to the scalar `cpu_num_threads`."""
    from vllm.config.offload import CotsOffloadConfig

    cfg = CotsOffloadConfig(
        f_cpu_store=0.10,
        cpu_runner="native",
        cpu_num_threads=12,
        cpu_num_threads_by_bucket=None,
    )
    off = cots.CotsOffloader(config=cfg)
    off._capture_buckets = [1, 4, 16]
    for b in [1, 4, 16, 64]:
        assert off._n_threads_for(b) == 12
    if off._runner is not None:
        off._runner.close()


def test_cpu_num_threads_by_bucket_rejects_bad_value() -> None:
    """`n_threads < 1` is invalid; surfaces at validate time, not at
    submit (where the C++ side would accept 0 and confuse oneDNN)."""
    from vllm.config.offload import CotsOffloadConfig

    cfg = CotsOffloadConfig(
        f_cpu_store=0.10,
        cpu_runner="native",
        cpu_num_threads=8,
        cpu_num_threads_by_bucket={1: 0},
    )
    off = cots.CotsOffloader(config=cfg)
    off._capture_buckets = [1]
    with pytest.raises(ValueError, match="must be >= 1"):
        off._validate_thread_policy()
    if off._runner is not None:
        off._runner.close()


def test_native_full_graph_routing_guard_detects_nonuniform_geometry() -> None:
    """§1c.35: task-id dispatch is now OOG, but routing geometry is not.

    FULL graph + native can proceed only when Python-side COTS routing
    geometry is uniform across buckets. Non-uniform Planner routing must
    fail loudly until that geometry moves behind the same dispatch boundary.
    """
    from vllm.config.offload import CotsOffloadConfig

    cfg = CotsOffloadConfig(f_cpu_store=0.0)
    off = cots.CotsOffloader(config=cfg)
    off._capture_buckets = (1, 4)

    class Handle:
        n_cpu = 10
        n_prefetch_by_bucket = {1: 0, 4: 0}
        n_cpu_compute_by_bucket = {1: 10, 4: 10}

    h = Handle()
    off._handles = [h]  # type: ignore[list-item]
    assert off._native_routing_uniform_across_buckets()

    h.n_prefetch_by_bucket = {1: 0, 4: 2}
    h.n_cpu_compute_by_bucket = {1: 10, 4: 8}
    assert not off._native_routing_uniform_across_buckets()


def test_native_operator_bucket_requires_dispatch_state() -> None:
    """Native operators must not fall back to shape-derived buckets."""
    from vllm.config.offload import CotsOffloadConfig

    cfg = CotsOffloadConfig(f_cpu_store=0.0)
    off = cots.CotsOffloader(config=cfg)
    off._capture_buckets = (1, 4)
    runner = cots.NativeCotsRunner(dry_run=True)
    try:
        off._runner = runner
        with pytest.raises(RuntimeError, match="dispatch state was published"):
            off._operator_bucket(8192)

        off._current_bucket = 4
        assert off._operator_bucket(8192) == 4
    finally:
        runner.close()


# --- Worker affinity ------------------------------------------------------


def test_set_worker_affinity_zero_mask_is_noop() -> None:
    """`set_worker_affinity(0)` is documented as a no-op (no
    affinity change). Sanity that no error is set and the worker
    continues to run."""
    ci = CotsCpuInfer()
    ci.install(
        n_slabs=1, max_num_tokens=1)
    keepalive = _populate_qkv_slab(ci, task_id=0, n_threads=1)
    ci.set_worker_affinity(0)
    _drive_one_qkv_slab(ci)
    assert not ci.has_error()
    del keepalive


def test_set_worker_affinity_accepts_high_bit() -> None:
    """Stage 4 review fix: the affinity mask is `uint64_t` end-to-end so
    bit 63 is representable. Earlier `int64_t` signature would reject
    `1 << 63 = 2^63` over pybind (overflows signed conversion) and
    `int64_t{1} << 63` was undefined behavior in C++. Confirms the
    uint64 fix actually accepts the high bit without error.

    The mask we pass intersects with `sched_getaffinity` per the C++
    implementation; if the process can't actually run on cpu 63, the
    intersection is empty and the C++ side warns-and-skips (no error).
    What we're testing is the ABI boundary, not the kernel-level
    behavior.
    """
    ci = CotsCpuInfer()
    ci.install(
        n_slabs=1, max_num_tokens=1)
    keepalive = _populate_qkv_slab(ci, task_id=0, n_threads=1)
    # 1 << 63 would fail under the prior int64_t binding with TypeError
    # ("int too big to convert"). Now passes through as uint64.
    ci.set_worker_affinity(1 << 63)
    _drive_one_qkv_slab(ci)
    assert not ci.has_error()
    del keepalive


def test_set_worker_affinity_with_intersected_mask_succeeds() -> None:
    """A non-empty mask intersected with the process's
    `sched_getaffinity` should succeed (the intersection is non-empty
    by construction — we use the process's own first allowed cpu).
    The C++ side enqueues a task that sets `pthread_setaffinity_np`;
    after the next dispatch, the worker is pinned and the GEMM
    completes without error.
    """
    import os

    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        pytest.skip("process has no affinity mask")
    cpu_id = allowed[0]
    if cpu_id >= 64:
        pytest.skip("test bitmask is int64 — skip cpu_id >= 64")
    mask = 1 << cpu_id

    ci = CotsCpuInfer()
    ci.install(
        n_slabs=1, max_num_tokens=1)
    keepalive = _populate_qkv_slab(ci, task_id=0, n_threads=1)
    ci.set_worker_affinity(mask)
    # Drive a real task — implicitly drains the affinity-set
    # task that set_worker_affinity enqueued.
    _drive_one_qkv_slab(ci)
    assert not ci.has_error()
    del keepalive
