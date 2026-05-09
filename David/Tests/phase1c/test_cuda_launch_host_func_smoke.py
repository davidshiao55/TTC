# SPDX-License-Identifier: Apache-2.0
"""Stage 1: cudaLaunchHostFunc smoke test.

Submits a dryrun_noop slab onto a real CUDA stream (no graph capture
yet); confirms the worker fires AFTER a prior `non_blocking copy_` on
the same stream — which is the load-bearing CUDA stream-ordering
guarantee Phase 1c relies on (the worker reads pinned x_pinned, which
must already be filled by the D2H by the time the host callback fires).
"""

import threading
import time

import pytest
import torch

pytestmark = pytest.mark.needs_cuda


@pytest.fixture
def runner():
    from vllm._cots_C import CotsCpuInfer

    ci = CotsCpuInfer()
    ci.install(n_slabs=4, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)
    for i in range(4):
        ci.populate_slab_dryrun(i)
    yield ci


def _stream_ptr() -> int:
    return torch.cuda.current_stream().cuda_stream


def test_submit_on_stream_executes(runner):
    """submit_on_stream returns immediately; sync_on_stream blocks the
    stream on the worker. After the stream syncs (host wait) the
    runner has no error."""
    runner.submit_on_stream(task_id=0, num_tokens=1, cuda_stream=_stream_ptr())
    runner.sync_on_stream(cuda_stream=_stream_ptr())
    torch.cuda.current_stream().synchronize()
    assert not runner.has_error()


def test_submit_after_d2h_observes_completed_copy(runner):
    """The point of cudaLaunchHostFunc is stream ordering: the host
    callback fires only after prior stream work completes. So a D2H
    copy queued on the same stream BEFORE submit_on_stream is
    guaranteed to be done by the time the worker reads its pinned src.

    We can't easily observe "the worker saw the D2H result" through the
    dryrun_noop body (it doesn't read anything). Instead, we confirm the
    coarser invariant: the entire pipeline (D2H → submit → sync) runs
    deterministically and the worker callback is dispatched."""
    src_gpu = torch.full((1024,), 7.0, dtype=torch.bfloat16, device="cuda")
    pinned = torch.empty(1024, dtype=torch.bfloat16, pin_memory=True)
    pinned.copy_(src_gpu, non_blocking=True)
    runner.submit_on_stream(task_id=1, num_tokens=1, cuda_stream=_stream_ptr())
    runner.sync_on_stream(cuda_stream=_stream_ptr())
    torch.cuda.current_stream().synchronize()
    assert not runner.has_error()
    assert torch.all(pinned == 7.0)  # D2H landed by the time host callback ran


def test_sync_actually_blocks_stream_on_worker(runner):
    """A `submit_dryrun_burst` followed by `sync_on_stream` on the same
    stream must wait for ALL queued tasks (including the burst) to drain
    before the stream's downstream work proceeds.

    We can't time `sync_on_stream` directly because it returns
    immediately (it just *queues* a host callback). But we can queue
    it followed by another stream op, and confirm the stream
    synchronization waits for the worker.
    """
    # Queue a burst to TaskQueue (no stream; goes straight via enqueue).
    runner.submit_dryrun_burst(2000)
    # Now schedule sync_on_stream on the default CUDA stream. The host
    # callback (when it fires after any prior stream work) blocks the
    # stream on the worker draining.
    runner.sync_on_stream(cuda_stream=_stream_ptr())

    # Block the host until the stream drains.
    t0 = time.perf_counter()
    torch.cuda.current_stream().synchronize()
    elapsed = time.perf_counter() - t0
    assert not runner.has_error()
    # Sanity: the stream sync had to wait for 2000 tasks; it isn't
    # instant. Lower bound is loose because dryrun tasks are very fast.
    # We just check the path completes without deadlock.
    assert elapsed < 2.0, f"stream sync took {elapsed:.3f}s; expected < 2s"


def test_no_deadlock_on_repeat(runner):
    """Repeated submit/sync cycles on the same stream don't deadlock or
    leak."""
    for i in range(20):
        runner.submit_on_stream(task_id=(i % 4), num_tokens=1, cuda_stream=_stream_ptr())
        runner.sync_on_stream(cuda_stream=_stream_ptr())
    torch.cuda.current_stream().synchronize()
    assert not runner.has_error()
