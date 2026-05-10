# SPDX-License-Identifier: Apache-2.0
"""§1c.29 commit 2 (review-fix-1) — worker exception + M3 must not deadlock.

The captured `m3_wait_kernel` spins on `done_slot` until it sees
`done >= req`. If the worker throws BEFORE reaching the
`*done_slot = seq` write, the wait kernel spins forever and the GPU
stream wedges — and Python-side `check_error()` never runs because
the next submit/sync is itself behind the wedged stream.

Commit 1 review-fix-1 mandates a try/finally publish: even on
exception, the worker writes `done_slot = seq` so the wait kernel
exits its spin loop. The next Python-side entry-point call surfaces
the worker's `what()` as a Python `RuntimeError`, but only after the
GPU stream is unblocked enough for that call to run at all.

This test is the canary: if the worker's finally-publish is removed
or the `seq != 0 && m3_installed` guard inverts, the stream
synchronize at the end will hang past the timeout — which is far
worse than the test failing fast.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

pytestmark = pytest.mark.needs_cuda


def _force_mlp_scratch_unavailable_error():
    """Same fixture as test_worker_exception_surfacing — install with
    `scratch_max_intermediate_per_half=0` so the worker's MLP
    `TORCH_CHECK(scratch_silu_up_.defined(), ...)` (vllm/csrc/cots/
    cots_cpu_infer.cpp) trips on submit. Pinned-host backing buffers
    must outlive the slab pointers, so we return them as a keepalive.
    """
    from vllm._cots_C import CotsCpuInfer

    ci = CotsCpuInfer()
    ci.install(n_slabs=1, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)

    in_dim = 4
    inter_per_half = 8
    out_dim = 2
    x_pin = torch.empty(2 * in_dim, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(2 * out_dim, dtype=torch.bfloat16, pin_memory=True)
    w_gate = torch.empty(inter_per_half * in_dim, dtype=torch.bfloat16, pin_memory=True)
    w_up = torch.empty(inter_per_half * in_dim, dtype=torch.bfloat16, pin_memory=True)
    w_down = torch.empty(out_dim * inter_per_half, dtype=torch.bfloat16, pin_memory=True)

    ci.populate_slab_mlp(
        task_id=0,
        n_threads=1,
        bucket_capacity_tokens=1,
        x_pinned_ptr=x_pin.data_ptr(),
        in_dim=in_dim,
        y_pinned_ptr=y_pin.data_ptr(),
        cpu_out_dim=out_dim,
        w_gate_ptr=w_gate.data_ptr(),
        w_gate_rows=inter_per_half,
        w_up_ptr=w_up.data_ptr(),
        w_up_rows=inter_per_half,
        w_down_ptr=w_down.data_ptr(),
        w_down_rows=out_dim,
        w_down_cols=inter_per_half,
        w_down_stride_row=inter_per_half,
        w_down_stride_col=1,
        intermediate_per_half=inter_per_half,
    )
    return ci, (x_pin, y_pin, w_gate, w_up, w_down)


def _bounded_stream_sync(stream: torch.cuda.Stream, timeout_s: float) -> bool:
    """Run `stream.synchronize()` on a watchdog thread; return True if
    it returned within timeout_s, False otherwise. We use a thread
    instead of CUDA timeouts because there is no native cudaStream
    sync timeout — torch.cuda.synchronize() blocks indefinitely. If
    the worker's finally-publish is missing, the wait kernel spins
    forever and this returns False; that's the failure mode the test
    is guarding against."""
    done = threading.Event()

    def waiter():
        try:
            stream.synchronize()
        finally:
            done.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    return done.wait(timeout_s)


def test_worker_throw_with_m3_does_not_hang_wait_kernel():
    """Force a worker exception while M3 is installed for the slab.
    Without the finally-publish in RunSlabOnWorker, the captured
    m3_wait_kernel would spin on `done < req` forever and stream
    sync would never return. With the publish, done_slot=seq is
    written from the worker's finally block, the wait kernel
    returns, stream sync returns, and the next entry-point call
    surfaces the error as a Python RuntimeError.
    """
    ci, _keepalive = _force_mlp_scratch_unavailable_error()
    ci.install_m3_for_task(0)
    # Drive everything on a non-default stream so we can sync only
    # the stream we care about and avoid waiting on whatever else
    # may be enqueued process-wide.
    s = torch.cuda.Stream()
    stream_ptr = s.cuda_stream

    with torch.cuda.stream(s):
        ci.submit_on_stream(
            task_id=0,
            num_tokens=2,
            cuda_stream=stream_ptr,
            x_gpu_ptr=0,
            x_cols=0,
            x_stride0=0,
            x_stride1=1,
        )
        ci.sync_or_wait_on_stream(task_id=0, cuda_stream=stream_ptr)

    # 5s is generous: the wait kernel either returns immediately
    # (done was published) or spins forever (no publish). There is
    # no in-between for this test shape.
    returned = _bounded_stream_sync(s, timeout_s=5.0)
    assert returned, (
        "stream.synchronize() did not return within 5s — the worker "
        "exception path likely skipped the done_slot=seq publish, so the "
        "captured m3_wait_kernel is spinning forever (GPU deadlock)."
    )

    # The error must have been recorded; a follow-up entry point
    # call surfaces it.
    assert ci.has_error()
    with pytest.raises(RuntimeError, match="scratch_silu_up_"):
        ci.check_error()


def test_done_slot_advanced_to_seq_after_worker_throw():
    """Sanity check that backs up the no-deadlock test: after the
    failing dispatch + sync, the host-mapped done_slot is at the seq
    that was written to req_slot — i.e., the finally-publish ran.
    Without the publish, done_slot would still be 0 from install."""
    ci, _keepalive = _force_mlp_scratch_unavailable_error()
    ci.install_m3_for_task(0)
    assert ci.m3_get_req_slot(0) == 0
    assert ci.m3_get_done_slot(0) == 0

    s = torch.cuda.Stream()
    stream_ptr = s.cuda_stream
    with torch.cuda.stream(s):
        ci.submit_on_stream(
            task_id=0,
            num_tokens=2,
            cuda_stream=stream_ptr,
            x_gpu_ptr=0,
            x_cols=0,
            x_stride0=0,
            x_stride1=1,
        )
        ci.sync_or_wait_on_stream(task_id=0, cuda_stream=stream_ptr)
    assert _bounded_stream_sync(s, timeout_s=5.0)

    # Tiny extra wait to let the worker's finally-publish complete
    # if it was scheduled but not yet observed (the wait kernel
    # already saw it, but the host-mapped slot read from CPU may
    # lag by a few hundred ns under heavy traffic).
    time.sleep(0.01)
    req = ci.m3_get_req_slot(0)
    done = ci.m3_get_done_slot(0)
    assert req == 1, (
        f"req_slot should have been bumped to 1 by the dispatch_cb "
        f"submit (got {req})"
    )
    assert done == req, (
        f"done_slot ({done}) should match req_slot ({req}) after the "
        f"worker exception's finally-publish; if done < req, the "
        f"finally-block was skipped on the throw path."
    )
