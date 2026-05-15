# SPDX-License-Identifier: Apache-2.0
"""Worker exception surfacing — regression for Stage 1 fix #2.

A C++ exception inside a worker task must:
  (a) be caught (TaskQueue::Worker decrements pending and notifies cv_
      so subsequent `sync()` doesn't deadlock — verified by the test
      reaching a normal stream-synchronize after the failing task);
  (b) be stored on the CotsCpuInfer (`has_error_` + `last_error_msg_`);
  (c) re-raise as a Python RuntimeError on the NEXT entry-point call
      (submit / sync / populate_slab) — the `check_error()` guard at
      the top of every entry point.

The Python runner naturally re-raises through `future.result()`; the
native runner needs the same surfacing so a silent worker death doesn't
hang the whole engine.
"""

import pytest
import torch

from vllm._cots_C import CotsCpuInfer

pytestmark = pytest.mark.needs_cuda


def _force_mlp_worker_error():
    """Install + populate an MLP slab with a deliberate K-dimension
    mismatch between the down-proj weight and the silu*up intermediate
    so the worker's bf16_gemm_transposed_at TORCH_CHECK trips. This is
    a deterministic, side-effect-free way to drive the worker into the
    catch handler.

    Pre-Stage-7-C this helper exploited a `scratch_silu_up_` undefined
    check (the worker used a per-instance scratch tensor); Stage 7-C
    removed the scratch usage and the silu*up result is allocated
    in-place per call. The new trip-wire is the shape-validation
    TORCH_CHECK inside the custom kernel.

    Returns ``(ci, keepalive)``. The caller MUST bind ``keepalive``
    locally so the pinned-host backing buffers outlive the C++ slab —
    `CotsCpuInfer` is a pybind class without a __dict__, so we can't
    smuggle the keepalive onto the instance.
    """
    ci = CotsCpuInfer()
    ci.install(n_slabs=1, max_num_tokens=0)

    in_dim = 4
    inter_per_half = 8
    out_dim = 2

    # Real pinned-host backing for the slab's pointer fields. Any deref
    # the worker does before it hits the TORCH_CHECK must land in valid
    # memory — we don't want a segfault confounding the test.
    x_pin = torch.empty(2 * in_dim, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(2 * out_dim, dtype=torch.bfloat16, pin_memory=True)
    w_gate = torch.empty(inter_per_half * in_dim, dtype=torch.bfloat16, pin_memory=True)
    w_up = torch.empty(inter_per_half * in_dim, dtype=torch.bfloat16, pin_memory=True)
    # Down-proj is column-strided over (out_dim, n_cpu_total) row-major
    # storage; for the test we use a contiguous (out_dim, inter_per_half)
    # buffer with stride_row=inter_per_half (no narrow offset).
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
    )
    keepalive = (x_pin, y_pin, w_gate, w_up, w_down)
    return ci, keepalive


def test_worker_throw_does_not_hang_sync():
    """Most important property: a thrown task must NOT leave
    `TaskQueue.sync()` waiting forever. We submit a guaranteed-failing
    task on the stream, schedule a sync host callback, then wait on
    the stream from the host. If the test completes within the timeout,
    the pending counter was decremented despite the throw.
    """
    ci, _keepalive = _force_mlp_worker_error()
    stream_ptr = torch.cuda.current_stream().cuda_stream

    ci.submit_on_stream(task_id=0, num_tokens=2, cuda_stream=stream_ptr, x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1)
    ci.sync_on_stream(cuda_stream=stream_ptr)
    # Drive the stream forward. The dispatch host callback fires →
    # worker runs → throws → catch sets has_error_. The sync host
    # callback fires next → blocks on task_queue_->sync(0); since
    # pending was decremented despite the throw, sync returns.
    torch.cuda.current_stream().synchronize()

    # Stream sync returned cleanly — invariant (a) holds.
    assert ci.has_error(), "worker error should have been recorded"


def test_check_error_raises_runtime_error_with_message():
    ci, _keepalive = _force_mlp_worker_error()
    stream_ptr = torch.cuda.current_stream().cuda_stream

    ci.submit_on_stream(task_id=0, num_tokens=2, cuda_stream=stream_ptr, x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1)
    ci.sync_on_stream(cuda_stream=stream_ptr)
    torch.cuda.current_stream().synchronize()
    assert ci.has_error()

    # Direct check_error() call surfaces the worker's std::runtime_error
    # as a Python RuntimeError. The message must carry the C++ what()
    # so callers can diagnose the failure.
    with pytest.raises(RuntimeError, match=r"\[cots worker\]"):
        ci.check_error()

    # check_error() consumes the error — has_error() now False, and a
    # second call doesn't re-raise.
    assert not ci.has_error()
    ci.check_error()  # no-op, no raise.


def test_next_submit_call_re_raises_after_worker_failure():
    """Surfacing path that mirrors Python runner's future.result(): the
    NEXT entry point call (after a worker failure) re-raises.
    """
    ci, _keepalive = _force_mlp_worker_error()
    stream_ptr = torch.cuda.current_stream().cuda_stream

    ci.submit_on_stream(task_id=0, num_tokens=2, cuda_stream=stream_ptr, x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1)
    ci.sync_on_stream(cuda_stream=stream_ptr)
    torch.cuda.current_stream().synchronize()

    # The next submit_on_stream call's check_error() surfaces.
    with pytest.raises(RuntimeError, match=r"\[cots worker\]"):
        ci.submit_on_stream(task_id=0, num_tokens=2, cuda_stream=stream_ptr, x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1)


def test_take_error_consumes_state():
    ci, _keepalive = _force_mlp_worker_error()
    stream_ptr = torch.cuda.current_stream().cuda_stream

    ci.submit_on_stream(task_id=0, num_tokens=2, cuda_stream=stream_ptr, x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1)
    ci.sync_on_stream(cuda_stream=stream_ptr)
    torch.cuda.current_stream().synchronize()
    assert ci.has_error()

    msg = ci.take_error()
    assert "[cots worker]" in msg
    assert not ci.has_error()
    # check_error() is now clean — the entry-point guard does not fire on
    # subsequent calls. We DON'T queue another submit_on_stream here:
    # leaving an un-fired CUDA host callback in flight while ci goes out
    # of scope would dereference a freed slab when the stream eventually
    # drains. The destructor's task_queue_->sync(0) only waits for
    # in-TaskQueue work, not for stream-pending host callbacks.
    ci.check_error()  # no-op, no raise — entry guard is clean.
