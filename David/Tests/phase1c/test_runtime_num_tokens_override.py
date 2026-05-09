# SPDX-License-Identifier: Apache-2.0
"""§1c.21 fix — `runtime_num_tokens` plumb-through.

Under vLLM full CUDA graph capture, `slab.num_tokens` is frozen at
the captured graph-bucket size (e.g., 256). Replays at B=1 decode
would otherwise run CPU GEMMs for the bucket size, costing ~17 ms
per GEMM instead of <1 ms (the §1c.21 regression).

Fix: `CotsCpuInfer.set_runtime_num_tokens(n)` overrides the
worker's row-count arithmetic. Set OUT OF GRAPH before each
captured replay; the worker reads it on the next host-callback
fire and uses it for at::from_blob shapes, GEMM input rows, and
scratch slicing. Bounded by `effective_n <= slab.num_tokens` so
the worker never reads past the slab's pinned buffer.

These tests pin the contract at the unit level so the smoke /
real-model anchor doesn't have to re-prove the worker behavior
each run.
"""

from __future__ import annotations

import pytest
import torch


def _new_runner_with_qkv_slab(
    bucket_size: int, in_dim: int, n_cpu: int
) -> tuple[object, torch.Tensor, torch.Tensor]:
    """Construct a NativeCotsRunner with one QKV slab. The slab's
    pinned buffers are sized for the full `bucket_size`; the worker
    can be told via runtime_num_tokens to only process the first N."""
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsRunner(dry_run=False)
    cots_ops.install_infer(
        r._runner_id,
        n_slabs=1,
        scratch_max_tokens=bucket_size,
        scratch_max_intermediate_per_half=0,
    )
    x_pin = torch.empty(bucket_size, in_dim, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(bucket_size, n_cpu, dtype=torch.bfloat16, pin_memory=True)
    w_cpu = torch.randn(n_cpu, in_dim, dtype=torch.bfloat16)
    infer = cots_ops._lookup_infer(r._runner_id, "test")
    infer.populate_slab_qkv(
        task_id=0,
        n_threads=1,
        x_pinned_ptr=x_pin.data_ptr(),
        in_dim=in_dim,
        y_pinned_ptr=y_pin.data_ptr(),
        cpu_out_dim=n_cpu,
        w_cpu_ptr=w_cpu.data_ptr(),
        w_cpu_rows=n_cpu,
    )
    return r, x_pin, y_pin


def test_set_runtime_num_tokens_smaller_than_bucket_processes_only_n_rows() -> None:
    """The reviewer's load-bearing test: set runtime=1 against a
    slab populated for bucket=256, run a real GEMM, and confirm
    only the first row of y_pin is touched (the rest stays at the
    pre-fill sentinel)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from vllm.model_executor.offloader import cots_ops

    bucket = 32
    in_dim = 16
    n_cpu = 8
    r, x_pin, y_pin = _new_runner_with_qkv_slab(bucket, in_dim, n_cpu)
    try:
        # Sentinel-fill y_pin so we can detect untouched rows.
        y_pin.fill_(-99.0)
        # Pre-fill x_pin's first row with known data; rows 1..bucket
        # are intentionally left as garbage so we can confirm the
        # worker doesn't read them.
        x_pin.zero_()
        x_pin[0] = torch.arange(in_dim, dtype=torch.bfloat16)
        # Stash slab.num_tokens at the FULL bucket via a no-op submit
        # (slab.num_tokens.store fires inside submit_on_stream). We
        # don't actually submit a GEMM yet — just need the slab cap
        # populated.
        infer = cots_ops._lookup_infer(r._runner_id, "test")
        # Force slab.num_tokens=bucket via a dryrun-style 0-ptr submit.
        stream = torch.cuda.current_stream().cuda_stream
        # First write slab.num_tokens=bucket. submit_on_stream needs
        # a real x_gpu_ptr to do a D2H, but with x_gpu_ptr=0 it skips.
        # We need slab.num_tokens.store to happen; that requires a
        # call. Use the runtime_num_tokens=0 path so the worker uses
        # slab capacity for any subsequent GEMM.
        infer.set_runtime_num_tokens(0)  # fall back to slab cap
        infer.submit_on_stream(
            task_id=0, num_tokens=bucket, x_gpu_ptr=0,
            x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
        )
        infer.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()
        # slab.num_tokens=bucket now. y_pin still sentinel
        # (dryrun_noop op, but our slab is QKV — no actually wait).
        # Actually populate_slab_qkv set op_kind=kQkv, so the worker
        # WILL run at::linear. Let me redo: clear y_pin again.
        y_pin.fill_(-99.0)

        # Now: set runtime=1 and re-submit. Worker should compute 1
        # row of GEMM and leave rows 1..bucket untouched.
        infer.set_runtime_num_tokens(1)
        infer.submit_on_stream(
            task_id=0, num_tokens=bucket, x_gpu_ptr=0,
            x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
        )
        infer.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()
        # y_pin[0] should be filled by the worker; y_pin[1:] should
        # still be the sentinel.
        assert not torch.all(y_pin[0] == -99.0), (
            "Row 0 not touched by worker — runtime override may be "
            "skipping the GEMM entirely."
        )
        assert torch.all(y_pin[1:] == -99.0), (
            f"Rows 1..{bucket} were modified — runtime_num_tokens=1 "
            f"should mean the worker processes only the first row. "
            f"Got non-sentinel values at rows: "
            f"{(y_pin[1:] != -99.0).any(dim=1).nonzero().flatten().tolist()}"
        )
    finally:
        r.close()


def test_set_runtime_num_tokens_zero_falls_back_to_slab_cap() -> None:
    """Sentinel: setting runtime=0 reverts to using slab.num_tokens.
    This is the default state at construction and also the explicit
    'clear override' value."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from vllm.model_executor.offloader import cots_ops

    bucket = 8
    r, x_pin, y_pin = _new_runner_with_qkv_slab(bucket, 8, 6)
    try:
        infer = cots_ops._lookup_infer(r._runner_id, "test")
        x_pin.zero_()
        y_pin.fill_(-99.0)
        # First a submit at full bucket to set slab.num_tokens.
        infer.set_runtime_num_tokens(0)
        stream = torch.cuda.current_stream().cuda_stream
        infer.submit_on_stream(
            task_id=0, num_tokens=bucket, x_gpu_ptr=0,
            x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
        )
        infer.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()
        # All rows of y_pin should be filled — fallback to slab cap.
        assert torch.all(y_pin != -99.0), (
            "runtime_num_tokens=0 should fall back to slab.num_tokens=bucket; "
            "all rows should be processed by the worker."
        )
    finally:
        r.close()


def test_set_runtime_num_tokens_above_slab_cap_hard_fails() -> None:
    """Reviewer's defensive test: runtime_num_tokens > slab.num_tokens
    must hard-fail. We never want the worker to read past the slab's
    pinned buffer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from vllm.model_executor.offloader import cots_ops

    bucket = 4
    r, _, _ = _new_runner_with_qkv_slab(bucket, 8, 6)
    try:
        infer = cots_ops._lookup_infer(r._runner_id, "test")
        # Set slab.num_tokens=bucket via submit.
        infer.set_runtime_num_tokens(0)
        stream = torch.cuda.current_stream().cuda_stream
        infer.submit_on_stream(
            task_id=0, num_tokens=bucket, x_gpu_ptr=0,
            x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
        )
        infer.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()

        # Now push runtime past the cap and submit again; worker
        # should fail the bound check.
        infer.set_runtime_num_tokens(bucket + 1)
        infer.submit_on_stream(
            task_id=0, num_tokens=bucket, x_gpu_ptr=0,
            x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
        )
        infer.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()
        # The worker error surfaces on the next Python-side call.
        with pytest.raises(RuntimeError, match="exceeds slab capacity|exceeds"):
            infer.set_runtime_num_tokens(0)
            # Force a check_error path (any populate / submit /
            # sync call surfaces it).
            infer.submit_on_stream(
                task_id=0, num_tokens=bucket, x_gpu_ptr=0,
                x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
            )
    finally:
        try:
            r.close()
        except Exception:
            pass


def test_set_runtime_num_tokens_negative_raises() -> None:
    """Defensive: passing a negative value at the Python boundary is
    rejected immediately."""
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsRunner(dry_run=False)
    try:
        cots_ops.install_infer(
            r._runner_id, n_slabs=0, scratch_max_tokens=0,
            scratch_max_intermediate_per_half=0,
        )
        infer = cots_ops._lookup_infer(r._runner_id, "test")
        with pytest.raises(RuntimeError, match="< 0"):
            infer.set_runtime_num_tokens(-1)
    finally:
        r.close()
