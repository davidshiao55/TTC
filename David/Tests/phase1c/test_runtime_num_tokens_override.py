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

import os

import pytest
import torch

# §1c.34 cleanup C: tests that read diagnostic counters
# (worker_clamp_override_count, d2h_replay_bucket_bytes, etc.) need
# VLLM_COTS_DIAG=1 set BEFORE process start because the env is read
# once at first call into the diag_enabled() lambda. Setting it
# mid-process via monkeypatch is a no-op against the cached read.
_DIAG_REQUIRED_SKIP_REASON = (
    "VLLM_COTS_DIAG=1 must be set before process start to populate the "
    "diagnostic counters this test reads (see §1c.34 cleanup C — "
    "production-default skips all observational counter increments). "
    "Re-run with VLLM_COTS_DIAG=1 pytest ..."
)


def _require_diag_env() -> None:
    if os.environ.get("VLLM_COTS_DIAG", "0") != "1":
        pytest.skip(_DIAG_REQUIRED_SKIP_REASON)


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
        max_num_tokens=bucket_size,
    )
    x_pin = torch.empty(bucket_size, in_dim, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(bucket_size, n_cpu, dtype=torch.bfloat16, pin_memory=True)
    w_cpu = torch.randn(n_cpu, in_dim, dtype=torch.bfloat16)
    infer = cots_ops._lookup_infer(r._runner_id, "test")
    infer.populate_slab_qkv(
        task_id=0,
        n_threads=1,
        bucket_capacity_tokens=bucket_size,
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


def test_set_runtime_num_tokens_above_slab_cap_clamps() -> None:
    """§1c.31 commit-3-real fix: runtime_num_tokens > slab.num_tokens
    is no longer a hard-fail; the worker treats the override as a
    CAP and clamps to slab_cap. A diagnostic counter
    (worker_clamp_override_count, diag-gated under VLLM_COTS_DIAG=1
    per §1c.34 cleanup C) records each clamp so the case is
    observable.

    Rationale: under eager mode, set_runtime_num_tokens applies
    globally per CotsCpuInfer to whatever slab fires next,
    regardless of which bucket sized that slab. Pre-fix behavior
    (TORCH_CHECK) hard-failed B=4 prefill at input_len=8 when a
    small-bucket slab fired with the global override set to 32
    (`runtime_num_tokens=25 exceeds slab capacity slab.num_tokens=8`).
    The slab's pinned buffer is sized for slab_cap so reading
    beyond it is UB; clamping is the safe interpretation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    _require_diag_env()
    from vllm.model_executor.offloader import cots_ops

    bucket = 4
    r, _, _ = _new_runner_with_qkv_slab(bucket, 8, 6)
    try:
        infer = cots_ops._lookup_infer(r._runner_id, "test")
        infer.reset_counters()
        infer.set_runtime_num_tokens(0)
        stream = torch.cuda.current_stream().cuda_stream
        infer.submit_on_stream(
            task_id=0, num_tokens=bucket, x_gpu_ptr=0,
            x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
        )
        infer.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()

        counters = dict(infer.get_counters())
        assert counters.get("worker_clamp_override_count", 0) == 0, (
            "baseline submit (no override) should not clamp"
        )

        # Override above slab capacity → clamp + counter increment.
        infer.set_runtime_num_tokens(bucket + 1)
        infer.submit_on_stream(
            task_id=0, num_tokens=bucket, x_gpu_ptr=0,
            x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
        )
        infer.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()

        counters = dict(infer.get_counters())
        assert counters.get("worker_clamp_override_count", 0) == 1, (
            f"clamp counter should have incremented; got {counters}"
        )
        # The worker did NOT set has_error_; subsequent calls should
        # succeed (vs the pre-§1c.31 hard-fail behavior).
        assert not infer.has_error(), (
            "worker should not record an error after a clamp; the "
            "override is a cap, not a row-count requirement"
        )
        infer.set_runtime_num_tokens(0)
        infer.submit_on_stream(
            task_id=0, num_tokens=bucket, x_gpu_ptr=0,
            x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
        )
        infer.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()
        # Reset to 0 → next submit doesn't clamp again.
        counters = dict(infer.get_counters())
        assert counters.get("worker_clamp_override_count", 0) == 1, (
            f"counter should stay at 1 (no new clamps); got {counters}"
        )
    finally:
        try:
            r.close()
        except Exception:
            pass


def test_clamp_b4_prefill_scenario_no_deadlock() -> None:
    """§1c.31 regression for the B=4 prefill scenario that
    motivated the clamp fix. Mirrors what the workload-grid bench
    saw: a small-bucket slab installed at capacity 8, then a
    global runtime_num_tokens of 25 (from a larger batch's prefill)
    submitted. Pre-fix: TORCH_CHECK raised, worker stopped, stream
    wedged at next sync. Post-fix: clamp + counter, worker
    completes normally, sync returns, downstream proceeds.

    Asserts the clamp counter increments — diag-gated, so this
    test requires VLLM_COTS_DIAG=1 at process start (§1c.34
    cleanup C)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    _require_diag_env()
    from vllm.model_executor.offloader import cots_ops

    bucket = 8  # matches the original error: slab.num_tokens=8
    r, _, _ = _new_runner_with_qkv_slab(bucket, 8, 6)
    try:
        infer = cots_ops._lookup_infer(r._runner_id, "test")
        infer.reset_counters()
        stream = torch.cuda.current_stream().cuda_stream
        # Submit at bucket=8 first to set slab.num_tokens.
        infer.set_runtime_num_tokens(0)
        infer.submit_on_stream(
            task_id=0, num_tokens=bucket, x_gpu_ptr=0,
            x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
        )
        infer.sync_on_stream(cuda_stream=stream)
        torch.cuda.current_stream().synchronize()
        # Now simulate the B=4 prefill: global override at 25.
        infer.set_runtime_num_tokens(25)
        for _ in range(5):  # multiple replays at the over-cap setting
            infer.submit_on_stream(
                task_id=0, num_tokens=bucket, x_gpu_ptr=0,
                x_cols=0, x_stride0=0, x_stride1=1, cuda_stream=stream
            )
            infer.sync_on_stream(cuda_stream=stream)
            torch.cuda.current_stream().synchronize()

        assert not infer.has_error(), (
            "stream should not have wedged on clamp scenario"
        )
        counters = dict(infer.get_counters())
        assert counters.get("worker_clamp_override_count", 0) == 5, (
            f"5 over-cap submits should have produced 5 clamps; "
            f"got {counters.get('worker_clamp_override_count', 0)}"
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
            r._runner_id, n_slabs=0, max_num_tokens=0,
        )
        infer = cots_ops._lookup_infer(r._runner_id, "test")
        with pytest.raises(RuntimeError, match="< 0"):
            infer.set_runtime_num_tokens(-1)
    finally:
        r.close()
