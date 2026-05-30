"""§1c.22 review-fix: prove `bucket_capacity_tokens` does not change
when the live-token cap or `slab.num_tokens` changes.

The TaskSlab has two distinct token-count fields:
  * `num_tokens` — mutable, written per submit_on_stream call (records
    capture-time/replay-time token count); also written when a captured
    custom-op replays via Python (PIECEWISE).
  * `bucket_capacity_tokens` — immutable, populated at install from the
    descriptor bucket. Replay byte counters MUST read this so attribution
    stays stable across mutations of the other state.

The test directly exercises CotsWeightTaskRunner (no Python runner involved):
  1. Populate slab with `bucket_capacity_tokens=N`.
  2. Drive the C++ `set_live_num_tokens` live-cap field with a
     different value → assert bucket_capacity_tokens unchanged.
  3. Drive `submit_on_stream` with a different num_tokens (mutates
     slab.num_tokens) → assert bucket_capacity_tokens unchanged.
"""

from __future__ import annotations

import pytest
import torch

torch_cuda_available = pytest.importorskip(
    "torch.cuda", reason="CUDA required"
)
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.fixture
def runner():
    from vllm._cots_C import CotsWeightTaskRunner

    ci = CotsWeightTaskRunner()
    ci.install(n_slabs=2, max_num_tokens=8)
    yield ci


def _populate_qkv(ci, task_id: int, bucket: int) -> dict:
    in_dim, out_dim = 64, 32
    x_pin = torch.empty(bucket, in_dim, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(bucket, out_dim, dtype=torch.bfloat16, pin_memory=True)
    w_cpu = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, pin_memory=True)
    ci.populate_slab_qkv(
        task_id=task_id,
        n_threads=1,
        bucket_capacity_tokens=bucket,
        x_pinned_ptr=x_pin.data_ptr(),
        in_dim=in_dim,
        y_pinned_ptr=y_pin.data_ptr(),
        cpu_out_dim=out_dim,
        w_cpu_ptr=w_cpu.data_ptr(),
        w_cpu_rows=out_dim,
    )
    return {"x_pin": x_pin, "y_pin": y_pin, "w_cpu": w_cpu}


def test_set_live_num_tokens_does_not_touch_bucket_capacity(runner):
    """The C++ live-token field is separate from slab capacity."""
    _populate_qkv(runner, task_id=0, bucket=8)
    assert runner.slab_bucket_capacity_tokens(0) == 8
    runner.set_live_num_tokens(1)
    assert runner.slab_bucket_capacity_tokens(0) == 8
    runner.set_live_num_tokens(4)
    assert runner.slab_bucket_capacity_tokens(0) == 8


def test_submit_on_stream_writes_num_tokens_but_not_bucket_capacity(runner):
    """submit_on_stream writes slab.num_tokens (the mutable field).
    bucket_capacity_tokens stays put."""
    keepalives = _populate_qkv(runner, task_id=0, bucket=8)
    assert runner.slab_bucket_capacity_tokens(0) == 8
    assert runner.slab_num_tokens(0) == 0  # not yet submitted

    stream = torch.cuda.current_stream().cuda_stream
    # x_gpu_ptr=0 with x_cols=0 means "no D2H copy" path; still bumps slab.num_tokens.
    runner.submit_on_stream(
        task_id=0, num_tokens=1, cuda_stream=stream,
        x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1,
    )
    runner.sync_on_stream(stream)
    torch.cuda.current_stream().synchronize()
    assert runner.slab_num_tokens(0) == 1
    assert runner.slab_bucket_capacity_tokens(0) == 8

    runner.submit_on_stream(
        task_id=0, num_tokens=3, cuda_stream=stream,
        x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1,
    )
    runner.sync_on_stream(stream)
    torch.cuda.current_stream().synchronize()
    assert runner.slab_num_tokens(0) == 3
    assert runner.slab_bucket_capacity_tokens(0) == 8
    del keepalives  # silence unused


def test_each_slab_keeps_its_own_bucket_capacity(runner):
    """Two slabs at different buckets — populating one doesn't disturb
    the other (catches an accidental cross-slab write to a shared field)."""
    _populate_qkv(runner, task_id=0, bucket=4)
    _populate_qkv(runner, task_id=1, bucket=16)
    assert runner.slab_bucket_capacity_tokens(0) == 4
    assert runner.slab_bucket_capacity_tokens(1) == 16
    runner.set_live_num_tokens(2)
    assert runner.slab_bucket_capacity_tokens(0) == 4
    assert runner.slab_bucket_capacity_tokens(1) == 16


def test_replay_bucket_counters_read_capacity_not_num_tokens():
    """The load-bearing assertion: prove `d2h_replay_bucket_bytes` and
    `uva_replay_bucket_bytes` are computed from `bucket_capacity_tokens`,
    not from the mutable `slab.num_tokens`. Without this test, the
    capacity field could be plumbed in but the counter could still
    accidentally read `slab.num_tokens` and we'd never know.

    Method:
      * Dryrun slab populated with bucket_capacity_tokens=64,
        in_dim=cpu_out_dim=32.
      * live-token cap = 1 → worker effective_n=1.
      * Submit on a CUDA stream with num_tokens=1 (so slab.num_tokens=1
        post-submit, distinct from bucket_capacity_tokens=64).
      * Sync, drain, dump counters.

    Expected:
      d2h_replay_bucket_bytes  = 64 × 32 × 2  = 4096
      uva_replay_bucket_bytes  = 64 × 32 × 2  = 4096
      worker_input_live_bytes  =  1 × 32 × 2  =   64
      worker_output_live_bytes =  1 × 32 × 2  =   64

    If the counters read `slab.num_tokens` instead, both replay-bucket
    counters would equal 64 (the live bytes), not 4096.

    Requires VLLM_COTS_DIAG=1 at process start — the
    d2h/uva_replay_bucket_bytes counters were diag-gated in
    §1c.34 cleanup C so the production-default hot path skips
    the atomic adds entirely.
    """
    import os

    if os.environ.get("VLLM_COTS_DIAG", "0") != "1":
        pytest.skip(
            "VLLM_COTS_DIAG=1 must be set before process start — "
            "d2h/uva_replay_bucket_bytes counters are diag-gated "
            "(§1c.34 cleanup C). Re-run with VLLM_COTS_DIAG=1 pytest ..."
        )
    from vllm._cots_C import CotsWeightTaskRunner

    BUCKET = 64
    IN_DIM = 32
    OUT_DIM = 32
    BF16 = 2
    LIVE = 1

    ci = CotsWeightTaskRunner()
    ci.install(n_slabs=1, max_num_tokens=BUCKET)

    # Pinned bufs sized for the bucket so the worker's at::from_blob
    # views have valid backing storage.
    x_pin = torch.empty(BUCKET, IN_DIM, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(BUCKET, OUT_DIM, dtype=torch.bfloat16, pin_memory=True)

    ci.populate_slab_dryrun(
        task_id=0,
        bucket_capacity_tokens=BUCKET,
        x_pinned_ptr=x_pin.data_ptr(),
        in_dim=IN_DIM,
        y_pinned_ptr=y_pin.data_ptr(),
        cpu_out_dim=OUT_DIM,
    )

    # Make slab.num_tokens diverge from bucket_capacity_tokens. The
    # counter MUST read capacity, not this.
    ci.reset_counters()
    ci.set_live_num_tokens(LIVE)
    stream = torch.cuda.current_stream().cuda_stream
    ci.submit_on_stream(
        task_id=0, num_tokens=LIVE, cuda_stream=stream,
        x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1,
    )
    ci.sync_on_stream(stream)
    torch.cuda.current_stream().synchronize()

    # Sanity: slab.num_tokens is now LIVE (1), bucket_capacity stays at 64.
    assert ci.slab_num_tokens(0) == LIVE
    assert ci.slab_bucket_capacity_tokens(0) == BUCKET

    counters = dict(ci.get_counters())
    bucket_in_bytes = BUCKET * IN_DIM * BF16
    bucket_out_bytes = BUCKET * OUT_DIM * BF16
    live_in_bytes = LIVE * IN_DIM * BF16
    live_out_bytes = LIVE * OUT_DIM * BF16

    assert counters["d2h_replay_bucket_bytes"] == bucket_in_bytes, (
        f"d2h_replay_bucket_bytes={counters['d2h_replay_bucket_bytes']}, "
        f"expected {bucket_in_bytes} (= bucket_capacity={BUCKET} × "
        f"in_dim={IN_DIM} × bf16); if counter reads slab.num_tokens it "
        f"would be {live_in_bytes}"
    )
    assert counters["uva_replay_bucket_bytes"] == bucket_out_bytes, (
        f"uva_replay_bucket_bytes={counters['uva_replay_bucket_bytes']}, "
        f"expected {bucket_out_bytes}; if counter reads slab.num_tokens "
        f"it would be {live_out_bytes}"
    )
    assert counters["worker_input_live_bytes"] == live_in_bytes
    assert counters["worker_output_live_bytes"] == live_out_bytes
