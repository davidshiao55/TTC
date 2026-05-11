# SPDX-License-Identifier: Apache-2.0
"""§1c.20 — Stride-aware D2H copy in the captured-graph submit path.

Real model paths (e.g., Qwen2 hidden_states from a sliced/padded base)
can hand the COTS QKV / MLP operators a row-strided x_gpu where
`stride(0) > shape[1]` but `stride(1) == 1`. The C++ side's
`submit_on_stream` dispatches `cudaMemcpyAsync` (1D) for the contiguous
case and `cudaMemcpy2DAsync` for the row-strided case. Transposed /
exotic layouts (`stride(1) != 1`) are rejected with a clear message.

These tests pin that behavior at the unit level so a future refactor
that drops the 2D path (or that re-introduces `is_contiguous()` as a
hard requirement) is caught immediately, without needing the full
real-model smoke to fail.
"""

from __future__ import annotations

import pytest
import torch


def _new_runner_with_qkv_slab(
    num_tokens: int, in_dim: int, n_cpu: int
) -> tuple[object, torch.Tensor, torch.Tensor]:
    """Construct a NativeCotsRunner with one QKV slab pointed at a real
    pinned input + output buffer. Returns (runner, x_pin, y_pin)."""
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsRunner(dry_run=False)
    cots_ops.install_infer(
        r._runner_id,
        n_slabs=1,
        max_num_tokens=num_tokens,
    )
    x_pin = torch.empty(num_tokens, in_dim, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(num_tokens, n_cpu, dtype=torch.bfloat16, pin_memory=True)
    w_cpu = torch.empty(n_cpu, in_dim, dtype=torch.bfloat16)
    infer = cots_ops._lookup_infer(r._runner_id, "test")
    infer.populate_slab_qkv(
        task_id=0,
        n_threads=1,
        bucket_capacity_tokens=num_tokens,
        x_pinned_ptr=x_pin.data_ptr(),
        in_dim=in_dim,
        y_pinned_ptr=y_pin.data_ptr(),
        cpu_out_dim=n_cpu,
        w_cpu_ptr=w_cpu.data_ptr(),
        w_cpu_rows=n_cpu,
    )
    return r, x_pin, y_pin


def test_contiguous_x_gpu_d2h_one_dim_path() -> None:
    """The fast 1D `cudaMemcpyAsync` path: x_gpu is contiguous, so
    stride(0) == shape[1]. The pinned destination should receive a
    bit-exact copy of x_gpu's data."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    num_tokens, in_dim, n_cpu = 4, 8, 6
    r, x_pin, _ = _new_runner_with_qkv_slab(num_tokens, in_dim, n_cpu)
    try:
        x_gpu = torch.arange(
            num_tokens * in_dim, dtype=torch.bfloat16, device="cuda"
        ).reshape(num_tokens, in_dim)
        assert x_gpu.is_contiguous()
        assert x_gpu.stride(0) == in_dim
        torch.ops.vllm.cots_submit_gemm(x_gpu, r._runner_id, 0, num_tokens)
        torch.cuda.synchronize()
        assert torch.equal(x_pin, x_gpu.cpu())
    finally:
        r.close()


def test_row_strided_x_gpu_d2h_two_dim_path() -> None:
    """The §1c.20-load-bearing case: x_gpu is a `[:, :in_dim]` slice
    over a wider base, so `stride(0) > shape[1]` but `stride(1) == 1`.
    The C++ side dispatches `cudaMemcpy2DAsync` and the pinned
    destination receives ONLY the leading in_dim columns of each
    row — not the padding bytes."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    num_tokens, in_dim, n_cpu = 4, 8, 6
    r, x_pin, _ = _new_runner_with_qkv_slab(num_tokens, in_dim, n_cpu)
    try:
        # Build a wider base. Filling with a known pattern so we can
        # tell the leading slice from the padding.
        wide = 16
        base = torch.arange(
            num_tokens * wide, dtype=torch.bfloat16, device="cuda"
        ).reshape(num_tokens, wide)
        x_gpu = base[:, :in_dim]  # row-strided view
        assert not x_gpu.is_contiguous()
        assert x_gpu.stride(0) == wide
        assert x_gpu.stride(1) == 1
        torch.ops.vllm.cots_submit_gemm(x_gpu, r._runner_id, 0, num_tokens)
        torch.cuda.synchronize()
        assert torch.equal(x_pin, x_gpu.cpu())
        # Sanity: the trailing-column data from `base` is NOT in x_pin.
        last_col_of_base = base[:, in_dim:].cpu().flatten()
        x_pin_flat = x_pin.flatten()
        for v in last_col_of_base.tolist():
            assert float(v) not in x_pin_flat.tolist(), (
                f"x_pin unexpectedly contains base[:, in_dim:] value {v}; "
                f"the 2D copy walked too far"
            )
    finally:
        r.close()


def test_transposed_x_gpu_is_rejected() -> None:
    """Transposed layout (stride(1) != 1) is rejected. The C++ D2H
    path doesn't have a sensible plan for stride(1) != 1; the
    Python validation catches it before we get to C++."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    num_tokens, in_dim, n_cpu = 4, 8, 6
    r, _, _ = _new_runner_with_qkv_slab(num_tokens, in_dim, n_cpu)
    try:
        # `.t()` swaps strides — for a (in_dim, num_tokens) base
        # tensor, `.t()` yields a (num_tokens, in_dim) view with
        # stride(1) != 1.
        base = torch.empty(in_dim, num_tokens, dtype=torch.bfloat16, device="cuda")
        x_gpu = base.t()
        assert x_gpu.shape == (num_tokens, in_dim)
        assert x_gpu.stride(1) != 1
        with pytest.raises(AssertionError, match="stride\\(1\\)"):
            torch.ops.vllm.cots_submit_gemm(x_gpu, r._runner_id, 0, num_tokens)
    finally:
        r.close()


def test_d2h_handles_partial_num_tokens() -> None:
    """`num_tokens` smaller than the slab's max-tokens preallocation:
    only the leading num_tokens rows are written; the rest of the
    pinned buffer is untouched."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    max_tokens, in_dim, n_cpu = 8, 8, 6
    r, x_pin, _ = _new_runner_with_qkv_slab(max_tokens, in_dim, n_cpu)
    try:
        # Pre-fill with a sentinel.
        x_pin.fill_(-99.0)
        # Run with HALF the rows.
        n = 4
        x_gpu = torch.arange(
            n * in_dim, dtype=torch.bfloat16, device="cuda"
        ).reshape(n, in_dim)
        torch.ops.vllm.cots_submit_gemm(x_gpu, r._runner_id, 0, n)
        torch.cuda.synchronize()
        assert torch.equal(x_pin[:n], x_gpu.cpu())
        # Trailing rows still hold the sentinel.
        assert torch.all(x_pin[n:] == -99.0)
    finally:
        r.close()
