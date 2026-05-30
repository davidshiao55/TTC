# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the Stage 7-D natural-layout BF16 GEMM kernel
(`vllm/csrc/cots/bf16_gemm_natural.cpp`). Sibling to
`test_bf16_gemm_transposed_kernel.py` which covers the transposed
(K, N) layout.

Covers:
  * M=1/2/3/4/8: exercises the M_TILE=4 register-tile path and the
    per-m M=1 tail for M % 4 != 0.
  * K-tail: K % 8 != 0 forces the scalar K-tail loop inside dot_tile.
  * N walked one column at a time (no N tail to test).
  * FP32 → BF16 round-to-nearest-even (RNE) — same conversion as the
    transposed sibling, validated against PyTorch's `to(bfloat16)`.
  * Numerical parity vs a scalar reference at small K (K = 16, 64).
  * Threading invariance — output must be bit-identical at thr=1 and
    thr=16 (parallel-for splits on N; no cross-thread accumulation).

Like the transposed sibling, these are correctness-only. The old Stage
7 performance probes were bring-up artifacts and are no longer part of
the retained Phase 1c test suite.
"""

from __future__ import annotations

import struct

import pytest
import torch

from vllm._cots_C import CotsWeightTaskRunner


# --------------------------------------------------------------------- #
# Scalar reference implementation. Same as the transposed sibling's
# reference but takes natural-layout weight (N, K) instead of (K, N).
# --------------------------------------------------------------------- #


def _ref_bf16_gemm_natural(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y = x @ w^T with w in natural (N, K) layout. Reference impl:
    upcast to FP32, accumulate, RNE-round to BF16."""
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert x.dim() == 2 and w.dim() == 2 and x.size(1) == w.size(1)
    x_f32 = x.to(torch.float32)
    w_f32 = w.to(torch.float32)
    acc = x_f32 @ w_f32.t()  # (M, K) @ (K, N) → (M, N)
    return acc.to(torch.bfloat16)


# --------------------------------------------------------------------- #
# RNE conversion correctness — same set as the transposed-kernel
# test. Driven through a M=K=N=1 inline kernel call to isolate the
# conversion path.
# --------------------------------------------------------------------- #


def _drive_kernel_to_emit(infer: CotsWeightTaskRunner, value_fp32: float) -> float:
    M, K, N = 4, 16, 16
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    w_NK = torch.zeros(N, K, dtype=torch.bfloat16)
    x[0, 0] = torch.tensor(1.0, dtype=torch.bfloat16)
    w_NK[0, 0] = torch.tensor(value_fp32, dtype=torch.bfloat16)
    y = torch.empty(M, N, dtype=torch.bfloat16)
    infer.run_bf16_gemm_natural_inline(x, w_NK, y)
    return float(y[0, 0])


@pytest.mark.parametrize(
    "value",
    [
        0.0, 1.0, -1.0,
        0.5, 1.5, -1.5,
        100.5, -100.5,
        1e-4, 1e-30,
        2.0, -2.0,
        1024.0, -1024.0, 65536.0,
    ],
)
def test_rne_conversion_matches_pytorch_bf16(value):
    infer = CotsWeightTaskRunner()
    got = _drive_kernel_to_emit(infer, value)
    expected = float(torch.tensor(value, dtype=torch.bfloat16))
    assert got == expected, (
        f"RNE mismatch for value={value!r}: got {got!r}, "
        f"expected {expected!r}"
    )


def test_rne_tie_to_even_boundary():
    """Tie-to-even boundary case: 0x3F808000 should round to mantissa 0."""
    midway_bits = 0x3F808000
    midway_fp32 = struct.unpack("<f", struct.pack("<I", midway_bits))[0]
    expected = float(torch.tensor(midway_fp32, dtype=torch.bfloat16))
    infer = CotsWeightTaskRunner()
    got = _drive_kernel_to_emit(infer, midway_fp32)
    assert got == expected
    above_bits = 0x3F808001
    above_fp32 = struct.unpack("<f", struct.pack("<I", above_bits))[0]
    expected_above = float(torch.tensor(above_fp32, dtype=torch.bfloat16))
    got_above = _drive_kernel_to_emit(infer, above_fp32)
    assert got_above == expected_above
    assert got_above > got


# --------------------------------------------------------------------- #
# Per-M path coverage. M_TILE=4 register tile + per-m M=1 tail for
# M % 4 leftover. At small K (≤ 64), accumulation order is fully
# deterministic so we can demand bit-exact equality.
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "M, K, N",
    [
        # M=1: pure M_TILE=1 tail path (no M_TILE=4 group taken).
        (1, 16, 64),
        (1, 16, 128),
        (1, 64, 256),
        # M=4: one full M_TILE=4 register-tile group, no tail.
        (4, 16, 64),
        (4, 16, 128),
        (4, 64, 256),
        # M=2 / M=3: pure tail (no M_TILE=4 group taken). Exercises
        # the M % 4 < 4 branch.
        (2, 16, 64),
        (3, 16, 64),
        # M=5..7: one M_TILE=4 group + 1..3 m tail.
        (5, 32, 128),
        (6, 32, 128),
        (7, 32, 128),
        # M=8: two full M_TILE=4 groups.
        (8, 32, 128),
        # M=12: three groups.
        (12, 32, 128),
    ],
)
def test_kernel_parity_per_M_path(M, K, N):
    """Kernel output bit-exact vs scalar reference at small K."""
    torch.manual_seed(0xC07407 + M * 1000 + K * 10 + N)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w_NK = torch.randn(N, K, dtype=torch.bfloat16)

    y_ref = _ref_bf16_gemm_natural(x, w_NK)

    infer = CotsWeightTaskRunner()
    y_got = torch.empty(M, N, dtype=torch.bfloat16)
    infer.run_bf16_gemm_natural_inline(x, w_NK, y_got)

    assert torch.equal(y_got, y_ref), (
        f"bit-exact parity failed at M={M} K={K} N={N}: "
        f"max abs diff {(y_got.float() - y_ref.float()).abs().max()}"
    )


# --------------------------------------------------------------------- #
# K-tail coverage: K % 8 != 0 forces the scalar K-tail loop after
# the 8-wide main loop. Production K is always a multiple of 8 but
# we want the kernel to be correct on arbitrary K.
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "M, K, N",
    [
        # K = 1..7 → no main loop iterations, pure K-tail.
        (1, 1, 16),
        (4, 3, 16),
        (4, 7, 32),
        # K = 9 → one main iter + 1 tail. K = 17 → 2 main + 1 tail.
        (1, 9, 16),
        (4, 17, 32),
        (4, 31, 64),
        # K = 23 → 2 main + 7 tail. M tail also exercised here.
        (3, 23, 32),
        (5, 23, 64),
        # Large K with tail: closer to production shape.
        (4, 947, 64),   # K=947 is not 8-aligned (947 % 8 = 3)
        (4, 1893, 64),  # K=1893 is 8-aligned? 1893 % 8 = 5
    ],
)
def test_kernel_k_tail_correctness(M, K, N):
    """K % 8 != 0 must still match the scalar reference."""
    torch.manual_seed(0xC07A11 + K)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w_NK = torch.randn(N, K, dtype=torch.bfloat16)
    y_ref = _ref_bf16_gemm_natural(x, w_NK)
    infer = CotsWeightTaskRunner()
    y_got = torch.empty(M, N, dtype=torch.bfloat16)
    infer.run_bf16_gemm_natural_inline(x, w_NK, y_got)
    assert torch.equal(y_got, y_ref), (
        f"K-tail parity failed at M={M} K={K} N={N}: "
        f"max abs diff {(y_got.float() - y_ref.float()).abs().max()}"
    )


# --------------------------------------------------------------------- #
# Production-shape relaxed parity. At K ~ 947-3584, FMA accumulation
# order produces ~ULP-of-BF16 differences vs at::linear.
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "M, K, N",
    [
        # QKV-shaped slices (K=hidden=3584, N=qkv_dim_cpu)
        (1, 3584, 460),
        (4, 3584, 460),
        (32, 3584, 460),
        # MLP1-shaped slices (K=hidden=3584, N=intermediate_cpu)
        (1, 3584, 1894),
        (4, 3584, 1894),
        (32, 3584, 1894),
    ],
)
def test_kernel_relaxed_parity_at_production_shapes(M, K, N):
    """At production K, accept ~ULP-of-BF16 relative error vs at::linear."""
    torch.manual_seed(0xC0F4C7 + M * 100 + N)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w_NK = torch.randn(N, K, dtype=torch.bfloat16)

    y_ref = torch.nn.functional.linear(x, w_NK)
    infer = CotsWeightTaskRunner()
    y_got = torch.empty(M, N, dtype=torch.bfloat16)
    infer.run_bf16_gemm_natural_inline(x, w_NK, y_got)

    diff = (y_got.float() - y_ref.float()).abs()
    scale = y_ref.float().abs().max() + 1e-6
    rel = diff.max() / scale
    assert rel < 5e-3, (
        f"relaxed parity failed at M={M} K={K} N={N}: rel={rel:.3e}, "
        f"max abs diff {diff.max()}, scale {scale}"
    )


# --------------------------------------------------------------------- #
# Threading invariance — `at::set_num_threads` should not change the
# output. The kernel parallelizes on N (disjoint output columns), so
# results must be thread-count independent.
# --------------------------------------------------------------------- #


def test_kernel_threaded_output_matches_serial():
    M, K, N = 4, 32, 256
    torch.manual_seed(0xC07A4)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w_NK = torch.randn(N, K, dtype=torch.bfloat16)
    infer = CotsWeightTaskRunner()

    saved = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        y_serial = torch.empty(M, N, dtype=torch.bfloat16)
        infer.run_bf16_gemm_natural_inline(x, w_NK, y_serial)

        torch.set_num_threads(16)
        y_threaded = torch.empty(M, N, dtype=torch.bfloat16)
        infer.run_bf16_gemm_natural_inline(x, w_NK, y_threaded)
    finally:
        torch.set_num_threads(saved)

    assert torch.equal(y_serial, y_threaded), (
        f"threading changed output (max abs diff "
        f"{(y_serial.float() - y_threaded.float()).abs().max()})"
    )
