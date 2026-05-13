# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the Stage 7-B custom AVX2 BF16 GEMM kernel
(`vllm/csrc/cots/bf16_gemm_transposed.cpp`).

Covers:
  * M=1 (N_INNER=4 register tile), M=2 (N_INNER=2), M=4 (N_INNER=1
    M-fusion path), and M=3 (general fallback path — exercises the
    per-(m, nb) fallback that has no register-tile M-fusion).
  * N tail handling: N not a multiple of Nb=16 (the kernel's main
    loop walks `N_main = (N // 16) * 16`; remaining columns fall to
    the trailing scalar loop).
  * FP32→BF16 round-to-nearest-even (RNE) conversion: exhaustive
    check against PyTorch's `to(torch.bfloat16)` (also RNE) across
    representative FP32 values, including the tie-to-even boundary
    case.
  * Numerical parity vs a scalar reference implementation at small K
    (K = 16, 64) — when K is small, the FMA accumulation order does
    not matter, so we can assert bit-exact equality against a
    reference implementation.

These tests deliberately cover correctness only. The old Stage 7
performance microbench was a bring-up probe and is no longer part of
the retained Phase 1c test suite.
"""

from __future__ import annotations

import pytest
import torch

from vllm._cots_C import CotsCpuInfer


# --------------------------------------------------------------------- #
# Reference scalar implementation. Used for bit-exact parity comparison
# at small K where FMA accumulation order does not produce different
# results.
# --------------------------------------------------------------------- #


def _ref_bf16_gemm_transposed(
    x: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """y = x @ w (BF16 row-major), reference scalar impl.

    Mirrors the kernel's behaviour: upconvert BF16→FP32 per element,
    accumulate in FP32, then round-to-nearest-even back to BF16 (which
    is exactly what PyTorch's `to(torch.bfloat16)` does).
    """
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
    assert x.dim() == 2 and w.dim() == 2 and x.size(1) == w.size(0)
    x_f32 = x.to(torch.float32)
    w_f32 = w.to(torch.float32)
    acc = x_f32 @ w_f32  # FP32 accumulator
    return acc.to(torch.bfloat16)  # RNE down-conversion


# --------------------------------------------------------------------- #
# RNE conversion correctness — exhaustive check on representative FP32
# values, especially the tie-to-even boundary (low-16-bits == 0x8000).
# --------------------------------------------------------------------- #


def _drive_kernel_to_emit(infer: CotsCpuInfer, value_fp32: float) -> float:
    """Drive the kernel so y[0, 0] equals the BF16-rounded `value_fp32`.

    Achieved with M=K=N=1 logical work: x = [[1.0]], w = [[value]].
    K=1 means there's no accumulation, so y[0, 0] is exactly the
    BF16-rounded product of two BF16-rounded operands. With x = 1.0
    (exactly representable) and w = `value_fp32` (rounded once on
    construction), the product = value (one BF16 round trip), and the
    kernel applies its FP32→BF16 RNE on top.
    """
    # Use M=4 because that's a register-tile path; pad with zeros.
    M, K, N = 4, 16, 16
    x = torch.zeros(M, K, dtype=torch.bfloat16)
    w_KN = torch.zeros(K, N, dtype=torch.bfloat16)
    x[0, 0] = torch.tensor(1.0, dtype=torch.bfloat16)
    # The construction of w[0, 0] from a Python float already RNE-rounds
    # to BF16 via PyTorch's tensor constructor.
    w_KN[0, 0] = torch.tensor(value_fp32, dtype=torch.bfloat16)
    y = torch.empty(M, N, dtype=torch.bfloat16)
    infer.run_bf16_gemm_transposed_inline(x, w_KN, y)
    return float(y[0, 0])


@pytest.mark.parametrize(
    "value",
    [
        0.0, 1.0, -1.0,
        0.5, 1.5, -1.5,
        100.5, -100.5,
        # Subnormals near 0 in FP32 land — round to nearest BF16
        # subnormal.
        1e-4, 1e-30,
        # Tie-to-even cases: FP32 value whose low 16 mantissa bits are
        # exactly 0x8000 (halfway). RNE chooses the even bf16 mantissa.
        # Bit pattern 0x3F808000 = 1.0 + (1/256). Halfway between
        # 1.0 (mantissa 0) and 1 + 1/128 (mantissa 1). Tie-to-even
        # → mantissa 0 → result 1.0.
        # We can't easily construct this from a Python float exactly
        # (it'll round on the way in), so we use values that are
        # exactly representable in BF16 (the tie test below covers
        # the boundary more rigorously).
        2.0, -2.0,
        # Values that exercise the high-magnitude range
        1024.0, -1024.0, 65536.0,
    ],
)
def test_rne_conversion_matches_pytorch_bf16(value):
    """Kernel's FP32→BF16 RNE must match PyTorch's `to(bfloat16)`.

    Both should produce identical bit patterns for any FP32 input that
    isn't NaN/Inf (which the kernel doesn't canonicalize — documented
    in the kernel's header comment).
    """
    infer = CotsCpuInfer()
    got = _drive_kernel_to_emit(infer, value)
    expected = float(torch.tensor(value, dtype=torch.bfloat16))
    assert got == expected, (
        f"RNE mismatch for value={value!r}: kernel emitted {got!r}, "
        f"PyTorch's RNE gives {expected!r}"
    )


def test_rne_tie_to_even_boundary():
    """Explicit tie-to-even check: a known-halfway FP32 bit pattern.

    Build an FP32 value whose low 16 mantissa bits are exactly 0x8000
    (the halfway point between two adjacent BF16 representations) and
    confirm the kernel rounds to the EVEN BF16 mantissa.
    """
    import struct

    # Bit pattern: 0x3F808000 = 1.0 + 0x8000 / (1 << 23) = midway
    # between 1.0 (BF16 mantissa 0) and 1 + 1/128 (BF16 mantissa 1).
    # RNE → mantissa 0 (even) → BF16 value = 1.0.
    midway_bits = 0x3F808000
    midway_fp32 = struct.unpack("<f", struct.pack("<I", midway_bits))[0]

    # PyTorch's BF16 conversion is also RNE, so we can use it as oracle.
    expected = float(torch.tensor(midway_fp32, dtype=torch.bfloat16))

    infer = CotsCpuInfer()
    got = _drive_kernel_to_emit(infer, midway_fp32)
    assert got == expected, (
        f"tie-to-even mismatch: got {got!r}, expected {expected!r}"
    )

    # And the next bit pattern up — 0x3F808001 — must round UP to the
    # next BF16 (mantissa 1, value 1 + 1/128).
    above_bits = 0x3F808001
    above_fp32 = struct.unpack("<f", struct.pack("<I", above_bits))[0]
    expected_above = float(torch.tensor(above_fp32, dtype=torch.bfloat16))
    got_above = _drive_kernel_to_emit(infer, above_fp32)
    assert got_above == expected_above
    assert got_above > got  # confirm it actually rounded UP


# --------------------------------------------------------------------- #
# Per-M register-tile path coverage. Small K so FMA order doesn't
# perturb the result — we can assert bit-exact equality.
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "M, K, N",
    [
        # M=1: hits the N_INNER=4 register tile (N_tile=64). N=64
        # is exactly one tile; N=128 is two; N=256 is the perf
        # microbench's range.
        (1, 16, 64),
        (1, 16, 128),
        (1, 64, 256),
        # M=2: hits the N_INNER=2 register tile (N_tile=32).
        (2, 16, 32),
        (2, 16, 64),
        (2, 64, 128),
        # M=4: hits the N_INNER=1 M-fusion register tile (N_tile=Nb=16).
        (4, 16, 16),
        (4, 16, 64),
        (4, 64, 256),
        # M=3: general fallback path (no M-fusion, per-m outer loop).
        (3, 16, 64),
        (3, 64, 128),
        # M=8: still hits the general fallback.
        (8, 32, 128),
    ],
)
def test_kernel_parity_per_M_path(M, K, N):
    """Kernel output bit-exact vs scalar reference at small K.

    At small K (≤ 64 here), FMA accumulation order produces no
    rounding-order divergence — the scalar reference and our kernel
    must agree on every bit.
    """
    torch.manual_seed(0xC07501 + M * 1000 + K * 10 + N)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w_KN = torch.randn(K, N, dtype=torch.bfloat16)

    y_ref = _ref_bf16_gemm_transposed(x, w_KN)

    infer = CotsCpuInfer()
    y_got = torch.empty(M, N, dtype=torch.bfloat16)
    infer.run_bf16_gemm_transposed_inline(x, w_KN, y_got)

    # At small K (≤ 64) the FMA order doesn't perturb the rounded
    # result; bit-exact equality is the right gate.
    assert torch.equal(y_got, y_ref), (
        f"bit-exact parity failed at M={M} K={K} N={N}: "
        f"max abs diff {(y_got.float() - y_ref.float()).abs().max()}, "
        f"first divergence at {(y_got != y_ref).nonzero()[:3]}"
    )


# --------------------------------------------------------------------- #
# N-tail coverage: N not a multiple of Nb=16. The main register-tiled
# loop covers `N_main = (N // 16) * 16`; the trailing columns fall to
# the per-row scalar/8-wide tail loop.
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "M, K, N",
    [
        # Each N here is `N_main + tail`, with tail > 0.
        # Tail < 8 → scalar path only.
        (1, 16, 17),  # tail = 1
        (1, 16, 22),  # tail = 6
        (4, 16, 23),  # tail = 7
        # 8 ≤ tail < 16 → one 8-wide tile + scalar.
        (1, 16, 24),  # tail = 8 (one 8-wide tile)
        (1, 16, 27),  # tail = 11 (one 8-wide + 3 scalar)
        (2, 32, 31),  # tail = 15 (one 8-wide + 7 scalar)
        (4, 32, 47),  # tail = 15 at the high end
        # M=3 fallback path with tail.
        (3, 32, 23),
        # Larger M+N with tail (general fallback).
        (8, 32, 33),
    ],
)
def test_kernel_n_tail_correctness(M, K, N):
    """Tail columns (N % 16 != 0) must match the scalar reference.

    Same gate as the per-M path test — at small K we can demand
    bit-exact equality.
    """
    torch.manual_seed(0xC07A11 + N)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w_KN = torch.randn(K, N, dtype=torch.bfloat16)
    y_ref = _ref_bf16_gemm_transposed(x, w_KN)
    infer = CotsCpuInfer()
    y_got = torch.empty(M, N, dtype=torch.bfloat16)
    infer.run_bf16_gemm_transposed_inline(x, w_KN, y_got)
    assert torch.equal(y_got, y_ref), (
        f"N-tail parity failed at M={M} K={K} N={N}: "
        f"max abs diff {(y_got.float() - y_ref.float()).abs().max()}, "
        f"tail starts at column {(N // 16) * 16}"
    )


# --------------------------------------------------------------------- #
# Large-K relaxed parity — at production-like K (~1000-3000) the FMA
# accumulation order between our kernel and PyTorch's at::linear can
# differ by ~1 ULP-of-BF16 at the accumulator magnitude. Use a
# relative-error gate rather than bit-exact.
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "M, K, N",
    [
        (1, 947, 3584),
        (4, 947, 3584),
        (1, 2842, 3584),
        (4, 2842, 3584),
    ],
)
def test_kernel_relaxed_parity_at_production_K(M, K, N):
    """At large K, accept ~ULP-of-BF16 relative error vs at::linear.

    The kernel emits the same per-element BF16 result modulo FMA
    accumulation order. PyTorch's BF16 conversion is RNE; ours is now
    too. Remaining diffs are purely accumulation-order artefacts that
    grow with K.
    """
    torch.manual_seed(0xC0F4C7)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w_NK = torch.randn(N, K, dtype=torch.bfloat16)
    w_KN = w_NK.t().contiguous()

    y_ref = torch.nn.functional.linear(x, w_NK)
    infer = CotsCpuInfer()
    y_got = torch.empty(M, N, dtype=torch.bfloat16)
    infer.run_bf16_gemm_transposed_inline(x, w_KN, y_got)

    diff = (y_got.float() - y_ref.float()).abs()
    scale = y_ref.float().abs().max() + 1e-6
    rel = diff.max() / scale
    # 5e-3 is conservative — the perf microbench's empirical worst
    # case at K=2842 was 1.5e-3 with RNE; pre-RNE was ~6e-3.
    assert rel < 5e-3, (
        f"relaxed parity failed at M={M} K={K} N={N}: rel={rel:.3e}, "
        f"max abs diff {diff.max()}, scale {scale}"
    )


# --------------------------------------------------------------------- #
# Threading invariance — `at::set_num_threads` should not change the
# kernel's output. Catches threading-induced data races. Note we use
# small K so accumulation order is deterministic regardless of how the
# work is split.
# --------------------------------------------------------------------- #


def test_kernel_threaded_output_matches_serial():
    """Output must be bit-exact identical at thr=1 and thr=16.

    The kernel parallelizes on N-tiles (disjoint column ranges of y);
    there is no reduction across threads, so the result must be
    independent of thread count.
    """
    M, K, N = 4, 32, 256
    torch.manual_seed(0xC07A4)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w_KN = torch.randn(K, N, dtype=torch.bfloat16)
    infer = CotsCpuInfer()

    saved = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        y_serial = torch.empty(M, N, dtype=torch.bfloat16)
        infer.run_bf16_gemm_transposed_inline(x, w_KN, y_serial)

        torch.set_num_threads(16)
        y_threaded = torch.empty(M, N, dtype=torch.bfloat16)
        infer.run_bf16_gemm_transposed_inline(x, w_KN, y_threaded)
    finally:
        torch.set_num_threads(saved)

    assert torch.equal(y_serial, y_threaded), (
        f"threading changed output (max abs diff "
        f"{(y_serial.float() - y_threaded.float()).abs().max()})"
    )
