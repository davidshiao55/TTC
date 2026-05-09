# SPDX-License-Identifier: Apache-2.0
"""Stage 3 standalone — strided down-proj slab parity.

Stage 1's `test_at_linear_microbench.py` proved C++ `at::linear` on a
strided `at::from_blob` view matches Python `F.linear` for the
load-bearing AVX2-BF16 dispatch. That microbench drove the test-only
helper `CotsCpuInfer.run_at_linear_inline(x, w, y)` directly, bypassing
the slab dispatch.

This test exercises the FULL public API path:
    populate_slab_mlp(..., w_down_ptr=POST-narrow, w_down_stride_row=N,
                      w_down_stride_col=1, ...)
        → submit_on_stream
        → cudaLaunchHostFunc → DispatchCallback
        → TaskQueue worker
        → at::from_blob(ptr, sizes, {stride_row, stride_col})
        → at::linear
        → y_view.copy_(...)

The non-zero `dn_n_pref` case (column slice starting at column N>0)
makes both the post-narrow `data_ptr()` and the (stride_row=n_cpu_total,
stride_col=1) layout load-bearing — accidentally passing the base
pointer or default strides would silently read the wrong rows.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm._cots_C import CotsCpuInfer

pytestmark = pytest.mark.needs_cuda


# Synthetic shapes self-contained to this test (no vLLM linear modules).
# The matched-index invariant the C++ MLP dispatcher relies on:
#     w_gate_rows == intermediate_per_half == w_down_cols (= dn_n_cpu)
# The worker computes
#     z = scratch.narrow(1, 0, intermediate_per_half).copy_(silu(gate)*up)
#     y = at::linear(z, w_down)              # z @ w_down.t()
# so z's K (intermediate_per_half) must equal w_down's K (w_down_cols).
#
# To exercise the load-bearing strided down-proj path we set:
#   * N_CPU_TOTAL_DN > dn_n_cpu  (so narrow has a positive offset)
#   * DN_N_PREF > 0              (the offset itself is non-zero)
# while keeping intermediate_per_half == dn_n_cpu (matched-index).
HIDDEN = 256
DN_N_PREF = 64       # offset into n_cpu_total — the load-bearing case
DN_N_CPU = 128        # = w_down_cols = intermediate_per_half (matched)
N_CPU_TOTAL_DN = DN_N_PREF + DN_N_CPU  # 192
INTERMEDIATE_PER_HALF = DN_N_CPU       # = w_gate_rows = w_up_rows
NUM_TOKENS = 16

BF16_RTOL = 5e-2
BF16_ATOL = 0.5


def _alloc_pinned(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, pin_memory=True)


def test_native_runner_strided_down_proj_matches_python():
    """Populate an MLP slab with `dn_n_pref > 0` so the down-proj weight
    pointer is offset and the strides differ from the contiguous case.
    Run one forward through the host-callback path and compare to
    Python `F.linear` on the equivalent narrow view."""
    torch.manual_seed(0)

    # Backing storage. `dn_w_cpu` is the full row-major (out_dim,
    # n_cpu_total) tensor that a real CotsLinearHandle would own; the
    # narrow-view pointer is what install passes to populate_slab_mlp.
    x_pinned_full = _alloc_pinned(NUM_TOKENS, INTERMEDIATE_PER_HALF)
    x_pinned_full.copy_(torch.randn_like(x_pinned_full))

    # We don't actually run gate / up here — we want to test the down
    # path in isolation. So populate the slab as if gate_out=up_out=1
    # and silu(1)*1 = 1*1 = 0.7311 ish — but the worker computes silu
    # internally. To make this a pure down-proj parity test, we'd need
    # to feed the slab gate/up weights that produce a known
    # silu(gate)*up. Easier: use the SAME gate/up weights for both
    # python reference AND native slab, so any divergence comes from
    # the down-proj strided path.
    gate_full = torch.randn(
        INTERMEDIATE_PER_HALF, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    up_full = torch.randn(
        INTERMEDIATE_PER_HALF, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    # Down storage: (hidden, n_cpu_total). The slab consumes
    # narrow(1, DN_N_PREF, DN_N_CPU) — a strided column slice.
    dn_w_full = torch.randn(
        HIDDEN, N_CPU_TOTAL_DN, dtype=torch.bfloat16, pin_memory=True
    )
    dn_w_view = dn_w_full.narrow(1, DN_N_PREF, DN_N_CPU)
    assert not dn_w_view.is_contiguous()
    assert dn_w_view.stride() == (N_CPU_TOTAL_DN, 1)

    # The MLP dispatch path expects gate/up of shape
    # (intermediate_per_half - n_pref_per_half, hidden) — i.e., the
    # post-prefetch CPU compute slice. For this test we pass the full
    # gate/up (n_pref_per_half == 0 on the gate/up side; only down has
    # the offset). intermediate_per_half stays at full INTERMEDIATE_PER_HALF.
    x_for_mlp = torch.randn(
        NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    y_pinned = _alloc_pinned(NUM_TOKENS, HIDDEN)

    ci = CotsCpuInfer()
    ci.install(
        n_slabs=1,
        scratch_max_tokens=NUM_TOKENS,
        scratch_max_intermediate_per_half=INTERMEDIATE_PER_HALF,
    )
    ci.populate_slab_mlp(
        task_id=0,
        n_threads=1,
        x_pinned_ptr=x_for_mlp.data_ptr(),
        in_dim=HIDDEN,
        y_pinned_ptr=y_pinned.data_ptr(),
        cpu_out_dim=HIDDEN,
        w_gate_ptr=gate_full.data_ptr(),
        w_gate_rows=INTERMEDIATE_PER_HALF,
        w_up_ptr=up_full.data_ptr(),
        w_up_rows=INTERMEDIATE_PER_HALF,
        # Load-bearing fields:
        w_down_ptr=dn_w_view.data_ptr(),  # POST-narrow (offset)
        w_down_rows=HIDDEN,  # = view.shape[0]
        w_down_cols=DN_N_CPU,  # = view.shape[1]
        w_down_stride_row=N_CPU_TOTAL_DN,  # = view.stride(0)
        w_down_stride_col=1,  # = view.stride(1)
        intermediate_per_half=INTERMEDIATE_PER_HALF,
    )

    # Drive through the public host-callback path on a real CUDA stream.
    stream = torch.cuda.current_stream().cuda_stream
    ci.submit_on_stream(task_id=0, num_tokens=NUM_TOKENS, cuda_stream=stream)
    ci.sync_on_stream(cuda_stream=stream)
    torch.cuda.current_stream().synchronize()
    assert not ci.has_error()

    # Python reference: same arithmetic, all on CPU, no slab.
    gate_out = F.linear(x_for_mlp, gate_full)
    up_out = F.linear(x_for_mlp, up_full)
    z = F.silu(gate_out) * up_out
    y_ref = F.linear(z, dn_w_view)

    torch.testing.assert_close(y_pinned, y_ref, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_native_runner_strided_down_proj_offset_pointer_is_load_bearing():
    """Sanity that `populate_slab_mlp(w_down_ptr=...)` accepting the
    POST-narrow pointer is what's load-bearing — passing the BASE
    pointer (offset 0) with the same strides would read the WRONG
    starting column and produce a divergent result.

    We populate one slab with the correct post-narrow pointer and a
    second slab with the base pointer (intentionally wrong), and
    confirm they produce different outputs. If the wrong-pointer slab
    happened to match (e.g., because dispatch silently ignored the
    pointer), this regression check would fail.
    """
    torch.manual_seed(1)
    x_for_mlp = torch.randn(
        NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    gate_full = torch.randn(
        INTERMEDIATE_PER_HALF, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    up_full = torch.randn(
        INTERMEDIATE_PER_HALF, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    dn_w_full = torch.randn(
        HIDDEN, N_CPU_TOTAL_DN, dtype=torch.bfloat16, pin_memory=True
    )
    dn_w_view = dn_w_full.narrow(1, DN_N_PREF, DN_N_CPU)
    y_correct = _alloc_pinned(NUM_TOKENS, HIDDEN)
    y_wrong = _alloc_pinned(NUM_TOKENS, HIDDEN)

    common_kwargs = dict(
        x_pinned_ptr=x_for_mlp.data_ptr(),
        in_dim=HIDDEN,
        cpu_out_dim=HIDDEN,
        w_gate_ptr=gate_full.data_ptr(),
        w_gate_rows=INTERMEDIATE_PER_HALF,
        w_up_ptr=up_full.data_ptr(),
        w_up_rows=INTERMEDIATE_PER_HALF,
        w_down_rows=HIDDEN,
        w_down_cols=DN_N_CPU,
        w_down_stride_row=N_CPU_TOTAL_DN,
        w_down_stride_col=1,
        intermediate_per_half=INTERMEDIATE_PER_HALF,
        n_threads=1,
    )

    ci = CotsCpuInfer()
    ci.install(
        n_slabs=2,
        scratch_max_tokens=NUM_TOKENS,
        scratch_max_intermediate_per_half=INTERMEDIATE_PER_HALF,
    )
    # Slab 0 — correct: post-narrow pointer.
    ci.populate_slab_mlp(
        task_id=0,
        y_pinned_ptr=y_correct.data_ptr(),
        w_down_ptr=dn_w_view.data_ptr(),  # post-narrow
        **common_kwargs,
    )
    # Slab 1 — wrong: base pointer (offset 0). Same strides; reads
    # cols [0:DN_N_CPU) instead of [DN_N_PREF:DN_N_PREF+DN_N_CPU).
    ci.populate_slab_mlp(
        task_id=1,
        y_pinned_ptr=y_wrong.data_ptr(),
        w_down_ptr=dn_w_full.data_ptr(),  # base, NOT post-narrow
        **common_kwargs,
    )

    stream = torch.cuda.current_stream().cuda_stream
    ci.submit_on_stream(task_id=0, num_tokens=NUM_TOKENS, cuda_stream=stream)
    ci.sync_on_stream(cuda_stream=stream)
    ci.submit_on_stream(task_id=1, num_tokens=NUM_TOKENS, cuda_stream=stream)
    ci.sync_on_stream(cuda_stream=stream)
    torch.cuda.current_stream().synchronize()
    assert not ci.has_error()

    diff = (y_correct.float() - y_wrong.float()).abs().max().item()
    # Different starting column → different cols of the down weight →
    # materially different output. Threshold is loose; the point is
    # the two are visibly distinct (not "off by one ULP").
    assert diff > 0.5, (
        f"Wrong-pointer slab produced y within {diff:.4f} of the correct "
        f"slab — populate_slab_mlp's w_down_ptr field appears to be "
        f"ignored, OR the offset isn't actually load-bearing for the "
        f"chosen DN_N_PREF / DN_N_CPU. Either way this regression "
        f"sentinel needs investigation."
    )
