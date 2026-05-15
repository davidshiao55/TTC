# SPDX-License-Identifier: Apache-2.0
"""Stage 7-C — contiguous-transposed-storage down-proj slab parity.

Stage 7-C's down-proj refactor flipped CPU storage from natural
`(out_dim, n_cpu_total)` (where a column-narrow gave a strided view —
the prior `test_strided_down_proj.py` premise) to transposed
`(n_cpu_total, out_dim)`. The CPU-compute slice is now
`w_cpu.narrow(0, dn_n_pref, dn_n_cpu)` — a CONTIGUOUS `(K, N)` view
that feeds the custom `bf16_gemm_transposed` kernel directly.

This test exercises the FULL public API path with the new layout:
    populate_slab_mlp(..., w_down_ptr=POST-narrow,
                      w_down_rows=K=dn_n_cpu,
                      w_down_cols=N=out_dim, ...)
        → submit_on_stream
        → cudaLaunchHostFunc → DispatchCallback
        → TaskQueue worker
        → ContigCpuViewFromBlob(ptr, K, N)
        → bf16_gemm_transposed_at(z, w_down, y_view)

The non-zero `dn_n_pref` case (starting row > 0) keeps the post-narrow
`data_ptr()` load-bearing — accidentally passing the base pointer
would silently read the wrong rows.
"""

from __future__ import annotations

import pytest
import torch

from vllm._cots_C import CotsCpuInfer

pytestmark = pytest.mark.needs_cuda


HIDDEN = 256
DN_N_PREF = 64       # offset into n_cpu_total — the load-bearing case
DN_N_CPU = 128       # = w_down_rows
N_CPU_TOTAL_DN = DN_N_PREF + DN_N_CPU  # 192
INTERMEDIATE_PER_HALF = DN_N_CPU       # = w_gate_rows = w_up_rows
NUM_TOKENS = 16

BF16_RTOL = 5e-2
BF16_ATOL = 0.5


def _alloc_pinned(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype, pin_memory=True)


def _ref_production_mlp_block(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,
    w_down_transposed: torch.Tensor,
) -> torch.Tensor:
    """Reference the production MLP CPU semantics from exposed GEMM helpers.

    The native worker rounds gate/up to BF16, computes SwiGLU into BF16
    scratch, then feeds the transposed down GEMM. Using the same GEMM helpers
    keeps this test focused on slab layout and post-narrow pointers rather
    than PyTorch/oneDNN's slightly different BF16 accumulation choices.
    """
    import torch.nn.functional as F

    infer = CotsCpuInfer()
    gate_out = torch.empty(
        x.shape[0], w_gate.shape[0], dtype=torch.bfloat16, pin_memory=True
    )
    up_out = torch.empty_like(gate_out)
    infer.run_bf16_gemm_natural_inline(x, w_gate, gate_out)
    infer.run_bf16_gemm_natural_inline(x, w_up, up_out)
    z = (F.silu(gate_out.float()) * up_out.float()).to(torch.bfloat16)
    y = torch.empty(
        x.shape[0],
        w_down_transposed.shape[1],
        dtype=torch.bfloat16,
        pin_memory=True,
    )
    infer.run_bf16_gemm_transposed_inline(z, w_down_transposed, y)
    return y


def test_native_runner_contig_down_proj_matches_python():
    """End-to-end parity: populate an MLP slab with the post-Stage-7-C
    transposed-storage down-proj layout, drive the slab through the
    host-callback path, and confirm output matches the production BF16
    scratch reference."""
    torch.manual_seed(0)

    gate_full = torch.randn(
        INTERMEDIATE_PER_HALF, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    up_full = torch.randn(
        INTERMEDIATE_PER_HALF, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )

    # Stage 7-C down storage: (n_cpu_total, out_dim) row-major. The CPU
    # compute slice is rows [DN_N_PREF, DN_N_PREF + DN_N_CPU) — a
    # contiguous (DN_N_CPU, HIDDEN) view. POST-narrow `data_ptr()` is
    # what the slab carries; reading the base pointer with the same
    # row count would touch the wrong rows.
    dn_w_full = torch.randn(
        N_CPU_TOTAL_DN, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    dn_w_view = dn_w_full.narrow(0, DN_N_PREF, DN_N_CPU)
    assert dn_w_view.is_contiguous(), (
        "Stage 7-C invariant: transposed-storage narrow on dim 0 is "
        "contiguous (no strided fallback path)"
    )
    assert dn_w_view.shape == (DN_N_CPU, HIDDEN)

    x_for_mlp = torch.randn(
        NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    y_pinned = _alloc_pinned(NUM_TOKENS, HIDDEN)

    ci = CotsCpuInfer()
    ci.install(
        n_slabs=1,
        max_num_tokens=NUM_TOKENS,
    )
    ci.populate_slab_mlp(
        task_id=0,
        n_threads=1,
        bucket_capacity_tokens=NUM_TOKENS,
        x_pinned_ptr=x_for_mlp.data_ptr(),
        in_dim=HIDDEN,
        y_pinned_ptr=y_pinned.data_ptr(),
        cpu_out_dim=HIDDEN,
        w_gate_ptr=gate_full.data_ptr(),
        w_gate_rows=INTERMEDIATE_PER_HALF,
        w_up_ptr=up_full.data_ptr(),
        w_up_rows=INTERMEDIATE_PER_HALF,
        # Stage 7-C field semantics:
        #   w_down_rows = K (= dn_n_cpu)
        #   w_down_cols = N (= out_dim = HIDDEN)
        w_down_ptr=dn_w_view.data_ptr(),  # POST-narrow
        w_down_rows=DN_N_CPU,
        w_down_cols=HIDDEN,
    )

    stream = torch.cuda.current_stream().cuda_stream
    ci.submit_on_stream(
        task_id=0, num_tokens=NUM_TOKENS, cuda_stream=stream,
        x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1,
    )
    ci.sync_on_stream(cuda_stream=stream)
    torch.cuda.current_stream().synchronize()
    assert not ci.has_error()

    y_ref = _ref_production_mlp_block(x_for_mlp, gate_full, up_full, dn_w_view)

    torch.testing.assert_close(y_pinned, y_ref, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_native_runner_contig_down_proj_offset_pointer_is_load_bearing():
    """Sanity check that `w_down_ptr` accepting the POST-narrow pointer
    is what's load-bearing — passing the BASE pointer (offset 0) would
    read the wrong STARTING ROW of the transposed-storage weight (rows
    [0, DN_N_CPU) instead of [DN_N_PREF, DN_N_PREF + DN_N_CPU)) and
    produce a divergent result.
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
        N_CPU_TOTAL_DN, HIDDEN, dtype=torch.bfloat16, pin_memory=True
    )
    dn_w_view = dn_w_full.narrow(0, DN_N_PREF, DN_N_CPU)
    y_correct = _alloc_pinned(NUM_TOKENS, HIDDEN)
    y_wrong = _alloc_pinned(NUM_TOKENS, HIDDEN)

    common_kwargs = dict(
        bucket_capacity_tokens=NUM_TOKENS,
        x_pinned_ptr=x_for_mlp.data_ptr(),
        in_dim=HIDDEN,
        cpu_out_dim=HIDDEN,
        w_gate_ptr=gate_full.data_ptr(),
        w_gate_rows=INTERMEDIATE_PER_HALF,
        w_up_ptr=up_full.data_ptr(),
        w_up_rows=INTERMEDIATE_PER_HALF,
        w_down_rows=DN_N_CPU,
        w_down_cols=HIDDEN,
        n_threads=1,
    )

    ci = CotsCpuInfer()
    ci.install(
        n_slabs=2,
        max_num_tokens=NUM_TOKENS,
    )
    # Slab 0 — correct: post-narrow pointer.
    ci.populate_slab_mlp(
        task_id=0,
        y_pinned_ptr=y_correct.data_ptr(),
        w_down_ptr=dn_w_view.data_ptr(),  # post-narrow
        **common_kwargs,
    )
    # Slab 1 — wrong: base pointer (offset 0). Reads rows [0:DN_N_CPU)
    # instead of [DN_N_PREF:DN_N_PREF+DN_N_CPU).
    ci.populate_slab_mlp(
        task_id=1,
        y_pinned_ptr=y_wrong.data_ptr(),
        w_down_ptr=dn_w_full.data_ptr(),  # base, NOT post-narrow
        **common_kwargs,
    )

    stream = torch.cuda.current_stream().cuda_stream
    ci.submit_on_stream(
        task_id=0, num_tokens=NUM_TOKENS, cuda_stream=stream,
        x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1,
    )
    ci.sync_on_stream(cuda_stream=stream)
    ci.submit_on_stream(
        task_id=1, num_tokens=NUM_TOKENS, cuda_stream=stream,
        x_gpu_ptr=0, x_cols=0, x_stride0=0, x_stride1=1,
    )
    ci.sync_on_stream(cuda_stream=stream)
    torch.cuda.current_stream().synchronize()
    assert not ci.has_error()

    diff = (y_correct.float() - y_wrong.float()).abs().max().item()
    assert diff > 0.5, (
        f"Wrong-pointer slab produced y within {diff:.4f} of the "
        f"correct slab — populate_slab_mlp's w_down_ptr field appears "
        f"to be ignored or the post-narrow offset isn't load-bearing "
        f"for the chosen DN_N_PREF / DN_N_CPU. Either way this "
        f"regression sentinel needs investigation."
    )
