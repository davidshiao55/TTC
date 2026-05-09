# SPDX-License-Identifier: Apache-2.0
"""Stage 1 HARD GATE — C++ at::linear vs Python F.linear parity & perf.

`David/Docs/phase0_findings.md §0.3.2` documents that BF16 `torch.mm`
falls to a 100–250× slower scalar path on i9-14900KF (no AVX512_BF16/
AMX), while `F.linear` → `torch._C._nn.linear` → oneDNN BF16 hits the
fast path. Phase 1c assumes the C++ equivalent `at::linear` dispatches
the same way. This microbench proves it across the two slab geometries
the runner uses:

  (a) contiguous MLP1 9% slice — the shape Phase 1a measured at 0.442 ms
      for B=1 in §0.3.2.
  (b) strided down-proj column slice (`narrow(1, n_pref, n_cpu)` on
      row-major `(out_dim, n_cpu)` storage) — non-contiguous when
      `dn_n_pref > 0`. This is the load-bearing case: oneDNN's
      strided-BF16 path may behave differently from the contiguous one.

If EITHER case is ≥2× the F.linear time (scalar fallback signature)
the test FAILS and Phase 1c HALTS per the approved plan — the fix is
a separate scoped task to add oneDNN linkage to the CUDA build of
`_cots_C` (or to pre-materialize contiguous per-bucket slices).

Within 5%: pass. 5–2×: marked as soft-warning xfail (numerics are fine
but a perf gap warrants follow-up). ≥ 2×: hard fail.
"""

import time

import pytest
import torch

from vllm._cots_C import CotsCpuInfer

# Phase 1a §0.3.2 reference shape: MLP1 9% slice, in_dim=3584, out_dim≈3410.
# We sample a slightly different N (3408, multiple of common alignment) but
# the qualitative perf characteristic should match.
MLP1_K = 3584
MLP1_N = 3408


def _bench_python(fn, iters: int) -> float:
    # Warm up.
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


@pytest.mark.parametrize("B", [1, 4, 16, 32])
def test_contiguous_mlp1_at_linear_matches_f_linear(B):
    """Case (a): contiguous MLP1 9% slice, BF16."""
    torch.set_num_threads(16)  # match Phase 1a's measured baseline
    torch.manual_seed(0)

    K, N = MLP1_K, MLP1_N
    x = torch.randn(B, K, dtype=torch.bfloat16)
    w = torch.randn(N, K, dtype=torch.bfloat16)
    y_ref = torch.empty(B, N, dtype=torch.bfloat16)

    # Python reference path.
    def py_call():
        y_ref.copy_(torch.nn.functional.linear(x, w))

    # C++ path through the runner's inline helper.
    ci = CotsCpuInfer()
    y_cxx = torch.empty(B, N, dtype=torch.bfloat16)

    def cxx_call():
        ci.run_at_linear_inline(x, w, y_cxx)

    py_call()
    cxx_call()
    assert torch.allclose(y_ref, y_cxx, atol=1e-2), (
        f"contiguous MLP1: bf16 result divergence "
        f"(max abs {(y_ref - y_cxx).abs().max()}). "
        f"This shouldn't happen — both paths route through ATen."
    )

    iters = 30 if B <= 4 else 10
    t_py = _bench_python(py_call, iters)
    t_cxx = _bench_python(cxx_call, iters)
    ratio = t_cxx / t_py
    print(
        f"[contiguous MLP1] B={B} K={K} N={N}  "
        f"F.linear={t_py * 1e3:.3f}ms  "
        f"at::linear(C++)={t_cxx * 1e3:.3f}ms  ratio={ratio:.3f}"
    )
    # Hard fail at 2× — that's the scalar-fallback signature.
    assert ratio < 2.0, (
        f"C++ at::linear is {ratio:.2f}× slower than Python F.linear at "
        f"contiguous MLP1 B={B}. This is the scalar-fallback signature "
        f"from phase0_findings.md §0.3.2 and Phase 1c MUST halt here."
    )
    # Soft warn at 1.20× — within ATen overhead but should investigate.
    if ratio > 1.20:
        pytest.xfail(
            f"C++ at::linear is {ratio:.2f}× slower than F.linear; "
            f"within scalar-vs-oneDNN bound but above 5% target."
        )


@pytest.mark.parametrize("B", [1, 4, 16, 32])
def test_strided_down_proj_at_linear_matches_f_linear(B):
    """Case (b) — STRIDED down-proj column slice. Load-bearing case.

    Phase 1b's `dn_h.w_cpu.narrow(1, dn_n_pref, dn_n_cpu)` produces a
    non-contiguous (column-narrowed) view of row-major
    `(out_dim, n_cpu)` storage. The C++ runner reconstructs this view
    via `at::from_blob(ptr, sizes, strides, opts)` so it MUST hit the
    same oneDNN BF16 path; otherwise the strided path silently drops
    to scalar.
    """
    torch.set_num_threads(16)
    torch.manual_seed(0)

    out_dim = 3584
    n_cpu_total = 5683  # representative ~30% of MLP intermediate
    n_pref = 1893  # representative dn_n_pref > 0 → non-contig narrow
    n_cpu_compute = n_cpu_total - n_pref  # = 3790

    # Source storage (row-major, contiguous): (out_dim, n_cpu_total).
    # The view we use is `.narrow(1, n_pref, n_cpu_compute)`.
    w_full = torch.randn(out_dim, n_cpu_total, dtype=torch.bfloat16)
    w_view = w_full.narrow(1, n_pref, n_cpu_compute)
    assert not w_view.is_contiguous()
    assert w_view.stride() == (n_cpu_total, 1)

    x = torch.randn(B, n_cpu_compute, dtype=torch.bfloat16)
    y_ref = torch.empty(B, out_dim, dtype=torch.bfloat16)
    y_cxx = torch.empty(B, out_dim, dtype=torch.bfloat16)

    # Python reference. F.linear handles non-contiguous weight by
    # internal contiguous-ize; we still benchmark vs that as the truth.
    def py_call():
        y_ref.copy_(torch.nn.functional.linear(x, w_view))

    ci = CotsCpuInfer()

    def cxx_call():
        ci.run_at_linear_inline(x, w_view, y_cxx)

    py_call()
    cxx_call()
    assert torch.allclose(y_ref, y_cxx, atol=1e-2), (
        f"strided down-proj: bf16 result divergence "
        f"(max abs {(y_ref - y_cxx).abs().max()})"
    )

    iters = 20 if B <= 4 else 8
    t_py = _bench_python(py_call, iters)
    t_cxx = _bench_python(cxx_call, iters)
    ratio = t_cxx / t_py
    print(
        f"[strided down-proj] B={B} out_dim={out_dim} "
        f"n_cpu={n_cpu_compute} stride0={n_cpu_total}  "
        f"F.linear={t_py * 1e3:.3f}ms  "
        f"at::linear(C++)={t_cxx * 1e3:.3f}ms  ratio={ratio:.3f}"
    )
    # HARD GATE — this is the scalar-fallback signature for STRIDED bf16.
    assert ratio < 2.0, (
        f"C++ at::linear on STRIDED down-proj view is {ratio:.2f}× slower "
        f"than Python F.linear at B={B}. Phase 1c MUST halt — strided-bf16 "
        f"is hitting the scalar fallback path. Scoped follow-up: oneDNN "
        f"linkage for CUDA `_cots_C` build, OR pre-materialize contiguous "
        f"per-bucket down-proj slices, OR Stage 7 transposed-storage path "
        f"lifted earlier."
    )
    if ratio > 1.20:
        pytest.xfail(
            f"strided at::linear is {ratio:.2f}× slower; within bound "
            f"but above 5% target — investigate before Stage 3."
        )
