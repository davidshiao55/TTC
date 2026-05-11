# SPDX-License-Identifier: Apache-2.0
"""Stage 7 GATE — transposed CPU storage for unifying down-proj
storage with the row-prefetch source buffer.

§1b.6 / §1b.7 keep an additional pinned-CPU duplicate
`w_row_prefetch_src_t: (max_n_prefetch, out_dim)` because:

  * row-major `w_cpu: (out_dim, n_cpu)` makes
    `w_cpu.narrow(1, 0, n_pref)` a STRIDED H2D — measured 1.85× slower
    than the contiguous transfer at f_prefetch=0.15 in
    `phase1b_findings.md §1b.7`.
  * The duplicate sidesteps the pitch issue: `(n_pref, out_dim)` rows
    are contiguous and `narrow(0, ...)` stays contiguous.
  * Cost: ~1 GiB pinned at f_prefetch=0.30 on Qwen2.5-7B (§1b.6).

Stage 7's premise: with Phase 1c's native CPU runner, the CPU-compute
side could in principle use the TRANSPOSED storage directly (via
`at::linear(input, w.t())`), eliminating the duplicate. The §1b.6
deferral note explicitly says: "primary CPU storage transpose is
deferred to Phase 1c (native CPU kernel). The duplicate is a
temporary cost paid until the CPU runner swap."

The blocker §1b.6 cites: "PyTorch eager F.linear on a transposed CPU
input was 100× slower in microbenches." That measurement was made
through Python F.linear pre-Phase-1c. This microbench answers the
gate question for the NATIVE runner's at::linear path:

  Does `at::linear` run efficiently when the weight is a transposed-
  storage view (i.e., `.t()` over `(n_cpu, out_dim)` storage), or does
  it fall to a scalar path the way pre-Phase-1c F.linear did?

Three paths compared at Qwen2.5-7B down-proj shape
(out_dim=3584, in_dim=18944):

  (A) BASELINE — Phase 1b row-major + strided narrow:
      w_cpu = (out_dim, n_cpu_total),  w_view = w_cpu.narrow(1, n_pref, n_cpu).
      strides = (n_cpu_total, 1). What the production native runner
      currently uses for the strided down-proj at::linear call.

  (B) STAGE-7 CANDIDATE — transposed storage + .t() view:
      w_cpu_t = (n_cpu_total, out_dim), w_view = w_cpu_t.narrow(0, n_pref, n_cpu).t().
      strides = (1, out_dim). Logical (out_dim, n_cpu) shape but
      column-major-equivalent stride pattern.

  (C) STAGE-7 ALTERNATIVE — transposed storage + materialize-on-use:
      w_view = w_cpu_t.narrow(0, n_pref, n_cpu).t().contiguous().
      Pay a one-shot transpose per submit. Sanity baseline — if (B)
      is too slow, this measures how much the per-submit transpose
      costs vs keeping the duplicate.

Gate:
  * (B) within 5% of (A): unify storage, drop the
    `w_row_prefetch_src_t` duplicate. Saves ~1 GiB at f_prefetch=0.30.
  * (B) within 20% of (A): borderline; weigh against the memory
    savings on a per-workload basis but DO NOT force the change.
  * (B) > 2× of (A): scalar fallback signature — Stage 7 stays
    deferred. Stays with the duplicate.

Plus (C) reports what the per-submit transpose costs as a
fallback.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch

from vllm._cots_C import CotsCpuInfer

# Where to write the per-case JSON summary. Single file accumulates
# entries across all parametrize cases; opened in append mode keyed
# by (B, f_prefetch, f_cpu_store) for downstream tools to consume.
_RESULTS_DIR = Path(__file__).parent / "results"
_RESULTS_JSON = _RESULTS_DIR / "stage7_layout_microbench.json"


def _bench(fn, iters: int) -> float:
    """Median-per-iter wall-clock over `iters` repeats."""
    # Warm up.
    for _ in range(3):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


# Qwen2.5-7B down-proj reference shape.
OUT_DIM = 3584
IN_DIM = 18944


@pytest.mark.stage7_perf
@pytest.mark.parametrize(
    "f_prefetch,f_cpu_store",
    [
        (0.05, 0.10),  # mild collaborative
        (0.15, 0.30),  # Phase 1b §1b.7 collaborative point
        (0.25, 0.30),  # heavy collaborative
    ],
    ids=["fpref_005_fcpu_010", "fpref_015_fcpu_030", "fpref_025_fcpu_030"],
)
@pytest.mark.parametrize("B", [1, 4])
def test_stage7_transposed_storage_at_linear_perf(B, f_prefetch, f_cpu_store):
    """Three-way bench at the down-proj shape. Reports A/B/C
    wall-clock per call; gates on B vs A ratio."""
    torch.set_num_threads(16)
    torch.manual_seed(0)

    n_cpu_total = int(IN_DIM * f_cpu_store)
    n_pref = int(IN_DIM * f_prefetch)
    n_cpu = n_cpu_total - n_pref
    assert n_cpu > 0, (
        f"f_cpu_store={f_cpu_store} f_prefetch={f_prefetch} gives "
        f"n_cpu={n_cpu} ≤ 0; pick a higher cpu_store"
    )

    # Path A: row-major storage with column-narrow view.
    w_rowmaj = torch.randn(OUT_DIM, n_cpu_total, dtype=torch.bfloat16)
    w_view_A = w_rowmaj.narrow(1, n_pref, n_cpu)
    assert w_view_A.shape == (OUT_DIM, n_cpu)
    assert w_view_A.stride() == (n_cpu_total, 1)

    # Path B: transposed storage with `.t()` over a row-narrow view.
    w_transposed = torch.randn(n_cpu_total, OUT_DIM, dtype=torch.bfloat16)
    # Make the bytes match path A so kernel inner loops see equivalent
    # values (avoids accidental perf differences from data distribution).
    w_transposed.copy_(w_rowmaj.t().contiguous())
    w_view_B = w_transposed.narrow(0, n_pref, n_cpu).t()
    assert w_view_B.shape == (OUT_DIM, n_cpu)
    assert w_view_B.stride() == (1, OUT_DIM), (
        f"expected column-major-equivalent (1, OUT_DIM); got {w_view_B.stride()}"
    )

    # Path C: transposed storage + per-submit materialize.
    def make_w_view_C():
        return w_transposed.narrow(0, n_pref, n_cpu).t().contiguous()

    # Path D: transposed storage used DIRECTLY via at::matmul (no .t()
    # re-transpose, no materialize). `w_view_D` has shape
    # `(n_cpu, out_dim)` row-major contiguous (it's a `.narrow(0, …)`
    # of the transposed storage). The compute becomes
    # `y = x @ w_view_D` — standard row-major GEMM, both operands
    # natively contiguous. This avoids the `.t()` re-transpose that
    # F.linear / at::linear inserts.
    w_view_D = w_transposed.narrow(0, n_pref, n_cpu)
    assert w_view_D.shape == (n_cpu, OUT_DIM)
    assert w_view_D.is_contiguous()

    x = torch.randn(B, n_cpu, dtype=torch.bfloat16)
    y = torch.empty(B, OUT_DIM, dtype=torch.bfloat16)
    ci = CotsCpuInfer()

    # Parity check.
    yA = torch.empty(B, OUT_DIM, dtype=torch.bfloat16)
    yB = torch.empty(B, OUT_DIM, dtype=torch.bfloat16)
    yD = torch.empty(B, OUT_DIM, dtype=torch.bfloat16)
    ci.run_at_linear_inline(x, w_view_A, yA)
    ci.run_at_linear_inline(x, w_view_B, yB)
    yD.copy_(torch.matmul(x, w_view_D))
    assert torch.allclose(yA, yB, atol=5e-2), (
        f"A vs B math divergence (max abs {(yA - yB).abs().max()})"
    )
    assert torch.allclose(yA, yD, atol=5e-2), (
        f"A vs D math divergence (max abs {(yA - yD).abs().max()})"
    )

    iters = 20 if B <= 4 else 8

    def call_A():
        ci.run_at_linear_inline(x, w_view_A, y)

    def call_B():
        ci.run_at_linear_inline(x, w_view_B, y)

    def call_C():
        w_view_C = make_w_view_C()
        ci.run_at_linear_inline(x, w_view_C, y)

    def call_D():
        # x @ w_view_D : (B, n_cpu) @ (n_cpu, out_dim) → (B, out_dim).
        # Both operands row-major contiguous; oneDNN should hit the
        # fast BF16 path without any internal transpose.
        y.copy_(torch.matmul(x, w_view_D))

    # ---- Stage 7 probe: oneDNN dispatch alternatives ----------------
    # Path E: pre-materialize row-major (out_dim, n_cpu_compute) slice
    #         once + at::linear. This is the "what if we paid one
    #         contiguous copy per submit and reused Path A's fast
    #         layout" baseline. The materialize cost is part of the
    #         measured wall.
    def call_E():
        w_mat = w_transposed.narrow(0, n_pref, n_cpu).t().contiguous()
        ci.run_at_linear_inline(x, w_mat, y)

    # Path F: torch._C._nn.linear directly on Path D's row-major
    #         transposed weight (passed via .t() into linear which
    #         does y = x @ w.t() internally). This routes through a
    #         different ATen dispatch than at::matmul; sometimes
    #         oneDNN is reached via mkldnn_linear instead of bmm.
    def call_F():
        # at::linear expects weight as (out, in); we pass w_view_D.t()
        # so internally we get x @ w_view_D.t().t() = x @ w_view_D.
        y.copy_(torch.nn.functional.linear(x, w_view_D.t()))

    # Path G: torch.compile(matmul) on Path D inputs. If Inductor
    #         emits a row-major BF16 GEMM, this would prove the
    #         kernel exists.
    @torch.compile(mode="reduce-overhead", dynamic=False)
    def _matmul_g(xx, ww):
        return xx @ ww
    # Warmup the compile (first call JITs).
    _ = _matmul_g(x, w_view_D)

    def call_G():
        y.copy_(_matmul_g(x, w_view_D))

    # Path H: our custom AVX2/FMA BF16 GEMM kernel
    # (csrc/cots/bf16_gemm_transposed.cpp) applied to Path D's
    # row-major (n_cpu, out_dim) = (K, N) transposed-storage layout
    # directly. Imitates oneDNN's f32:bf16 BRGEMM inner load sequence
    # (`vpmovzxwd + vpslld 16` per 8-wide tile) which oneDNN itself
    # does NOT dispatch for the bf16:bf16 case on AVX2 hardware (no
    # AVX2_VNNI_2 / AVX512_BF16 / AMX_BF16 on the target CPU). See
    # bf16_gemm_transposed.cpp for the design notes.
    # Parity check (BF16 noise ~5e-3 relative).
    yH = torch.empty(B, OUT_DIM, dtype=torch.bfloat16)
    ci.run_bf16_gemm_transposed_inline(x, w_view_D, yH)
    # BF16 reduction across K~1700 produces ~1 ULP-of-BF16-at-magnitude
    # accumulation noise. Use rtol-dominant tolerance like the worker's
    # output parity bands.
    rel_H = (yA.float() - yH.float()).abs().max() / (
        yA.float().abs().max() + 1e-6
    )
    assert rel_H < 2e-2, f"A vs H rel divergence {rel_H:.3e} (max abs {(yA - yH).abs().max()})"

    def call_H():
        ci.run_bf16_gemm_transposed_inline(x, w_view_D, y)

    # Single-thread Path A baseline for an apples-to-apples comparison
    # against Path H. The custom kernel is single-threaded; oneDNN's
    # at::linear auto-parallelizes. Reporting both lets the verdict
    # separate "kernel quality" (single-thread head-to-head) from
    # "single-thread vs multi-thread" (which is solvable by adding
    # OMP to the custom kernel as a follow-up).
    _saved_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    t_A_1t = _bench(call_A, iters)
    t_H_1t = _bench(call_H, iters)
    torch.set_num_threads(_saved_threads)
    ratio_H_vs_A_1t = t_H_1t / t_A_1t

    t_A = _bench(call_A, iters)
    t_B = _bench(call_B, iters)
    t_C = _bench(call_C, iters)
    t_D = _bench(call_D, iters)
    t_E = _bench(call_E, iters)
    t_F = _bench(call_F, iters)
    t_G = _bench(call_G, iters)
    t_H = _bench(call_H, iters)
    ratio_B = t_B / t_A
    ratio_C = t_C / t_A
    ratio_D = t_D / t_A
    ratio_E = t_E / t_A
    ratio_F = t_F / t_A
    ratio_G = t_G / t_A
    ratio_H = t_H / t_A
    print(
        f"\n[Stage 7 layout] B={B} f_pref={f_prefetch} f_cpu={f_cpu_store}  "
        f"shape=(out={OUT_DIM},n_cpu={n_cpu})\n"
        f"  (A) row-major+narrow     : {t_A * 1e3:.3f} ms\n"
        f"  (B) transposed+.t() view : {t_B * 1e3:.3f} ms   "
        f"ratio={ratio_B:.3f}\n"
        f"  (C) transposed+materialize: {t_C * 1e3:.3f} ms   "
        f"ratio={ratio_C:.3f}\n"
        f"  (D) transposed+matmul    : {t_D * 1e3:.3f} ms   "
        f"ratio={ratio_D:.3f}\n"
        f"  (E) per-submit mat+at::linear: {t_E * 1e3:.3f} ms   "
        f"ratio={ratio_E:.3f}\n"
        f"  (F) F.linear on .t() view: {t_F * 1e3:.3f} ms   "
        f"ratio={ratio_F:.3f}\n"
        f"  (G) torch.compile matmul : {t_G * 1e3:.3f} ms   "
        f"ratio={ratio_G:.3f}\n"
        f"  (H) custom AVX2 BF16 GEMM: {t_H * 1e3:.3f} ms   "
        f"ratio={ratio_H:.3f}\n"
        f"  -- single-thread parity check --\n"
        f"  (A) 1-thread             : {t_A_1t * 1e3:.3f} ms\n"
        f"  (H) 1-thread             : {t_H_1t * 1e3:.3f} ms   "
        f"ratio_H_vs_A_1t={ratio_H_vs_A_1t:.3f}"
    )

    # Outcome interpretation (does NOT fail the test — both outcomes
    # are valid information for the Stage 7 verdict):
    #
    #   best ratio < 1.05  → Stage 7 unification is a clean win.
    #   1.05 ≤ best ratio < 2.0 → borderline; weigh memory savings.
    #   best ratio ≥ 2.0  → defer (no fast transposed-storage path).
    #
    # Post-Stage-7-B-review-fix empirical result on i9-14900KF: the
    # custom kernel (Path H) wins at every (B, f_prefetch, f_cpu_store)
    # tested, with ratio in the 0.39–0.80 range.
    best_transposed = min(
        ratio_B, ratio_C, ratio_D, ratio_E, ratio_F, ratio_G, ratio_H
    )
    if best_transposed < 1.05:
        verdict = "WIN"
    elif best_transposed < 2.0:
        verdict = "BORDERLINE"
    else:
        verdict = "DEFERRED"
    print(
        f"  best transposed path ratio: {best_transposed:.3f}  "
        f"verdict: {verdict}"
    )

    # JSON artifact. One entry per parametrize case, keyed by the
    # (B, f_prefetch, f_cpu_store) tuple, accumulated across the
    # parametrize sweep. The file is rewritten with the full set
    # each time pytest runs the perf suite; downstream tools should
    # consume this rather than scraping stdout.
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if _RESULTS_JSON.exists():
        try:
            data = json.loads(_RESULTS_JSON.read_text())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    key = f"B{B}_fpref{f_prefetch:.2f}_fcpu{f_cpu_store:.2f}"
    data[key] = {
        "B": B,
        "f_prefetch": f_prefetch,
        "f_cpu_store": f_cpu_store,
        "n_cpu": n_cpu,
        "out_dim": OUT_DIM,
        "in_dim": IN_DIM,
        "times_ms": {
            "A_row_major_narrow": t_A * 1e3,
            "B_transposed_dotT_view": t_B * 1e3,
            "C_transposed_materialize": t_C * 1e3,
            "D_transposed_matmul": t_D * 1e3,
            "E_per_submit_materialize_at_linear": t_E * 1e3,
            "F_F_linear_on_dotT": t_F * 1e3,
            "G_torch_compile_matmul": t_G * 1e3,
            "H_custom_avx2_bf16_gemm": t_H * 1e3,
            "A_1thread": t_A_1t * 1e3,
            "H_1thread": t_H_1t * 1e3,
        },
        "ratios_vs_A": {
            "B": ratio_B, "C": ratio_C, "D": ratio_D, "E": ratio_E,
            "F": ratio_F, "G": ratio_G, "H": ratio_H,
            "H_vs_A_1t": ratio_H_vs_A_1t,
        },
        "verdict": verdict,
        "best_transposed_ratio": best_transposed,
        "num_threads": torch.get_num_threads(),
        "num_iters": iters,
    }
    _RESULTS_JSON.write_text(json.dumps(data, indent=2))
