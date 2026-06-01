"""Phase 1b §2 — Per-bucket prefetch geometry on `CotsLinearHandle`.

Validates `apply_prefetch_split_per_bucket` for the COTS linear roles
(qkv / mlp_gate_up / mlp_down). Maps to
`weight_offload_design.md §Tensor Granularity` and
`planner_design.md §4.2`.

Per-bucket geometry must satisfy:
  * `n_prefetch + n_cpu_compute == n_cpu` (every CPU-stored byte dispatched).
  * `prefetch_indices ∪ cpu_compute_indices == cpu_indices` (set equality).
  * Disjoint subsets — no double-routing of any output column.
  * For `qkv`: prefetch is the contiguous prefix of cpu_indices in
    `[Q_tail | K | V]` order; spills into K/V at high f_prefetch.
  * For `mlp_gate_up`: prefetch picks the FIRST `n_prefetch_per_half` of each
    half's CPU range — preserves the matched-index invariant with paired
    `mlp_down`.
  * For matched MLP output-split/input-split pair under uniform `f_prefetch`:
    `gu.n_prefetch_by_bucket[b] // 2 == dn.n_prefetch_by_bucket[b]`.
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.offloader.cots import (
    CotsLinearHandle,
    MLP_DOWN_ROLE,
    MLP_GATE_UP_ROLE,
    QKV_ROLE,
    _complement,
    _qkv_kv_biased_counts,
    _qkv_kv_biased_indices,
)


# Qwen2.5-7B GQA shapes for realistic K/V-biased picker tests.
HEAD_DIM_7B = 128
Q_SIZE_7B = 28 * HEAD_DIM_7B  # 3584
KV_SIZE_7B = 4 * HEAD_DIM_7B  # 512
QKV_OUT_7B = Q_SIZE_7B + 2 * KV_SIZE_7B  # 4608
HIDDEN_7B = 3584
INTERMEDIATE_7B = 18944


def _fake_linear(out_dim: int, in_dim: int, dtype=torch.bfloat16) -> nn.Module:
    class _FakeLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(out_dim, in_dim, dtype=dtype, device="cuda")
            )

    return _FakeLinear()


def _make_qkv(n_cpu_raw, q_size=Q_SIZE_7B, kv_size=KV_SIZE_7B,
              head_dim=HEAD_DIM_7B, in_dim=HIDDEN_7B):
    """Build a qkv handle with `n_cpu` snapped to head boundaries — the
    picker's actual snap, mirroring what `for_qkv` does at production."""
    out_dim = q_size + 2 * kv_size
    n_q_tail, n_k, n_v = _qkv_kv_biased_counts(
        q_size, kv_size, n_cpu_raw, head_dim=head_dim
    )
    n_cpu = n_q_tail + n_k + n_v
    linear = _fake_linear(out_dim, in_dim)
    cpu_indices = _qkv_kv_biased_indices(q_size, kv_size, n_cpu, head_dim=head_dim)
    handle = CotsLinearHandle(
        role=QKV_ROLE,
        linear=linear, qualified_name="qkv.test",
        in_dim=in_dim, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=_complement(cpu_indices, out_dim),
        dtype=torch.bfloat16, q_size=q_size, kv_size=kv_size, head_dim=head_dim,
    )
    handle.install(torch.device("cuda"))
    return handle


def _make_col(n_cpu_per_half, half=INTERMEDIATE_7B, in_dim=HIDDEN_7B):
    out_dim = 2 * half
    n_cpu = 2 * n_cpu_per_half
    linear = _fake_linear(out_dim, in_dim)
    base = torch.arange(half - n_cpu_per_half, half, dtype=torch.long)
    cpu_indices = torch.cat([base, base + half])
    handle = CotsLinearHandle(
        role=MLP_GATE_UP_ROLE,
        linear=linear, qualified_name="col.test",
        in_dim=in_dim, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=_complement(cpu_indices, out_dim),
        dtype=torch.bfloat16, merged_partition_sizes=(half, half),
    )
    handle.install(torch.device("cuda"))
    return handle


def _make_row(n_cpu, in_dim=INTERMEDIATE_7B, out_dim=HIDDEN_7B):
    linear = _fake_linear(out_dim, in_dim)
    cpu_indices = torch.arange(in_dim - n_cpu, in_dim, dtype=torch.long)
    handle = CotsLinearHandle(
        role=MLP_DOWN_ROLE,
        linear=linear, qualified_name="row.test",
        in_dim=in_dim, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=_complement(cpu_indices, in_dim),
        dtype=torch.bfloat16,
    )
    handle.install(torch.device("cuda"))
    return handle


def _check_invariants(handle, bucket):
    """Common per-bucket invariants: n sum, set equality, disjointness."""
    n_pref = handle.n_prefetch_by_bucket[bucket]
    n_cpu_compute = handle.n_cpu_compute_by_bucket[bucket]
    pref_idx = handle.prefetch_indices_cuda_by_bucket[bucket].cpu()
    cpu_idx = handle.cpu_compute_indices_cuda_by_bucket[bucket].cpu()

    assert n_pref + n_cpu_compute == handle.n_cpu, (
        f"sum mismatch: n_pref={n_pref} + n_cpu_compute={n_cpu_compute} "
        f"!= n_cpu={handle.n_cpu}"
    )
    assert pref_idx.numel() == n_pref, "prefetch index count mismatch"
    assert cpu_idx.numel() == n_cpu_compute, "cpu_compute index count mismatch"

    # Set equality with cpu_indices, disjoint subsets.
    union = torch.cat([pref_idx, cpu_idx])
    assert torch.equal(
        torch.sort(union).values, torch.sort(handle.cpu_indices).values
    ), "prefetch ∪ cpu_compute != cpu_indices"
    assert len(set(pref_idx.tolist()) & set(cpu_idx.tolist())) == 0, (
        "prefetch and cpu_compute indices overlap"
    )


# ---------------------------------------------------------------------------
# QKV — split invariants
# ---------------------------------------------------------------------------
def test_qkv_split_invariants_below_kv_boundary():
    """f_cpu_store=0.22 → exactly K+V on CPU (n_q_tail=0). Prefetch consumes
    rows from cpu_indices in order [K | V]."""
    n_cpu = 2 * KV_SIZE_7B  # exactly all K+V
    handle = _make_qkv(n_cpu)
    assert handle.n_q_tail == 0
    table = {16: (0.22, 0.05), 64: (0.22, 0.10)}
    handle.apply_prefetch_split_per_bucket(table)
    for b in table:
        _check_invariants(handle, b)


def test_qkv_split_invariants_above_kv_boundary():
    """f_cpu_store=0.50 → K+V + Q-tail on CPU. Prefetch is the contiguous
    prefix of cpu_indices (Q_tail first, then K, then V)."""
    f_cpu = 0.50
    n_cpu_raw = round(f_cpu * QKV_OUT_7B)
    handle = _make_qkv(n_cpu_raw)
    assert handle.n_q_tail > 0
    assert handle.n_k > 0 and handle.n_v > 0

    table = {16: (0.30, 0.20), 64: (0.10, 0.40)}
    handle.apply_prefetch_split_per_bucket(table)

    for b in table:
        _check_invariants(handle, b)


def test_qkv_pure_prefetch_consumes_all_cpu_rows():
    """f_cpu == 0 with f_prefetch > 0: pure-prefetch fast path consumes the
    full snapped n_cpu; CPU compute is exactly empty."""
    handle = _make_qkv(round(0.30 * QKV_OUT_7B))
    handle.apply_prefetch_split_per_bucket({1: (0.0, 0.30)})
    assert handle.n_prefetch_by_bucket[1] == handle.n_cpu
    assert handle.n_cpu_compute_by_bucket[1] == 0


# ---------------------------------------------------------------------------
# Col (MergedCol gate_up)
# ---------------------------------------------------------------------------
def test_col_split_picks_first_n_per_half():
    """Prefetch picks the FIRST n_prefetch_per_half of each half's CPU range.
    Total n_prefetch = 2 * n_prefetch_per_half."""
    n_cpu_per_half = 1024
    handle = _make_col(n_cpu_per_half)

    # f_prefetch=0.05 → n_pref_per_half ≈ 947 at half=18944
    table = {16: (0.05, 0.05)}
    handle.apply_prefetch_split_per_bucket(table)
    _check_invariants(handle, 16)

    n_pref = handle.n_prefetch_by_bucket[16]
    assert n_pref == 2 * (n_pref // 2), "n_prefetch must be even (per-half)"
    n_per_half = n_pref // 2
    # Prefetch indices are the FIRST n_per_half of each half's cpu_indices range.
    pref_idx = handle.prefetch_indices_cuda_by_bucket[16].cpu()
    expected = torch.cat([
        handle.cpu_indices[:n_per_half],
        handle.cpu_indices[n_cpu_per_half : n_cpu_per_half + n_per_half],
    ])
    assert torch.equal(pref_idx, expected)


def test_col_caps_at_n_cpu_per_half():
    """f_prefetch large enough to demand more than n_cpu_per_half — capped."""
    n_cpu_per_half = 256
    handle = _make_col(n_cpu_per_half)
    handle.apply_prefetch_split_per_bucket({1: (0.0, 1.0)})
    # Capped: every CPU row prefetched.
    assert handle.n_prefetch_by_bucket[1] == handle.n_cpu
    assert handle.n_cpu_compute_by_bucket[1] == 0


# ---------------------------------------------------------------------------
# Row (down_proj)
# ---------------------------------------------------------------------------
def test_row_split_picks_first_n():
    n_cpu = 1024
    handle = _make_row(n_cpu)
    table = {16: (0.05, 0.05)}
    handle.apply_prefetch_split_per_bucket(table)
    _check_invariants(handle, 16)

    n_pref = handle.n_prefetch_by_bucket[16]
    pref_idx = handle.prefetch_indices_cuda_by_bucket[16].cpu()
    assert torch.equal(pref_idx, handle.cpu_indices[:n_pref])


def test_row_caps_at_n_cpu():
    handle = _make_row(256)
    handle.apply_prefetch_split_per_bucket({1: (0.0, 1.0)})
    assert handle.n_prefetch_by_bucket[1] == handle.n_cpu
    assert handle.n_cpu_compute_by_bucket[1] == 0


# ---------------------------------------------------------------------------
# Matched-index invariant: col gate_up + paired row down
# ---------------------------------------------------------------------------
def test_matched_index_col_row_under_uniform_dispatch():
    """Under uniform `f_prefetch` applied to both MLP1 (col) and MLP2 (row),
    the prefetched intermediate-dim index sets must match: MLP1's prefetched
    output cols (per half) == MLP2's prefetched input cols.
    """
    half = INTERMEDIATE_7B  # 18944
    n_cpu_per_half = 1024
    f_prefetch = 0.03  # n_per_half ≈ 568

    gu = _make_col(n_cpu_per_half=n_cpu_per_half)
    dn = _make_row(n_cpu=n_cpu_per_half, in_dim=half)
    table = {16: (0.0, f_prefetch)}
    gu.apply_prefetch_split_per_bucket(table)
    dn.apply_prefetch_split_per_bucket(table)

    # Phase 1b matched-index check: gu.n_prefetch (total over both halves) ==
    # 2 * dn.n_prefetch.
    assert gu.n_prefetch_by_bucket[16] == 2 * dn.n_prefetch_by_bucket[16]

    # gate's prefetched intermediate indices == dn's prefetched intermediate cols.
    gu_pref = gu.prefetch_indices_cuda_by_bucket[16].cpu()
    n_pref_per_half = gu.n_prefetch_by_bucket[16] // 2
    gate_intermediate_idx = gu_pref[:n_pref_per_half]  # output cols 0..half
    dn_pref = dn.prefetch_indices_cuda_by_bucket[16].cpu()
    assert torch.equal(gate_intermediate_idx, dn_pref)


# ---------------------------------------------------------------------------
# max_n_prefetch
# ---------------------------------------------------------------------------
def test_max_n_prefetch_tracks_largest_bucket():
    handle = _make_row(1024)
    table = {1: (0.10, 0.02), 16: (0.05, 0.05), 64: (0.0, 0.10)}
    handle.apply_prefetch_split_per_bucket(table)
    expected = max(handle.n_prefetch_by_bucket.values())
    assert handle.max_n_prefetch == expected


def test_empty_table_yields_zero_max_n_prefetch():
    handle = _make_row(1024)
    handle.apply_prefetch_split_per_bucket({})
    assert handle.max_n_prefetch == 0
    assert handle.n_prefetch_by_bucket == {}


# ---------------------------------------------------------------------------
# Phase 1a regression: f_prefetch=0 across all buckets degenerates correctly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("role", [QKV_ROLE, MLP_GATE_UP_ROLE, MLP_DOWN_ROLE])
def test_zero_f_prefetch_yields_no_prefetch(role):
    """f_prefetch=0 → n_prefetch=0, cpu_compute_indices == cpu_indices.
    Phase 1a regression sentinel."""
    if role == QKV_ROLE:
        handle = _make_qkv(round(0.10 * QKV_OUT_7B))
    elif role == MLP_GATE_UP_ROLE:
        handle = _make_col(n_cpu_per_half=1024)
    else:
        handle = _make_row(n_cpu=1024)

    table = {1: (0.10, 0.0), 16: (0.10, 0.0)}
    handle.apply_prefetch_split_per_bucket(table)
    for b in table:
        assert handle.n_prefetch_by_bucket[b] == 0
        assert handle.n_cpu_compute_by_bucket[b] == handle.n_cpu
        cpu_idx = handle.cpu_compute_indices_cuda_by_bucket[b].cpu()
        assert torch.equal(cpu_idx, handle.cpu_indices)
