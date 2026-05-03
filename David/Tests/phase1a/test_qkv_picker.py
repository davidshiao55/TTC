"""Phase 1a §2 — K/V-biased column picker for WQKV.

Validates that the picker assigns columns in priority K+V groups first, then
Q tail. Maps to `weight_offload_design.md §WQKV Column Choice` and the
`kv_biased_cpu_columns` reference in `bench_split_correctness.py`.

Head-group alignment (per `weight_offload_design.md §201-205`): K and V cols
on CPU are whole head groups, never sub-head splits. The picker rounds the
requested n_cpu_cols to the nearest KV head pair below the boundary.
"""

import pytest
import torch

from vllm.model_executor.offloader.cots import (
    _qkv_kv_biased_counts,
    _qkv_kv_biased_indices,
)


# Qwen2.5-7B GQA: 28 Q heads, 4 KV heads, head_dim=128.
HEAD_DIM_7B = 128
Q_SIZE_7B = 28 * HEAD_DIM_7B  # 3584
KV_SIZE_7B = 4 * HEAD_DIM_7B  # 512
TOTAL_7B = Q_SIZE_7B + 2 * KV_SIZE_7B  # 4608


def _pick(n_cpu, *, kv_biased=True):
    return _qkv_kv_biased_indices(
        Q_SIZE_7B, KV_SIZE_7B, n_cpu,
        head_dim=HEAD_DIM_7B, kv_biased=kv_biased,
    )


def test_picker_zero_columns():
    idx = _pick(0)
    assert idx.numel() == 0


def test_picker_full_assignment():
    idx = _pick(TOTAL_7B)
    assert idx.numel() == TOTAL_7B
    # Set equality with [0, total) — every column appears exactly once.
    assert torch.equal(torch.sort(idx).values, torch.arange(TOTAL_7B))


def test_picker_below_kv_boundary_snaps_to_head_pair():
    """At requested f=0.09 (raw n_cpu=414), head-aligned picker snaps to 2 KV
    head pairs = 512 cols (effective f≈11.1%). All assigned indices come from
    the K+V range and form whole heads.
    """
    requested = round(0.09 * TOTAL_7B)  # 414
    idx = _pick(requested)
    # 2 head pairs = 2 * 2 * 128 = 512.
    assert idx.numel() == 2 * 2 * HEAD_DIM_7B
    assert (idx >= Q_SIZE_7B).all(), (
        "below kv boundary: all indices must be in K+V range"
    )
    assert (idx < TOTAL_7B).all()


def test_picker_at_exact_kv_boundary():
    """Requested = full K+V (1024): all of K and all of V on CPU, no Q."""
    n_cpu = 2 * KV_SIZE_7B
    idx = _pick(n_cpu)
    assert idx.numel() == n_cpu
    assert torch.equal(
        torch.sort(idx).values, torch.arange(Q_SIZE_7B, TOTAL_7B)
    )


def test_picker_above_kv_boundary_dips_into_q_tail():
    """f=0.5: 2304 cols. K+V (1024) full + 1280 from Q tail.

    Above the K+V boundary, K and V are full (head-aligned trivially), so
    head-alignment doesn't change the count. Q tail picks the LAST n_q_tail
    cols of Q (Q has no head-boundary requirement at the layer-shape level —
    only K/V pair preservation matters per design `§201-205`).
    """
    n_cpu = round(0.5 * TOTAL_7B)
    n_q_tail_expected = n_cpu - 2 * KV_SIZE_7B
    idx = _pick(n_cpu)
    assert idx.numel() == n_cpu

    # All K+V columns must be present.
    kv_range = set(range(Q_SIZE_7B, TOTAL_7B))
    assert kv_range.issubset(set(idx.tolist()))

    # The remaining n_q_tail picks must come from the END of Q.
    q_picks = sorted(int(i) for i in idx.tolist() if i < Q_SIZE_7B)
    assert len(q_picks) == n_q_tail_expected
    assert q_picks == list(range(Q_SIZE_7B - n_q_tail_expected, Q_SIZE_7B))


def test_picker_unbiased_ablation():
    """kv_biased=False = TP-style proportional, no head alignment. Each shard
    contributes round(f * shard) LAST cols; Q absorbs rounding residual.
    """
    n_cpu = 256
    idx = _pick(n_cpu, kv_biased=False)
    assert idx.numel() == n_cpu
    assert len(set(idx.tolist())) == n_cpu  # disjoint

    n_k = round(n_cpu * KV_SIZE_7B / TOTAL_7B)
    n_v = round(n_cpu * KV_SIZE_7B / TOTAL_7B)
    n_q_tail = n_cpu - n_k - n_v

    expected_q = torch.arange(Q_SIZE_7B - n_q_tail, Q_SIZE_7B)
    expected_k = torch.arange(
        Q_SIZE_7B + KV_SIZE_7B - n_k, Q_SIZE_7B + KV_SIZE_7B
    )
    expected_v = torch.arange(
        Q_SIZE_7B + 2 * KV_SIZE_7B - n_v, Q_SIZE_7B + 2 * KV_SIZE_7B
    )
    assert torch.equal(idx, torch.cat([expected_q, expected_k, expected_v]))


def test_picker_no_duplicates():
    for f in (0.05, 0.09, 0.22, 0.30, 0.50, 0.75):
        n_cpu = round(f * TOTAL_7B)
        idx = _pick(n_cpu)
        assert len(set(idx.tolist())) == idx.numel(), (
            f"duplicates at f={f} (n_cpu={n_cpu}): {idx.tolist()}"
        )


def test_picker_head_alignment():
    """For every f in the planner-relevant range, K, V, AND Q-tail CPU column
    counts are multiples of `head_dim`. K and V are equal (paired KV heads)
    and Q tail is whole heads — required by `weight_offload_design.md` and
    Phase 2's per-head suffix-attention iteration.
    """
    for f in (0.0, 0.03, 0.05, 0.09, 0.15, 0.22, 0.30, 0.50, 0.75, 1.0):
        n_cpu = round(f * TOTAL_7B)
        n_q_tail, n_k, n_v = _qkv_kv_biased_counts(
            Q_SIZE_7B, KV_SIZE_7B, n_cpu, head_dim=HEAD_DIM_7B,
        )
        assert n_k == n_v, f"f={f}: n_k={n_k} != n_v={n_v}"
        assert n_k % HEAD_DIM_7B == 0, (
            f"f={f}: n_k={n_k} not a multiple of head_dim={HEAD_DIM_7B}"
        )
        assert n_v % HEAD_DIM_7B == 0
        assert n_q_tail % HEAD_DIM_7B == 0, (
            f"f={f}: n_q_tail={n_q_tail} not a multiple of head_dim={HEAD_DIM_7B}"
        )
        assert 0 <= n_q_tail <= Q_SIZE_7B


def test_picker_head_alignment_specific_values():
    """Concrete snap targets for the comparison-relevant f values."""
    # f=0.09 → 414 raw → 2 pairs = 512 cols (effective 11.1%)
    _, n_k, n_v = _qkv_kv_biased_counts(
        Q_SIZE_7B, KV_SIZE_7B, 414, head_dim=HEAD_DIM_7B
    )
    assert (n_k, n_v) == (256, 256)
    # f=0.05 → 230 raw → 1 pair = 256 cols (effective 5.55%)
    _, n_k, n_v = _qkv_kv_biased_counts(
        Q_SIZE_7B, KV_SIZE_7B, 230, head_dim=HEAD_DIM_7B
    )
    assert (n_k, n_v) == (128, 128)
    # f=0.22 → 1014 raw → snaps to full K+V at boundary
    n_q_tail, n_k, n_v = _qkv_kv_biased_counts(
        Q_SIZE_7B, KV_SIZE_7B, 1014, head_dim=HEAD_DIM_7B
    )
    assert (n_q_tail, n_k, n_v) == (0, 512, 512)


def test_picker_rejects_out_of_range():
    with pytest.raises(ValueError):
        _pick(-1)
    with pytest.raises(ValueError):
        _pick(TOTAL_7B + 1)
