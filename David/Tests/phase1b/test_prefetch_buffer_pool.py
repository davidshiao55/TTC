"""Phase 1b §3 — `CotsPrefetchBufferPool` slot allocation.

Validates K=2 slot rotation and per-handle slot shape matching `w_cpu`'s
prefetched-portion layout. Per-kind shape:
  col / qkv : `(max_n_prefetch, in_dim)`     — narrow(0, ...) contiguous
  row       : `(max_n_prefetch, out_dim)`    — narrow(0, ...) contiguous
                                               (Phase 1b row-prefetch fix:
                                                transposed vs `w_cpu` to
                                                avoid pitched H2D)
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.offloader.cots import (
    CotsLinearHandle,
    CotsPrefetchBufferPool,
    _complement,
    _qkv_kv_biased_counts,
    _qkv_kv_biased_indices,
)


HEAD_DIM = 128
Q_SIZE = 28 * HEAD_DIM        # 3584
KV_SIZE = 4 * HEAD_DIM         # 512
QKV_OUT = Q_SIZE + 2 * KV_SIZE  # 4608
HIDDEN = 3584
INTERMEDIATE = 18944
DTYPE = torch.bfloat16


def _fake_linear(out_dim, in_dim):
    class _FakeLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(out_dim, in_dim, dtype=DTYPE, device="cuda")
            )

    return _FakeLinear()


def _qkv_handle(n_cpu_raw=round(0.30 * QKV_OUT)):
    n_q_tail, n_k, n_v = _qkv_kv_biased_counts(
        Q_SIZE, KV_SIZE, n_cpu_raw, head_dim=HEAD_DIM
    )
    n_cpu = n_q_tail + n_k + n_v
    linear = _fake_linear(QKV_OUT, HIDDEN)
    cpu_indices = _qkv_kv_biased_indices(Q_SIZE, KV_SIZE, n_cpu, head_dim=HEAD_DIM)
    h = CotsLinearHandle(
        kind="qkv", linear=linear, qualified_name="qkv",
        in_dim=HIDDEN, out_dim=QKV_OUT, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=_complement(cpu_indices, QKV_OUT),
        dtype=DTYPE, q_size=Q_SIZE, kv_size=KV_SIZE, head_dim=HEAD_DIM,
    )
    h.install(torch.device("cuda"))
    return h


def _col_handle(n_cpu_per_half=1024, half=INTERMEDIATE):
    out_dim = 2 * half
    n_cpu = 2 * n_cpu_per_half
    linear = _fake_linear(out_dim, HIDDEN)
    base = torch.arange(half - n_cpu_per_half, half, dtype=torch.long)
    cpu_indices = torch.cat([base, base + half])
    h = CotsLinearHandle(
        kind="col", linear=linear, qualified_name="col",
        in_dim=HIDDEN, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=_complement(cpu_indices, out_dim),
        dtype=DTYPE, merged_partition_sizes=(half, half),
    )
    h.install(torch.device("cuda"))
    return h


def _row_handle(n_cpu=1024, in_dim=INTERMEDIATE):
    out_dim = HIDDEN
    linear = _fake_linear(out_dim, in_dim)
    cpu_indices = torch.arange(in_dim - n_cpu, in_dim, dtype=torch.long)
    h = CotsLinearHandle(
        kind="row", linear=linear, qualified_name="row",
        in_dim=in_dim, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=_complement(cpu_indices, in_dim),
        dtype=DTYPE,
    )
    h.install(torch.device("cuda"))
    return h


def test_pool_allocates_k2_slots_per_handle():
    """Each non-empty handle gets exactly K=2 slot views."""
    qkv = _qkv_handle()
    col = _col_handle()
    row = _row_handle()
    table = {1: (0.10, 0.10)}
    for h in (qkv, col, row):
        h.apply_prefetch_split_per_bucket(table)

    pool = CotsPrefetchBufferPool([qkv, col, row], torch.device("cuda"))

    for h in (qkv, col, row):
        assert len(h.w_prefetch_slots) == CotsPrefetchBufferPool.K == 2


def test_slot_shape_matches_kind_layout():
    """col/qkv slot is `(max_n_prefetch, in_dim)`; row slot is
    `(max_n_prefetch, out_dim)` — Phase 1b row-prefetch fix transposes
    the row slot so contiguous H2D from `w_row_prefetch_src_t` lands
    on a contiguous destination."""
    qkv = _qkv_handle()
    col = _col_handle()
    row = _row_handle()
    table = {1: (0.10, 0.10)}
    for h in (qkv, col, row):
        h.apply_prefetch_split_per_bucket(table)

    CotsPrefetchBufferPool([qkv, col, row], torch.device("cuda"))

    for h in (qkv, col):
        assert tuple(h.w_prefetch_slots[0].shape) == (h.max_n_prefetch, h.in_dim)
        assert tuple(h.w_prefetch_slots[1].shape) == (h.max_n_prefetch, h.in_dim)
    assert tuple(row.w_prefetch_slots[0].shape) == (row.max_n_prefetch, row.out_dim)
    assert tuple(row.w_prefetch_slots[1].shape) == (row.max_n_prefetch, row.out_dim)


def test_slots_are_distinct_buffers():
    """Slot 0 and slot 1 must occupy disjoint memory — slot rotation is the
    point. Write to one, read the other, expect no crosstalk."""
    qkv = _qkv_handle()
    qkv.apply_prefetch_split_per_bucket({1: (0.10, 0.10)})
    CotsPrefetchBufferPool([qkv], torch.device("cuda"))

    s0, s1 = qkv.w_prefetch_slots
    s0.fill_(1.0)
    s1.fill_(2.0)
    assert not torch.allclose(s0, s1)
    assert s0.data_ptr() != s1.data_ptr()


def test_total_bytes_matches_layout():
    """K * Σ_unique_shape (slot_numel) * dtype_bytes — slots are shared
    across handles with the same (kind, slot_shape). Three distinct shapes
    here (qkv / col / row) → three groups, each contributing K=2 slots."""
    qkv = _qkv_handle()
    col = _col_handle()
    row = _row_handle()
    table = {1: (0.10, 0.10)}
    for h in (qkv, col, row):
        h.apply_prefetch_split_per_bucket(table)

    pool = CotsPrefetchBufferPool([qkv, col, row], torch.device("cuda"))
    elem_bytes = torch.empty(0, dtype=DTYPE).element_size()  # 2 for bf16

    expected = 2 * (
        (qkv.max_n_prefetch * qkv.in_dim)
        + (col.max_n_prefetch * col.in_dim)
        + (row.out_dim * row.max_n_prefetch)
    ) * elem_bytes
    assert pool.total_bytes == expected


def test_slots_are_shared_across_layers_with_same_shape():
    """Two handles of the same (kind, slot_shape) — e.g., qkv handles from
    different decoder layers — must SHARE the K slot views. Slot rotation
    happens at the handle level via `slot_idx = layer_idx % K`. This is
    the load-bearing shape-grouping behavior that mirrors native
    `StaticBufferPool`. Without it, pool size scales with N_layers and
    OOMs at high f_prefetch."""
    qkv0 = _qkv_handle()
    qkv1 = _qkv_handle()
    table = {1: (0.10, 0.10)}
    qkv0.apply_prefetch_split_per_bucket(table)
    qkv1.apply_prefetch_split_per_bucket(table)
    # Same (kind, max_n_prefetch, in_dim) → must share slots.
    assert qkv0.kind == qkv1.kind
    assert qkv0.max_n_prefetch == qkv1.max_n_prefetch
    assert qkv0.in_dim == qkv1.in_dim

    pool = CotsPrefetchBufferPool([qkv0, qkv1], torch.device("cuda"))

    # Both handles' w_prefetch_slots reference the SAME physical buffers.
    for k in range(CotsPrefetchBufferPool.K):
        assert qkv0.w_prefetch_slots[k].data_ptr() \
               == qkv1.w_prefetch_slots[k].data_ptr(), (
            f"slot {k} should be shared between same-shape handles"
        )

    # Pool size = K × ONE shape's slot bytes (NOT × num_handles).
    elem_bytes = torch.empty(0, dtype=DTYPE).element_size()
    expected = 2 * qkv0.max_n_prefetch * qkv0.in_dim * elem_bytes
    assert pool.total_bytes == expected, (
        f"shared-slot pool should be K × per-shape, "
        f"not K × per-handle (got {pool.total_bytes}, expected {expected})"
    )


def test_pool_size_independent_of_layer_count():
    """At f_prefetch fixed, allocating a pool over N copies of the same
    handle shape must produce the SAME `total_bytes` regardless of N
    (slots are shared). Repro for the f=0.50 OOM at 28 layers."""
    table = {1: (0.10, 0.10)}

    def _pool_for_n_handles(n):
        handles = []
        for _ in range(n):
            h = _row_handle()
            h.apply_prefetch_split_per_bucket(table)
            handles.append(h)
        return CotsPrefetchBufferPool(handles, torch.device("cuda")).total_bytes

    bytes_1 = _pool_for_n_handles(1)
    bytes_28 = _pool_for_n_handles(28)
    assert bytes_1 == bytes_28, (
        f"pool size should not scale with N_handles when shapes match: "
        f"got {bytes_1} for 1 handle, {bytes_28} for 28 handles"
    )


def test_pool_skips_handles_with_zero_prefetch():
    """Handle with `max_n_prefetch == 0` gets an empty slot list and
    contributes 0 bytes."""
    qkv_offload = _qkv_handle()
    col_no_prefetch = _col_handle()

    qkv_offload.apply_prefetch_split_per_bucket({1: (0.10, 0.10)})
    col_no_prefetch.apply_prefetch_split_per_bucket({1: (0.10, 0.0)})  # f_pref=0
    assert col_no_prefetch.max_n_prefetch == 0

    pool = CotsPrefetchBufferPool(
        [qkv_offload, col_no_prefetch], torch.device("cuda")
    )

    assert col_no_prefetch.w_prefetch_slots == []
    assert len(qkv_offload.w_prefetch_slots) == 2
    # Pool's bytes account only for qkv_offload.
    elem_bytes = torch.empty(0, dtype=DTYPE).element_size()
    expected = 2 * qkv_offload.max_n_prefetch * qkv_offload.in_dim * elem_bytes
    assert pool.total_bytes == expected


def test_empty_pool_is_legal():
    """If every handle has max_n_prefetch == 0, the pool allocates nothing
    and binds no slots."""
    col = _col_handle()
    col.apply_prefetch_split_per_bucket({1: (0.10, 0.0)})  # all-cpu, no prefetch
    pool = CotsPrefetchBufferPool([col], torch.device("cuda"))
    assert pool.total_bytes == 0
    assert col.w_prefetch_slots == []


def test_runtime_narrow_works_at_smaller_buckets():
    """Phase 1b runtime path narrows the slot view to the active bucket's
    `n_prefetch_by_bucket[b]`. Smoke-test that narrow returns a view of the
    expected shape for both kinds of layout."""
    qkv = _qkv_handle()
    row = _row_handle()
    # Build a table with one large bucket and one small bucket — qkv guard
    # caps n_pref at n_q_tail; row picks first n_pref of input cols.
    table = {1: (0.05, 0.05), 64: (0.02, 0.02)}
    for h in (qkv, row):
        h.apply_prefetch_split_per_bucket(table)

    CotsPrefetchBufferPool([qkv, row], torch.device("cuda"))

    # Narrow at smaller bucket (1) for col/qkv → narrow on dim 0.
    n_pref_64 = qkv.n_prefetch_by_bucket[64]
    narrowed = qkv.w_prefetch_slots[0].narrow(0, 0, n_pref_64)
    assert tuple(narrowed.shape) == (n_pref_64, qkv.in_dim)

    # Row → narrow on dim 0 (Phase 1b row-prefetch fix: transposed slot).
    n_pref_64_row = row.n_prefetch_by_bucket[64]
    narrowed_row = row.w_prefetch_slots[0].narrow(0, 0, n_pref_64_row)
    assert tuple(narrowed_row.shape) == (n_pref_64_row, row.out_dim)


def test_h2d_copy_into_slot_smoke():
    """End-to-end smoke: pinned CPU source → GPU slot via copy_. col/qkv
    use the contiguous narrow(0, ...) on `w_cpu`. Row uses the
    transposed `w_row_prefetch_src_t` source (Phase 1b row-prefetch
    fix); both source and slot are narrowed on dim 0, both contiguous."""
    col = _col_handle(n_cpu_per_half=64, half=128)  # small for fast test
    row = _row_handle(n_cpu=64, in_dim=128)
    table = {1: (0.20, 0.10)}
    for h in (col, row):
        h.apply_prefetch_split_per_bucket(table)

    CotsPrefetchBufferPool([col, row], torch.device("cuda"))
    # Transposed source — populated manually in this test (offloader
    # would do this in `_install_prefetch_machinery` + the row loader).
    row.w_row_prefetch_src_t = torch.empty(
        (row.max_n_prefetch, row.out_dim),
        dtype=row.dtype, device="cpu", pin_memory=True,
    )

    torch.manual_seed(0)
    # Fill w_cpu with deterministic values.
    col.w_cpu.copy_(torch.randn_like(col.w_cpu).to(DTYPE))
    row.w_cpu.copy_(torch.randn_like(row.w_cpu).to(DTYPE))
    # Mirror the loader: transposed prefix.
    row.w_row_prefetch_src_t.copy_(
        row.w_cpu[:, : row.max_n_prefetch].transpose(0, 1).contiguous()
    )

    # H2D into slot 0. Both narrows are on dim 0 → contiguous.
    col_n = col.n_prefetch_by_bucket[1]
    row_n = row.n_prefetch_by_bucket[1]
    col.w_prefetch_slots[0].narrow(0, 0, col_n).copy_(
        col.w_cpu.narrow(0, 0, col_n)
    )
    row.w_prefetch_slots[0].narrow(0, 0, row_n).copy_(
        row.w_row_prefetch_src_t.narrow(0, 0, row_n)
    )
    torch.cuda.synchronize()

    # Bit-equality (no compute, just memcpy).
    col_dst = col.w_prefetch_slots[0].narrow(0, 0, col_n).cpu()
    col_src = col.w_cpu.narrow(0, 0, col_n).cpu()
    assert torch.equal(col_dst, col_src)

    # Transposed slot back to (out_dim, row_n) for comparison with w_cpu.
    row_dst = row.w_prefetch_slots[0].narrow(0, 0, row_n).cpu().T.contiguous()
    row_src = row.w_cpu.narrow(1, 0, row_n).contiguous()
    assert torch.equal(row_dst, row_src)
