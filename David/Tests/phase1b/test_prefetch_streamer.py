"""Phase 1b §4a — `WeightPrefetchStreamer` execution-layer behavior.

Validates the H2D streaming primitive in isolation:
  * `start(layer_idx, handles)` issues one H2D per non-zero handle on
    `copy_stream`, narrowing slots/sources to the active bucket.
  * `wait(layer_idx)` is a no-op when nothing was started; otherwise blocks
    the compute stream on the layer's copy-done event.
  * `join_after_forward` clears in-capture flags.
  * Slot rotation: layer i and layer i+2 share `slot_idx % 2`; consecutive
    layers occupy distinct slots.
  * Bucket-aware: H2D copy size matches `n_prefetch_by_bucket[current_bucket]`.

Operates on synthetic handles wired with `slot_idx` manually — integration
with `CotsOffloader.wrap_modules` is Step 4b's concern.
"""

import torch
import torch.nn as nn

from vllm.model_executor.offloader.cots import (
    CotsLinearHandle,
    CotsPrefetchBufferPool,
    WeightPrefetchStreamer,
    _complement,
)


HIDDEN = 256       # tiny shapes for fast tests
INTERMEDIATE = 384
DTYPE = torch.bfloat16


def _fake_linear(out_dim, in_dim):
    class _FakeLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(out_dim, in_dim, dtype=DTYPE, device="cuda")
            )

    return _FakeLinear()


def _make_col(layer_idx, n_cpu_per_half=64, half=INTERMEDIATE, in_dim=HIDDEN):
    out_dim = 2 * half
    n_cpu = 2 * n_cpu_per_half
    linear = _fake_linear(out_dim, in_dim)
    base = torch.arange(half - n_cpu_per_half, half, dtype=torch.long)
    cpu_indices = torch.cat([base, base + half])
    h = CotsLinearHandle(
        kind="col", linear=linear, qualified_name=f"layer{layer_idx}.col",
        in_dim=in_dim, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=_complement(cpu_indices, out_dim),
        dtype=DTYPE, merged_partition_sizes=(half, half),
    )
    h.install(torch.device("cuda"))
    h.layer_idx = layer_idx
    h.slot_idx = layer_idx % CotsPrefetchBufferPool.K
    return h


def _make_row(layer_idx, n_cpu=64, in_dim=INTERMEDIATE, out_dim=HIDDEN):
    linear = _fake_linear(out_dim, in_dim)
    cpu_indices = torch.arange(in_dim - n_cpu, in_dim, dtype=torch.long)
    h = CotsLinearHandle(
        kind="row", linear=linear, qualified_name=f"layer{layer_idx}.row",
        in_dim=in_dim, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=_complement(cpu_indices, in_dim),
        dtype=DTYPE,
    )
    h.install(torch.device("cuda"))
    h.layer_idx = layer_idx
    h.slot_idx = layer_idx % CotsPrefetchBufferPool.K
    return h


def _setup_layers(n_layers=4, table=None):
    """Build n_layers worth of (col, row) handle pairs, apply the dispatch
    table to populate per-bucket geometry, and allocate a buffer pool.
    Returns (layer_handles, streamer)."""
    if table is None:
        table = {1: (0.20, 0.20)}

    layer_handles: list[list[CotsLinearHandle]] = []
    flat_handles: list[CotsLinearHandle] = []
    for i in range(n_layers):
        c = _make_col(i)
        r = _make_row(i)
        c.apply_prefetch_split_per_bucket(table)
        r.apply_prefetch_split_per_bucket(table)
        layer_handles.append([c, r])
        flat_handles.extend([c, r])

    CotsPrefetchBufferPool(flat_handles, torch.device("cuda"))
    # Phase 1b row-prefetch fix: allocate the transposed pinned-CPU
    # source for row handles (mirrors `_install_prefetch_machinery`).
    for h in flat_handles:
        if h.kind == "row" and h.max_n_prefetch > 0:
            h.w_row_prefetch_src_t = torch.empty(
                (h.max_n_prefetch, h.out_dim),
                dtype=h.dtype, device="cpu", pin_memory=True,
            )
    streamer = WeightPrefetchStreamer(n_layers=n_layers)
    streamer.set_current_bucket(1, lambda _n: 1)
    return layer_handles, streamer


# ---------------------------------------------------------------------------
def test_start_h2d_copies_correct_bytes():
    """`start(layer_idx, handles)` populates the slot per kind:
      qkv : slot[:n] == w_cpu[:n]                  (contiguous prefix)
      col : slot[:n//2] == w_cpu[:n//2]            (gate prefix)
            slot[n//2:n] == w_cpu[n_cpu_per_half_total : n_cpu_per_half_total+n//2]
                                                   (up prefix — matched-index)
      row : slot[:n, :].T == w_cpu[:, :n]          (Phase 1b transposed
                                                   prefetch source —
                                                   contiguous narrow on
                                                   dim 0)
    """
    layers, streamer = _setup_layers(n_layers=4, table={1: (0.20, 0.20)})

    torch.manual_seed(0)
    for handles in layers:
        for h in handles:
            h.w_cpu.copy_(torch.randn_like(h.w_cpu).to(DTYPE))
            if h.kind == "row" and h.w_row_prefetch_src_t is not None:
                # Mirror the loader closure: transposed prefix.
                m = h.max_n_prefetch
                h.w_row_prefetch_src_t.copy_(
                    h.w_cpu[:, :m].transpose(0, 1).contiguous()
                )

    streamer.start(0, layers[0])
    streamer.copy_stream.synchronize()

    for h in layers[0]:
        n = h.n_prefetch_by_bucket[1]
        slot = h.w_prefetch_slots[h.slot_idx]
        if h.kind == "row":
            # Transposed slot: slot[:n, :] is contiguous and equals
            # w_cpu[:, :n].T.
            dst = slot.narrow(0, 0, n).cpu()
            src = h.w_cpu[:, :n].T.contiguous().cpu()
            assert torch.equal(dst, src), f"row H2D mismatch on {h.qualified_name}"
        elif h.kind == "col":
            # Fixed-max layout: gate at `[0:max_half]`, up at
            # `[max_half:2*max_half]`. Active bucket consumes the
            # per-half prefix `[:n_per_half]` of each region.
            n_per_half = n // 2
            max_half = h.max_n_prefetch // 2
            n_cpu_per_half_total = h.n_cpu // 2
            assert torch.equal(
                slot[:n_per_half, :].cpu(), h.w_cpu[:n_per_half, :].cpu()
            ), f"col gate prefix mismatch on {h.qualified_name}"
            assert torch.equal(
                slot[max_half : max_half + n_per_half, :].cpu(),
                h.w_cpu[
                    n_cpu_per_half_total : n_cpu_per_half_total + n_per_half, :
                ].cpu(),
            ), f"col up prefix mismatch on {h.qualified_name}"
        else:
            dst = slot.narrow(0, 0, n).cpu()
            src = h.w_cpu.narrow(0, 0, n).cpu()
            assert torch.equal(dst, src), f"qkv H2D mismatch on {h.qualified_name}"


def test_start_records_copy_done_event():
    layers, streamer = _setup_layers(n_layers=4)
    # Pre-condition: no events have been recorded; wait_event would crash on
    # an unrecorded event under CUDA's normal rules.
    assert not streamer._event_valid_for_eager[0]
    streamer.start(0, layers[0])
    assert streamer._event_valid_for_eager[0]
    streamer.copy_stream.synchronize()


def test_start_skips_when_no_prefetch():
    """Layer with no handles needing prefetch is a no-op — no event recorded."""
    layers, streamer = _setup_layers(
        n_layers=4, table={1: (0.20, 0.0)}
    )  # f_prefetch=0
    assert all(h.max_n_prefetch == 0 for h in layers[0])
    streamer.start(0, layers[0])
    assert not streamer._event_valid_for_eager[0]


def test_wait_falls_back_to_wait_stream_when_unrecorded():
    """If start was never called for layer i, wait should fall back to
    `wait_stream` (drains all copy_stream work, safe but conservative)."""
    layers, streamer = _setup_layers(n_layers=4)
    # Don't call start; event is unrecorded.
    streamer.wait(2)  # must not raise


def test_wait_uses_event_when_valid():
    layers, streamer = _setup_layers(n_layers=4)
    streamer.start(0, layers[0])
    # wait should succeed without raising — event is valid.
    streamer.wait(0)
    torch.cuda.synchronize()


def test_join_after_forward_clears_in_capture_flags():
    layers, streamer = _setup_layers(n_layers=4)
    # Manually flag layer 1 as in-capture (we're not actually capturing here).
    streamer._prefetch_in_capture[1] = True
    streamer._copy_done_events[1].record(streamer.copy_stream)
    streamer.join_after_forward()
    assert not streamer._prefetch_in_capture[1]


def test_slot_rotation_layer_i_and_i_plus_2_share_slot():
    """K=2: layers 0 and 2 share slot 0; layers 1 and 3 share slot 1."""
    layers, _ = _setup_layers(n_layers=4)
    assert layers[0][0].slot_idx == layers[2][0].slot_idx == 0
    assert layers[1][0].slot_idx == layers[3][0].slot_idx == 1


def test_consecutive_layer_starts_use_distinct_slots():
    """Layer 0's prefetch writes slot 0; layer 1's writes slot 1. Layer 1's
    H2D must NOT clobber slot 0 (still being read by layer 0's compute in
    the real path; here we just verify the addresses are different)."""
    layers, streamer = _setup_layers(n_layers=4)
    h0_col = layers[0][0]
    h1_col = layers[1][0]
    assert h0_col.w_prefetch_slots[h0_col.slot_idx].data_ptr() != \
           h1_col.w_prefetch_slots[h1_col.slot_idx].data_ptr()


def test_set_current_bucket_drives_h2d_size():
    """At a smaller bucket, less is copied — narrow is per-bucket."""
    table = {1: (0.05, 0.05), 64: (0.20, 0.20)}
    layers, streamer = _setup_layers(n_layers=4, table=table)
    h = layers[0][0]
    assert h.n_prefetch_by_bucket[1] < h.n_prefetch_by_bucket[64]

    # Switch to the large bucket and start.
    streamer.set_current_bucket(64, lambda _n: 64)
    torch.manual_seed(0)
    for handles in layers:
        for hh in handles:
            hh.w_cpu.copy_(torch.randn_like(hh.w_cpu).to(DTYPE))

    streamer.start(0, layers[0])
    streamer.copy_stream.synchronize()

    n_64 = h.n_prefetch_by_bucket[64]
    slot_view = h.w_prefetch_slots[h.slot_idx].narrow(0, 0, n_64).cpu()
    src_view = h.w_cpu.narrow(0, 0, n_64).cpu()
    assert torch.equal(slot_view, src_view)


def test_available_rows_records_active_bucket():
    """After start() H2Ds, prefetch_available_rows_in_slot[slot_idx] equals
    the per-half row count for col handles or the total prefix row count
    for qkv/row. Initialized to 0; only the targeted slot is touched."""
    table = {1: (0.05, 0.05), 64: (0.20, 0.20)}
    layers, streamer = _setup_layers(n_layers=4, table=table)
    h = layers[0][0]  # col handle (gate_up)
    # Slots initialize with available_rows == 0 for all K slots.
    assert h.prefetch_available_rows_in_slot == [0, 0]

    streamer.set_current_bucket(64, lambda _n: 64)
    streamer.start(0, layers[0])
    streamer.copy_stream.synchronize()
    n_64 = h.n_prefetch_by_bucket[64]
    expected = n_64 // 2 if h.kind == "col" else n_64
    assert h.prefetch_available_rows_in_slot[h.slot_idx] == expected
    assert h.prefetch_owner_in_slot[h.slot_idx] is h
    # The OTHER slot is still empty.
    assert h.prefetch_available_rows_in_slot[1 - h.slot_idx] == 0


def test_available_rows_preserved_across_no_op_start():
    """When start() is a no-op (active bucket has n_prefetch=0 for all
    handles), the slot's prior available_rows is preserved — the slot
    still holds the previous start()'s data. Operators read the active
    bucket's compute shape regardless; this test only asserts the
    bookkeeping survives."""
    # Two buckets: A=64 has n_prefetch>0, B=1 has n_prefetch=0.
    table = {1: (0.20, 0.0), 64: (0.20, 0.20)}
    layers, streamer = _setup_layers(n_layers=4, table=table)
    h = layers[0][0]
    assert h.n_prefetch_by_bucket[1] == 0
    assert h.n_prefetch_by_bucket[64] > 0

    streamer.set_current_bucket(64, lambda _n: 64)
    streamer.start(0, layers[0])
    streamer.copy_stream.synchronize()
    expected_64 = (
        h.n_prefetch_by_bucket[64] // 2 if h.kind == "col"
        else h.n_prefetch_by_bucket[64]
    )
    assert h.prefetch_available_rows_in_slot[h.slot_idx] == expected_64

    # Switch to bucket 1; start() is a no-op. available_rows preserved.
    streamer.set_current_bucket(1, lambda _n: 1)
    streamer.start(0, layers[0])
    streamer.copy_stream.synchronize()
    assert h.prefetch_available_rows_in_slot[h.slot_idx] == expected_64


def test_slot_metadata_is_shape_group_shared():
    """Handles in the same shape group MUST share the SAME owner /
    available_rows list objects. start() from one handle is visible to
    all other handles sharing the physical slot — this is what catches
    the case where layer 2's start overwrites slot 0 while layer 0's
    per-handle metadata would otherwise still claim ownership."""
    layers, streamer = _setup_layers(n_layers=4)
    h0_col = layers[0][0]
    h2_col = layers[2][0]
    # Layer 0 and layer 2 share slot 0 (= layer_idx % K). Same shape
    # group → SAME list objects.
    assert h0_col.prefetch_owner_in_slot is h2_col.prefetch_owner_in_slot
    assert (
        h0_col.prefetch_available_rows_in_slot
        is h2_col.prefetch_available_rows_in_slot
    )
    # Different shape group (col vs row) → DIFFERENT list objects.
    h0_row = layers[0][1]
    assert h0_col.prefetch_owner_in_slot is not h0_row.prefetch_owner_in_slot


def test_layer2_overwrite_makes_layer0_owner_check_fail():
    """Concrete bug scenario this fix addresses: layer 0 and layer 2 share
    physical slot 0 (K=2). After layer 0's start writes slot 0 (owner=h0),
    layer 2's start overwrites slot 0 (owner=h2). At this point a hypothetical
    re-read by layer 0 would find owner != self — exactly the case the
    operator's `assert prefetch_owner_in_slot[k] is self` catches."""
    layers, streamer = _setup_layers(n_layers=4)
    h0_col = layers[0][0]
    h2_col = layers[2][0]
    assert h0_col.slot_idx == h2_col.slot_idx == 0

    streamer.start(0, layers[0])
    streamer.copy_stream.synchronize()
    assert h0_col.prefetch_owner_in_slot[0] is h0_col

    streamer.start(2, layers[2])
    streamer.copy_stream.synchronize()
    # Same physical slot — owner is now h2_col, even when read via h0_col.
    assert h0_col.prefetch_owner_in_slot[0] is h2_col
    assert h0_col.prefetch_owner_in_slot[0] is not h0_col
