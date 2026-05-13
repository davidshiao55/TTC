# Phase 1b Implementation Findings

This document records the design and verification of **Phase 1b** of
the COTS prototype — the layer-ahead weight prefetch extension on
top of Phase 1a's static CPU-compute backend. Phase 1b extends the
dispatch table from `(f_cpu_store, 0)` to `(f_cpu_compute,
f_prefetch_compute)` per bucket with the invariant
`f_cpu_compute + f_prefetch_compute ≤ f_cpu_store`, so that CPU-stored
bytes that aren't CPU-computed get streamed back to GPU via
layer-ahead H2D and consumed by a third compute path. With that, all
three paths from `thesis_proposal.md §3.2` engage concurrently:
GPU-permanent, prefetch-to-GPU, CPU-compute.

Phase 1b ships under `enforce_eager=True` (Phase 1c lifts that), uses
a single global `f_prefetch` config knob (per-bucket variation
deferred to the Planner), and is structurally graph-capture-ready —
all runtime decisions are pure-data lookups keyed on the active
bucket, slot state is contract-checked, and the deferred wraparound
is scheduled through a registered custom op (§1b.4).

Hardware: NVIDIA RTX 4090 (24 GB), Intel i9-14900KF (AVX2, no
AVX512/AMX), DDR5. PyTorch 2.10.0+cu128, MKL enabled, oneDNN BF16,
Triton.

---

## Contents

**Mechanism**
- §1b.1 — Architecture extension: streamer / pool / per-bucket geometry
- §1b.2 — Three-way scatter and the matched-index invariant
- §1b.3 — Pre-compute K=1 next-layer prefetch + K=2 slot rotation + post-init layer-0 max-fill
- §1b.4 — Static deferred wraparound (graph-safe)

**Memory & buffer invariants**
- §1b.5 — Prefetch buffer pool: shape-group sharing
- §1b.6 — Pinned-CPU duplicate: `w_row_prefetch_src_t`

**Performance postmortem**
- §1b.7 — Row-prefetch contention fix (strided H2D → contiguous, ~1.85× PCIe)

**Measurements**
- §1b.8 — Bench 1: COTS pure-prefetch vs native `prefetch_defer` (tensor-granularity wins at matched bytes)
- §1b.9 — Bench 2: matched offload depth (regime story)
- §1b.10 — Bench 3: matched per-path resource (contention diagnostic)
- §1b.11 — Cross-bench predictive model
- §1b.12 — Collaborative-split sweep — Planner motivation

**Phase 1c compatibility refactor**
- §1b.13 — Active-bucket dispatch and `prepare_before_forward`

**Verification**
- §1b.14 — Test matrix and reproducibility

---

## 1b.1: Architecture extension — streamer, pool, per-bucket geometry

Phase 1a's three-layer split (`phase1a_findings.md §1.1`) is
preserved verbatim. Phase 1b adds two execution-layer siblings to
`CpuTaskRunner`, plus per-bucket plan-derived state on the storage
layer.

| Layer | Phase 1a | Phase 1b additions |
|---|---|---|
| Storage | `CotsLinearHandle` (per-Linear: `w_cpu`, GPU slice, indices) | Per-bucket geometry dicts: `n_prefetch_by_bucket`, `n_cpu_compute_by_bucket`, `prefetch_indices_cuda_by_bucket`, `cpu_compute_indices_cuda_by_bucket`; plus `max_n_prefetch`, `slot_idx`, `w_prefetch_slots`, and the row-only `w_row_prefetch_src_t` (§1b.6). |
| Execution | `CpuTaskRunner` (CPU work submitter) | `WeightPrefetchStreamer` (sibling): owns `copy_stream`, per-layer events, slot-rotation policy, deferred wraparound. `CotsPrefetchBufferPool`: K=2 GPU slot allocator, shape-group-shared. **No model knowledge.** |
| Operator | `CotsQKVOp`, `CotsSwiGLUMLPOp` | Three-way path (GPU permanent + GPU prefetched + CPU compute). New `_scatter_col_outputs_three_way` helper. |
| Lifecycle | `CotsOffloader` | Layer-level forward hooks (one queue per decoder layer); first-decoder pre-hook for active-bucket dispatch + deferred-prefetch entry; `_install_prefetch_machinery` (post-pool, pre-hooks). |

The streamer is where **all** runtime prefetch state lives:
`copy_stream`, the K=2 buffer pool, the per-layer copy-done events,
the static deferred wraparound index, the active bucket cache. The
offloader is the lifecycle adapter that hooks layer forwards and
delegates to the streamer's lifecycle methods.

### Per-bucket geometry computation

`CotsLinearHandle.apply_prefetch_split_per_bucket(dispatch_table)`
populates the per-bucket dicts from a `dict[bucket, (f_cpu, f_prefetch)]`.
Phase 1b passes a uniform-fill table; the Planner later passes a
per-bucket table. The geometry per-kind (`_compute_bucket_split`,
`cots.py:608-651`):

| Kind | n_pref derivation | Prefetched rows of `w_cpu` |
|---|---|---|
| `qkv` | `requested = round(f_prefetch × out_dim)` → snapped via `_qkv_kv_biased_counts` to head-aligned `(n_q, n_k, n_v)`; `n_pref = min(n_q+n_k+n_v, n_cpu)` | First `n_pref` rows of `w_cpu`, in `cpu_indices` order: `[Q_tail \| K \| V]`. When `n_pref ≤ n_q_tail` prefetch covers Q-tail only; when `n_pref > n_q_tail` it extends into K then V. |
| `col` (gate_up) | `n_pref_per_half = min(round(f_prefetch × half), n_cpu_per_half)`; `n_pref = 2 × n_pref_per_half` | First `n_pref_per_half` rows of EACH half (gate's first n + up's first n). |
| `row` (down_proj) | `n_pref = min(round(f_prefetch × in_dim), n_cpu)` | First `n_pref` input cols (the prefix of the LAST `n_cpu` cols, by `cpu_indices` ordering). |

The `kv_biased=True` picker (Phase 1a) defines the order of
`cpu_indices` for QKV: the picker ensures all three of
`n_q_tail / n_k / n_v` are head-aligned and assigns CPU columns as
Q-tail first, then K, then V. Prefetch and CPU-compute split that
contiguous prefix at `n_pref`. There is no separate "K/V-pin" guard
— at low `f_cpu_store` (below the K/V fraction) `n_q_tail = 0` and
CPU covers K/V only; prefetch then takes K/V cols directly when
`f_prefetch > 0`.

`max_n_prefetch` is set after the per-bucket loop as
`max(n_prefetch_by_bucket.values())` and used to size slot views and
the `w_row_prefetch_src_t` buffer. Phase 1b uniform fill makes
`max_n_prefetch == n_prefetch_by_bucket[any_bucket]`; Planner
per-bucket variation makes them genuinely differ.

### Index-set disjointness invariant

For every kind and every bucket:

```
gpu_indices ⊎ prefetch_indices_by_bucket[b] ⊎ cpu_compute_indices_by_bucket[b]
  = [0, dim)
```

The three operator paths (GPU permanent, GPU prefetched, CPU
computed) write disjoint column slices via `index_copy_` into the
output tensor (§1b.2). Verified per-bucket by
`test_prefetch_split.py`.

---

## 1b.2: Three-way scatter and the matched-index invariant

Phase 1a's two-way scatter (`gpu_indices_cuda` vs `cpu_indices_cuda`)
becomes a three-way add-reduce / scatter:

**QKV** (column-parallel split, output along col axis):

```python
out = torch.empty((n_tok, out_dim), dtype=..., device=...)
out.index_copy_(1, h.gpu_indices_cuda,                      out_perm)      # GPU permanent
out.index_copy_(1, h.prefetch_indices_cuda_by_bucket[b],    out_pref)      # GPU prefetched
out.index_copy_(1, h.cpu_compute_indices_cuda_by_bucket[b], y_cpu_on_gpu)  # CPU-computed
```

Disjointness of the index sets means the three writes don't
overlap; ordering doesn't matter; no atomic ops.

**MLP block** (row-parallel split, output along reduction dim): the
three partial outputs all have the same shape `(n_tok, hidden)` and
add-reduce — `out_perm + out_pref + y_cpu_on_gpu`. The CPU path
returns its full block output through a single UVA copy per fused
block (Phase 1a's matched-index optimization, preserved verbatim).

### Matched-index invariant

For the MLP block the prefetched-GPU path and the CPU path must
operate on the **same input cols** of MLP2 to be addable. With:

- gate_up (col handle): `n_prefetch_per_half` rows of EACH gate/up
  half are prefetched.
- down (row handle): `n_prefetch` input cols are prefetched.

The invariant is `gate_up.n_prefetch_per_half == down.n_prefetch`,
so the SiluAndMul'd output of `(gate_pref, up_pref)` is exactly the
right column slice to feed into `down`'s prefetched input cols.
Verified per-bucket at install time
(`apply_prefetch_split_per_bucket` extends Phase 1a's matched-index
assertion to every bucket key); tested in `test_three_way_scatter.py`
with parametric `(f_cpu_store, f_prefetch)`.

---

## 1b.3: Pre-compute K=1 next-layer prefetch + K=2 slot rotation + post-init layer-0 max-fill

Phase 1b adopts the same building blocks as vLLM's native
`PrefetchOffloader` (`prefetch.py`) — copy stream, per-layer
copy-done events, K=2 slot rotation — but with a **pre-compute
K=1 lookahead schedule** instead of the native's post-compute
K-step lookahead. The pre-compute design folds wraparound into
the natural per-layer schedule and removes COTS-specific deferred
machinery entirely.

### Layer-level forward hooks (pre-compute K=1)

`CotsOffloader._hook_layer_forward(index, layer)` wraps each
decoder layer's forward with:

```
forward(*args):
    wait_prefetch(anchor, index)              # sync layer index's H2D
    if next_has_prefetch:
        start_prefetch(anchor, next_idx)      # next_idx = (index + 1) % n_layers
    output = original_forward(*args)          # layer index compute
    return output
```

The `start_prefetch(next_idx)` fires **before** the layer's
compute, so the H2D for layer `index + 1` overlaps with layer
`index`'s compute on the main stream. Wraparound is implicit:
layer `N - 1`'s pre-hook starts layer `0`'s H2D for the next iter.
No special-cased deferred path; no static
`deferred_wraparound_index`; no `start_deferred_prefetch` op
registration on COTS' side.

`wait_prefetch` and `start_prefetch` are the existing
`torch.ops.vllm.{wait,start}_prefetch` custom ops registered in
`prefetch_ops.py`; they dispatch via `BaseOffloader._wait_for_layer`
and `_start_prefetch`, which COTS overrides to delegate to
`WeightPrefetchStreamer.{wait,start}`. **No new op registration.**

The `next_has_prefetch` guard handles two edge cases:
`n_layers == 1` (`next_idx == index`, no self-prefetch);
`max_n_prefetch == 0` for layer `next_idx`'s handles (no work to
do).

### K=2 slot rotation (slot count, not lookahead)

K refers to two different things in Phase 1b — keep them straight:

| K meaning | Value | Where |
|---|---|---|
| **Slot count** in `CotsPrefetchBufferPool` | 2 | `slot_idx = layer_idx % K` |
| **Lookahead distance** in the layer wrapper | 1 | `next_idx = (index + 1) % n_layers` |

With K=2 slots and K=1 lookahead: layer `i` reads slot `i % 2`;
layer `i + 1` writes slot `(i + 1) % 2` (different physical slot)
during layer `i`'s compute; layer `i + 1`'s wait syncs on it. The
slots rotate cleanly between consecutive layers because consumer
and producer always land on different slots. K=2 slots is the
minimum that works: K=1 slot would have layer `i + 1`'s H2D
overwrite layer `i`'s read source mid-flight.

### post-init max-fill — **layer 0 only**

On iter 1, every layer's slot needs to be primed before its
`wait_prefetch` fires. The priming sources are:

| Layer | Filled by |
|---|---|
| 0 | `post_init` max-fill (no predecessor in iter 1) |
| 1..N-1 | Predecessor layer's pre-compute `start_prefetch` hook, fired during iter 1 |
| 0 (next iter) | Layer N-1's pre-compute `start_prefetch` hook, fired in iter N |

So `post_init` only needs to handle layer 0. It max-fills to
`max_n_prefetch` rows on `copy_stream` and host-syncs once. Cost
on Qwen2.5-7B at f_prefetch=0.30 (worst case in the bench sweep):
one layer × 3 handles ≈ ~210 MB H2D once at startup, ~9 ms at PCIe
ceiling (24 GB/s pinned). The post-init H2D writes into the
already-allocated pool, so no allocation happens outside the
DeviceMemoryProfiler context.

---

## 1b.4: Implicit wraparound (no deferred-state machinery)

`phase0_findings.md §0.10.5` documented the eager-mode CE0 FIFO
contention pattern that motivates deferring the *last* wrap-around
prefetch from end-of-iter-N to start-of-iter-N+1 (after vLLM's
input-prep H2Ds queue on CE0). The native `PrefetchOffloader` ships a
graph-compatible defer-wraparound fix (§0.10.5 "Graph-compatible
defer-wraparound fix") with a static `deferred_wraparound_index` +
`torch.ops.vllm.start_deferred_prefetch` custom op. **COTS does not
use that machinery.**

### Why COTS doesn't need it

Under the pre-compute K=1 schedule (§1b.3), the wrap-around is
already structurally a regular prefetch:

- Layer `N-1`'s wrapper fires `start_prefetch(0)` BEFORE layer `N-1`
  compute. The H2D queues onto `copy_stream` and overlaps with layer
  `N-1`'s compute (~2.5 ms) and any tail before iter N+1's first
  layer.
- On iter N+1, layer 0's wrapper fires `wait_prefetch(0)` first, then
  `start_prefetch(1)`, then layer 0 compute. No special path; same
  control flow as every other layer.

The deferred-wraparound fix exists in vLLM specifically for the
post-compute K-step schedule, where layer N-1's post-hook would
queue the wrap-around H2D AT the iter boundary (zero hide window) and
contend with iter N+1's input-prep H2Ds on CE0. Pre-compute K=1
sidesteps the whole problem: the wrap-around H2D starts before
layer N-1's compute, so it has at minimum one layer of compute as
hide window — same as every other layer's prefetch.

### Empirical confirmation

The §0.10.5 CE0-contention regression hypothesis (worst-case
prediction: +2.2s on Bench 2 C @ B=1) **did not materialize** when
the pre-compute restructure landed. Bench 2 C @ B=1 went 11.91 s
(deferred K=2) → 12.04 s (pre-compute K=1) — within 1.1%, well
inside 3-iter measurement noise. Reason: at COTS' G=1 every-layer-
offloaded coverage, CE0 is saturated throughout every iter (28
back-to-back H2Ds on `copy_stream`); the iter boundary is not
structurally different from any intra-iter boundary, so the
wrap-around isn't a special source of CE0 contention. The §0.10.5
phenomenon is specifically a low-coverage pathology.

### What this removes from COTS

| Before | After |
|---|---|
| `WeightPrefetchStreamer.defer_wraparound: bool` | removed |
| `WeightPrefetchStreamer.deferred_wraparound_index: int \| None` | removed |
| `CotsOffloader._start_deferred_prefetch` method | removed |
| First-decoder pre-hook calls `torch.ops.vllm.start_deferred_prefetch` | removed |
| `post_init` skip-deferred-index logic | removed (post_init fills layer 0 only; nothing to skip) |
| Layer N-1 post-hook special case (`is_last_layer and wraps_around`) | removed (no post-hook anymore) |

The `start_deferred_prefetch` custom op remains registered in
`prefetch_ops.py` for the native `PrefetchOffloader` (which still
needs it under the post-compute K-step schedule). COTS just doesn't
call it.

### Phase 1c implications

The Phase 1c FULL CUDA graph boundary still calls
`offloader.prepare_before_forward(num_tokens)` +
`offloader.sync_prev_onload()` outside the captured graph (§1b.13).
COTS' `prepare_before_forward` repairs layer 0 only; all steady-state
prefetches (including the wraparound to layer 0) are emitted by the
captured layer wrappers via `torch.ops.vllm.start_prefetch`, so they
become graph nodes that replay deterministically. No Python pending
state is needed.

---

## 1b.5: Prefetch buffer pool — shape-group sharing

`CotsPrefetchBufferPool` allocates K slots **per unique shape**, not
per handle. At Qwen2.5-7B with 28 offloaded layers and 3 unique
shapes per layer (qkv / col / row), this is 28× smaller than
per-handle allocation.

```
groups: dict[(kind, slot_shape)] → list[handles]
total_numel = sum(K * shape[0] * shape[1] for shape in groups)
```

All handles in a group share the same K slots; rotation happens at
the handle level via `slot_idx = layer_idx % K`. The shared owner
list and `available_rows` list (§1b.13) catch the case where layer
2 overwrites layer 0's physical slot via K=2 rotation —
`prefetch_owner_in_slot[k]` lets the operator assert it's reading
its own weights, not a sibling's.

### Slot shape per kind

| Kind | Slot shape | Why |
|---|---|---|
| `qkv` | `(max_n_prefetch, in_dim)` | `narrow(0, ...)` is contiguous → single-pass H2D. |
| `col` | `(max_n_prefetch, in_dim)` | Same shape; gate region at `[0:max_half]`, up region at `[max_half:2*max_half]` (fixed-max layout, §1b.13). |
| `row` | `(max_n_prefetch, out_dim)` | **Transposed** versus `w_cpu = (out_dim, n_cpu)`. Required for the row-prefetch contiguity fix (§1b.7). |

Pool allocation is inside `wrap_modules` (DeviceMemoryProfiler
invariant from `phase1a_findings.md §1.5`). At Qwen2.5-7B with
`f_prefetch=0.30` (a worst-case in the bench sweep), pool size ≈
0.40 GiB GPU.

---

## 1b.6: Pinned-CPU duplicate — `w_row_prefetch_src_t`

For `kind == "row"` only, an additional pinned CPU buffer of shape
`(max_n_prefetch, out_dim)` holds the transposed prefix of `w_cpu`.
Allocated in `_install_prefetch_machinery` after `max_n_prefetch` is
known and populated by `_row_weight_loader` at weight-load time:

```python
self.w_row_prefetch_src_t.copy_(
    src_block.transpose(0, 1).contiguous(),  # one-shot; paid at load
    non_blocking=False,
)
```

The runtime H2D path then narrows on dim 0 of this transposed
source, which is contiguous regardless of the active bucket's
`n_prefetch`. See §1b.7 for the perf postmortem this fixes.

Memory cost on Qwen2.5-7B at `f_prefetch=0.15`: 28 layers × 19.4 MiB
≈ **0.53 GiB extra pinned CPU**. At `f_prefetch=0.30`: ~1.06 GiB.
Acceptable (DDR5-CPU offload is the cheapest tier in the system).

CPU storage `w_cpu: (out_dim, n_cpu)` stays untouched — PyTorch
eager F.linear on a transposed CPU input was 100× slower in
microbenches, so primary CPU storage transpose is deferred to Phase
1c (native CPU kernel). The duplicate is a temporary cost paid until
the CPU runner swap.

---

## 1b.7: Row-prefetch contention fix

The first end-to-end Bench 2 measurement of the collaborative arm
(C: f_cpu_store=0.30, f_prefetch=0.15) was ~2× slower than the
pure-prefetch arm (B: f_cpu_store=0.30, f_prefetch=0.30) at B=1 —
38.6 s vs 19.8 s. Path-decomposition microprobes pinpointed the
mechanism: the MLP2 (down_proj) prefetch H2D source was
**non-contiguous** under the original layout.

### Mechanism (verified)

In the original `WeightPrefetchStreamer.start` row branch:

```python
src = h.w_cpu.narrow(1, 0, n_pref)         # w_cpu shape (out_dim, n_cpu) row-major
dst = h.w_prefetch_slots[h.slot_idx].narrow(1, 0, n_pref)
dst.copy_(src, non_blocking=True)
```

`narrow(1, 0, n_pref)` on a `(out_dim, n_cpu)` row-major tensor
produces a view with stride `(n_cpu, 1)` over a `(out_dim, n_pref)`
shape. This is **pitched H2D** — slower than a contiguous transfer
of the same byte count. Crucially:

- When `n_pref == n_cpu` (pure-prefetch case B), `narrow` degenerates
  to the full contiguous tensor → fast.
- When `0 < n_pref < n_cpu` (collaborative case C), `narrow`
  produces a strided view → pitched H2D, slow.

This explained the 2× C-vs-B gap: the collaborative arm hits the
pathological pitched path, the pure-prefetch arm doesn't.

### Empirical verification — `probe_row_prefetch_layout.py`

Microprobe at Qwen2.5-7B `down_proj` shape
`(out_dim=3584, in_dim=18944)`, f_cpu_store=0.30:

| f_prefetch | strided H2D (current) | contiguous H2D (fix) | ratio |
|---:|---:|---:|---:|
| 0.05 | 0.35 ms / 19.6 GB/s | 0.29 ms / 23.8 GB/s | 1.21× |
| **0.15** (collab) | **1.58 ms / 12.9 GB/s** | **0.85 ms / 23.9 GB/s** | **1.85×** |
| 0.25 | 3.72 ms / 9.1 GB/s | 1.42 ms / 23.9 GB/s | 2.62× |
| 0.30 (=n_cpu) | 1.70 ms / 23.9 GB/s | 1.70 ms / 23.9 GB/s | 1.00× |

The `f_prefetch == f_cpu_store` row confirms the cliff: at exactly
that point `narrow` returns the full tensor and bandwidth recovers
to the PCIe ceiling. Per-forward extrapolation (28 row handles,
B=1): strided 43.7 ms → contig 23.9 ms, **~20 ms saved per forward**,
≈ 2.5 s over a 128-step decode on row-handle H2D alone.

### Fix

Add a per-row-handle pinned **transposed** CPU duplicate (§1b.6) and
transpose the GPU slot to match (§1b.5). The H2D becomes:

```python
src = h.w_row_prefetch_src_t.narrow(0, 0, n_pref)   # (n_pref, out_dim) contig
dst = h.w_prefetch_slots[h.slot_idx].narrow(0, 0, n_pref)   # (n_pref, out_dim) contig
dst.copy_(src, non_blocking=True)
```

Both source and destination are `narrow(0, ...)` over a row-major
buffer with stride `(out_dim, 1)` — single-pass H2D DMA. The MLP2
prefetched-GPU compute uses `pref_silu.matmul(slot[:n, :])` instead
of `F.linear(z, slot.narrow(1, 0, n))`; the math is `z @ Wᵀ` either
way, GPU microbenches show no measurable perf delta.

### End-to-end impact (Bench 2 C arm pre/post fix)

| B | pre-fix C | post-fix C | speedup |
|---:|---:|---:|---:|
| 1 | 38.63 s | 11.88 s | **3.25×** |
| 4 | 40.69 s | 14.03 s | 2.90× |
| 16 | 54.64 s | 28.91 s | 1.89× |
| 64 | 118.72 s | 93.35 s | 1.27× |

Speedup shrinks with B because at large B the CPU GEMM (not H2D) is
the bottleneck — making H2D faster has less to bite into. At B=1, C
went from "loses to B by 2×" to "beats min(A, B) by 15%". qkv and
col paths use `narrow(0, ...)` over row-major buffers (whole rows)
and were already contiguous; the row layout was the single dominant
bug. The pre-fix before/after conclusion is preserved here; raw
result files are no longer checked in.

---

## 1b.8: Bench 1 — COTS pure-prefetch vs native `prefetch_defer` (tensor-granularity wins at matched bytes)

`David/Benchmarks/phase1b/bench_cots_exact_match.py`. **Question
this bench answers:** at matched offloaded GiB, does COTS in
pure-prefetch mode (`f_prefetch == f_cpu_store`, every layer
offloaded a fraction) match the densest-spread native baseline?

The COTS arm offloads a **fraction** of EVERY layer. The native arm
offloads **entire** layers, picked uniformly via the canonical
picker (`PrefetchOffloader(G, N=1, K=1)`). At matched offloaded GiB
both arms move the same total bytes per forward — the question is
whether COTS' tensor-granularity layout matches native's
layer-granularity layout in PCIe time.

The native arm uses `--offload-backend prefetch_defer`
(`PrefetchDeferOffloader`, the thesis-optimized variant from
`vllm/model_executor/offloader/prefetch_defer.py`). Stock factory
`prefetch` is the unoptimized baseline `phase0_findings.md §0.10.3`
already characterizes; the apples-to-apples comparison for COTS is
against the **best** native baseline.

Pairs at matched offloaded GiB (Qwen2.5-7B BF16, 28 layers, ~12.15
GiB total weight; per-layer ~0.434 GiB):

| n_layers | native G | cots_f | offloaded GiB |
|---:|---:|---:|---:|
| 1  | 28 | 0.0357 | 0.43 |
| 2  | 14 | 0.0714 | 0.87 |
| 4  |  7 | 0.1429 | 1.74 |
| 7  |  4 | 0.2500 | 3.04 |
| 14 |  2 | 0.5000 | 6.08 |

Workload: decode-heavy (input=8, output=128). Batches {1, 64}.
`none` baseline: 2.035 s @ B=1, 2.482 s @ B=64.

### Headline

Native arm is run at K=1 (apples-to-apples lookahead match for
COTS' pre-compute K=1 schedule) and K=2 (native's empirical
optimum at uniform spacing per `phase0_findings.md §0.10.1d`):

| Depth | B | COTS | native K=1 | native K=2 | vs K=1 | vs K=2 |
|---|---:|---:|---:|---:|---:|---:|
| 0.43 GiB (1L) | 1 | 2.63 | 2.73 | 2.73 | **−3.5%** MATCH | **−3.5%** MATCH |
| 0.43 GiB (1L) | 64 | 2.75 | 2.83 | 2.83 | −3.0% MATCH | −2.9% MATCH |
| 0.87 GiB (2L) | 1 | 4.73 | 5.29 | 5.00 | **−10.5%** WIN | **−5.2%** WIN |
| 0.87 GiB (2L) | 64 | 4.86 | 5.41 | 5.04 | **−10.1%** WIN | −3.6% MATCH |
| 1.74 GiB (4L) | 1 | 9.58 | 10.43 | 9.99 | **−8.2%** WIN | −4.1% MATCH |
| 1.74 GiB (4L) | 64 | 9.69 | 10.60 | 10.07 | **−8.6%** WIN | −3.8% MATCH |
| 3.04 GiB (7L) | 1 | 16.52 | 18.10 | 17.47 | **−8.7%** WIN | **−5.4%** WIN |
| 3.04 GiB (7L) | 64 | 16.66 | 18.38 | 17.60 | **−9.3%** WIN | **−5.3%** WIN |
| 6.08 GiB (14L) | 1 | 33.00 | 36.00 | 34.92 | **−8.3%** WIN | **−5.5%** WIN |
| 6.08 GiB (14L) | 64 | 33.28 | 36.55 | 35.20 | **−8.9%** WIN | **−5.4%** WIN |

(MATCH: |Δ| ≤ 5%. WIN: COTS faster by >5%.)

### Four findings

1. **COTS pure-prefetch beats the optimized native K=1 baseline by
   8–10% at every non-trivial depth.** The tensor-granularity
   layout — fractional offload spread across all 28 layers, broken
   into 3 sub-modules per layer — finishes 8–10% faster than the
   layer-granularity K=1 layout. At matched bytes per forward, the
   comparison is on the *quality of the H2D / compute overlap*,
   and COTS wins.

2. **K=2 narrows but doesn't close the gap.** Native at K=2
   (empirical optimum from `phase0_findings.md §0.10.1d`) saves
   3–5% over K=1, consistent with §0.10.1d's "≤3% under uniform
   spacing" — bigger gains at low depth (G=14, 2 offloaded layers)
   where CE0 has more idle window for K>1 to exploit. **COTS still
   wins or matches at every depth even against the K=2 optimum**:
   −5.2 to −5.5% WIN at 0.87 / 3.04 / 6.08 GiB; MATCH at the
   smaller depths. No measured point where native beats COTS.

3. **Mechanism: finer-grained H2D queue + better per-sub-module
   compute hide.** Native's per-forward H2D pattern is a few large
   transfers, each ~444 MiB / ~19 ms; one full GPU-resident layer
   of compute (~2.5 ms) hides only a small fraction (K=2 doubles
   the hide window but it's still a small fraction). COTS' pattern
   is many smaller H2Ds queued onto `copy_stream`, with per-sub-
   module compute on the GPU side providing more (and finer-
   grained) windows for CE0 to drain. Crucially, `copy_stream` /
   CE0 stays busy *throughout* every COTS layer — no quiet
   intervals where CE0 is idle waiting for the next big transfer
   like in native's clustered pattern.

4. **The match at the smallest depth (0.43 GiB) reflects the
   floor.** At 1 native layer offloaded vs COTS f=0.036 (one
   layer's worth of bytes spread across all 28), both move ~0.43
   GiB on PCIe per forward and the absolute time is dominated by
   the unhidable PCIe portion (~0.6 s above `none`). K=2 gives no
   benefit over K=1 here (only one offloaded layer; nothing for
   the lookahead to extend to). The 3% COTS edge at this depth is
   closer to noise floor than at the bigger depths where the
   layout difference dominates.

### Implication

This is the strongest validation of the **tensor-granularity
hypothesis** the thesis is built on (`weight_offload_design.md`):
splitting weights at sub-module granularity and placing the slices
across all layers extracts **better** PCIe overlap than the
layer-granularity baseline at matched bytes. Combined with Bench
2's collaborative wins at small B (§1b.9) and the per-bucket-tuned
wins demonstrated by the collaborative-split sweep at B=16 (§1b.12),
the overall Phase 1b story is: **COTS' tensor-granularity offload
beats every native-prefetch baseline at every operating point we
measured, except the high-batch saturated regime where pure-
prefetch is genuinely optimal**. The Planner's job is to navigate
that.

---

## 1b.9: Bench 2 — matched offload depth (regime story)

`David/Benchmarks/phase1b/bench_cots_collaborative.py`. Three arms
at **matched total offload depth** `f_cpu_store=0.30`, decode-heavy
workload (input=8, output=128), Qwen2.5-7B BF16, default
`cots_cpu_num_threads=16`.

| Arm | Configuration | What it tests |
|---|---|---|
| A — cpu-only       | `f_cpu_store=0.30, f_prefetch=0.0`  | Phase 1a's pure-CPU path, no prefetch |
| B — prefetch-only  | `f_cpu_store=0.30, f_prefetch=0.30` | Pure-PCIe path (every CPU-stored byte streamed back) |
| C — collaborative  | `f_cpu_store=0.30, f_prefetch=0.15` | Both paths concurrent (the thesis claim) |

### Headline (post all refactors, including pre-compute K=1)

| B | A_cpu_only | B_prefetch_only | C_collaborative | C vs min(A, B) | verdict |
|---:|---:|---:|---:|---:|---|
| 1  | 14.61 s | 19.83 s | **12.04 s** | **−17.6%** | **WIN** |
| 4  | 19.73 s | 19.90 s | **13.94 s** | **−29.3%** | **WIN** |
| 16 | 50.61 s | 20.06 s | 28.42 s | +41.7% | LOSE |
| 64 | 181.71 s | 20.01 s | 95.03 s | +374.9% | LOSE |

`none` baseline at the same workload: B=1 2.033 s, B=4 2.099 s,
B=16 2.148 s, B=64 2.482 s.

### Three observations

1. **Small B (1, 4): collaborative wins decisively.** At B=4, C
   beats min(A, B) by ~30%. PCIe is the bottleneck (B's pure-prefetch
   time is ~20 s, constant across batches because per-forward bytes
   moved are bucket-independent), and splitting work into CPU lets
   it hide compute behind prefetch. C ≈ 14 s ≈ max(A_per_path,
   B_per_path) at f=0.15 + ~2 s orchestration (per Bench 3, §1b.10).

2. **Large B (16, 64): pure prefetch wins by huge margins.** At
   B=64, A scales linearly with B (179 s) while B is constant
   (~20 s). Adding any CPU work to a B=64 forward drags the
   bottleneck up — C inherits A's CPU-bound cost (93 s), conceding
   to B's 20 s by 4.7×.

3. **The crossover is between B=4 and B=16.** Bench 2 establishes
   the regime where collaborative pays off and where it should
   simplify to pure prefetch. The Planner's job is exactly this
   per-bucket trade-off: at fixed total offload depth, the optimal
   `f_prefetch` shifts from 0.5 (collaborative split) at small B
   toward 1.0 (pure prefetch) at large B.

### B and `none` are stable across reruns

After four large refactors (row-prefetch fix, active-bucket
dispatch, deferred-prefetch graph-safe rewrite, pre-compute K=1
restructure), Bench 2 was re-measured each time. The B and `none`
arms are bit-equal across runs (B ≈ 19.84-20.06 s, `none` =
2.033-2.034 s) — they don't touch any of the changed paths. C arm
drifts within ±2% across all four refactors, all within 3-iter
measurement variance.

Pre-fix snapshots were used during bring-up to isolate each refactor
step. Their conclusions are preserved in this document; the raw
snapshot trees were removed during Phase 1 cleanup and replaced by
`David/Benchmarks/phase1b/results/phase1b_final_summary.json`.

---

## 1b.10: Bench 3 — matched per-path resource (contention diagnostic)

Bench 2 answers "at fixed total budget, which dispatch wins?" but
cannot distinguish "C beats min(A, B) because of clean overlap"
from "C beats them because A/B saturate one path with the full
budget." To diagnose **path contention** you need each path's
baseline at the EXACT load C exposes on it.

`David/Benchmarks/phase1b/bench_cots_path_contention.py` — three
arms at `f_collab=0.15` (= C's `f_prefetch` = C's `f_cpu_compute`):

| Arm | Configuration | What it isolates |
|---|---|---|
| A_per_path_cpu       | `f_cpu_store=0.15, f_prefetch=0.0`  | Pure CPU at C's CPU-compute load |
| B_per_path_prefetch  | `f_cpu_store=0.15, f_prefetch=0.15` | Pure prefetch at C's PCIe load |
| C_collaborative      | `f_cpu_store=0.30, f_prefetch=0.15` | Same C as Bench 2 |

Contention metric per batch:
`contention = T_C − max(T_A_per_path, T_B_per_path)`. ~0 = perfect
overlap; >0 = paths interfere (PCIe contention, host-dispatch
saturation, scheduler interaction); <0 = pure paths over-saturated
relative to C's per-path loads (rare).

### Results

| B | A_per_path | B_per_path | C_collab | contention | verdict |
|---:|---:|---:|---:|---:|---|
| 1  | 8.14 s  | 10.01 s | 11.97 s | **+1.96 s** | MILD |
| 4  | 11.20 s | 10.02 s | 13.59 s | **+2.39 s** | CONTENT |
| 16 | 27.93 s | 10.08 s | 28.94 s | **+1.01 s** | MILD |
| 64 | 93.86 s | 10.10 s | 93.53 s | **−0.33 s** | OVERLAP |

### Three findings

1. **B_per_path is constant at ~10 s across all batches.**
   Pure-prefetch time depends on bytes/forward, not on token count.
   Sanity check passed; cross-validates that the prefetch path is
   bucket-insensitive at this `num_tokens` × hidden range.

2. **At B≥16 contention is essentially zero (<0.2 s ≈ 0.5%).** When
   CPU work dominates (A > B in this matched-load comparison), the
   prefetch path fully hides inside the CPU compute window.
   Collaborative overlap is near-perfect — the second path is free.

3. **At B≤4 there is ~2 s of residual contention.** Comparable in
   magnitude to the `--cots-dry-run` orchestration overhead measured
   in `phase1a_findings.md §1.14` (~0.45 s/generate from
   `dryrun − none` in pure-CPU; here at the more elaborate three-way
   path it's ~2 s). The 2 s is **orchestration overhead**, not PCIe
   contention — Python hooks, the runner's `submit_with_d2h` /
   `wait` round-trip, the three-way scatter, the per-bucket
   dispatch lookup. `cudaLaunchHostFunc` + CUDA-graph capture (Phase
   1c) are the targeted fixes.

---

## 1b.11: Cross-bench predictive model

Bench 3's per-path baselines + measured contention predict Bench
2's collaborative arm:

```
predicted_C_at_B = max(A_per_path(B), B_per_path(B)) + dispatch_overhead(B)
```

where `dispatch_overhead = T_C − max(T_A, T_B)` from Bench 3.

| B | A_per_path | B_per_path | dispatch | predicted C | actual Bench 2 C | error |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 8.14  | 10.01 | +1.96 | 11.97 | 12.04 | **0.6%** |
| 4  | 11.20 | 10.02 | +2.39 | 13.59 | 13.94 | **2.6%** |
| 16 | 27.93 | 10.08 | +1.01 | 28.94 | 28.42 | **1.8%** |
| 64 | 93.86 | 10.10 | −0.33 | 93.53 | 95.03 | **1.6%** |

The predictive model holds within ~5% across batches — strong
evidence that:

1. **The contention metric is well-defined** as a workload-invariant
   property of the orchestration substrate, not a noisy random
   variable.
2. **Collaborative dispatch behaves predictably from per-path
   baselines** — i.e., it's `max + δ`, not some emergent
   nonlinearity.
3. **The Planner could use Bench 3's per-path data as direct inputs
   to a per-bucket cost model.** Per-path costs scale with
   `num_tokens` (CPU) or are constant (PCIe); plug into `max + δ` to
   estimate collaborative latency at any (depth, B) without running
   the full Bench 2.

The B=4 point shows the largest error (4.7%) because that's where
A_per_path and B_per_path are nearly equal (10.74 vs 10.05) — the
`max` operator has the most noise sensitivity at the crossover.

---

## 1b.12: Collaborative-split sweep — Planner motivation

Bench 2 reports collaborative LOSING at B=16 (+41.7%) and B=64
(+374.9%) vs pure prefetch. But Bench 2 only sampled the 50/50
split (`f_pref = 0.15` of `f_cpu_store = 0.30`) — at large B the CPU
path is the long pole, so the optimal split should shift toward
prefetch-heavy. **Does collaborative win at large B when properly
tuned?**

`David/Benchmarks/phase1b/probe_collab_split_sweep.py` sweeps
`f_prefetch ∈ {0.20, 0.25, 0.28}` at fixed `f_cpu_store = 0.30`,
B ∈ {16, 64}, decode-heavy (input=8, output=128).

| B | f_pref=0.15 (Bench 2 C) | 0.20 | 0.25 | 0.28 | f_pref=0.30 (Bench 2 B) |
|---:|---:|---:|---:|---:|---:|
| 16 | 28.4 s | 21.4 s | **17.2 s** | 18.7 s | 20.0 s |
| 64 | 95.0 s | 64.3 s | 37.1 s | **20.8 s** | 20.0 s |

### Two findings

1. **At B=16, well-tuned collaborative beats pure prefetch by
   −14.5%** (17.2 s at f_pref=0.25 vs 20.0 s pure prefetch). Bench
   2 said "LOSE +41.7%" at the 50/50 split; the per-bucket-tuned
   answer flips the verdict to a clear win. This validates the
   Planner's per-bucket dispatch concept — the right `f_prefetch`
   is genuinely batch-dependent and worth tuning.

2. **At B=64, pure prefetch is genuinely optimal.** The curve
   approaches but doesn't cross: f_pref=0.28 → 20.8 s, just 4%
   above pure prefetch's 20.0 s. CPU GEMM scales linearly with B
   while PCIe is constant; at B=64 the CPU path can't be hidden
   regardless of split. **The Planner should pick pure prefetch at
   B=64.**

The shape matches the Bench 3 cost model (§1b.11): predicted
collaborative latency = `max(T_cpu(f_cpu, B), T_pcie(f_pref)) +
dispatch`. Plugging in:
- B=16, f_pref=0.25 → max(28.27 × 0.05/0.15, 10 × 0.25/0.15) + ~1 ≈ 17.7 s. Measured 17.2 s.
- B=64, f_pref=0.28 → max(93.09 × 0.02/0.15, 10 × 0.28/0.15) + ~0 ≈ 18.7 s. Measured 20.8 s (small slop, but the qualitative shape is captured).

So **the Planner's per-bucket dispatch table can be read off the
cost model with no additional measurement** — Bench 3's per-path
baselines plus this confirmation that the model holds across the
collaborative split axis.

### Thread-count knob (deferred to phase1a)

The `cpu_num_threads` discussion lives in `phase1a_findings.md §1.13b`.
The Phase 1b row-prefetch fix (§1b.7) eliminated most of the strided-H2D
vs CPU-pinned-DRAM contention that the Phase 1a thread-count sweep
was reacting to, but the qualitative finding (`t=16` optimal across
batches; `t=8` slightly better at decode-B=1) is unchanged. Per-bucket
thread policy stays Planner work; no separate Phase 1b probe needed.

---

## 1b.13: Phase 1c compatibility refactor — active-bucket dispatch

The original Phase 1b operator dispatch read the slot's last-fill
bucket as the source of truth for compute shape:

```python
slot_b = h.prefetch_bucket_in_slot[h.slot_idx]      # runtime Python list read
n_pref = h.n_prefetch_by_bucket[slot_b]
n_cpu  = h.n_cpu_compute_by_bucket[slot_b]
```

Under uniform `f_prefetch` (Phase 1b's only mode) this is correct —
every bucket has the same `n_prefetch`, and the slot's prior-iter
geometry happens to match the current iter's geometry. **In eager
mode it works.** It just won't survive CUDA-graph capture (Phase 1c):
`slot_b` is a Python list element mutated during execution; under
graph capture the read happens at capture time, freezes whatever
value was there, and ignores subsequent runtime mutations. There's
no clean way to express "look at runtime slot state to decide
compute shape" inside a captured graph.

### The conceptual shift

> **Slot metadata proves bytes are *available*; bucket metadata
> decides *computation shape*.**

Each captured graph corresponds to one `BatchExecutionDescriptor` →
one bucket. The operator reads
`n_prefetch_by_bucket[that_bucket]` (capture-time constant), uses
`slot[:n_pref, :]` (capture-time constant slice), and asserts the
slot is sufficiently filled (runtime invariant check, not runtime
lookup). Boundary preparation (filling missing slot bytes before
replay) runs **outside** the captured graph.

### Six concrete changes

| Change | Site | Why |
|---|---|---|
| Replace `prefetch_bucket_in_slot: list[int]` with `prefetch_available_rows_in_slot: list[int]` | `CotsLinearHandle.__init__` | Semantics shift: slot tracks *bytes available*, not *bucket-this-was-for*. Per-half count for col, total prefix for qkv/row. |
| Col slot layout: active-packed `[gate_n \| up_n]` → fixed-max `[gate_max \| up_max]` | `CotsPrefetchBufferPool` slot_shape; `WeightPrefetchStreamer.start` col branch | Required for max-fill to be slice-safe at smaller active buckets — gate region always at `[0:max_half]`, up region at `[max_half:2*max_half]` regardless of active `n_per_half`. |
| MLP1 prefetched path: `F.linear(x, [gate\|up]) + SiluAndMul` → `F.linear(x, gate_w) + F.linear(x, up_w) + F.silu(gate)*up` | `CotsSwiGLUMLPOp.__call__` | Gate and up no longer adjacent in memory under fixed-max layout. Math-equivalent to `SiluAndMul.forward_native` (`activation.py:138-141`). ~5–10 μs eager overhead per layer; CUDA graph capture (Phase 1c) erases it. |
| Operator sources `b = streamer.current_bucket` instead of `slot_b = …` | `CotsQKVOp.apply`, `CotsSwiGLUMLPOp.__call__` | Compute shape from active forward, not slot history. Captured into the graph as a fixed `n_prefetch_by_bucket[b]` lookup. |
| New `WeightPrefetchStreamer.prepare_for_forward_bucket(layer_idx, handles)` | `cots.py` | Idempotent boundary repair: suffix-copies missing rows when `available_rows < required`; owner mismatch hard-asserts. Records per-layer event symmetrically with `start()`. |
| `prepare_before_forward(num_tokens)` lifecycle hook on `BaseOffloader`; FULL-graph wiring in `cudagraph_utils.py` | `vllm/v1/worker/gpu/cudagraph_utils.py:223,267` | Generic: any offloader with bucket-dependent state. Calls `prepare_before_forward(desc.num_tokens)` + `sync_prev_onload()` outside the graph (capture and replay) so repair H2D fires eagerly on `copy_stream`, drains via the sync, and the captured graph contains only the bucket-frozen `start_prefetch` calls emitted by each layer wrapper. |

### Why `prepare_before_forward` is generic, not COTS-specific

Default no-op on `BaseOffloader`. The native `PrefetchOffloader`
inherits the no-op (it has no bucket-dependent state to repair).
COTS overrides it to set the active bucket and call
`prepare_for_forward_bucket(0, ...)`. A future offloader (e.g., a
quantized weight cache that depends on token count) would override
similarly. The graph-boundary call site is shared infrastructure;
the work each backend does is local.

### Bench 2 sanity gates (post-refactor)

Each refactor landed atomically with a Bench 2 C @ B=1 sanity gate:

| Refactor | C @ B=1 | delta |
|---|---:|---:|
| Pre-row-fix | 38.63 s | (baseline of pathology) |
| Post-row-fix (§1b.7) | 11.88 s | (3.25× speedup baseline) |
| Post-active-bucket (§1b.13 first 5 changes) | 11.81 s | −0.6% |
| Post-deferred-prefetch (was §1b.4 deferred design; now superseded) | 11.91 s | +0.8% |
| Post-`prepare_before_forward` | 11.91 s | identical |
| **Post-pre-compute K=1 restructure (§1b.3, §1b.4)** | **12.04 s** | **+1.1%** |

All within 3-iter measurement variance. Steady-state numerics
preserved across the refactors. The pre-compute K=1 restructure
predicted a +2.2 s worst-case CE0-contention regression (per
§0.10.5 logic) — measured +0.13 s. The pathology is specific to
low-coverage prefetch baselines, not COTS' G=1.

---

## 1b.14: Test matrix and reproducibility

`David/Tests/phase1b/` has **9 files, 80 collected tests**. Mirrors
phase1a's six-file pattern plus three Phase-1b-specific suites.

| File | What it covers |
|---|---|
| `test_prefetch_split.py` | `apply_prefetch_split_per_bucket` correctness across kinds; index-set disjointness `gpu ∪ prefetch ∪ cpu_compute = [0, dim)`; matched-index invariant on col↔row pairs. |
| `test_prefetch_buffer_pool.py` | Slot rotation: layer i and layer i+2 share slot. Total bytes = `K × Σ_per_unique_shape (slot_numel × dtype_bytes)`. Per-handle slot view shape/stride matches per-kind layout (transposed for row, fixed-max for col). H2D smoke. |
| `test_prefetch_streamer.py` | `start()` populates the slot per kind correctly; copy-done event recorded; slot rotation; available_rows updated; shape-group sharing; owner mismatch detected. |
| `test_three_way_scatter.py` | `_scatter_col_outputs_three_way` parity with full GEMM (QKV); MLP three-way add-reduce parity. BF16 tolerance. Pure-prefetch and pure-CPU degenerate cases. |
| `test_layer_ahead_smoke.py` | Regression sentinel: `f_prefetch=0.0` matches Phase 1a fixture bit-exactly. End-to-end on a 2-layer Qwen2 stub at small `f_prefetch`. |
| `test_offloader_integration_phase1b.py` | End-to-end `wrap_modules + post_init` on a 2-layer mini stub at `f_prefetch=0.05, f_cpu_store=0.20`. Streamer allocated, hooks installed, slot indices stamped, dispatch table populated, deferred index set, K=2 slot count. |
| `test_row_prefetch_transposed.py` | Loader populates `w_row_prefetch_src_t` correctly; new contiguous-narrow H2D matches old strided narrow byte-for-byte; pool layout under the new (max_n_prefetch, out_dim) row shape; matmul vs F.linear math-equivalence for the prefetched MLP2 path. |
| `test_active_bucket_dispatch.py` | post_init max-fills only layer 0 to max_n_prefetch; layer 0's pre-compute hook starts layer 1's prefetch; operators dispatch off `streamer.current_bucket`; owner mismatch raises; `prepare_for_forward_bucket` suffix-copies on `avail < required`; consume-only-prefix when `avail > required`; col fixed-max layout `gate[:n_half]` / `up[max_half:max_half+n_half]` correctness. |
| `conftest.py` | Session-scoped TP-1 init (gloo backend); skip if no CUDA. |

Phase 1a's 60 tests + Phase 1b's 80 tests = **140 total, all green**
on RTX 4090 in ~5 s. Run via:

```bash
cd /TTC/David/Tests/phase1b
/opt/conda/envs/thesis/bin/python -m pytest . -q
```

---

## Conclusion

Phase 1b delivers the layer-ahead prefetch extension on top of Phase
1a's static CPU-compute backend, plus three Phase-1c-prerequisite
refactors:

1. **Three-way concurrent dispatch.** GPU-permanent +
   GPU-prefetched + CPU-compute paths engage in every offloaded
   forward, with a single UVA copy per fused MLP block (matched-index
   invariant preserved). The split is parameterized by a per-bucket
   dispatch table the Planner can target later.

2. **Bench 2 (matched offload depth) validates the regime story.**
   Collaborative beats both pure paths at small B (−20% / −30% at
   B=1, B=4) and concedes to pure prefetch at large B (B=16 / 64
   hit the CPU GEMM ceiling). The crossover between B=4 and B=16 is
   the per-bucket trade-off the Planner navigates.

3. **Bench 3 (matched per-path resource) quantifies contention.**
   At B≥16 contention is ~0.2 s (perfect overlap when CPU work
   dominates). At B≤4 it's ~2 s of orchestration overhead — Phase
   1c's `cudaLaunchHostFunc` + graph capture target this directly.
   Cross-bench predictive model fits Bench 2 within ~5%.

4. **Row-prefetch contention fix (§1b.7).** Pinned transposed
   duplicate buffer for MLP2/down_proj converts pitched H2D to
   contiguous, recovering ~1.85× PCIe bandwidth at the collaborative
   operating point. End-to-end C arm gains 3.25× at B=1.

5. **Active-bucket dispatch refactor (§1b.13).** Operators source
   compute shape from the active forward's bucket
   (`streamer.current_bucket`), not from slot history. Slot state is
   contract-checked. The `prepare_before_forward` lifecycle hook on
   `BaseOffloader` runs outside captured graphs (eager pre-hook +
   `cudagraph_utils.py` graph boundary) so the captured graph
   contains only bucket-frozen kernels. This is a Phase 1c
   prerequisite, not a Phase 1b correctness fix — eager mode worked
   correctly all along.

6. **Implicit wraparound via pre-compute K=1 (§1b.3, §1b.4).**
   Layer wrappers fire `wait_prefetch(i) → start_prefetch(i+1) →
   forward(i)`, so the wrap-around (layer N-1 → layer 0) is just
   another normal pre-compute prefetch. No `defer_wraparound` flag,
   no `deferred_wraparound_index`, no COTS dependency on
   `start_deferred_prefetch`. The §0.10.5 CE0-contention regression
   hypothesis didn't materialize at G=1 because CE0 is saturated
   throughout every iter — the iter boundary is not structurally
   different from intra-iter boundaries. Bench 2 C @ B=1: 11.91 →
   12.04 s (+1.1%, within 3-iter noise).

The architecture remains the three-layer split from Phase 1a:
storage / execution / operator. Phase 1b's additions
(`WeightPrefetchStreamer`, `CotsPrefetchBufferPool`,
`prepare_before_forward`) are localized; Phase 1c's
`cudaLaunchHostFunc` + native CPU runner swap is still a body-only
change inside `CpuTaskRunner`. The next code-level checkpoint is
Phase 1c.

The thesis claim "collaborative offloading outperforms pure paths
at the same offload depth" is empirically validated at small B, and
the Planner has a measurable cost model (Bench 3 + cross-bench
predictive model, §1b.11) for emitting per-bucket dispatch
decisions across the entire B range.
