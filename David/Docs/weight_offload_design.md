# Weight Offload Design
Following FlexGen's terminology, weight offloading can be partitioned at three granularity levels. From coarse to fine:

| Granularity | Unit of placement | Example | Used by |
|---|---|---|---|
| **Group** | Groups of layers | Assign each layer to GPU or CPU, spaced in groups | vLLM PrefetchOffloader |
| **Layer** | Tensors within a layer | Assign each tensor (WQ, WK, ...) to GPU or CPU | FlexGen |
| **Tensor** | Elements within a tensor | Assign columns within a tensor to GPU or CPU | **Ours** |

---

## Storage Is Static; Compute Dispatch Is Per-Bucket

Before describing the three-way split, an important distinction: this thesis separates **where weight bytes live** (static, set once at model load) from **which compute path consumes them** (decided per CUDA graph bucket by the Planner).

- **Storage** is two-tier and static:
  - `W_gpu_permanent` — pinned on GPU for the lifetime of the engine
  - `W_cpu` — on CPU for the lifetime of the engine (no movement)
  - Plus `B_prefetch`, a fixed-size GPU scratch buffer for streaming
- **Compute dispatch** is three-way and **varies per `BatchDescriptor` bucket**:
  - GPU-permanent path — compute on GPU from `W_gpu_permanent`
  - Prefetch path — stream `W_cpu` into `B_prefetch` via `cudaMemcpyAsync` on CE0 (H2D copy engine), compute on GPU
  - CPU-compute path — compute on CPU from `W_cpu`, return result via SM-issued UVA copy kernel (bypasses CE0; runs concurrently with bg prefetch — see `phase0_findings.md §0.5`)

The same `W_cpu` bytes feed both the prefetch path and the CPU-compute path at different times in different buckets. Nothing moves between storage tiers at runtime.

Per bucket, the Planner's dispatch table specifies `(f_cpu_compute, f_prefetch_compute)` under the invariant `f_cpu_compute + f_prefetch_compute ≤ f_cpu_store`. See `planner_design.md` for how this table is produced.

---

## Three-Way Split (Applies to All Granularities)

At **any** granularity, each placement unit can be assigned one of three compute-dispatch roles (storage is two-tier per above):

```
f_gpu      : pinned on GPU permanently (GPU-permanent compute path)
f_prefetch : stored on CPU, streamed to GPU and computed there
f_cpu      : stored on CPU, computed on CPU in parallel with GPU

f_gpu + f_prefetch + f_cpu = 1 (per placement unit)
```

Under the bucket-aware framing, `f_prefetch` and `f_cpu` vary per `BatchDescriptor`; `f_gpu` is their complement for that bucket. Storage placement sets the upper bound `f_cpu_store` on `f_cpu + f_prefetch` across all buckets.

The granularity determines the **unit** to which this split is applied:

| Granularity | f_gpu unit | f_prefetch unit | f_cpu unit |
|---|---|---|---|
| Group | Entire layers permanent on GPU | Entire layers prefetched | Entire layers computed on CPU |
| Layer | Whole tensors permanent on GPU | Whole tensors prefetched | Whole tensors computed on CPU |
| Tensor | Column-portions pinned on GPU | Column-portions prefetched | Column-portions computed on CPU |

---

## Group Granularity (vLLM)

Entire layers are placed on GPU or CPU. Layers are organized into groups of size G; the last M layers per group are offloaded. Binary per-layer decision.

```
G=4, M=1, L=28:
[GPU][GPU][GPU][CPU][GPU][GPU][GPU][CPU]...
 ← group 0  →   ← group 1  →
```

21 layers fully GPU-resident, 7 layers fully offloaded.

**Placement**: `layer_idx % G >= G - M → offload`

**Pipeline**: Resident layers compute while PCIe prefetches the next offloaded layer. (G−1) resident layers provide hiding time.

```
Layer 0 (res): GPU compute ──── │ PCIe: prefetch layer 3 weights
Layer 1 (res): GPU compute ──── │ PCIe: prefetch (continues)
Layer 2 (res): GPU compute ──── │ PCIe: prefetch (continues)
Layer 3 (off): wait + compute   │ PCIe: idle
```

**Hiding constraint**: `(G-1) × resident_layer_time ≥ f_prefetch × layer_weight / PCIe_BW`

With the three-way split at group level: resident layers have f_gpu=1. Offloaded layers can mix f_prefetch and f_cpu — the prefetched portion runs on GPU after transfer, the f_cpu portion is computed on CPU in parallel. Higher f_cpu on offloaded layers extends their compute time but reduces the PCIe transfer requirement, relaxing the hiding constraint.

**Key parameters**: `group_size` (G), `num_in_group` (M=1 preferred), `prefetch_step` (P=1 sufficient for M=1).

---

## Layer Granularity (FlexGen)

Within each layer, individual tensors are placed on GPU or CPU. The same assignment is applied to every layer — no layer is fully resident.

```
w_gpu_percent=50, SelfAttention:
Tensors:    WQ(h×h)   WK(h×h)   WV(h×h)   WO(h×h)
Cumulative: ~25%      ~50%      ~75%      ~100%
            ├── GPU ──┤├───── CPU ────────┤

Every layer:
[WQ,WK on GPU | WV,WO on CPU] × L layers
```

**Placement**: `cumulative_size_percent < w_gpu_percent → GPU, else CPU`. Applied per-module (SelfAttention, MLP) per-layer via `init_weight_list()`.

**Pipeline**: Load next layer's CPU-placed tensors during current layer's compute (single-layer overlap).

```python
for j in range(num_layers):
    load_weight(j+1)     # prefetch next layer's CPU tensors
    compute_layer(j)     # compute current layer
    sync()
```

With the three-way split at layer level: each tensor is assigned to one of f_gpu (permanent GPU), f_prefetch (transferred to GPU just-in-time), or f_cpu (computed on CPU). Since the unit is a whole tensor, each tensor is fully in one category — no partial splits within a tensor.

**Key parameters**: `w_gpu_percent`, `w_cpu_percent` (remainder on disk), `overlap` (enable load/compute overlap), `gpu_batch_size` (batch chunking for weight reuse).

---

## Tensor Granularity (Ours)

Each individual weight matrix is split across GPU and CPU. The **split axis is per-sub-module**, chosen to match vLLM's tensor-parallel conventions and to minimize activation transfer on the PCIe path:

```
WQKV   — column-parallel (shard output dim, K/V-biased picker)
MLP1   — column-parallel (shard output dim, intermediate direction)
MLP2   — row-parallel    (shard input dim, matches MLP1's intermediate sharding)
WO     — not offloaded in Phase 1/2 (fully GPU-resident); see §WO Split Axis Decision
```

Both col-split and row-split are **mathematically exact** — they produce identical results to single-device computation, differing only in the assembly operation at the boundary:

```
Col-split (output-sharded):   Y = concat(X @ W_gpu, X @ W_cpu)           → concat on matching output cols
Row-split  (input-sharded):   Y = X[:, gpu_cols] @ W_gpu + X[:, cpu_cols] @ W_cpu  → add-reduce
```

For WQKV specifically, the column dimension corresponds to the head dimension — the same dimension tensor parallelism splits across GPU ranks (`QKVParallelLinear` inherits from `ColumnParallelLinear` in vLLM). For MLP2, the input dim corresponds to the intermediate expansion — the same dim `RowParallelLinear` shards in `down_proj`. Our CPU offload maps directly onto these existing vLLM conventions — nothing invented; just applied across the device boundary instead of across GPU ranks.

**WQKV, MLP1, and MLP2 share the `[f_gpu, f_prefetch, f_cpu]` three-path structure and receive the same per-bucket `(f_cpu, f_prefetch)` pair from the Planner**. WO is fully GPU-resident in Phase 1/2 (not offloaded) and carries no dispatch state. Storage is set uniformly: at load time, `f_cpu_store` is a single scalar applied to WQKV, MLP1, and MLP2 alike. This uniform dispatch is empirically justified by the measured uniform CPU μs/MB across sub-modules at decode-regime batch sizes (`phase0_findings.md §0.3.4` — 1.02–1.07× spread at B ∈ {16, 64, 128}); per-sub-module optimization buys at most ~1 percentage point of extra f efficiency, within the rounding error of reasonable operating points.

Prefetch distance is committed to **layer-ahead**: the Planner schedules prefetch against the full layer compute-time budget, with a single prefetch queue per layer and one sync per layer boundary. Tensor-ahead (per-sub-phase prefetch) was considered and rejected — its topological constraint (`f_prefetch_{m+1} × W_{m+1} ≤ compute_time_m × PCIe_BW`) would *force* per-sub-module f tuning to avoid prefetch starvation on short-compute sub-modules, a cost we avoid by committing to uniform dispatch. `pcie_bandwidth_allocation_design.md` carries the full comparison.

**Key parameters**: one load-time scalar `f_cpu_store` (applied to WQKV/MLP1/MLP2); per-bucket `(f_cpu, f_prefetch)` pair emitted by the Planner; split axis fixed per sub-module (not a Planner knob).

### Per-Sub-Module Split Axis

The axis choice follows vLLM's `ColumnParallelLinear`/`RowParallelLinear` pairing. The structural reason is that TP places the shard on whichever dim is larger so the per-rank activation slice stays small; for our single-device offload the same logic reduces PCIe activation bytes.

For Qwen2.5-7B at `f_cpu = 10%`, per forward pass, per sub-module:

| Sub-module | in_dim | out_dim | Axis | CPU activation bytes (H2D in + D2H out) |
|---|---|---|---|---|
| WQKV   | 3584  | 5120  | **col** (out) | ~10 KB/token |
| WO     | 3584  | 3584  | col   | ~4 KB/token (symmetric with row) |
| MLP1   | 3584  | 37888 | **col** (out) | ~7 KB/token |
| MLP2   | 18944 | 3584  | **row** (in)  | ~3 KB/token (vs col's ~38 KB — **~12× reduction**) |

For WQKV, the col axis is *required* — the K/V-biased picker (§WQKV Column Choice) needs the head-structured output dim, and row-split would have no analogous bias story. For MLP1, col is byte-optimal because output (2·intermediate) is larger than input. For MLP2, **row is byte-optimal** for the same structural reason (input is 5× larger than output), and pairs with MLP1 for the pipelining save below. For WO, col and row are byte-symmetric (in=out=hidden), and the offload decision is a separate question handled in §WO Split Axis Decision.

### MLP1→MLP2 Pipelining

Because MLP1 is col-parallel and MLP2 is row-parallel, **the intermediate activation never needs to cross the device boundary between them.** This mirrors vLLM's TP FFN pattern (`qwen2.py:113-117`): `gate_up_proj` (col, `gather_output=False`) → local elementwise SwiGLU on each rank's slice → `down_proj` (row) consumes the local slice directly → one all-reduce at the end.

For our single-device offload, the analogous flow is:

```
GPU:  x → W_MLP1_gpu @ x → [N, intermediate·(1-f)] slice     ┐
CPU:  x → W_MLP1_cpu @ x → [N, intermediate·f] slice         │   H2D of x (shared)
      ↓                                                       │
CPU:  SwiGLU on local slice                                  │   no comm
      ↓                                                       │
CPU:  W_MLP2_cpu @ intermediate_slice → [N, hidden] partial  │   D2H of partial sum
GPU:  W_MLP2_gpu @ intermediate_slice → [N, hidden] partial  │
GPU:  Y = GPU_partial + CPU_partial                          ┘   add-reduce
```

Compared to uniform column-split (2 H2D + 2 D2H per MLP block, with the intermediate round-tripping GPU→CPU for MLP2 input), col→row is 1 H2D + 1 D2H. At B=64, Qwen7B, this saves ~2.4 MB of activation PCIe per layer (~67 MB/decode step across 28 layers) — the intermediate re-send from GPU back to CPU for MLP2's input is entirely eliminated.

**Pipelining invariant — automatic under uniform dispatch.** For the save to materialize, the CPU column index set chosen for MLP1's output must exactly equal the CPU column index set chosen for MLP2's input (i.e., the same intermediate indices). Because the Planner emits a single per-bucket `(f_cpu, f_prefetch)` pair applied uniformly to both MLP1 and MLP2, and `f_cpu_store` is set uniformly across MLP1/MLP2 at load time, the two sub-modules select identical intermediate index sets by construction. No separate coupling constraint is needed — the invariant is a consequence of uniform dispatch, not an enforced rule.

SwiGLU-on-slice is safe because `MergedColumnParallelLinear` (gate_up) stores gate and up columns with matching indices: if intermediate index `j` is on CPU, then both `gate[j]` and `up[j]` are on CPU, and `SiLU(gate_j) * up_j` is purely elementwise-local.

### WO Split Axis Decision

**WO is not offloaded in Phase 1/2 — fully GPU-resident, no CPU path, no prefetch path.** This is stricter than the original §0.4.3 Alt B (which only ruled out CPU compute for WO): we additionally exclude `f_prefetch_WO` to avoid the load-time dispatch asymmetry that would arise if WO participated in prefetch-only while other sub-modules have the full three-path structure. At 7B on RTX 4090, WO weight is ~686 MB across layers — a small fraction of the 24 GB budget after MLP/WQKV offload, not worth the design complication.

§0.4.2 measured both alternatives that were on the table:

**Alternative A — WO col-split with merge-before-WO.** Col-split weight offload applied to the merged attention output: GPU merges prefix + suffix via online softmax, sends merged `attn_out` slice to CPU, CPU computes its WO slice, returns partial for concat. Saves `f_cpu_store_WO · 686 MB` of GPU weight (~69 MB at f=10%, ~206 MB at f=30%). **Rejected**: 3 PCIe round trips per layer + ~0.4 ms added critical-path latency per layer (~11 ms per decode step across 28 layers).

**Alternative B — WO GPU-resident, no CPU compute.** GPU does full WO after merge in its idle slack during CPU attention. Single D2H per layer. **Adopted, and extended** to also exclude `f_prefetch_WO` (see above) — WO has no dispatch at all in Phase 1/2.

**Phase 3 revisit condition.** At 14B, WO weight is ~2.5 GB (48 layers × 52 MB). If 14B memory pressure exceeds what MLP+WQKV offload can relieve, Phase 3 can add `f_prefetch_WO` as a late extension — adds a prefetch-only dispatch path for WO, no CPU compute (Alt B's no-f_cpu rule still stands). The asymmetry is accepted in exchange for the 2.5 GB of GPU memory relief.

**Note on the withdrawn "merge-after-WO fusion" idea.** An earlier design considered duplicating WO on CPU and merging at the WO-output level via the linearity of WO (`WO @ merge(a_p, a_s) = α_p · WO(a_p) + α_s · WO(a_s)`). Withdrawn after timing analysis: adds CPU_WO (~2–3 ms for 7B) to CPU's critical path, which is already the bottleneck from memory-bound CPU attention. D2H bytes identical either way, no comm win. See Phase 0 §0.5 and §0.4.2 for the measurement.

**Key parameters**: f_gpu, f_prefetch, f_cpu per tensor (WQKV, WO, MLP1, MLP2).

### WQKV Column Choice: K/V-Biased Slice

WQKV is column-parallel, and unlike MLP1/MLP2 it has a meaningful *which columns* decision because its output columns correspond to distinct logical roles (Q, K, V heads). (For the other col-parallel sub-modules, any column subset of equal size produces an equivalent result.)

**Policy: assign CPU columns in priority order — KV-head groups (K+V together per head), then Q heads.** For Qwen2.5-7B GQA (28 Q heads, 4 KV heads), K+V together is 22% of WQKV output. Below `f_cpu_store_WQKV = 22%`, only some KV-head groups go to CPU; at 22%, all KV heads on CPU and Q stays fully on GPU (the strict Q | K | V split emerges naturally); above 22%, remaining budget starts consuming Q heads.

**K and V for the same head always move together** because attention uses them as a pair (`scores = Q @ K^T`, then `softmax(scores) @ V`). Pairing by KV-head group keeps K[h] and V[h] co-located in the suffix KV cache layout and matches the GQA group structure.

(A K-over-V variation — fill all K heads before V heads — is a near-equivalent alternative. Under our "all suffix on CPU" Phase 2 design, any K or V still on GPU after the WQKV projection must D2H to reach the suffix cache, so both schemes end up with K[h] and V[h] reunited on CPU before attention runs. Continuous analysis gives identical transfer volume `(1 − X/0.22)·(K+V)`; in the discrete case K-over-V can pack 1 extra head-unit of K+V on CPU at some X values because heads are half the atomic size of pairs. The difference is ≤ ~2 KB per layer per step — we pick KV-group for memory layout simplicity, not volume savings.)

**Why K/V-biased is the default (two advantages):**

Phase 2 attention offload requires full new K/V on CPU (for the suffix KV cache write) and full Q on both devices (GPU for prefix attention, CPU for suffix attention). Against this requirement:

1. **Volume saving on D2H transfers.** The GPU-resident K/V portion must D2H to CPU at every decode step. Under K/V-biased, this portion is smaller — `max(0, 1 − f/0.22)·(K+V)` vs. uniform's `(1−f)·(K+V)` — reducing total K/V D2H volume at every `f`.

2. **Minimizing H2D contention with weight prefetch** (the more important advantage). PCIe H2D is an invariant-allocated resource for weight prefetch (see `pcie_bandwidth_allocation_design.md`); D2H is mostly idle. Q heads on CPU must H2D to GPU (for prefix attention); K/V-biased keeps Q *maximally on GPU*, driving Q H2D volume to zero below `f = 22%` and minimizing it above. Uniform would push `f·Q` H2D bytes onto the path contended with weight prefetch.

**Direction asymmetry** is the design point: K/V-biased pushes mandatory Phase-2 transfers onto the idle D2H direction, aligning the weight-offload mechanism with the attention-offload mechanism and the PCIe invariant (100% H2D → weight prefetch).

| Per layer per step (Qwen2.5-7B, B=16) | Uniform, f=22% | K/V-biased, f=22% |
|---|---|---|
| H2D volume (contends with prefetch) | 25 KB | **0 KB** |
| D2H volume (idle direction) | 112 KB | 137 KB |
| Total volume | 137 KB | 112 KB |

Absolute numbers are small in the decode regime; the value is in the direction of the transfer, not the magnitude.

**Other sub-modules (WO, MLP1, MLP2)** have no "K/V-like" column semantics — any column slice produces the same result. The column-choice question is WQKV-specific.

---

## What Granularity Changes

### 1. Non-Uniform Sub-Module Compute

Layers have roughly uniform total compute time. But within a layer, sub-modules have vastly different sizes:

| Sub-module | Size (7B) | Fraction of layer |
|---|---|---|
| WQKV | 41 MB | 8.8% |
| WO | 18 MB | 3.9% |
| MLP1 (gate_up) | 271 MB | 58.2% |
| MLP2 (down) | 136 MB | 29.2% |

**Group granularity** treats the layer as a single 466 MB block. The prefetch/compute schedule cannot exploit the internal structure — the entire layer must be prefetched before any computation starts, and f_cpu/f_prefetch applies uniformly to the whole layer.

**Tensor granularity with layer-ahead prefetch** (our committed design) assigns one per-bucket `(f_cpu, f_prefetch)` pair uniformly to WQKV, MLP1, MLP2 (see opening of this section for the empirical justification). The prefetch budget is global across the full layer time; any sub-module's `f_prefetch × W_m` draw from the shared pool. This is simpler than per-sub-module tuning and — given uniform CPU μs/MB at decode — sacrifices no meaningful optimality.

The alternative, **tensor-ahead prefetch with per-sub-module tuning** (rejected), would assign different `f` values per sub-module to navigate the topological constraint `f_prefetch_{m+1} × W_{m+1} ≤ compute_time_m × PCIe_BW`. WO's short compute phase (~0.02 ms at small decode) would starve MLP1's prefetch budget, forcing MLP1 to high `f_cpu` just to extend its own compute phase and unblock MLP2's prefetch (a "self-bootstrapping" cascade). Tensor-ahead's one remaining advantage — fine-grained per-phase control over PCIe contention — is moot under our layer-grain mechanism: `phase0_findings.md §0.5` shows that fg activation returns route through SM-issued UVA loads on a separate hardware path from CE0 (the H2D copy engine), so fg events don't queue behind bg prefetches regardless of how bg is scheduled (fg_s2c stays at ~30–35 μs across all bg chunk sizes). With the queue dependency eliminated at the layer grain, tensor-ahead's fine control has nothing to control. We avoid this complexity entirely by using layer-ahead and uniform dispatch.

Group granularity cannot exploit sub-module non-uniformity — it sees one 466 MB block, not four differently-sized sub-modules.

### 2. Buffer Size

Our committed design is **tensor partition + layer-ahead prefetch**, with buffer size `sum_m (f_prefetch_m × W_m)` — all prefetched weights held concurrently until consumed within the layer.

| Partition | Prefetch distance | Buffer size | Example (7B, 50% offload) |
|---|---|---|---|
| Group (vLLM) | Multi-layer-ahead | layer_weight (full offloaded layer) | 466 MB |
| Layer (FlexGen) | Layer-ahead | f_prefetch × layer_weight | 233 MB |
| **Tensor (ours)** | **Layer-ahead** | sum(f_prefetch_i × W_i) | **~233 MB** |
| Tensor (rejected) | Tensor-ahead | max(f_prefetch_i × W_i) | 73 MB |

Tensor-ahead has the smallest buffer but imposes the per-sub-module topological constraint discussed above. The extra 160 MB for layer-ahead (~1% of 24 GB) is negligible on RTX 4090 and pays for the design simplification.

### 3. Interaction with KV Prefetch

Our design **rejects KV prefetch** (see `pcie_bandwidth_allocation_design.md`): the full PCIe H2D budget is allocated to weight prefetch. Under layer-ahead, weight prefetch keeps PCIe continuously busy across the layer, leaving no idle window to exploit for KV traffic anyway. This interaction question therefore does not apply to the committed design — included here only to document that the rejection is consistent across granularities.

### 4. Scheduling Complexity and Overhead

| | Group | Layer | Tensor |
|---|---|---|---|
| PCIe transfers per layer | 0 (resident) or 1 (full layer) | 1 (CPU-placed tensors) | Up to 5 (per sub-phase) |
| Sync points per layer | 1 event (offloaded only) | 1 sync | ~5 events |
| Scheduling logic | Group index math | Cumulative % threshold | Per-tensor three-way optimizer |

**Overhead magnitude**: Tensor granularity adds ~5 sync points per layer (~5 μs). Smallest PCIe transfer is ~4 MB (200 μs at 22 GB/s), so latency overhead < 2%. At layer times of 1–5 ms, total overhead is **< 0.5%**. Coarser granularities have lower overhead but the difference is negligible in absolute terms.

**CUDA Graph compatibility**: vLLM captures the entire decode forward pass as a CUDA Graph, eliminating per-kernel launch overhead (~5–10 μs each). The existing PrefetchOffloader is graph-compatible (uses async copy stream + `cudaStreamWaitEvent`). In our tensor-granularity pipeline:
- PCIe prefetch (copy stream + events): **graph-compatible** (same as PrefetchOffloader)
- GPU kernels: **graph-compatible**
- CPU task submission + result sync: **NOT graph-compatible** — requires host intervention

Without CUDA Graphs, kernel launch overhead adds ~2–3 ms per decode step (28 layers × ~10 kernels × ~10 μs). This is significant relative to 1–5 ms layer times.

**KTransformers solution**: Encapsulate CPU submit/sync in `cudaLaunchHostFunc` callbacks, which are embeddable in CUDA Graphs. A CPU control thread pushes tasks into a lock-free queue; background workers execute in parallel. The GPU stream stalls at the sync callback until CPU finishes — this stall is the `max(GPU, CPU)` bottleneck we already model, not additional overhead. The entire decode path fits in one CUDA Graph, preserving vLLM's graph-based execution.

This is effectively a **prerequisite** for production integration with vLLM, not an optional optimization. For prototyping, CUDA Graphs can be disabled as a stepping stone.

### 5. CPU Compute Utilization

All three granularities can use f_cpu to compute on CPU in parallel. But finer granularity allows **more precise tuning**:

- **Group**: f_cpu is per-layer — the entire layer uses the same CPU fraction
- **Layer**: f_cpu is per-tensor — each tensor (WQ, MLP1, ...) is fully GPU or fully CPU. Can't partially split a tensor.
- **Tensor**: f_cpu is per-tensor and continuous — each tensor can have any fraction on CPU, tuned to the preceding phase's prefetch budget

---

## Comparison Table

| | Group (vLLM) | Layer (FlexGen) | Tensor (Ours) |
|---|---|---|---|
| Placement unit | Layer | Whole tensor | Tensor columns (col-parallel) or input cols (row-parallel), per sub-module |
| f_gpu/f_prefetch/f_cpu | Per-layer (uniform within layer) | Per-tensor (binary: 0 or 1) | Uniform across WQKV/MLP1/MLP2 per bucket; WO not offloaded |
| Prefetch distance | Multi-layer-ahead (G−1) | Layer-ahead | **Layer-ahead** (committed) |
| Exploits sub-module non-uniformity | No | Partially (different tensors, different placement) | Yes — different split axes per sub-module; uniform dispatch over them |
| Buffer size | layer_weight | f_prefetch × layer_weight | sum_m (f_prefetch × W_m) ≈ f_prefetch × layer_weight |
| KV prefetch | None — PCIe fully occupied by weights | Same | Same — 100% PCIe H2D to weight prefetch |
| Overhead | Lowest | Low | Low (~0.5%) |
| Scheduling | Simple | Simple | One prefetch queue per layer boundary (1 sync) |

---

## Implementation Note: WQKV K/V-Pin Optimization

*This is a runtime implementation detail in `CpuComputeDispatcher`, not part of the thesis-level design narrative above.*

Under the uniform per-bucket dispatch story, the Planner emits one `(f_cpu, f_prefetch)` pair applied to WQKV, MLP1, and MLP2. At runtime, `CpuComputeDispatcher` applies an optimization specifically to WQKV: the K/V portion of WQKV (the first `2 × num_kv_heads × head_dim` output columns, ordered by the K/V-biased picker) is **always CPU-computed, never prefetched**. Prefetch on WQKV only applies to Q columns above the K/V boundary.

**Why this optimization exists.** If the per-bucket `f_cpu` lands below the K/V fraction (e.g., at 7B decode the Planner's optimum is ~1–2% while the K/V fraction is ~22%), uniform dispatch would route part of K/V through the prefetch path: weight H2D to GPU, K/V computed on GPU, K/V output D2H back to the CPU suffix cache. That's a full PCIe round-trip for K/V every forward — wasteful since CPU-computing K/V in-place produces output directly on CPU with no transfer at all. The K/V-pin guard avoids it.

**When the guard binds vs no-ops.** At `f_cpu ≥ K/V-fraction` the guard is a no-op (K/V is already CPU-computed under uniform dispatch). At `f_cpu < K/V-fraction` the guard raises WQKV's effective `f_cpu_compute` to cover K/V, with WQKV's `f_prefetch` correspondingly reduced. The Planner's cost model should account for this nudge — see `planner_design.md §7.3`.

**Why this is an implementation footnote, not a design claim.** The thesis-level story is "uniform dispatch across WQKV/MLP1/MLP2." The K/V-pin guard is a small runtime optimization that happens to live inside WQKV's dispatcher without changing the design-level interface. It's documented here so readers of the implementation can reconstruct the behavior, not as part of the mechanism's contribution story.
