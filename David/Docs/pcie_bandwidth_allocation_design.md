# PCIe Bandwidth Allocation Design

> **Note on terminology.** This doc uses `f_gpu_kv`, `f_prefetch_kv`, `f_cpu_kv` to frame a three-way KV split for the *analysis* of why we rejected KV prefetch. The thesis's actual KV mechanism is a simpler **two-pool model** (`KV_gpu_bytes` + `KV_cpu_bytes`, with prefix-on-GPU / suffix-on-CPU as a fixed topology — see `attention_offload_design.md`). The three-way framing here is kept as-is because it's the cleanest way to present the contention analysis and explain why the two-pool simplification is correct.

## The Contention

Both weight offloading and attention offloading use a three-way split (f_gpu, f_prefetch, f_cpu). The f_gpu and f_cpu components are **orthogonal** — they use GPU memory and CPU compute respectively, with no PCIe contention. Only f_prefetch competes:

```
Orthogonal (no contention):
  f_gpu_weight   ↔  f_gpu_kv      : both use GPU memory, no PCIe
  f_cpu_weight   ↔  f_cpu_kv      : both use CPU compute, no PCIe

Contention (shared resource):
  f_prefetch_weight  ↔  f_prefetch_kv  : both need PCIe H2D (22 GB/s on RTX 4090)
```

The question: how should the limited PCIe H2D bandwidth be allocated between weight prefetch and KV prefetch?

---

## Key Insight: Prefetch Serves Different Purposes at Different Scales

Prefetch can replace two different components, depending on what's available:

**Replaces f_gpu** (takes from permanent GPU storage): 1 MB prefetched = 1 MB of GPU memory freed. The freed memory is fungible — available for KV cache (more beams). This applies to both weight and KV prefetch equally: **at the memory level, they are equivalent**.

**Replaces f_cpu** (takes from CPU compute): no GPU memory is freed. The benefit is **latency reduction** — work moves from the slow CPU path to fast GPU compute via streaming. Here, weight and KV prefetch are **not equivalent**: weight prefetch reduces per-layer latency unconditionally (every step, every layer), while KV prefetch only helps when CPU attention is already the bottleneck.

Which replacement happens depends on the model size — and in both cases, weight prefetch is preferred.

---

## Weight Prefetch by Model Size

### Small model (weights mostly fit on GPU)

Prefetch **replaces f_gpu** — moves weight from permanent GPU storage to on-demand streaming. GPU still computes it, but the permanent memory is freed for KV.

```
f_gpu: 90% → 80%  (freed 10% of weight memory → more KV → more beams)
f_prefetch: 0% → 10%  (still computed on GPU, just not permanently stored)
f_cpu: 10% → 10%  (unchanged, layer time similar)
```

Purpose: **free GPU memory for KV**. Layer time barely changes (GPU still computes the prefetched portion). The gain is more beams.

### Large model (weights don't fit on GPU)

Prefetch **replaces f_cpu** — moves weight from slow CPU compute to fast GPU compute via streaming. f_gpu can't increase (no room), so no GPU memory is freed.

```
f_gpu: 30% → 30%  (unchanged, can't increase — no room)
f_prefetch: 0% → 10%  (shifted from CPU to GPU compute)
f_cpu: 70% → 60%  (less CPU bottleneck → faster layers)
```

Purpose: **reduce latency**. GPU memory is unchanged, but each layer is faster because less work falls on the slow CPU path. This compounds over every decode step across every layer.

### Summary

| | Small model | Large model |
|---|---|---|
| Prefetch replaces | f_gpu | f_cpu |
| What's gained | GPU memory (→ more KV) | Faster layers (→ less CPU bottleneck) |
| What's unchanged | Layer time (~same) | GPU memory (~same) |

---

## Why Weight Prefetch Over KV Prefetch

### 1. Batch size is our control variable

The f_cpu_kv bottleneck (CPU suffix attention) scales with B × S. But batch size B is **our choice** — if CPU attention becomes too slow, we reduce B (more scheduling rounds, each faster). The bottleneck is not a hard constraint; it's a dial we control.

Weight f_cpu penalty, on the other hand, hits **every decode step regardless of batch size**. There is no knob to dial it away.

### 2. KV prefetch cannot match weight prefetch at either scale

- **Small models**: KV prefetch frees GPU KV space, but so does weight prefetch (freed weight memory → KV space). Both achieve the same goal. But weight prefetch doesn't increase CPU attention load, while KV prefetch provides no additional benefit over weight prefetch.
- **Large models**: KV prefetch frees KV space, enabling larger batch — but larger batch increases f_cpu_kv load, worsening the CPU attention bottleneck. Weight prefetch directly attacks the latency problem without this feedback loop.

### 3. Weight offloading is often mandatory; KV prefetch is always optional

For models that don't fit on GPU (14B+), weight offloading is a **necessity to run at all**. PCIe for weight prefetch reduces the latency penalty of this mandatory offloading.

KV prefetch is never mandatory — suffix KV can always fall back to f_cpu_kv (CPU attention), with batch size as the release valve.

| Model | Total weight | Fits on 24 GB GPU? | Weight offloading |
|---|---|---|---|
| 7B | 13.0 GB | Yes | Optional |
| 14B | 26.5 GB | No | Mandatory |
| 32B | 62.4 GB | No | Mandatory (massive) |

### 4. Weight prefetch benefit is universal; KV prefetch is situational

- **Weight prefetch**: benefits every decode step from step 1, regardless of batch size or suffix length
- **KV prefetch**: only matters when suffix is long enough AND batch is large enough that CPU attention is the bottleneck. Early in generation (short suffix), CPU attention is cheap — KV prefetch provides no benefit

---

## Design Decision: No KV Prefetch

**All PCIe H2D bandwidth is allocated to weight prefetch. Suffix KV uses f_cpu_kv (CPU attention) exclusively.**

The attention split simplifies to:
- **GPU**: prefix attention (f_gpu_kv, always GPU-resident)
- **CPU**: suffix attention (f_cpu_kv, computed on CPU in parallel)
- **Merge**: online softmax via `merge_attn_states` (exact, no approximation)

### What this eliminates

- No KV prefetch scheduling (per-step decisions about which blocks to transfer)
- No per-slot weight-vs-KV PCIe assignment
- No GPU buffer for prefetched suffix KV
- No split block table management (GPU-resident vs CPU-resident suffix blocks)

### What remains

- Weight three-way split (f_gpu, f_prefetch, f_cpu) with full PCIe for prefetch pipeline
- CPU suffix attention with batch size as the release valve for CPU bottleneck
- GPU prefix attention (unchanged from existing cascade pattern)

---

## Prefetch Distance

Partition granularity (see `weight_offload_design.md`) determines **how** weights are split. Prefetch distance determines **when** prefetched data is transferred. These are two independent dimensions of the PCIe scheduling design.

### The Spectrum

| Prefetch distance | Hiding time | What's happening |
|---|---|---|
| **Tensor-ahead** | Preceding tensor's compute | Prefetch next tensor during current tensor's compute |
| **Layer-ahead** | 1 full layer's compute | Prefetch next layer during current layer's compute |
| **Multi-layer-ahead (K)** | K *offloaded modules* of compute | Prefetch K offloaded modules ahead (vLLM's `prefetch_step`, default K=1; configurable). With one layer offloaded per group (N=1) this collapses to "(G−1)-actual-layer-ahead"; with N>1, hiding is non-uniform across the group (see `phase0_findings.md §0.10.2`). |

Longer prefetch distance = more hiding time = can prefetch more data. But also = larger buffer (must hold prefetched data until consumed).

### Constraint: Partition Limits Minimum Prefetch Distance

Coarser partition forces longer minimum prefetch distance — you can't start consuming a unit until the entire unit has arrived:

| Partition | Minimum prefetch distance | Why |
|---|---|---|
| Group | Multi-layer-ahead | Entire layer is the unit — must arrive before layer executes. Need (G−1) layers of hiding. |
| Layer | Layer-ahead | Whole tensors are the unit — all CPU-placed tensors must arrive before layer starts. Could also do tensor-ahead if tensors are executed individually. |
| Tensor | Tensor-ahead | Column slices are the unit — only need to arrive before that tensor's compute. |

**Finer partition enables shorter prefetch distance, but doesn't require it.** You can always prefetch farther ahead than the minimum.

### Combinations Used by Each System

| System | Partition | Prefetch distance |
|---|---|---|
| vLLM PrefetchOffloader | Group | K offloaded-modules ahead (default K=1; with N=1, ≈ group-ahead = (G−1) actual layers) |
| FlexGen | Layer | Layer-ahead |
| **Ours** | Tensor | **Layer-ahead** (committed) |

### Our Committed Design: Tensor Partition + Layer-Ahead

Prefetch all of next layer's f_prefetch portions during the entire current layer's compute. One prefetch queue per layer boundary, one sync per layer.

```
Layer N:   GPU+CPU compute full layer   │ PCIe: prefetch layer N+1's f_prefetch (all sub-modules)
Layer N+1: GPU+CPU compute (uses prefetched) │ PCIe: prefetch layer N+2's f_prefetch
```

- **Global budget per layer**: `Σ_m (f_prefetch × W_m) ≤ layer_time × PCIe_BW` over the offloaded sub-modules {WQKV, MLP1, MLP2}.
- **Buffer**: `Σ_m (f_prefetch × W_m)` — all prefetched weights resident when the layer begins. At 7B f=50%, ~233 MB (~1% of 24 GB budget — negligible).
- **Scheduling**: single prefetch queue populated at the layer boundary; one `cudaStreamWaitEvent` to gate compute on prefetch completion. Clean Phase 1c CUDA-graph integration (one sync per layer instead of per sub-phase).

### Alternatives Considered and Rejected

#### Tensor-ahead (per-sub-phase prefetch)

Prefetch next sub-module's weights during the current sub-module's compute. Smaller buffer (`max(f_prefetch_i × W_i)` ≈ 73 MB at 7B f=50%) but introduces a **topological constraint**:

```
f_prefetch_{m+1} × W_{m+1} ≤ compute_time_m × PCIe_BW
```

At decode with WO's short compute phase (~0.018 ms), the MLP1 prefetch budget starves to ~0.4 MB. The only way out is per-sub-module f tuning — force MLP1 to high `f_cpu` so its compute phase lengthens, bootstrapping MLP2's prefetch budget. A cascade the Planner would have to solve via a recurrence.

**Why rejected:**
1. `phase0_findings.md §0.3.4` shows CPU μs/MB is uniform across sub-modules at decode B ≥ 16. Per-sub-module f tuning (tensor-ahead's main purpose) buys effectively nothing.
2. fg-wait against bg prefetch is already bounded **at the layer grain** by routing fg activation returns through SM-issued UVA loads (`phase0_findings.md §0.5`). The UVA copy kernel uses a different PCIe path than CE0 (the H2D copy engine), so fg events don't queue behind bg DMA regardless of how bg is scheduled — fg_s2c stays at ~30–35 μs whether bg is a single 4 MB H2D or many 64 KB chunks. Tensor-ahead's per-sub-phase contention control adds no value when the layer-grain mechanism already eliminates the queue dependency.
3. Buffer savings (~160 MB) are negligible on a 24 GB GPU.
4. Per-sub-phase scheduling (~5 sync points per layer) adds Phase 1c CUDA-graph complexity for no measurable benefit.

#### Multi-layer-ahead (K > 1)

Same as layer-ahead but starts K layers in advance. Budget grows to `K × layer_time × PCIe_BW`, buffer to `K × Σ_m (f_prefetch × W_m)`.

**Why not adopted**: 1-layer-ahead already provides enough budget (`layer_time × 22 GB/s` ≈ 10–30 MB at decode, scales with batch). Going deeper just inflates buffer without adding a benefit we need. **Empirically confirmed in `phase0_findings.md §0.10.2c`**: at G=14 N=4 on Qwen2.5-7B, K=1 → 7.00 s @ B=64; K=2 → 6.60 s (5.6% improvement); K=3 → 6.60 s (no further); **K=4 OOMs at B=64** because the prefetch buffer pool grows linearly with K (~1.86 GiB extra GPU residency at K=4).

---

## Summary

1. **Prefetch serves different purposes by scale**: replaces f_gpu (frees memory) for small models; replaces f_cpu (reduces latency) for large models
2. **Weight prefetch wins at both scales**: when replacing f_gpu, weight and KV prefetch are equivalent — but weight is simpler. When replacing f_cpu, weight prefetch is strictly better (unconditional latency reduction vs situational)
3. **KV prefetch has no regime where it's preferred**: at best equivalent (small models), at worst inferior (large models)
4. **Batch size controls f_cpu_kv cost** — CPU attention bottleneck is a dial, not a wall
5. **Design simplification**: all PCIe → weight prefetch. Suffix attention → CPU only. Merge via online softmax.
6. **Prefetch distance committed to layer-ahead**: one prefetch queue per layer boundary, one sync per layer. Tensor-ahead rejected because its topological constraint would force per-sub-module f tuning — needless given uniform CPU μs/MB (`phase0_findings.md §0.3.4`) and the layer-grain fg-wait bypass via SM-issued UVA copy kernel (`phase0_findings.md §0.5`). Buffer cost (~160 MB extra at 7B f=50%) is ~1% of GPU budget — negligible.
