# Weight Offload Design
Following FlexGen's terminology, weight offloading can be partitioned at three granularity levels. From coarse to fine:

| Granularity | Unit of placement | Example | Used by |
|---|---|---|---|
| **Group** | Groups of layers | Assign each layer to GPU or CPU, spaced in groups | vLLM PrefetchOffloader |
| **Layer** | Tensors within a layer | Assign each tensor (WQ, WK, ...) to GPU or CPU | FlexGen |
| **Tensor** | Elements within a tensor | Assign columns within a tensor to GPU or CPU | **Ours** |

---

## Three-Way Split (Applies to All Granularities)

At **any** granularity, each placement unit can be assigned one of three roles:

```
f_gpu      : pinned on GPU permanently
f_prefetch : stored on CPU, prefetched to GPU via PCIe during preceding computation
f_cpu      : stored on CPU, computed on CPU in parallel with GPU

f_gpu + f_prefetch + f_cpu = 1 (per placement unit)
```

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

Each individual weight matrix is split at the column level across GPU and CPU. Both devices compute their portion in parallel, with results concatenated.

```
W = [W_gpu_permanent | W_gpu_prefetched | W_cpu]
Y = concat(X @ W_gpu_permanent, X @ W_gpu_prefetched, X @ W_cpu)
```

Column-parallel split is mathematically exact — produces identical results to single-device computation.

Each sub-module has independently tuned f_gpu, f_prefetch, f_cpu. The prefetch scheduling (tensor-ahead, layer-ahead, or multi-layer-ahead) is an independent design dimension — see `pcie_bandwidth_allocation_design.md` for the full analysis.

**Key parameters**: f_gpu, f_prefetch, f_cpu per tensor (WQKV, WO, MLP1, MLP2).

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

**Layer and tensor granularity** can assign different f_gpu/f_prefetch/f_cpu per sub-module, exploiting the non-uniformity. This enables:

- **Per-sub-module tuning**: MLP1 (large, preceded by short WO phase) can have high f_gpu to avoid the prefetch bottleneck, while MLP2 (preceded by long MLP1 phase) can rely more on f_prefetch.
- **Self-bootstrapping pipeline** (tensor-ahead only): A sub-module with high f_cpu creates a long compute phase, generating prefetch budget for the next sub-module. This cascading effect only works with tensor-ahead prefetch distance.

Example (tensor granularity, tensor-ahead, S=500):
```
WO is small → short phase → MLP1 forced to high f_cpu
→ MLP1 phase is long (3.34 ms) → MLP2 can use large f_prefetch (54% on GPU)
→ moderate MLP2 phase → WQKV(next) gets decent prefetch budget
```

With layer-ahead prefetch distance, this bottleneck disappears — MLP1's f_prefetch comes from the global layer budget, not WO's tiny phase.

Group granularity cannot exploit sub-module non-uniformity — it sees one 466 MB block, not four differently-sized sub-modules.

### 2. Buffer Size

Buffer size depends on **both** partition granularity and prefetch distance (see `pcie_bandwidth_allocation_design.md` for prefetch distance analysis). The buffer must hold all prefetched weights before they can be consumed.

| Partition | Prefetch distance | Buffer size | Example (7B, 50% offload) |
|---|---|---|---|
| Group | Multi-layer-ahead | layer_weight (full offloaded layer) | 466 MB |
| Layer | Layer-ahead | f_prefetch × layer_weight (all CPU-placed tensors) | 233 MB |
| Tensor | Tensor-ahead | max(f_prefetch_i × W_i) (reused between phases) | 73 MB |
| Tensor | Layer-ahead | sum(f_prefetch_i × W_i) (all held at once) | ~233 MB |

Tensor partition with tensor-ahead has the smallest buffer (phases are sequential, buffer reused). With layer-ahead, buffer is larger but prefetch budget is global — no per-tensor bottleneck.

### 3. Interaction with KV Prefetch

If KV prefetch is used (see `pcie_bandwidth_allocation_design.md` for the allocation decision), it competes with weight prefetch for PCIe H2D. The granularity determines how much room exists:

**Group granularity**: During (G−1) resident layers, PCIe H2D is fully occupied by weight prefetch. **No room for KV prefetch** without contention.

**Layer granularity**: PCIe is occupied transferring weight tensors every layer. **Same problem.**

**Tensor granularity**: Sub-layer pipeline has a naturally free slot during the WQKV phase (WQKV already on GPU from previous layer's last phase). **KV prefetch could use this slot without contending with weight prefetch.**

| Granularity | KV prefetch room |
|---|---|
| Group | None — PCIe fully occupied by weight during resident layers |
| Layer | None — PCIe occupied by weight tensors every layer |
| Tensor | Slot 1 (WQKV phase) naturally free — could be used for KV |

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
| Placement unit | Layer | Whole tensor | Tensor columns |
| f_gpu/f_prefetch/f_cpu | Per-layer (uniform within layer) | Per-tensor (binary: 0 or 1) | Per-tensor (continuous 0–1) |
| Min prefetch distance | Multi-layer-ahead (G−1) | Layer-ahead | Tensor-ahead (most flexible) |
| Exploits sub-module non-uniformity | No | Partially (different tensors, different placement) | Fully (different f per tensor + pipeline) |
| Buffer size | layer_weight | f_prefetch × layer_weight | Depends on prefetch distance (see PCIe doc) |
| KV prefetch room | **None** | **None** | **Slot 1 free** (WQKV phase, tensor-ahead only) |
| Overhead | Lowest | Low | Low (~0.5%) |
| Scheduling | Simple | Simple | Per-sub-phase or per-layer (depends on prefetch distance) |
