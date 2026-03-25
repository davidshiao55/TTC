# Test-Time Compute on Consumer GPU

## 1. Problem Statement

**Workload**: Test-time compute (TTC) inference — beam search, tree search, best-of-N sampling.
**Computing Platform**: Consumer GPU (NVIDIA RTX 4090, 24 GB GDDR6X)

**Existing Methods**: Locality-Aware Beam Scheduling, FastTTS — these optimize KV-cache handling (movement, scheduling, reuse), but assume model weights and active KV cache stay GPU-resident throughout inference.

**Limitations**:
- **Model size**: Max model size ≤ VRAM.
- **Batch size / throughput**: With GPU-resident weights, batch size is bounded by leftover GPU VRAM, so throughput is fundamentally constrained.

**Objective**: Maximize utilization of all resources (GPU compute/memory, CPU compute/memory, PCIe bandwidth) to achieve maximum TTC throughput.

| Resource | Capacity | Used in GPU-only inference? |
|---|---|---|
| GPU memory bandwidth | 1,008 GB/s | Yes (primary) |
| CPU memory bandwidth | ~80 GB/s DDR5 | **Idle** |
| PCIe 4.0 x16 | ~22 GB/s per direction | **Idle** |
| System RAM | 32–128 GB | **Idle** |

---

## 2. Background & Motivation

### Test-Time Compute

Rather than relying on ever-larger pretraining budgets, test-time methods use dynamic inference strategies that allow models to "think longer" on harder problems. The common paradigm is *search against a verifier*: generate multiple candidate answers, then use a verifier to select the best one. Search strategies include:

- Best-of-N
- Beam search
- Diverse verifier tree search (DVTS)

However, maintaining two models (generator + verifier) and KV cache from multiple search paths is challenging for edge deployment.

### Edge Deployment for TTC

Existing work has attempted to improve test-time scaling at the edge through:

- Speculative beam extension
- Dynamic Prefix-Aware Scheduling
- Asymmetric Multi-Model Memory Allocation

But these methods assume model weights and active KV cache stay GPU-resident throughout inference.

#### Model Size Limitation

A consumer GPU like the RTX 4090 has only 24 GB of memory, which can fit models under ~12B parameters. TTC pipelines typically require two models (a generator and a verifier), further limiting deployable model size.

#### Throughput Limitation

With vLLM's continuous batching, maximum batch size is constrained by GPU memory available for KV cache. FastTTS submits all beams in one `generate()` call; vLLM's scheduler (`scheduler.py:541`) allocates KV blocks per request and processes excess beams round by round:

```
wall_time ≈ ⌈total_beams / batch_capacity⌉ × time_per_step
```

### Challenges

The common methods for enabling larger models and higher throughput — **weight offloading** and **attention offloading** — are not directly applicable to TTC.

#### Weight Offloading

Existing offloading methods are typically optimized for one of two regimes:

- **Hybrid offloading** (online edge inference, low-batch): prioritizes latency but neglects throughput and KV cache memory implications of batching.
- **GPU-centric offloading** (offline inference, large-batch): prioritizes throughput but neglects latency and underutilizes CPU computation.

Neither regime applies to TTC because:
1. TTC requires efficient handling of multi-path generation while maintaining low latency — neither regime's strategy addresses both.
2. Existing methods do not consider that even when a model fully fits on GPU, weight offloading can be beneficial to enlarge batch size and therefore throughput. <!-- TODO: still considering whether to include this point — need to verify with experiments -->
3. Existing methods are designed for a single model, but TTC pipelines have two models with different characteristics (generator vs. verifier).

#### Attention Offloading

Existing attention offloading methods assume the model fits in GPU memory. To overlap CPU/GPU computation, they split the batch in two and overlap their respective computations.

These methods don't apply to TTC because:
1. Splitting the batch conflicts with our goal of maximizing batch size.
2. With weight offloading in the pipeline, there are better opportunities for CPU-GPU overlap than batch splitting.

---

## 3. Proposed Solution: Resource Partitioning

We partition computation and data across GPU, CPU, and PCIe to maximize utilization of all resources. The three methods below determine *where* each piece of work executes; Section 4 determines the optimal partition fractions.

### 3.1 Attention Offloading

TTC tree search naturally exhibits prefix-sharing. We partition KV cache by topology:
- **Shared prefix** → GPU (KV cache + attention)
- **Per-beam suffix** → CPU (KV cache + attention)

```
                [Shared Prefix - GPU]
                /        |         \
        [Suffix A]  [Suffix B]  [Suffix C]   ← CPU
```

Merge via online softmax (mathematically exact). Already implemented in vLLM's `cascade_attention()` and `merge_attn_states()`.

### 3.2 Hybrid CPU-GPU Weight Computation

The CPU has compute resources idle during GPU-only inference. Each weight matrix is split into two compute partitions via column-parallel partitioning:

```
W = [W_gpu | W_cpu]
Y = concat(X @ W_gpu, X @ W_cpu)
(column-parallel, mathematically exact)
```

- **f_gpu**: stored and computed on GPU
- **f_cpu**: stored on CPU, computed on CPU in parallel with GPU

At small f_cpu (~9%), CPU compute fits within GPU idle time (GPU is memory-bandwidth-bound), so the split is effectively free — GPU memory is freed with no latency cost. The split is applied at **tensor granularity** — each sub-module (WQKV, WO, MLP1, MLP2) has independently tuned fractions.

### 3.3 PCIe Weight Prefetch

Sections 3.1 and 3.2 partition *computation* between GPU and CPU. This section addresses how to use the PCIe bandwidth that remains available.

Rather than splitting PCIe between weight prefetch and KV prefetch, we dedicate **all PCIe H2D bandwidth to weight prefetch**. A portion of f_gpu weights need not be permanently GPU-resident — they can be stored on CPU and streamed to GPU via PCIe during preceding computation. This introduces a three-way weight split:

```
W = [W_gpu_permanent | W_gpu_prefetched | W_cpu]
```

- **f_gpu_permanent**: pinned on GPU
- **f_prefetch**: stored on CPU, streamed to GPU via PCIe during preceding computation, computed on GPU after arrival
- **f_cpu**: stored and computed on CPU (unchanged from 3.2)

The role of prefetch depends on model size:
- **Small models** (weights fit on GPU): prefetch **replaces f_gpu** — frees GPU memory for KV cache (more beams) with minimal latency impact.
- **Large models** (weights don't fit): prefetch **replaces f_cpu** — shifts work from the slow CPU path to GPU, reducing per-layer latency.

KV prefetch is not used because: (1) batch size is our control variable for CPU attention cost, (2) weight prefetch is universal (benefits every decode step) while KV prefetch is situational, and (3) eliminating KV prefetch simplifies scheduling. See `pcie_bandwidth_allocation_design.md` for the full analysis.

---

## 4. Proposed Solution: Performance Modeling

The partition methods in Section 3 introduce tunable fractions (f_gpu, f_prefetch, f_cpu) for each sub-module. A performance model determines the optimal values, taking into account:
- Hardware characteristics (GPU/CPU compute bandwidth, PCIe bandwidth, memory capacities)
- Model architecture differences between generator and verifier
- Target batch size and search configuration

All methods compose to maximize utilization of all available resources.

---

## 5. Reference Architecture

### Qwen2.5 Model Specs

| | 7B | 14B | 32B |
|---|---|---|---|
| hidden / heads / kv_heads | 3584 / 28 / 4 | 5120 / 40 / 8 | 5120 / 40 / 8 |
| intermediate / layers | 18944 / 28 | 13824 / 48 | 27648 / 64 |
| Layer weight (BF16) | 466 MB | 551 MB | 975 MB |
| Total BF16 weights | ~14.1 GB | ~28 GB | ~64 GB |
| KV per token (all layers) | 56 KB | 192 KB | 256 KB |

### Per-Layer Timing (BF16, small-batch decode)

| Metric | 7B | 14B | 32B |
|---|---|---|---|
| GPU compute (memory-BW-bound) | ~0.5 ms | ~0.6 ms | ~1.1 ms |
| PCIe transfer (full layer) | 21.2 ms | 25.0 ms | 44.3 ms |
| CPU compute (full layer) | 5.8 ms | 6.9 ms | 12.2 ms |
| **PCIe : GPU ratio** | **~42×** | **~42×** | **~40×** |

The ~40× ratio is the fundamental problem: **pure prefetch cannot hide latency for BF16 decode.**

---

## 6. Analysis

### 6.1 Attention Offloading

With attention offloading, GPU KV holds only the shared prefix (e.g., ~0.5 GB for 4096 prefix tokens). All per-beam suffix KV moves to CPU — attention over suffix is computed on CPU in parallel with GPU prefix attention, merged via online softmax. No suffix KV is prefetched to GPU; all PCIe bandwidth is reserved for weight prefetch (see `pcie_bandwidth_allocation_design.md`).

**Tradeoff**: GPU prefix attention time is fixed (shared cache, constant size). As batch size grows, CPU suffix attention time grows proportionally. Beyond a crossover point, CPU attention becomes the bottleneck and the GPU idles after its own attention finishes. **Batch size is our control variable** — if CPU attention becomes too slow, reduce B (more scheduling rounds, each faster). The optimal batch size balances linear layer efficiency gains against CPU attention overhead.

### 6.2 Resident Hybrid: Free GPU Memory (f_cpu ≈ 9%)

For GPU-resident layers, permanently place ~9% of weights on CPU. CPU computes its fraction in parallel with GPU. At small f_cpu, CPU compute fits within GPU idle time (GPU is memory-bound):

```
Crossover: f_cpu ≤ CPU_BW / (CPU_BW + GPU_BW) + adjustment_for_overhead ≈ 8–10%
```

| f_cpu | Layer time vs baseline | Memory saved (7B, L=28) |
|---|---|---|
| 0% | baseline | 0 |
| 5% | −3% (faster) | 0.65 GB |
| **9%** | **~0% (free)** | **1.17 GB** |
| 15% | +40% | 1.96 GB |

**At f_cpu ≈ 9%: save ~1.2 GB for free.** Universally applicable — even models that fit benefit from increased batch capacity.

### 6.3 Offloaded Hybrid: Freeing More GPU Memory

For offloaded layers (necessity or elective), hybrid compute reduces the PCIe prefetch requirement. This can operate at different granularities (see `weight_offload_design.md` for full comparison):

**Group granularity (vLLM PrefetchOffloader)**: Controlled by **group_size (G)**: with M=1, there are (G−1) resident layers between each offloaded layer, providing hiding time for prefetch.

**Hiding constraint** (with `prefetch_step=1`, `M=1`):
```
(G-1) × resident_layer_time ≥ (1-f_cpu) × layer_weight / PCIe_BW
```

| Group size G | Hiding time (7B) | Min f_cpu | Buffer (1 slot) | # offloaded (L=28) | Memory freed | Decode |
|---|---|---|---|---|---|---|
| 4 | 1.5 ms | 93% | 33 MB | 7 | 2.9 GB | 49 ms |
| 7 | 3.0 ms | 86% | 65 MB | 4 | 1.6 GB | 32 ms |
| 14 | 6.5 ms | 69% | 144 MB | 2 | 0.7 GB | 22 ms |
| 28 | 13.5 ms | 36% | 298 MB | 1 | 0.1 GB | 16 ms |

- **Small G**: many offloaded layers, high f_cpu, tiny buffer → maximum memory freed, large latency cost
- **Large G**: few offloaded layers, lower f_cpu, larger buffer → less freed, near-baseline latency

The ~40× BF16 ratio means offloaded layers always need f_cpu > ~36% (even at G=28) — they can never match resident-layer speed. Uniform f_cpu across all layers is not viable.

**Tensor granularity (our approach)**: Instead of binary per-layer decisions, each sub-module (WQKV, WO, MLP1, MLP2) has independently tuned f_gpu/f_prefetch/f_cpu. This enables:
- **Variable f_cpu per sub-module**: MLP1 (large, preceded by short WO phase) can have high f_gpu; MLP2 (preceded by long MLP1 phase) can rely more on f_prefetch. The self-bootstrapping pipeline cascades prefetch budgets.
- **Smaller buffers**: only max(f_prefetch_i × W_i) vs full layer_weight for group granularity.
- **All PCIe for weight**: sub-layer pipeline dedicates all PCIe to weight prefetch (no KV prefetch contention).

**Elective offloading**: Even models that fit on GPU can offload a few layers. This is a latency-throughput tradeoff: per-step time increases, but larger batch capacity reduces scheduling rounds.

### 6.4 vLLM PrefetchOffloader

Key implementation (`vllm/model_executor/offloader/prefetch.py`):

- **Group-based selection**: `group_size=G`, `num_in_group=M` → offload last M layers per group of G. **Use M=1** — M>1 creates consecutive offloaded layers with no hiding between them, requiring extra buffer slots for no benefit (G=8,M=2 ≡ G=4,M=1 in offload ratio, but worse hiding).
- **Circular buffer**: `StaticBufferPool` with `prefetch_step` slots. Buffer = `prefetch_step` × per-offloaded-layer GPU weight.
- **Pipeline**: `wait_prefetch(N)` → compute → `start_prefetch(N + prefetch_step)`. Indices are over **offloaded layers only**. With M=1, `prefetch_step=1` means "1 group ahead" — (G−1) resident layers of natural hiding time.
- **prefetch_step=1 is sufficient for M=1**. Higher values are only needed for M>1 (consecutive offloaded layers).

---

## 7. End-to-End Scenarios

Overhead: ~3 GB (embeddings, activations, prefix KV, framework). All resident layers at f_cpu=9%. Offloaded layers use M=1, P=1; f_cpu set to minimum that hides prefetch for the chosen G.

### 7.1 Qwen2.5-7B (L=28, 14.1 GB)

Fits on GPU. Resident hybrid + attention offloading maximize batch capacity.

| Config | Res | Off | GPU weight | Decode | Free GPU |
|---|---|---|---|---|---|
| Baseline | 28 | 0 | 14.1 GB | 15.4 ms | 6.9 GB |
| **Resident hybrid** | **28** | **0** | **12.9 GB** | **14.7 ms** | **8.1 GB** |
| + Elective 2 off | 26 | 2 | 12.0 + 0.1 GB | 24.1 ms | 8.9 GB |
| + Elective 4 off | 24 | 4 | 11.1 + 0.1 GB | 33.5 ms | 9.9 GB |

Elective offloading: +4.7 ms and −424 MB GPU per offloaded layer. Worthwhile when beams exceed batch capacity.

### 7.2 Qwen2.5-14B (L=48, 28 GB)

Must offload. Resident hybrid saves ~2.3 GB but model still exceeds 24 GB.

| Config | Res | Off | GPU weight | Decode | Free GPU |
|---|---|---|---|---|---|
| Min offload (7 off) | 41 | 7 | 20.5 + 0.1 GB | 70 ms | 0.4 GB |
| Partial (13 off) | 35 | 13 | 17.5 + 0.1 GB | 105 ms | 3.4 GB |
| Full offload | 0 | 48 | 0.1 GB | 300 ms | 20.9 GB |

### 7.3 Qwen2.5-32B (L=64, 64 GB)

Heavy offload. Max ~23 resident layers (887 MB each).

| Config | Res | Off | GPU weight | Decode | Free GPU |
|---|---|---|---|---|---|
| Min offload (41 off) | 23 | 41 | 20.4 + 0.2 GB | 476 ms | 0.4 GB |
| Partial (48 off) | 16 | 48 | 14.2 + 0.2 GB | 546 ms | 6.6 GB |
| Full offload | 0 | 64 | 0.2 GB | 709 ms | 20.8 GB |

---

## 8. Implementation

### 8.1 Existing vLLM Building Blocks

| Component | File |
|---|---|
| Cascade attention + online softmax merge | `flash_attn.py:1038`, `merge_attn_states.py` |
| CPU attention + KV offload | `cpu_attn.py`, `kv_offload/` |
| Prefetch offloader + circular buffer | `offloader/prefetch.py` |
| QKVParallelLinear, MergedColumnParallelLinear | `layers/linear.py` |

### 8.2 Engineering Gaps

1. Column-parallel weight split + CPU matmul + partial result concat (tensor granularity three-way split)
2. CPU attention kernel returning per-head LSE values (required for online softmax merge)
3. CUDA Graph compatibility with CPU compute — `cudaLaunchHostFunc` callbacks (KTransformers approach) to embed CPU task submission in CUDA Graphs. Prerequisite for production vLLM integration.
4. Per-tensor f_gpu/f_prefetch/f_cpu configuration and sub-layer pipeline scheduling

### 8.3 Roadmap

| Phase | Goal | Impact |
|---|---|---|
| 1. Resident hybrid | f_cpu≈9% on all layers, column-parallel split | Free ~1.2 GB, zero cost |
| 2. Attention offloading | Suffix KV → CPU, CPU attention with LSE, online softmax merge | Free GPU KV for more beams |
| 3. Tensor-granularity offloading | Per-sub-module three-way split with sub-layer pipeline | Enable 14B+ models, all PCIe for weight |
| 4. CUDA Graph integration | `cudaLaunchHostFunc` for CPU compute in CUDA Graphs | Production-ready vLLM integration |
| 5. Benchmarking | RTX 4090: 7B / 14B / 32B | Validate analysis |

---

## 9. Key Risks

| Risk | Mitigation |
|---|---|
| CPU GEMM below theoretical bandwidth | Profile; adjust f_cpu |
| CPU attention + weight matmul contend for memory BW | Batch size is our control variable — reduce B when CPU saturates |
| Column-parallel numerical differences | Mathematically exact; unit tests |
| CUDA Graph breaks with CPU compute (f_cpu) | `cudaLaunchHostFunc` (KTransformers approach) — prerequisite for production. CUDA Graphs can be disabled for prototyping. |

---

## 10. Summary

| | |
|---|---|
| **Problem** | BF16 on 24 GB consumer GPU: weights dominate memory, PCIe ~40× slower than GPU compute. TTC needs large batch sizes; vLLM processes excess beams round by round. |
| **Attention offloading** | Partition KV by topology: prefix on GPU, suffix on CPU. Batch size is the control variable for CPU attention bottleneck. |
| **Hybrid weight computation** | Column-parallel GPU/CPU split. f_cpu≈9%: CPU computes ~9% of weights for free (hides within GPU idle time). Saves ~1.2 GB. Universal. |
| **PCIe weight prefetch** | All PCIe → weight prefetch. Small models: replaces f_gpu (frees memory). Large models: replaces f_cpu (reduces latency). No KV prefetch. |
| **Performance model** | Determines optimal partition fractions (f_gpu, f_prefetch, f_cpu) per sub-module, accounting for generator vs. verifier characteristics. |
| **Implementation** | (1) Resident hybrid → (2) Attention offloading → (3) Tensor-granularity offloading → (4) CUDA Graph integration → (5) Benchmarking |

---

## Detailed Design Documents

| Document | Scope |
|---|---|
| `weight_offload_design.md` | Granularity comparison (group/layer/tensor), three-way split, sub-layer pipeline, buffer sizing, CUDA Graph compatibility |
| `attention_offload_design.md` | KV three-way split, CPU suffix attention, batch size tradeoff, implementation gaps (CPU attention LSE) |
| `pcie_bandwidth_allocation_design.md` | Why all PCIe goes to weight prefetch — equivalence analysis, model size regimes, no KV prefetch rationale |
