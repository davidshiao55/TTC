# Attention Offloading Design

## Motivation

In TTC beam search, all beams share a common **prefix** (problem statement) but each beam has a unique **suffix** (generated tokens). Suffix KV cache = B × S × 2 KB per layer — grows linearly with both batch size and suffix length, consuming GPU memory.

**Why offload suffix KV**: By moving suffix KV to CPU, we free GPU memory for KV of additional beams. More beams per scheduling round = fewer rounds = faster wall-clock TTC.

**Two methods for offloaded suffix attention**:
- **f_cpu_kv**: suffix KV stays on CPU, attention computed on CPU in parallel with GPU. No GPU memory cost, but CPU attention grows with B × S.
- **f_prefetch_kv**: suffix KV prefetched from CPU back to GPU, attention computed on GPU. Uses PCIe bandwidth + GPU buffer, but avoids CPU attention bottleneck for prefetched portion.

This document analyzes both methods. The design decision on PCIe allocation (whether to use f_prefetch_kv at all) is made in `pcie_bandwidth_allocation_design.md`.

### Why Larger Batch Helps (and Where It Breaks)

Increasing batch size (more beams) improves **linear layer efficiency**: decode matmuls are memory-bandwidth-bound at small batch, so adding beams increases arithmetic intensity with near-zero extra cost until the GPU becomes compute-bound. More beams per scheduling round = fewer rounds = faster wall-clock TTC.

However, the **attention compute scales differently**:
- **f_gpu_kv (prefix)**: Prefix KV is fixed-size (shared across beams). GPU prefix attention scales with batch size, but prefix length is constant. GPU handles this efficiently.
- **f_prefetch_kv + f_cpu_kv (suffix)**: Suffix KV is per-beam. Total suffix KV = B × S × 2 KB/layer. As B increases, total suffix attention grows linearly.

The constraint: **GPU memory for KV is finite**. Prefix KV is fixed, but prefetched suffix KV buffer competes with weight memory. At some point, increasing B means:
- More suffix KV than can be prefetched → f_cpu_kv fraction grows
- CPU attention over f_cpu_kv grows linearly with B
- **CPU suffix attention becomes the bottleneck** — adding more beams no longer reduces wall-clock time

This is the fundamental tradeoff: larger batch improves linear layer throughput but eventually hits the CPU attention ceiling. KV prefetch raises this ceiling (by moving more suffix work to GPU), but cannot eliminate it entirely.

---

## Three-Way KV Split

Analogous to the weight three-way split, each layer's KV cache is split:

```
f_gpu_kv      : KV pinned on GPU (prefix KV, always GPU-resident)
f_prefetch_kv : suffix KV stored on CPU, prefetched to GPU via PCIe
f_cpu_kv      : suffix KV stored on CPU, attention computed on CPU
```

- **GPU attention** covers prefix + prefetched suffix → output_gpu, lse_gpu
- **CPU attention** covers remaining suffix → output_cpu, lse_cpu
- **Merge** via online softmax: `merge_attn_states(output_gpu, lse_gpu, output_cpu, lse_cpu)` — exact, no approximation

The split is per-layer, per-step (suffix grows each decode step).

### KV Data Sizes (Qwen2.5-7B)

kv_heads=4, head_dim=128, BF16. KV per token per layer = 2 × 4 × 128 × 2 bytes = **2 KB**.

| Component | Tokens | KV/layer | Notes |
|---|---|---|---|
| Prefix (f_gpu_kv) | P=4,000 | 8 MB | Shared across beams, always GPU |
| Suffix (f_prefetch + f_cpu) | B×S | B×S×2 KB | Per-beam, on CPU |

| B | S | Suffix KV/layer | Total suffix (28 layers) |
|---|---|---|---|
| 16 | 500 | 16 MB | 448 MB |
| 16 | 1,000 | 32 MB | 896 MB |
| 16 | 2,000 | 64 MB | 1.8 GB |

---

## KV Prefetch Granularity

KV prefetch and weight prefetch compete for the same PCIe H2D bandwidth. When they can happen depends on the prefetch granularity:

### Layer-Ahead Prefetch

Prefetch layer N+1's suffix KV during layer N's **entire** compute. PCIe is dedicated to KV transfer for the full layer time.

```
Layer N:   GPU+CPU compute ──────────────── │ PCIe H2D: KV prefetch for layer N+1
Layer N+1: GPU+CPU compute (uses prefetched KV) │ PCIe H2D: KV prefetch for layer N+2
```

**Budget**: full layer compute time × 22 GB/s.

**Tradeoff**: PCIe is occupied the entire layer → **blocks weight prefetch**. Only viable when weights don't need PCIe (i.e., all weights are f_gpu — permanent GPU resident, or f_cpu — CPU compute only, no transfer).

### Tensor-Ahead Prefetch

Prefetch layer N's suffix KV during layer N's **WQKV phase** only. WQKV weights are already on GPU (prefetched during previous layer's last phase), so PCIe is free during this specific sub-phase.

```
Layer N:
GPU :  WQKV compute → (QK^T)V attn  → WO compute   → MLP1 compute  → MLP2 compute
PCIe:  KV prefetch  → WO prefetch   → MLP1 prefetch → MLP2 prefetch → WQKV prefetch (N+1)
CPU :  WQKV compute → (QK^T)V attn  → WO compute   → MLP1 compute  → MLP2 compute
```

**Budget**: WQKV phase time × 22 GB/s.

**Tradeoff**: Small budget but **coexists with weight prefetch** — KV uses the WQKV slot, weight prefetch uses the other slots. No contention.

### Budget Comparison (Qwen2.5-7B, B=16, S=500)

Need 16 MB suffix KV per layer.

| Prefetch granularity | Budget/layer | % of suffix | Weight prefetch compatible? |
|---|---|---|---|
| **Layer-ahead** (pure hybrid, f_gpu_w=91%) | 15.8 MB | ~100% | **No** — but not needed (weights permanent) |
| **Layer-ahead** (with weight prefetch active) | 0 MB | 0% | **N/A** — PCIe already consumed by weights |
| **Tensor-ahead** (f_gpu_w=0, f_cpu_WQKV=58%) | 6.5 MB | 40% | **Yes** — other phases free for weights |
| **Tensor-ahead** (pure hybrid, f_cpu_WQKV=9%) | 1.0 MB | 6% | **Yes** — but small budget |

**Key insight**: Layer-ahead gives large KV budget but is mutually exclusive with weight prefetch. Tensor-ahead gives smaller budget but coexists. The choice depends on the weight offloading strategy:

- **If weights are f_gpu-heavy (pure hybrid)**: use layer-ahead — PCIe is free anyway, get ~100% KV prefetch
- **If weights use f_prefetch (weight prefetch pipeline)**: use tensor-ahead — get 40% KV prefetch during WQKV, weight prefetch during other phases

### Suffix Length Scaling (Layer-Ahead, Pure Hybrid)

At longer suffixes, KV need grows linearly but PCIe budget grows only with attention time:

| S | Suffix KV/layer | Layer time | KV budget | % prefetch |
|---|---|---|---|---|
| 500 | 16 MB | 0.72 ms | 15.8 MB | ~100% |
| 1,000 | 32 MB | 0.92 ms | 20.2 MB | 63% |
| 2,000 | 64 MB | 1.32 ms | 29.0 MB | 45% |
| 5,000 | 160 MB | 2.52 ms | 55.4 MB | 35% |

Partial prefetch is still valuable: 40% KV prefetch reduces CPU attention burden by ~40%, allowing ~1.7× more beams before CPU saturates.

---

## Interaction with Weight Offloading

**KV prefetch and weight prefetch compete for the same PCIe H2D bandwidth.** The design space ranges from all-KV to all-weight:

| Weight strategy | KV prefetch strategy | KV budget | Weight memory savings |
|---|---|---|---|
| Pure hybrid (f_gpu_w=91%) | Layer-ahead | ~16 MB/layer | Minimal (11.9 GB GPU) |
| Weight prefetch pipeline | Tensor-ahead (WQKV slot) | ~1–6.5 MB/layer | Maximum (0–7 GB GPU) |
| No weight offload | Layer-ahead | ~10 MB/layer* | None (all 13 GB on GPU) |

*Without weight offload, layer time is ~0.46 ms (pure GPU), budget = 22 × 0.46 = 10 MB.

The analysis of how to allocate PCIe bandwidth between weight and KV prefetch is in `pcie_bandwidth_allocation_design.md`.

---

## GPU Buffer for Prefetched KV

Layers are processed sequentially, so the KV buffer is reused:

```
GPU KV buffer = B × S_prefetch × 2 KB (per layer, reused across layers)
```

| B | S_prefetch | Buffer |
|---|---|---|
| 16 | 500 (100%) | 16 MB |
| 16 | 200 (40%) | 6.4 MB |
| 16 | 50 (10%) | 1.6 MB |

Small relative to weight buffers. Pre-allocated on GPU, managed like weight prefetch buffer pool.

---

## Implementation Feasibility

### Existing vLLM Building Blocks

| Component | Status | Location |
|---|---|---|
| Cascade attention (prefix/suffix split) | **Exists** | `flash_attn.py:1038` — two-pass (prefix causal=False, suffix causal=True), both return LSE |
| Online softmax merge | **Exists** | `merge_attn_states.py` — merges (output_a, lse_a, output_b, lse_b), optional output_lse for chaining |
| CPU attention kernel | **Exists** | `cpu_attn.py:349` — `ops.cpu_attention_with_kv_cache()` |
| KV offload manager | **Exists** | `kv_offload/abstract.py` — tracks CPU block locations, prepare_load/store |
| CPU↔GPU block transfer | **Exists** | `kv_offload/worker/cpu_gpu.py` — pinned memory, async CUDA stream transfers |

### Gaps

**Gap 1: CPU attention does not return LSE (CRITICAL)**

`cpu_attention_with_kv_cache()` writes only the output tensor — no LSE. Without LSE, CPU attention output cannot be merged with GPU attention via online softmax.

**Fix**: Modify the C++ kernel to expose per-head LSE values. The softmax denominator (from which LSE is derived) is already computed internally. Need to output it as an additional tensor.

Difficulty: Medium. C++ kernel + Python binding modification.

**Gap 2: Separate block tables for GPU prefix and CPU suffix**

vLLM's `block_table_tensor` contains only GPU block IDs. CPU blocks are tracked separately by `OffloadingManager`.

**Approach**: Two-pass attention with separate block tables (extends existing cascade pattern):
- **GPU pass**: prefix blocks (GPU block table)
- **CPU pass**: suffix blocks (CPU block table)
- **Merge**: `merge_attn_states(gpu_output, gpu_lse, cpu_output, cpu_lse)`

No need to modify block table format. Each attention kernel sees a homogeneous block table (all blocks on its device). If f_prefetch_kv is used, the GPU pass would also include prefetched suffix blocks.

Difficulty: Medium.

**Gap 3: Per-step KV prefetch scheduling (only if f_prefetch_kv is used)**

Suffix grows each decode step. If KV prefetch is enabled, need to decide:
- How many suffix tokens to prefetch (PCIe budget)
- Which blocks to transfer
- Update block tables accordingly

Difficulty: Medium. Not needed if all suffix KV uses f_cpu_kv only (see `pcie_bandwidth_allocation_design.md`).

### Attention Flow

**With f_cpu_kv only (no KV prefetch — recommended, see `pcie_bandwidth_allocation_design.md`):**

```
Per layer:
1. [GPU] Flash attention over prefix KV → gpu_output, gpu_lse
2. [CPU] CPU attention over suffix KV   → cpu_output, cpu_lse  (requires Gap 1 fix)
3. [GPU] merge_attn_states(gpu_output, gpu_lse, cpu_output, cpu_lse)
```

Steps 1 and 2 run in parallel. Step 3 uses existing online softmax merge (exact).

**With f_prefetch_kv (if KV prefetch is enabled):**

```
Per layer:
1. [PCIe] Prefetch selected suffix KV blocks: CPU → GPU buffer
2. [GPU] Flash attention over prefix + prefetched suffix KV → gpu_output, gpu_lse
3. [CPU] CPU attention over remaining suffix KV → cpu_output, cpu_lse
4. [GPU] merge_attn_states(gpu_output, gpu_lse, cpu_output, cpu_lse)
```

### Comparison to Hybrid Weight Computation

| | Hybrid weight | KV prefetch |
|---|---|---|
| Split type | Static (fixed at model load) | Dynamic (suffix grows each step) |
| What's modified | Linear layer forward | Attention backend + block tables |
| Scheduling | None (predetermined) | Per-step: how much to prefetch |
| Missing kernel | CPU matmul (standard BLAS) | CPU attention with LSE |
| Block management | None | Split block tables per step |
| Merge | concat (column-parallel) | online softmax (exists) |
| Difficulty | **Medium** | **Medium-High** |

### Verdict

Attention offloading (f_cpu_kv) is **feasible**. KV prefetch (f_prefetch_kv) is also feasible but adds complexity. The main effort for the base case (CPU-only suffix attention):
1. CPU attention LSE (kernel modification — medium, high impact)
2. Separate block tables for GPU prefix / CPU suffix (medium, extends cascade pattern)

Additional effort if f_prefetch_kv is enabled:
3. Per-step KV prefetch scheduling (medium, new logic)
4. Mixed GPU/CPU suffix block table management

---

## Summary

1. **Three-way KV split**: f_gpu (prefix, always GPU), f_prefetch (suffix prefetched to GPU), f_cpu (suffix on CPU) — mirrors weight split framework
2. **f_cpu_kv is always viable**: CPU suffix attention with batch size as the control variable for bottleneck management
3. **f_prefetch_kv is viable but competes with weight prefetch for PCIe** — layer-ahead gives large budget but blocks weight prefetch; tensor-ahead gives smaller budget but coexists
4. **PCIe allocation decision** is in `pcie_bandwidth_allocation_design.md` — determines whether f_prefetch_kv is used
5. **Feasible to implement**: extends existing cascade_attention + merge_attn_states; main gap is CPU attention LSE output (Gap 1)
