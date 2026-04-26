# Attention Offloading Design

## Motivation

In TTC beam search, all beams share a common **prefix** (problem statement + shared ancestry) but each beam has a unique **suffix** (divergent generated tokens). Suffix KV cache = B × S × 2 KB per layer — grows linearly with both batch size and suffix length, consuming GPU memory.

**Why offload suffix KV**: By moving suffix KV to CPU, we free GPU memory for KV of additional beams. More beams per scheduling round = fewer rounds = faster wall-clock TTC.

**Method**: suffix KV stays on CPU; attention over suffix KV is computed on CPU in parallel with GPU prefix attention; results merged via online softmax. PCIe bandwidth is not used for KV transfer — it is allocated entirely to weight prefetch (see `pcie_bandwidth_allocation_design.md`).

### Why Larger Batch Helps (and Where It Breaks)

Increasing batch size (more beams) improves **linear layer efficiency**: decode matmuls are memory-bandwidth-bound at small batch, so adding beams increases arithmetic intensity with near-zero extra cost until the GPU becomes compute-bound. More beams per scheduling round = fewer rounds = faster wall-clock TTC.

However, the **attention compute scales differently**:
- **Prefix attention (GPU)**: Prefix KV is fixed-size (shared across beams). GPU prefix attention scales with batch size, but prefix length is constant. GPU handles this efficiently.
- **Suffix attention (CPU)**: Suffix KV is per-beam. Total suffix KV = B × S × 2 KB/layer. As B increases, total suffix attention grows linearly.

As B grows, CPU suffix attention eventually becomes the bottleneck — adding more beams no longer reduces wall-clock time. Batch size is the control variable for this bottleneck: if CPU attention saturates, reduce B (more scheduling rounds, each faster).

---

## Two-Pool KV Model

Each model's KV is allocated across two pools, sized by the Planner at engine launch:

```
KV_gpu_bytes : GPU KV pool
KV_cpu_bytes : CPU KV pool (extension of the GPU pool)
```

Two candidate partitioning axes are on the table — both produce a two-pool split, but they carve the KV cache differently. The final mechanism is not yet committed; §0.4 of `phase0_findings.md` measures both before Phase 2 locks it in.

**Candidate A — Position-split (prefix/suffix-style).** KV positions `[0 : split_point]` live on GPU; positions `[split_point : ]` live on CPU. Attention runs on each device over its position range; results merge via online softmax (`merge_attn_states`). The special case "prefix on GPU, suffix on CPU" is the common shape but not invariant — `split_point` can extend past the prompt boundary when GPU has headroom, and can spill earlier when pressure grows.

**Candidate B — Head-split (TP-style).** Some KV heads' full cache lives on CPU, others on GPU, mirroring vLLM's TP sharding (`flash_attn.py:121-130`, per-rank KV shape `(2, num_blocks, block_size, num_kv_heads/tp_size, head_size)`). Each device runs attention over its local head group; no online-softmax merge because heads concatenate, not combine.

### Tradeoffs Between A and B

| Axis | Position-split (A) | Head-split (B) |
|---|---|---|
| Merge comm | Required per layer (`[N, H] + LSE`) | **Not required** (head outputs concatenate) |
| Prefix reuse across beams | Full win (one GPU copy serves all beams) | Per-head allocation; prefix sharing still works per head but with coarser memory packing |
| Arithmetic intensity match | High-AI (compute-bound, prefix-like) on GPU, low-AI (memory-bound, suffix-like) on CPU | Each device does mixed AI for its heads — less ideal |
| Granularity | Continuous (any split point) | Discrete (few KV heads → few split points) |
| GPU KV footprint at high beam × long seq | Small (can hold just shared prefix) | Larger (must hold its heads' full per-beam KV) |

The Planner decides pool sizes (and, once committed, the partitioning axis) from workload target (`n`, `max_context`, search strategy) and shared budgets. See `planner_design.md` for pool sizing.

### Attention Flow (Candidate A — Position-Split)

```
GPU: flash attention over GPU-side KV positions  → gpu_output, gpu_lse
CPU: attention over CPU-side KV positions        → cpu_output, cpu_lse
GPU: merge_attn_states(gpu_output, gpu_lse, cpu_output, cpu_lse)  (exact, online softmax)
```

GPU and CPU attention run in parallel over their position ranges. Merge on GPU is a single kernel call using `merge_attn_states` (already in vLLM). Candidate B (head-split) would replace the merge with a head-dim concatenation; the remaining machinery is similar.

### KV Data Sizes (Qwen2.5-7B)

kv_heads=4, head_dim=128, BF16. KV per token per layer = 2 × 4 × 128 × 2 bytes = **2 KB**.

| Component | Tokens | KV/layer | Notes |
|---|---|---|---|
| Prefix | P=4,000 | 8 MB | Shared across beams, GPU pool |
| Suffix | B×S | B×S×2 KB | Per-beam, CPU pool |

| B | S | Suffix KV/layer | Total suffix (28 layers) |
|---|---|---|---|
| 16 | 500 | 16 MB | 448 MB |
| 16 | 1,000 | 32 MB | 896 MB |
| 16 | 2,000 | 64 MB | 1.8 GB |

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

No need to modify block table format. Each attention kernel sees a homogeneous block table (all blocks on its device).

Difficulty: Medium.

### Attention Flow

```
Per layer:
1. [GPU] Flash attention over prefix KV → gpu_output, gpu_lse
2. [CPU] CPU attention over suffix KV   → cpu_output, cpu_lse  (requires Gap 1 fix)
3. [GPU] merge_attn_states(gpu_output, gpu_lse, cpu_output, cpu_lse)
```

Steps 1 and 2 run in parallel. Step 3 uses existing online softmax merge (exact).

### Comparison to Hybrid Weight Computation

| | Hybrid weight compute | Attention offload (suffix on CPU) |
|---|---|---|
| Split type | Storage static, dispatch per-bucket | KV topology fixed (prefix→GPU, suffix→CPU); pool sizes set by Planner |
| What's modified | Linear layer forward | Attention backend + block tables |
| Missing kernel | CPU matmul (standard BLAS) | CPU attention with LSE |
| Block management | None | Split block tables (GPU prefix / CPU suffix) |
| Merge | concat (column-parallel) | online softmax (exists) |
| Difficulty | **Medium** | **Medium** |

### Verdict

Attention offloading (suffix on CPU, CPU attention computed in parallel with GPU) is **feasible**. Main effort:

1. CPU attention LSE output (Gap 1 — kernel modification, medium, high impact)
2. Separate block tables for GPU prefix / CPU suffix (Gap 2 — medium, extends cascade pattern)

---

## Summary

1. **Two-pool KV model**: `KV_gpu_bytes` for shared prefix, `KV_cpu_bytes` as the extension for per-beam suffix. Pool sizes are Planner outputs; topology (prefix-GPU, suffix-CPU) is a mechanism invariant.
2. **No KV prefetch**: all PCIe H2D allocated to weight prefetch (see `pcie_bandwidth_allocation_design.md`). Suffix attention is computed on CPU.
3. **Batch size is the release valve** for CPU attention bottleneck — Scheduler clamps if CPU saturates.
4. **Feasible to implement**: extends existing cascade_attention + merge_attn_states; main gap is CPU attention LSE output.
