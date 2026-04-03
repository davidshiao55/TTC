# Phase 0 Benchmark Findings

This document records findings from Phase 0 pre-implementation benchmarking.
Hardware: NVIDIA RTX 4090 (24 GB), Intel i9-14900KF (AVX2, no AVX512/AMX), DDR5.
PyTorch 2.10.0+cu128, MKL enabled, oneDNN enabled.

---

## 0.1+0.2: CPU/GPU Overlap Feasibility

### Column-Parallel Split: Same Dimension as Tensor Parallelism

Our CPU offloading split operates along the **same output dimension** as tensor
parallelism (TP). TP shards Q/K/V heads across multiple GPUs; we shard to CPU
instead. Both are column-parallel splits of the weight matrix's output dimension.

```
TP=2 (multi-GPU):
  GPU0: [ Q_h0..h13 | K_h0..h1 | V_h0..h1 ]   GPU1: [ Q_h14..h27 | K_h2..h3 | V_h2..h3 ]

Our split (GPU+CPU):
  GPU:  [ (1-f_cpu) of output columns ]          CPU:  [ f_cpu of output columns ]
```

The key difference: TP splits to a fast GPU with NVLink/PCIe interconnect; our
split sends columns to a slower CPU with the partial result returned via PCIe.
The overlap question is whether CPU finishes its portion before the GPU finishes
its portion — if yes, the split is free (like TP where both GPUs finish together).

### Critical Discovery: `F.linear` vs `torch.mm` for BF16 on CPU

**`torch.mm` with BF16 on CPU is catastrophically slow** — 100-250x slower than FP32.
It falls back to a naive scalar path because the i9-14900KF lacks AMX/AVX512-BF16
hardware. Effective bandwidth: 1-6 GB/s vs DDR5 theoretical ~80 GB/s.

**`F.linear` with BF16 on CPU is fast** — it routes through `torch._C._nn.linear`
which dispatches to oneDNN's optimized BF16 kernel. This kernel reads BF16 weights
(half memory bandwidth), converts to FP32 internally via AVX2 integer shifts, and
does FP32 FMA accumulation.

| Approach | MLP1 9% slice, B=1 | Path |
|---|---|---|
| `torch.mm` BF16 | 22.3 ms | `aten::mm` → naive scalar fallback |
| `torch.mm` FP32 | 0.17 ms | `aten::mm` → MKL SGEMM |
| **`F.linear` BF16** | **0.087 ms** | `torch._C._nn.linear` → oneDNN BF16 |
| `F.linear` FP32 | 0.166 ms | `torch._C._nn.linear` → MKL SGEMM |

At B=1, BF16 `F.linear` is **2x faster than FP32** because the operation is
memory-bandwidth-bound and BF16 reads half the bytes. FP16 `F.linear` shows
similar performance (0.098 ms) but BF16 is slightly faster and is the native
model weight format.

**Design decision: use `F.linear` with BF16 weights on CPU.** No FP32 weight
duplication needed. The `CpuComputeDispatcher` must use `F.linear`, not `torch.mm`.

### BF16 F.linear Scaling with Batch Size

BF16 `F.linear` loses its advantage at higher batch sizes as the operation
transitions from memory-BW-bound to compute-bound. The BF16→FP32 software
conversion overhead (no AMX) becomes significant.

MLP1 9% slice (3584 × 3410):

| B | BF16 F.linear | FP32 F.linear | BF16/FP32 |
|---|---|---|---|
| 1 | 0.087 ms | 0.166 ms | 0.52x (BF16 wins) |
| 4 | 0.339 ms | 0.260 ms | 1.3x |
| 8 | 0.674 ms | 0.348 ms | 1.9x |
| 16 | 1.340 ms | 0.482 ms | 2.8x |
| 32 | 2.688 ms | 0.818 ms | 3.3x |

Despite the scaling penalty, BF16 is the correct choice because:
1. No memory doubling (critical for 14B+ where CPU RAM is constrained)
2. TTC decode batch sizes are moderate (4-32 beams)
3. The absolute times still allow useful overlap with GPU at small f_cpu

### GPU Layer Time Baseline (RTX 4090, BF16 F.linear)

GPU is memory-bandwidth-bound during decode — layer time is nearly constant
across batch sizes, determined by weight read time at ~1008 GB/s.

| Sub-module | Weight (BF16) | GPU time (B=1) | GPU time (B=16) | GPU BW% |
|---|---|---|---|---|
| WQKV | 33.0 MB | 0.019 ms | 0.020 ms | ~83% |
| WO | 25.7 MB | 0.017 ms | 0.019 ms | ~80% |
| MLP1 | 271.6 MB | 0.287 ms | 0.295 ms | ~94% |
| MLP2 | 135.8 MB | 0.148 ms | 0.153 ms | ~92% |
| **Total** | **466.1 MB** | **0.470 ms** | **0.487 ms** | |

WQKV and WO achieve >100% of DRAM bandwidth at B≥4 due to L2 cache effects
(33 MB and 26 MB weights vs RTX 4090's 72 MB L2).

### Per-Sub-Module Overlap Results

Each CPU sub-module runs in parallel with its GPU counterpart. The constraint
is per-sub-module: CPU time ≤ GPU time for zero overhead.

**B=1 (single token decode):**

| f_cpu | WQKV | WO | MLP1 | MLP2 | Layer overhead | GPU freed (×28 layers) |
|---|---|---|---|---|---|---|
| 3% | ✓ | ✓ | ✓ | ✓ | **FREE** | 0.39 GB |
| 5% | ✓ | ✓ | ✓ | ✓ | **FREE** | 0.65 GB |
| 9% | ~ | ✓ | ✓ | ✓ | **FREE** | 1.17 GB |
| 15% | ~ | ~ | ✓ | ✓ | +0.007 ms | 1.96 GB |
| 30% | ✗ | ✗ | ✗ | ✗ | +0.776 ms | 3.91 GB |

**B=4 (minimum TTC batch):**

| f_cpu | WQKV | WO | MLP1 | MLP2 | Layer overhead | GPU freed (×28 layers) |
|---|---|---|---|---|---|---|
| 3% | ✓ | ✓ | ✓ | ✓ | **FREE** | 0.39 GB |
| 5% | ~ | ~ | ✓ | ✓ | +0.006 ms | 0.65 GB |
| 9% | ✗ | ✗ | ~ | ~ | +0.131 ms | 1.17 GB |
| 15% | ✗ | ✗ | ✗ | ~ | +0.412 ms | 1.96 GB |

**B=8:**

| f_cpu | WQKV | WO | MLP1 | MLP2 | Layer overhead |
|---|---|---|---|---|---|
| 3% | ✗ | ✓ | ✓ | ✓ | +0.007 ms |
| 5% | ✗ | ✓ | ✗ | ✓ | +0.119 ms |
| 9% | ✗ | ✗ | ✗ | ✗ | +0.674 ms |

Key observations:
- **WQKV and WO are the first to overflow** because their GPU times are tiny
  (0.02 ms). Even small CPU work exceeds this at B≥4.
- **MLP1 and MLP2 have large GPU times** (0.29 ms, 0.15 ms) providing generous
  overlap budgets. MLP1 tolerates f_cpu=9% up to B=4.
- **MLP1 dominates offloaded-layer cost.** At f_cpu=30%, MLP1 accounts for ~78%
  of the layer overhead.

### Throughput Tradeoff Framing

The original thesis claim of "f_cpu≈9% is free" needs qualification:
- At B=1, it is genuinely free (all sub-modules fit within GPU time)
- At TTC batch sizes (B≥4), it is a **throughput tradeoff**: small per-step
  latency increase in exchange for more KV cache → more beams per scheduling
  round → fewer rounds → potentially faster wall-clock TTC

The value of freeing GPU memory is to increase batch capacity. But the memory
is only useful at higher batch sizes, where the CPU overhead is no longer zero.
The thesis should frame this as an optimization tradeoff, not a free lunch.

### Implications for Implementation

1. **Always use `F.linear` for CPU compute, never `torch.mm`** — this is the
   single most important implementation detail from Phase 0.

2. **BF16 weights on CPU** — no FP32 conversion needed, saving memory.

3. **Per-sub-module f_cpu is important** — WQKV/WO have tiny GPU budgets,
   MLP1/MLP2 have large ones. Non-uniform f_cpu per sub-module is optimal.

4. **For resident layers (7B), WQKV should stay fully on GPU** — transfer
   the small k,v result (32 KB at B=16) to CPU suffix cache rather than
   computing K/V on CPU. The Q|K|V split only makes sense for offloaded
   layers where the weights must be on CPU anyway.

5. **Phase 3 (PCIe weight prefetch) should prioritize MLP1** — it is the
   dominant bottleneck for offloaded layers at any f_cpu above 15%.

6. **Hardware dependency**: CPUs with AMX (Sapphire Rapids+) would eliminate
   the BF16 scaling penalty, making higher f_cpu viable at larger batch sizes.
   The architectural contribution (column-parallel split, CpuComputeDispatcher)
   is hardware-independent; only the optimal f_cpu values change.
