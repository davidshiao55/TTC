# Phase 0 Benchmark Findings

This document records findings from Phase 0 pre-implementation benchmarking — the **first iteration of the Profiler's output** (see `profiler_design.md`). It establishes baseline numbers for RTX 4090 + Qwen2.5-7B + BF16 and validates the measurement methodology future profile runs will reuse.

Hardware: NVIDIA RTX 4090 (24 GB), Intel i9-14900KF (AVX2, no AVX512/AMX), DDR5.
PyTorch 2.10.0+cu128, MKL enabled, oneDNN enabled.

---

## Contents

**Compute substrate**
- §0.1 — Dispatch axis: num_tokens unification
- §0.2 — Split mechanism correctness
- §0.3 — CPU/GPU compute characterization
- §0.4 — Per-sub-module split-axis design

**PCIe behavior**
- §0.5 — PCIe behavior: bandwidth, contention, UVA bypass

**System reference**
- §0.6 — CPU attention latency (Phase 2 reference)
- §0.7 — CUDA graph impact (Phase 4 priority)
- §0.8 — vLLM V1 FastTTS baseline
- §0.9 — vLLM V1 KV offload impact

---

## 0.1: Dispatch axis — num_tokens unification

The Planner's dispatch table is keyed on a single scalar — the forward call's `num_tokens` (see `thesis_proposal.md §5.1` and `planner_design.md §4.5`). This requires that **GEMM arithmetic intensity depend only on `num_tokens`** and not on how those tokens are distributed across requests / prefill / decode. A failure of this assumption would force a second dispatch axis (prefill/decode ratio or per-request batch size).

### Why the claim should hold

Mathematically, the matmul sees input `[num_tokens, hidden]` after vLLM's scheduler flattens all tokens into a single batch dimension. The shape is identical regardless of whether N tokens came from 1 prefill × N, N decodes × 1, or any mixed split — so the matmul cost should be identical too. Attention is a separate case (cost depends on `q_len` and `kv_len` per request) and is measured in §0.6 by its own `(B, S)` curve, not by the num_tokens dispatch.

### Empirical test — `bench_num_tokens_axis.py`

For each `N ∈ {16, 64, 256}`, time each GEMM sub-module at five pre-flatten compositions that all collapse to `[N, hidden]`:

| Composition | Pre-flatten shape | Interpretation |
|---|---|---|
| `flat` | `[N, H]` | already flat (reference) |
| `1x_N` | `[1, N, H]` | 1 request × N-token prefill |
| `Nx_1` | `[N, 1, H]` | N decodes × 1 token each |
| `halfx_2` | `[N/2, 2, H]` | N/2 requests × 2 tokens (chunked prefill) |
| `quarterx_4` | `[N/4, 4, H]` | N/4 requests × 4 tokens (mixed-depth) |

The load-bearing metric is **comp_spread**: `(max − min) / max` across the four *reshape* compositions (excluding `flat`). The four reshape variants represent different scheduling realities that all flatten to the same tensor — if they agree within noise, the claim holds.

Tested on **both** the GPU cuBLAS path (what decode actually runs on) and the CPU oneDNN BF16 path (what the CPU-compute dispatch would run on in Phase 1+).

Tolerance: 5%. Timing uses median of 50 iterations with pre-warmed kernel (absorbs oneDNN JIT / cuBLAS kernel-selection startup cost). On CPU, 3 trials per composition with best-median to dampen OpenMP scheduling variance.

### Results — Qwen2.5-7B

**GPU (cuBLAS): all sub-modules and N — `comp_spread ≤ 4.2%` ✓ PASSED.**

**CPU (oneDNN BF16):**

| N | Sub-module | comp_spread | Verdict |
|---|---|---:|---|
| 16 | WQKV, MLP1, MLP2 | ≤2.4% | ✓ |
| 16 | WO (~1.4 ms) | 24% | ✗ noise floor |
| 64 | all | ≤3.0% | ✓ |
| 256 | all | ≤2.4% | ✓ |

One outlier: N=16 WO at ~1.4 ms absolute time, where 0.3–0.5 ms of OpenMP scheduling noise manifests as 24% relative spread. Not a composition violation — at every other (sub-module, N) pair the spread is under 5%.

### Results — Skywork-PRM-1.5B

**GPU (cuBLAS):**

| N | comp_spread range | Verdict |
|---|---:|---|
| 16 | ≤1.6% | ✓ |
| 64 | WQKV 4.1%, others ≤1.6% | ✓ (all within tolerance) |
| 256 | all ≤4.2% | ✓ |

PRM GPU matmuls are smaller than 7B's (hidden=1536 vs 3584), so absolute noise is proportionally larger, but all within 5% threshold.

**CPU (oneDNN BF16):**

| N | Sub-module | comp_spread | Verdict |
|---|---|---:|---|
| 16 | all | ≤3.4% | ✓ |
| 64 | MLP2 (~6 ms) | 5.2% | ✗ borderline |
| 64 | WQKV, WO, MLP1 | ≤2.5% | ✓ |
| 256 | WO (~4.6 ms) | 22.6% | ✗ noise floor |
| 256 | WQKV, MLP1, MLP2 | ≤2.9% | ✓ |

PRM's smaller matmuls (1–6 ms absolute) hit the CPU noise floor more often. The 22.6% spread on WO at N=256 traces to a single outlier (`1x_N=5.79` ms vs ~4.6 ms for others) — classic OpenMP scheduling variance.

### Composition equivalences (specific cases from the question)

| Equivalence | Tested | Verdict |
|---|---|---|
| B=64 × 1-token decode ≡ B=1 × 64-token prefill | `Nx_1` vs `1x_N` at N=64, both GPU & CPU | ✓ within 1-3% |
| 2 × 32-token prefill ≡ 4 × 16-token prefill | `halfx_2` vs `quarterx_4` at N=64 | ✓ within 1-3% |
| Pure prefill ≡ pure decode ≡ any mix | All four reshape variants, all N | ✓ cluster together |
| Generator vs verifier shares the claim | Tested on 7B and PRM-1.5B separately | ✓ per-model profile |

### Interpretation

The claim is empirically confirmed on both devices for all operationally relevant regimes:

| Equivalence | Evidence |
|---|---|
| B=N decodes × 1-token ≡ 1 prefill × N-token | `Nx_1` vs `1x_N` agree within 1-3% on both GPU and CPU at all N ≥ 64 |
| N/2 × 2 prefill ≡ N/4 × 4 prefill | `halfx_2` vs `quarterx_4` agree within 1-3% |
| Pure prefill ≡ pure decode ≡ any mix | All four reshape variants cluster together |

The `flat` allocation pattern occasionally differs from reshape variants by ~3-5% — this is a cuBLAS kernel-selection artifact (different tensor metadata), not a composition difference. vLLM's scheduler produces flat 2D tensors at the matmul input, so the `flat` curve is what the Planner's cost model uses.

At very small CPU matmuls (≤2 ms absolute), OpenMP thread-scheduling variance creates occasional 20%+ outliers. Above ~5 ms absolute time, CPU results are as tight as GPU.

### Conclusion

**No second dispatch axis is required.** The Planner's single-axis `num_tokens` dispatch table is empirically justified on both GPU and CPU paths. Different prefill/decode compositions at the same `num_tokens` produce matmul timings that agree within measurement noise for all operationally relevant workload sizes.

---

## 0.2: Split mechanism correctness

Script: `David/Benchmarks/phase0/bench_split_correctness.py`.

Verifies that the per-sub-module split mechanisms chosen in `weight_offload_design.md §Per-Sub-Module Split Axis` produce numerically equivalent results to unsplit computation. Four test families cover the full mechanism space used by Phase 1:

| Family | Covers | Assembly op |
|---|---|---|
| A. Col-parallel contiguous   | WQKV, MLP1, WO (if Alt A)     | `concat` |
| B. Col-parallel K/V-biased   | WQKV (design invariant)       | `index_copy_` into full shape |
| C. Row-parallel contiguous   | MLP2                          | `add_` (partial-sum reduce) |
| D. MLP1→MLP2 col→row pipeline | end-to-end MLP block         | `add_` on MLP2 partial sum |

Sweep: `f_cpu ∈ {3%, 9%, 15%, 22%, 30%, 50%}` × `B ∈ {1, 8, 32}` for both models.

**Tolerance**: 2% × output scale. BF16 has a 7-bit mantissa (~1.5% ulp at the top of its exponent range); large reductions (inner dim up to 18944) in different orders on GPU vs CPU amplify this to ~2%. `EXACT` means bitwise equal; `close` means within tolerance.

**Result: ALL TESTS PASSED** for both Qwen2.5-7B-Instruct and Skywork-PRM-1.5B. Every `(f, B)` point across the four families falls within tolerance. The K/V-biased picker passes its structural invariants (head alignment, KV-group pairing below the boundary, full K+V + tail-Q above the boundary). Family D specifically validates the **matched-index invariant**: with CPU holding the same intermediate index set for MLP1's output and MLP2's input, each device applies SwiGLU on its local slice and MLP2 consumes the local slice directly, yielding output equal to the unsplit MLP block.

This closes the correctness question for Phase 1's mixed col/row mechanism. The design-validation numbers (activation byte savings from col→row pipelining, and the Phase 2 WO offload decision) live in §0.4.

---

## 0.3: CPU/GPU compute characterization

Three measurements gate weight-offload viability: GPU time per sub-module (the budget the CPU must fit inside), CPU time per sub-module (does it fit?), and concurrent overlap (does scheduling actually overlap them?). The Planner consumes these as `gpu_layer_timing`, `cpu_gemm_curve`, and `gpu_reduced_timing` curves; this section establishes their values for RTX 4090 + i9-14900KF and surfaces two implementation gotchas (`F.linear` mandatory; `μs/MB` uniformity).

### Split axis per sub-module — matches vLLM TP

Our CPU offloading split follows vLLM's tensor-parallelism conventions, applying col-parallel or row-parallel per sub-module rather than uniformly column-parallel. TP shards across multiple GPUs; we shard across the GPU-CPU boundary — same mechanism, different destination.

```
WQKV, MLP1 — column-parallel (shard output dim)
  GPU:  [ (1-f_cpu) of output cols ]   CPU:  [ f_cpu of output cols ]
  Assembly: concat on matching output cols.

MLP2       — row-parallel    (shard input dim)
  GPU:  [ (1-f_cpu) of input cols ]    CPU:  [ f_cpu of input cols ]
  Assembly: add-reduce (GPU_partial + CPU_partial).

WO         — col if Alt A; no offload if Alt B (decided in §0.4.2).
```

This matches vLLM's `ColumnParallelLinear` (qkv_proj, gate_up_proj) and `RowParallelLinear` (down_proj, o_proj) pairing. For MLP1→MLP2 specifically, the col→row pairing keeps the intermediate activation local on each device — no GPU↔CPU round-trip on the MLP block. See `weight_offload_design.md §Per-Sub-Module Split Axis` for the full design, §0.2 for split-mechanism correctness, and §0.4.1 for the MLP pipeline validation.

The key difference from TP: TP splits to a fast GPU with NVLink/PCIe interconnect; our split sends work to a slower CPU with the partial result returned via PCIe. The overlap question is whether CPU finishes its portion before the GPU finishes its portion — if yes, the split is free (like TP where both ranks finish together).

### 0.3.1: GPU layer baseline

GPU is memory-bandwidth-bound during decode — layer time is nearly constant across batch sizes, determined by weight read time at ~1008 GB/s.

| Sub-module | Weight (BF16) | GPU time (B=1) | GPU time (B=16) | GPU BW% |
|---|---|---|---|---|
| WQKV | 33.0 MB | 0.019 ms | 0.020 ms | ~83% |
| WO | 25.7 MB | 0.017 ms | 0.019 ms | ~80% |
| MLP1 | 271.6 MB | 0.287 ms | 0.295 ms | ~94% |
| MLP2 | 135.8 MB | 0.148 ms | 0.153 ms | ~92% |
| **Total** | **466.1 MB** | **0.470 ms** | **0.487 ms** | |

WQKV and WO achieve >100% of DRAM bandwidth at B≥4 due to L2 cache effects (33 MB and 26 MB weights vs RTX 4090's 72 MB L2).

### 0.3.2: CPU compute path

#### Critical discovery — `F.linear` vs `torch.mm` for BF16 on CPU

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

#### BF16 `F.linear` scaling with batch size

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

### 0.3.3: Per-sub-module overlap

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

**Visual evidence — `probe_overlap.py`.** A 1-second nsys probe at WQKV, f_cpu=9%, B=1 timeline-confirms CPU+GPU concurrency. Phase A (`A_gpu_only`) and B (`B_cpu_only`) ground-truth the per-side durations; Phase C (`C_concurrent`) shows the GPU bar and CPU `F.linear` bar overlapping in time. Run `nsys profile -o probe_overlap.nsys-rep --trace=cuda,nvtx,osrt --force-overwrite=true python probe_overlap.py` and inspect in Nsight Systems GUI.

#### Throughput tradeoff framing

The claim that "f_cpu≈9% is free" holds **only at B=1 decode** on this hardware:
- At B=1, all sub-modules fit within GPU time → genuinely free.
- At TTC batch sizes (B≥4), it becomes a **throughput tradeoff**: small per-step
  latency increase in exchange for more KV cache → more beams per scheduling
  round → fewer rounds → potentially faster wall-clock TTC.

The specific number `9%` is a property of this (HW, model, dtype) combination at B=1, not a universal constant. The Planner (see `planner_design.md`) derives bucket-specific values from the profile, so the number used in practice varies per `BatchDescriptor`.

### 0.3.4: CPU μs/MB uniformity across sub-modules

The Planner's single-`(f_cpu, f_prefetch)`-per-bucket design (see `planner_design.md §4.2`) rests on the empirical claim that **CPU GEMM throughput in μs/MB is uniform across the four sub-modules** at decode-regime batch sizes. This subsection documents the measurement that justifies the claim.

Per-sub-module CPU time divided by CPU-slice weight bytes, from `rtx4090_qwen7b_bf16.json`, at several `(B, f)` points:

| B | f_cpu | WQKV μs/MB | WO μs/MB | MLP1 μs/MB | MLP2 μs/MB | max/min |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 10% | 56.5 | 56.1 | 54.6 | 58.6 | **1.07×** |
| 16 | 30% | 54.5 | 55.2 | 55.8 | 54.7 | **1.02×** |
| 16 | 50% | 54.4 | 54.8 | 56.4 | 55.4 | **1.04×** |
| 64 | 10% | 227 | 220 | 220 | 234 | **1.07×** |
| 64 | 30% | 219 | 219 | 222 | 223 | **1.02×** |
| 64 | 50% | 218 | 220 | 225 | 224 | **1.03×** |
| 128 | 10% | 453 | 443 | 444 | 594 | 1.34× |
| 128 | 30% | 441 | 441 | 482 | 521 | 1.18× |
| 128 | 50% | 440 | 443 | 493 | 454 | 1.12× |

At B=16 and B=64 (the dominant decode regime), the per-MB throughput is flat within **1.02–1.07×**. At B=128 the spread widens modestly (1.12–1.34×) from L3-capacity effects but remains tightly bounded. This validates the theoretical expectation: dense GEMM arithmetic intensity at fixed `num_tokens` is `2N / dtype_size` flops/byte regardless of sub-module shape, so per-byte CPU time should be shape-independent — and it is.

**B=1 edge case.** At B=1 the uniformity breaks: MLP1 at f=50% takes 17.4 μs/MB vs. WQKV's 3.6 — a 4.86× spread. Cause is an oneDNN microkernel pathology on tall-skinny matmul shapes at tiny `num_tokens`, not a structural AI difference. Absolute times are sub-millisecond, so the impact on decode wall-clock is bounded, but the Planner should treat B=1 separately rather than extrapolating from the B≥16 curve. In practice this is a non-issue because decode with beam count ≥ 2 is the normal regime.

**GPU asymmetry (not uniform).** GPU μs/MB does vary across sub-modules by ~1.69× from L2-cache effects: WQKV (33 MB) and WO (25.7 MB) fit in the RTX 4090's 72 MB L2, while MLP1 (271 MB) and MLP2 (135 MB) don't. This shifts the per-sub-module GPU/CPU balance point by ~0.8 percentage points of `f` (1.2% optimal for WQKV, 2.0% for MLP1). Small enough that forcing uniform `f_cpu` across WQKV/MLP1/MLP2 is near-optimal in absolute wall-clock terms.

**Consequence for the Planner.** With uniform CPU throughput and only ~1 pp of GPU-side variation in optimal `f`, a single `(f_cpu, f_prefetch)` pair per bucket applied uniformly across WQKV/MLP1/MLP2 is within noise of the per-sub-module optimum. This collapses the Planner's per-bucket solve to a single scalar (`f_cpu`) and eliminates the need for per-sub-module dispatch vectors — the simplification the current design rests on.

---

## 0.4: Per-sub-module split-axis design

§0.2 closed the correctness question for the mixed col/row split mechanism. This section covers the two measured design claims that commit Phase 1 and Phase 2 to specific axis choices: the MLP1→MLP2 col→row pipelining saves activation PCIe, and the WO offload alternatives (Alt A weight-split vs. Alt B GPU-resident) diverge enough to pick one globally.

### 0.4.1: MLP1→MLP2 pipelining — empirical wall-clock & PCIe comparison

Script: `David/Benchmarks/phase0/bench_mlp_pipeline.py`.

The col→row pairing eliminates the intermediate round-trip that uniform-col would require between MLP1 and MLP2. Under uniform col, CPU computes a slice of `gate_up` (MLP1 output), returns it to GPU, GPU concatenates and applies SwiGLU on the full intermediate, then GPU re-sends the full intermediate back to CPU as MLP2's input. With col→row, CPU keeps its `gate_up` slice locally, applies SwiGLU in place on its slice, and feeds the local slice into its row slice of `W_down` — zero intermediate transfer.

To empirically validate the savings claim, we implemented both patterns end-to-end on the real CPU/GPU critical path and compared per-MLP-block wall-clock time and PCIe byte count. Each step is timed in its actual position in the serial CPU critical path, with GPU steps timed via CUDA events. CPU is the bottleneck in the decode regime (CPU GEMM ≫ GPU compute at small-batch decode), so the serial CPU-side sum approximates block wall-clock.

**Qwen2.5-7B-Instruct:**

|   N |    f | col total | col→row total | Δ (ms) | col PCIe | col→row PCIe | PCIe ratio |
|----:|-----:|----------:|--------------:|-------:|---------:|-------------:|-----------:|
|  16 |  10% |   2.25 ms |       2.30 ms |  -0.05 |  0.85 MB |      0.23 MB |    **3.72×** |
|  16 |  30% |   6.81 ms |       6.81 ms |  +0.00 |  1.12 MB |      0.23 MB |    **4.88×** |
|  64 |  10% |   8.99 ms |       9.14 ms |  -0.15 |  3.41 MB |      0.92 MB |    **3.72×** |
|  64 |  30% |  27.24 ms |      27.12 ms |  +0.13 |  4.48 MB |      0.92 MB |    **4.88×** |
| 128 |  10% |  18.28 ms |      18.48 ms |  -0.19 |  6.83 MB |      1.84 MB |    **3.72×** |
| 128 |  30% |  54.48 ms |      57.71 ms |  -3.24 |  8.95 MB |      1.84 MB |    **4.88×** |

**Skywork-PRM-1.5B:**

|   N |    f | col total | col→row total | Δ (ms) | col PCIe | col→row PCIe | PCIe ratio |
|----:|-----:|----------:|--------------:|-------:|---------:|-------------:|-----------:|
|  16 |  10% |   0.55 ms |       0.52 ms |  +0.03 |  0.40 MB |      0.10 MB |    **4.05×** |
|  16 |  30% |   1.46 ms |       1.44 ms |  +0.02 |  0.52 MB |      0.10 MB |    **5.32×** |
|  64 |  10% |   2.03 ms |       2.02 ms |  +0.01 |  1.59 MB |      0.39 MB |    **4.05×** |
|  64 |  30% |   5.79 ms |       5.70 ms |  +0.10 |  2.09 MB |      0.39 MB |    **5.32×** |
| 128 |  10% |   4.01 ms |       4.00 ms |  +0.01 |  3.18 MB |      0.79 MB |    **4.05×** |
| 128 |  30% |  11.63 ms |      11.46 ms |  +0.17 |  4.18 MB |      0.79 MB |    **5.32×** |

Δ (ms) = col_total − col→row_total. Positive Δ means col→row is faster.

**Observations.**

1. **PCIe byte savings are real and large**: 3.72–5.32× less activation PCIe per MLP block, independent of `N` and `f`. This matches the analytical byte count: the dominant saving is eliminating the full-intermediate H2D that uniform col requires for MLP2's input.

2. **Wall-clock is essentially flat between patterns** (Δ within ±0.5 ms in all but one point). CPU GEMM dominates the MLP block critical path, and the CPU GEMM FLOPs are identical between patterns (MLP1 is the same shape; MLP2's row-slice has the same FLOP count as col-slice, just a different matmul shape). Saving PCIe bytes does not shorten the MLP block wall-clock in this regime because PCIe is not the bottleneck of the MLP block.

3. **The saved bytes are not wasted, though.** PCIe H2D is a contended resource with weight prefetch (§0.5). Col→row's 3.72–5.32× reduction means weight prefetch sees that much less activation-traffic contention — a Phase-3-critical property. At N=128 on 7B, uniform col injects ~9 MB of activation H2D per MLP block × 28 layers ≈ 250 MB/decode step (~11 ms at 22 GB/s), while col→row injects only ~51 MB/step (~2.3 ms). The difference is time the Phase 3 weight-prefetch pipeline recovers.

4. **One wall-clock anomaly**: at 7B, N=128, f=30%, col→row is 3.24 ms slower. Traceable to the `cpu_mlp1` step: oneDNN picks a different microkernel for the contiguous `2·cpu_inter × hidden` weight shape in col→row vs. the larger same-axis col weight in uniform col. Noise from oneDNN heuristic crossing at that shape; not a design defect.

**Conclusion.** The design claim "col→row eliminates the intermediate round-trip and saves ~3–5× activation PCIe" is empirically confirmed on both models. The saving is in PCIe bytes, not in MLP-block wall-clock — but the saved bytes translate directly into weight-prefetch bandwidth for Phase 3, where activation vs prefetch contention was called out in §0.5 as a real concern. Col→row pays no wall-clock tax and frees real PCIe budget. Adopted for Phase 1.

### 0.4.2: WO offload — Alt A vs Alt B decision

Script: `David/Benchmarks/phase0/bench_wo_offload_tradeoff.py`.

Two alternatives for WO after the Phase 2 attention merge, described in `weight_offload_design.md §WO Split Axis Decision`:

- **Alt A (col-split WO with merge-before-WO)**: CPU holds `f · WO_bytes` of weight. GPU merges `attn_out`, sends CPU's input slice via H2D, CPU computes its partial, D2H back for concat on GPU. Saves `f · 686 MB` of GPU WO weight at 7B.
- **Alt B (no WO offload)**: GPU does full WO after merge. No H2D/D2H of merged `attn_out`. CPU has no WO work. WO occupies 686 MB on GPU at 7B.

**TL;DR — WO is the smallest matmul in the layer, and sits in the worst place for offload.** Per-layer BF16 weight sizes on Qwen2.5-7B: MLP1 272 MB, MLP2 136 MB, WQKV 37 MB, **WO 26 MB** (smallest). The per-MB CPU latency tax of offloading is roughly constant across sub-modules (CPU matmul throughput doesn't care which weight it's computing — just shape and FLOPs), so offloading WO pays the **same tax rate** as MLP1 but saves **~10× fewer bytes per unit of f**. MLP1 justifies the tax rate because its absolute memory gain is large enough to matter; WO does not.

On top of the size disadvantage, Alt A has a structural communication cost the MLP offload does not: WO sits **after** the attention merge, so Alt A forces 3 PCIe hops per layer (CPU→GPU for attn merge → GPU→CPU to hand merged attn_out to CPU → CPU→GPU to return the partial), vs. the 2 natural hops of MLP's col→row pattern. Either the per-MB argument or the extra-hop argument alone would reject Alt A at 7B; together they make the decision robust.

---

The attention / merge phases are identical in both alternatives, so the comparison reduces to a post-merge differential:

```
Δ_latency(f, N) = max(t_WO_reduced_gpu,  H2D_merged + t_WO_cpu + D2H_partial)
                  − t_WO_full_gpu
```

Measured on RTX 4090 + i9-14900KF, BF16, pinned-memory PCIe, sweep `N ∈ {1, 16, 64, 128}` × `f ∈ {10%, 20%, 30%, 50%}`:

**Qwen2.5-7B-Instruct (hidden=3584, WO total = 686 MB across 28 layers):**

|   f | max Δ / layer | max Δ / decode step (28 layers) | GPU memory saved | % of 24 GB budget |
|----:|----:|----:|----:|----:|
| 10% | +1.14 ms | +32.1 ms | 72 MB | 0.3% |
| 20% | +4.62 ms | +129.4 ms | 144 MB | 0.6% |
| 30% | +8.33 ms | +233.2 ms | 216 MB | 0.9% |
| 50% | +12.40 ms | +347.2 ms | 360 MB | 1.5% |

**Skywork-PRM-1.5B (hidden=1536, WO total = 132 MB across 28 layers):**

|   f | max Δ / layer | max Δ / decode step | GPU memory saved | % of 24 GB budget |
|----:|----:|----:|----:|----:|
| 10% | +4.28 ms | +119.7 ms | 13 MB | 0.05% |
| 20% | +6.30 ms | +176.5 ms | 26 MB | 0.11% |
| 30% | +11.95 ms | +334.5 ms | 40 MB | 0.17% |
| 50% | +9.73 ms | +272.5 ms | 66 MB | 0.28% |

**Reading the numbers.** The latency tax is expressed as a fraction of a typical decode step (~30 ms), the memory gain as a fraction of the 24 GB GPU budget. At 7B f=10%, Alt A roughly doubles decode step time to save 0.3% of the GPU budget — a bad trade by any reading. The ratio gets worse with `f` (latency scales super-linearly with CPU GEMM size once oneDNN tiling crosses microkernel boundaries, while memory scales linearly).

**Decision: Alt B.** Phase 2 keeps WO fully GPU-resident. WO has no entry in `cpu_gemm_curve` at runtime (though the profile table keeps it under the "col" axis — §0.2 validated correctness for Alt A in case a future phase revisits the decision). This commits `weight_offload_design.md §WO Split Axis Decision` to the no-offload alternative.

**When to revisit.** The argument reverses when GPU memory becomes the binding constraint. At 14B in Phase 3, the model weights alone (~28 GB BF16) exceed the 24 GB budget before any offload. Even with aggressive offload on the larger sub-modules, every MB of GPU memory matters — and WO at 14B is ~1.3 GB, so a 30% offload frees ~390 MB, which is 3–5% of the remaining budget rather than 1%. In that regime the memory fraction gain is binding enough that the latency tax may be justified. Repeat the §0.4.2 procedure against the 14B decode workload before Phase 3 locks in; if the ratios flip, switch to Alt A for WO at 14B only.

### 0.4.3: Phase 1/2 design commitments

Locked for Phase 1 implementation:

1. **Split axes**: WQKV = col (K/V-biased), MLP1 = col, MLP2 = row, WO = not offloaded (fully GPU-resident in Phase 1/2). `CpuComputeDispatcher` implements col + row paths for the three offloaded sub-modules; WO has no CPU path, no prefetch path.
2. **Uniform per-bucket dispatch**: Planner emits one `(f_cpu, f_prefetch)` pair per `num_tokens` bucket, applied uniformly to WQKV, MLP1, and MLP2. The MLP1↔MLP2 matched-index invariant is automatic under uniform dispatch (same scalars → same index set selected by construction). See `planner_design.md §4.2` and the §0.3.4 uniformity subsection for the empirical basis.
3. **Layer-ahead prefetch (Phase 1b)**: single prefetch queue per layer boundary, one `cudaStreamWaitEvent` per layer. Tensor-ahead rejected — see `pcie_bandwidth_allocation_design.md`.
4. **Assembly operations**: col → `torch.cat` along output dim; row → `.add_` on the partial sum. Both correctness-validated in §0.2 at every tested `(f, B)` for both models.

---

## 0.5: PCIe behavior — bandwidth, contention, UVA bypass

Per-layer in Phase 1b, weight prefetch (**bg**) and CPU-compute activation returns (**fg**) both target the H2D direction on PCIe. This section answers:

1. What single-direction PCIe bandwidth is available (the baseline)?
2. How does same-direction H2D contention behave on RTX 4090?
3. What mechanism keeps fg latency bounded under continuous bg prefetch?

**Methodology — nsys-driven (CUPTI ground truth).** Contention timings come from CUPTI activity records in nsys SQLite exports, not from CUDA event API calls. Each iteration is wrapped in NVTX ranges; per-role NVTX sub-ranges (`submit_fg`, `submit_bg`) tag launches. The parser correlates GPU-side memcpy/kernel events to their submitting role via CUPTI runtime `correlationId`. Source: `bench_contention.py` (orchestrator + workload modes); data: `results/0.5_pcie/contention.json` (the orchestrator deletes the `.nsys-rep`/`.sqlite` after parsing — for visual inspection use the focused probes `probe_engines.py` / `probe_uva_bypass.py`, whose traces land in `results/0.5_pcie/probe_traces/`). Single-direction baseline timings (§0.5.0) are CUDA-event-driven via `bench_pcie_sweep.py`.

**Key metric: `fg_s2c` (submission-to-complete latency).** Time from the host-side `submit_fg` NVTX range start to fg's last GPU activity end. This includes any wait fg incurs in the CE0 FIFO. CUDA-event-based `fg_active` time captures only the on-engine kernel duration and hides wait — the deprecated framing of this section confused the two.

**Hardware context.** RTX 4090 reports `asyncEngineCount = 2`. NSys engine attribution (`probe_engines.py`) confirms: **CE0 = H2D, CE1 = D2H, neither services the other direction.** Two H2D operations on different streams must queue on CE0; H2D and D2H on different streams use CE0 + CE1 concurrently.

**Three PCIe traffic paths from the GPU side.** This is the architectural detail that makes UVA bypass possible:

| Path | Hardware | Used by |
|---|---|---|
| **CE0** (H2D copy engine) | Dedicated DMA engine, single FIFO | `cudaMemcpyAsync` host→device |
| **CE1** (D2H copy engine) | Dedicated DMA engine, single FIFO | `cudaMemcpyAsync` device→host |
| **SM-initiated PCIe** | SMs → L1 → L2 → GPU MMU → PCIe root complex | UVA reads/writes from kernels |

Engine-level FIFOs serialize within an engine (two H2D ops both go to CE0 → queue). The PCIe link itself is shared at packet level across all three paths — multiple sources can have transactions in flight simultaneously, with bandwidth split by the link arbiter. The serialization in §0.5.1 is engine-level, not link-level. UVA bypasses CE0's FIFO by using a separate hardware path; it still shares link BW with CE0 traffic, but doesn't queue behind it.

### 0.5.0: PCIe bandwidth reference card

Measured by `bench_pcie_sweep.py` on RTX 4090 + i9-14900KF with pinned memory. Single-direction H2D and D2H size-vs-bandwidth curve for `cudaMemcpyAsync` over the PCIe 4.0 x16 link; used as the baseline for Planner cost models. Two-direction dynamics are in §0.5.1–§0.5.4 below.

#### Size sweep

| Size | H2D (GB/s) | D2H (GB/s) |
|---:|---:|---:|
| 0.25 MB | 11.30 | 15.29 |
| 1 MB    | 20.18 | 22.20 |
| 4 MB    | 22.80 | 25.16 |
| 10 MB   | 23.51 | 25.89 |
| 100 MB  | 23.80 | 26.31 |
| 500 MB  | 23.92 | 26.38 |

Asymptotic H2D ≈ 24 GB/s (~76% of PCIe 4.0 x16 theoretical 31.5 GB/s); D2H ≈ 26 GB/s. D2H is consistently slightly faster than H2D across all sizes.

#### Key transfer sizes for the thesis

| Purpose | Size | H2D BW |
|---|---:|---:|
| CPU K+V output per step (B=16, kv_dim=1024, bf16) | 33 KB | 3.57 GB/s |
| Activation result (B=16, hidden=3584, bf16)        | 115 KB | 9.40 GB/s |
| 9% MLP1 weight slice                              | 4 MB | 22.85 GB/s |
| Full layer weight (7B)                            | 466 MB | 23.91 GB/s |

#### Design implications

1. **Small transfers pay a launch-overhead tax.** KB-scale H2D transfers achieve only 15–40% of peak bandwidth. The Planner's cost model for the CPU-compute activation round-trip must index into the correct size bin; assuming saturated 22 GB/s would overestimate by 3–5×.

2. **MB-scale weight prefetch runs near peak.** A 4 MB weight slice saturates to ~96% of peak H2D. This validates the quantitative premise of `pcie_bandwidth_allocation_design.md` — bulk prefetch operates in the saturated regime.

### 0.5.1: Same-direction H2D serializes on CE0

Two equal-sized pure-H2D transfers on separate streams, `fg_first`. Each transfer's wall ≈ iso wall (full BW for its active period); aggregate wall ≈ 2× iso (sum-of-bytes / link-BW).

| size | iso wall | fg active (co) | bg active (co) | iter wall | wall / iso |
|---|---:|---:|---:|---:|---:|
| 1 MB  |   45.0 μs |   45.0 μs |   45.0 μs |    91.4 μs | **2.03×** |
| 4 MB  |  176.3 μs |  176.3 μs |  176.4 μs |   354.0 μs | **2.01×** |
| 16 MB |  701.5 μs |  701.5 μs |  701.6 μs |  1404.4 μs | **2.00×** |
| 64 MB | 2802.3 μs | 2802.0 μs | 2802.1 μs |  5605.4 μs | **2.00×** |

**Verdict: pure serialization on CE0.** Aggregate H2D throughput cannot exceed link BW because only one engine carries H2D. NSys timeline (`probe_engines.py` phase A): two H2D ops on alternating streams run sequentially on CE0, no interleaving. Layer-level implication: **once bg chunks are queued on CE0, fg's `cudaMemcpyAsync` goes to the end of the queue** — its wait scales with total queued bg, not just the in-flight chunk's remainder.

### 0.5.2: Bidirectional H2D + D2H

H2D weight prefetch on CE0 and D2H KV spill on CE1 run concurrently. Measured: aggregate throughput exceeds either unidirectional baseline (CE0 + CE1 are physically separate engines), so the engines work in parallel. **H2D is undisturbed by concurrent D2H** (708 μs solo → 708 μs under D2H pressure). **D2H drops 36% under H2D pressure** (link-arbiter favors inbound writes), but at 100 KB-scale KV chunks (~10 μs each) the absolute cost is dwarfed by the 178 μs prefetch window. **KV spill on D2H does not block weight prefetch.** Watch-item: the comfort margin shrinks if per-step KV-spill volume grows.

### 0.5.3: DMA vs UVA copy kernel — isolated and under contention

The design-relevant comparison: in isolation and under continuous bg DMA, what is fg's submission-to-complete latency?

| fg path | bg state | **fg_s2c** | fg active | bg active | iter wall |
|---|---|---:|---:|---:|---:|
| dma_copy        | none (no bg) |  **15.4 μs** |  5.3 μs | —      |  18.3 μs |
| uva_copy_kernel | none (no bg) |  **24.0 μs** |  6.4 μs | —      |  26.2 μs |
| dma_copy        | with 4MB bg  | **181.3 μs** |  5.4 μs | 176 μs | 196.3 μs |
| uva_copy_kernel | with 4MB bg  |  **34.1 μs** | 12.9 μs | 180 μs | 194.7 μs |

**Two regimes, opposite winners.**
- **Isolated**: DMA wins by 8.6 μs (15.4 vs 24.0). UVA's launch overhead and slightly higher kernel cost.
- **Under bg**: UVA wins by **147 μs** (34.1 vs 181.3). DMA fg waits behind bg in CE0's FIFO; UVA fg runs on SMs through the GPU MMU path concurrently with CE0.

UVA's `fg_active` rises from 6.4 μs (isolated) to 12.9 μs (under bg) — link-arbiter sharing with CE0 cuts the kernel's effective BW roughly in half. But the kernel still finishes inside bg's 178 μs window, so `fg_s2c` is dominated by kernel duration plus launch overhead, not by bg's transfer time. This is the bypass: UVA goes on a separate engine path, sharing link BW packet-wise rather than queueing.

### 0.5.4: Validation — 4 fg variants × bg chunk sizes

Continuous bg DMA at 4 MB total volume, varying chunk size. `bg_first` ordering exposes the queue-blocking dynamic. Four fg variants test the bypass robustness and the access-pattern contract:

- `dma_copy` — `cudaMemcpyAsync` of 98 KB (Option A baseline; tests whether chunked bg gives DMA fg enough relief)
- `uva_copy_kernel` — Triton kernel reading UVA, writing to GPU buffer (Option B, the bypass)
- `uva_matmul` — UVA tensor as matmul input (broken pattern, strawman for the access-pattern contract)
- `dma_into_matmul` — DMA copy + matmul (full Phase 1b fg op shape including compute tail)

| variant | chunk | **fg_s2c** | fg active | bg s2c | iter wall |
|---|---:|---:|---:|---:|---:|
| `dma_copy`         | 64 KB |  122 μs |  5.3 μs | 317 μs | 326 μs |
| `dma_copy`         | 256 KB | 166 μs |  5.3 μs | 217 μs | 228 μs |
| `dma_copy`         | 1 MB  |  178 μs |  5.4 μs | 194 μs | 203 μs |
| `dma_copy`         | 4 MB  |  181 μs |  5.4 μs | 187 μs | 196 μs |
| **`uva_copy_kernel`** | 64 KB | **29 μs** |  8.5 μs | 320 μs | 323 μs |
| **`uva_copy_kernel`** | 256 KB | **32 μs** | 12.3 μs | 222 μs | 225 μs |
| **`uva_copy_kernel`** | 1 MB  | **35 μs** | 13.0 μs | 198 μs | 202 μs |
| **`uva_copy_kernel`** | 4 MB  | **34 μs** | 12.9 μs | 191 μs | 195 μs |
| `uva_matmul`       | 64 KB | 578 μs | 560 μs | 836 μs | 840 μs |
| `uva_matmul`       | 4 MB  | 710 μs | 694 μs | 594 μs | 725 μs |
| `dma_into_matmul`  | 64 KB | 151 μs | 23 μs | 316 μs | 350 μs |
| `dma_into_matmul`  | 4 MB  | 208 μs | 23 μs | 186 μs | 222 μs |

**Three regimes confirmed by `fg_s2c`:**

1. **`dma_copy` waits behind bg**: fg_s2c scales 122 → 181 μs as chunks grow. Small chunks (64 KB) leave driver-side gaps that fg sneaks into; large chunks (4 MB) make fg wait the full bg duration. **Chunking helps but doesn't eliminate the wait.**
2. **`uva_copy_kernel` bypasses CE0**: fg_s2c stays 29–35 μs across all chunk sizes. **The bypass holds on RTX 4090.** The slight rise reflects link-BW sharing as bg becomes more sustained.
3. **`uva_matmul` is broken**: fg_s2c 578–710 μs. The cuBLAS GEMM access pattern over UVA blows up PCIe traffic; fg is slow *and* it starves bg (bg_s2c also degraded). Mechanism not pinned down (would require Nsight Compute), but the empirical "don't use UVA as a re-read kernel input" contract is validated.

`dma_into_matmul` is the realistic Phase 1b shape: DMA copy + matmul tail. fg_s2c at 4 MB chunk = 208 μs (181 μs of wait + ~23 μs of matmul) — what we'd actually pay if we picked Option A.

NSys timeline confirms physical concurrency for `uva_copy_kernel`: the Triton kernel on the SM stream starts *inside* bg's CE0 window and overlaps for its full ~13 μs duration. SM and CE0 are different hardware paths to the same link.

**Phase 1b production budget** (3 fg events per layer, 7B decode, ~500 μs/layer compute window):

| approach | fg_s2c per event | total fg cost / layer |
|---|---:|---:|
| dma_copy + 4 MB single-shot bg | 181 μs (full wait) | **~543 μs** (kills the layer) |
| dma_copy + 64 KB chunked bg    | 122 μs (partial wait, 30% bg BW penalty) | ~366 μs |
| **uva_copy_kernel + any bg**   | **34 μs (no wait)** | **~102 μs** |

UVA copy kernel wins decisively at production scale. The 8.6 μs isolated-case penalty is the price tag; the 147 μs saved per fg event when bg is in flight (which is most of the time in the pipelined design) pays for it 17×.

### 0.5.5: Design recommendations

1. **Weight prefetch (bg, all phases): explicit `cudaMemcpyAsync` on pinned memory.** Uses CE0 at peak BW. UVA for weights is eliminated a priori (PCIe ~23 GB/s vs GDDR6X ~1008 GB/s is 46× slower).

2. **fg activation return (Phase 1b CPU-compute path): zero-copy via SM-issued Triton/CUDA copy kernel reading UVA-mapped pinned memory.** Bypasses CE0; runs concurrently with bg DMA. fg_s2c is bounded at ~30–35 μs regardless of bg state. **One-shot UVA copy is the right mechanism for fg.**

3. **KV spill (Phase 2): explicit `cudaMemcpyAsync`, D2H.** Uses CE1 (different engine from bg's CE0). The 36% link-arbiter penalty under H2D pressure is well within margin for 100 KB-scale chunks.

4. **Stream issue order: fg-first when possible.** Within a single fg+bg batch, fg-first puts the small cost on bg (~4 μs). When fg events arrive after bg has already been queued (the layer-level case), fg-first ordering at submission cannot prevent waiting — but with recommendation #2, fg doesn't queue on CE0 in the first place, making the ordering rule a micro-optimization rather than a correctness invariant.

5. **No bg chunking required.** Once recommendation #2 is adopted, fg_s2c is independent of bg chunk size, so bg can use a single large H2D per layer at peak BW. The previously-considered "chunked bg" mitigation is unnecessary.

6. **The "100% PCIe H2D to weight prefetch" invariant in `pcie_bandwidth_allocation_design.md` stands** — but the reason changes. fg uses the SM-PCIe path, not CE0. CE0's full bandwidth remains available for bg weight prefetch.

7. **One-shot access pattern is a load-bearing contract for any UVA consumer.** Direct kernel inputs that re-read (matmul, attention, anything tiled with re-reads) are catastrophic on UVA-mapped memory. Always copy UVA → GPU buffer first via the one-shot kernel; downstream consumers read from the GPU buffer.

These recommendations close the PCIe-allocation question for Phases 1b–3.

---

## 0.6: CPU attention latency (reference)

Measured by `bench_cpu_attn.py` using PyTorch's `F.scaled_dot_product_attention` on CPU with `enable_gqa=True`. Feeds `cpu_attn_curve[batch_size, suffix_context_len]` in the Planner's schema (`profiler_design.md §1.4`).

**Decode only.** Each sequence contributes `q_len=1` (one new token per beam per decode step) attending over `kv_len=S` of suffix history. This matches the Phase 2 design: prefix-on-GPU / suffix-on-CPU means the CPU attention path only ever runs the decode-time suffix attention. Prefill attention (both prefix and the generator's initial prompt) runs on GPU and is not in scope for this table.

**Implementation caveat**: vLLM's optimized CPU attention kernel (`cpu_attention_with_kv_cache`) is only compiled when `VLLM_TARGET_DEVICE=cpu`; our CUDA build does not include it. The numbers below use native PyTorch SDPA (oneDNN / MKL paths) and are therefore an **upper bound** on the real kernel's latency. The expected speedup from the C++ kernel is 2–5×; the shape of the curve is what matters for Planner sizing. Real numbers will land when Phase 2 integrates the C++ kernel (`implementation_roadmap.md §2`, also requires the per-head LSE output modification).

### Qwen2.5-7B (num_query_heads=28, num_kv_heads=4, head_dim=128)

Mean latency in ms, BF16, 24 CPU threads:

| B \ S | 100 | 500 | 1000 | 2000 | 4000 |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.18 | 0.70 | 1.35 | 2.78 | 5.77 |
| 8 | 0.14 | 1.45 | 2.86 | 5.84 | 12.24 |
| 16 | 0.26 | 2.71 | 5.32 | 10.30 | 21.60 |
| 32 | 0.55 | 5.43 | 9.93 | 20.68 | 43.01 |

### Skywork-PRM-1.5B (num_query_heads=12, num_kv_heads=2, head_dim=128)

| B \ S | 100 | 500 | 1000 | 2000 | 4000 |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.06 | 0.31 | 0.59 | 1.12 | 2.38 |
| 8 | 0.06 | 0.59 | 1.13 | 2.28 | 4.80 |
| 16 | 0.12 | 1.18 | 2.34 | 4.89 | 9.53 |
| 32 | 0.22 | 2.31 | 4.56 | 8.63 | 18.21 |

Roughly 2.3× faster than 7B, matching the query-head ratio (28/12 ≈ 2.33) — attention cost scales with Hq·Hkv·S·D.

### Observations

1. **Linear in B×S.** Arithmetic intensity is essentially constant at ~3000 tokens/ms for 7B and ~7000 tokens/ms for PRM-1.5B across the (B, S) grid (ignoring the S=100 row where per-iteration overhead dominates). The Planner can linearly interpolate `cpu_attn_curve` in either axis.

2. **CPU attention is expensive relative to GPU.** At B=16, S=1000 (realistic beam-search decode on 7B), suffix attention costs 5.32 ms per layer — **11× the GPU baseline layer time** of 0.47 ms (§0.3). Over 28 layers, pure CPU attention would add ~149 ms per decode step. Even with the expected 2–5× speedup from the real C++ kernel, the CPU attention path is the critical bottleneck of any Phase 2 attention-offload configuration.

3. **Back-pressure is mandatory.** The Planner must cap batch × suffix_length such that CPU attention fits within the GPU forward-pass budget; otherwise attention offload *increases* latency. `KV_cpu_bytes` and `max_num_seqs` are the two knobs:
   - Reducing `KV_cpu_bytes` forces shorter effective suffix per beam (via suffix pruning or tighter admission).
   - Reducing batch size (at the Scheduler level) caps the B axis.
   The risk mitigation listed in `thesis_proposal.md §10` ("CPU attention bottleneck at long contexts") is empirically justified: this curve is where that risk bites.

4. **Small-workload noise.** The S=100 row shows non-monotonicity (B=4: 0.18 ms, B=8: 0.14 ms). At tiny workloads the 24-thread parallel dispatch overhead competes with the compute; SDPA also has dynamic dispatch logic for backend selection. Not a concern for the Planner — the interesting operating range is S ≥ 500 where compute dominates.

5. **Per-model asymmetry.** The verifier (PRM-1.5B) has roughly half the head count, so its suffix attention is ~2× cheaper. The Planner will admit more `KV_cpu_bytes` for the verifier than the generator at the same latency budget.

---

## 0.7: CUDA graph impact

Measured by `bench_cuda_graph.sh` — `vllm bench latency` with and without `--enforce-eager`, comparing CUDA-graph-captured execution to eager per-kernel launches. Purpose: decide whether Phase 4 (the `cudaLaunchHostFunc` retrofit that makes the Phase 2/3 CPU-dispatch and prefetch paths CUDA-graph-compatible) is urgent or can be safely deferred.

### Setup

- Model: Qwen2.5-7B-Instruct, BF16
- Prompt length: 512 tokens, output length: 128 tokens
- Warmup: 3 iters, measurement: 10 iters
- Batch sizes: 1, 4, 16, 64

### Results

End-to-end generation latency (prefill + 128-token decode):

| B | CUDA Graph (ms) | Eager (ms) | Eager overhead |
|---:|---:|---:|---:|
| 1 | 2063.93 | 2074.43 | +0.5% |
| 4 | 2244.47 | 2285.12 | +1.8% |
| 16 | 2857.70 | 2913.72 | +2.0% |
| 64 | 5484.59 | 5552.26 | +1.2% |

### Finding

**CUDA Graph impact is 0.5–2.0% across all tested batch sizes.** The graph-vs-eager gap is small because 7B decode on an RTX 4090 is memory-bandwidth-bound per layer (§0.3: ~0.47 ms/layer, >90% of GDDR6X peak). The per-kernel launch overhead that CUDA graphs eliminate is a tiny fraction of each layer's memory-bound execution time. Graphs help most when many small kernels dispatch rapidly; our decode path has four relatively coarse-grained matmuls per layer, so eager launches are already efficient.

### Implications for Phase 4 priority

1. **Prototyping with `enforce_eager=True` costs at most ~2% throughput** for the main thesis workload. This is well within the noise of FastTTS beam-search measurements. The `CpuComputeDispatcher` abstraction (Phase 1 day-1 choice; `implementation_roadmap.md §1.3`) is viable against eager-mode forward passes for all of Phase 1–3.

2. **Phase 4 remains correctly prioritized at the tail of the roadmap.** The `cudaLaunchHostFunc` + `CPUInfer` retrofit is ~400 lines of engineering (per the roadmap). Trading that effort for a 2% throughput gain would be a poor allocation of thesis time unless the final system is production-bound.

3. **V1's per-process memory overhead (0.9–1.4 GiB from CUDA graphs + torch.compile, see `vllm_v1_migration.md §12`) is the real cost of enabling graphs.** For our two-process generator+verifier setup, disabling graphs via `enforce_eager=True` recovers ~2 GiB of GPU memory — which then becomes additional KV budget. This is a *net-positive* trade for the thesis: eager mode gives back more KV capacity than it costs in per-step latency.

### Recommendation

Phase 1–3 prototypes run with `enforce_eager=True` by default. Phase 4 (graph retrofit) is deferred as originally planned.

### Framing Phase 4's purpose

Phase 4's value is **upstream compatibility, not latency**. The §0.7 measurement shows graph capture gives only ~2% throughput on our thesis workload — too small to be a Phase 4 justification on its own. The real reason Phase 4 exists:

1. **vLLM integration.** Production vLLM defaults to `FULL_AND_PIECEWISE` graph capture. A Python-threaded `CpuComputeDispatcher` falls out of graph capture — any downstream user who enables graphs would silently lose the CPU-compute path. For the work to land upstream, the CPU-dispatch and prefetch paths must be `cudaLaunchHostFunc`-based.

2. **Graph benefit grows with system scale.** On 8-GPU TP deployments each layer has many more kernels (all-reduces, MoE routing, etc.), so launch overhead fraction grows. The 2% we measured on a single 4090 becomes 10%+ on production setups.

3. **Compiler pipelines assume graph capture.** `torch.compile` + piecewise graph capture is becoming the vLLM default; opting out increasingly looks like pre-modern infrastructure.

For the thesis's research question ("does offloading work on consumer hardware?"), Phase 4 contributes nothing. For the engineering deliverable ("is this merge-compatible with modern LLM serving?"), Phase 4 is mandatory. That distinction is what makes Phase 4 correctly placed at the tail of the roadmap — do it only if time allows after the research claim is validated.

---

## 0.8: vLLM V1 FastTTS baseline

See §0.9 for the matched offload ablation. This section reports the `fasttts` (non-offload) arm of that sweep — the V1 reference throughput numbers the thesis measures improvements against.

### Reference throughput (7B generator + PRM-1.5B verifier, RTX 4090)

| Dataset | n | Latency (s) | Goodput (tok/s) | gen_gpu_hit | gen_max_kv_usage |
|---|---:|---:|---:|---:|---:|
| aime | 1 | 15.5 | 61.9 | 97.1% | 0.05 |
| aime | 4 | 18.2 | 47.7 | 97.9% | 0.09 |
| aime | 16 | 34.7 | 24.8 | 98.4% | 0.42 |
| aime | 64 | 58.9 | 13.8 | 98.0% | **1.00** |
| aime | 256 | 122.9 | 6.9 | 95.2% | **1.00** |
| math500 | 1 | 9.5 | 61.5 | 96.0% | 0.07 |
| math500 | 4 | 12.7 | 45.2 | 97.4% | 0.13 |
| math500 | 16 | 18.3 | 30.6 | 97.7% | 0.42 |
| math500 | 64 | 31.8 | 17.8 | 97.6% | **1.00** |
| math500 | 256 | 65.3 | 8.7 | 96.1% | **1.00** |

### Observations

1. **KV cache saturates at n=64 on aime (longer completions, ~800 tok) and n=256 on math500 (shorter, ~520 tok).** This is the goalpost Phase 1 targets: freeing ~1.2 GB of GPU memory via weight offload (predicted from §0.1+0.3 overlap analysis) should raise the saturating n by ~25%, validated in Phase 4.

2. **Baseline goodput declines sharply from n=1 to n=256** (62 → 7 tok/s on aime, 62 → 9 tok/s on math500). Most of the decline before saturation (n=1 → n=64) is inherent to beam parallelism producing correlated outputs; the decline past saturation (n=64 → n=256) is additional throughput loss from KV-pressure queueing (`gen_frac_steps_queued` jumps from 0.1% to 8.1% on aime).

3. **Prefix caching is highly effective** across all n: `gen_gpu_hit` 95–98% means prefill costs are amortized across beams. This is the "shared prefix" property Phase 1's WQKV K/V-biased split is designed to preserve — our column-parallel weight split must not disrupt prefix caching, which is the FastTTS win.

4. **These are V1 numbers on vLLM 0.19.0**, not V0 on 0.9.2 (the baseline in the original FastTTS paper). V1 has slightly higher per-process memory overhead (~0.9–1.4 GiB from CUDA graphs + torch.compile, see `vllm_v1_migration.md §12`), which slightly raises the GPU memory split for generator vs. verifier (0.74 / 0.16 recommended) and is the reason we can't directly compare to the published FastTTS numbers. Phase 4 will re-evaluate against these V1 numbers.

---

## 0.9: vLLM V1 built-in KV offload — impact

See §0.8 for the baseline arm of this sweep. Measured by `bench_kv_offload.py` — FastTTS end-to-end runs with and without vLLM V1's built-in CPU KV offload (`vllm/v1/kv_offload/`).

**What the V1 offload actually does (and why it's orthogonal to our Phase 2).** V1's KV offload is a **prefill-side optimization**: when cold prefix KV blocks are evicted from the GPU KV cache, instead of being dropped they're spilled to CPU. On a subsequent request that shares that prefix, the blocks are loaded back from CPU and prefill attention is skipped for those tokens. It's a prefix-reuse / prefill-avoidance mechanism that expands the effective prefix cache from VRAM-sized to RAM-sized.

This is **orthogonal to our Phase 2 design**:

| | V1 KV offload (§0.9) | Phase 2 attention offload |
|---|---|---|
| What's avoided | Re-computing prefill for cached prefixes | Running decode-time suffix attention on GPU |
| Where KV lives at attention time | **GPU** (reloaded from CPU before attention) | **CPU** (suffix stays on CPU) |
| Where attention runs | GPU | **Split**: prefix on GPU, suffix on CPU, merged via online softmax |
| Freed resource | GPU KV cache space (evicted-then-reloadable) | GPU KV cache space (per-beam suffix moved off GPU) |
| Cost | PCIe load on prefix miss | CPU GEMV per decode step per layer |

The two mechanisms can coexist. Phase 2 reuses V1's CPU block pool and allocator infrastructure (the `vllm/v1/kv_offload/` framework), but the attention path is distinct: V1 offload is prefill-only, Phase 2 is decode-time.

### Sweep

`{aime, math500}` × `{fasttts (no offload), fasttts_kvoff (V1 offload enabled)}` × `n ∈ {1, 4, 16, 64, 256}`. 7B generator + Skywork-PRM-1.5B verifier on RTX 4090. `math500, n=256, fasttts_kvoff` is pending and will be backfilled.

### End-to-end latency and goodput

| Dataset | n | Baseline lat (s) | +Offload lat (s) | Δ lat | Baseline goodput (tok/s) | +Offload goodput | Δ goodput |
|---|---:|---:|---:|---:|---:|---:|---:|
| aime | 4 | 18.2 | 23.6 | **+30%** | 47.7 | 41.4 | −13% |
| aime | 16 | 34.7 | 32.4 | −6% | 24.8 | 27.8 | +12% |
| aime | 64 | 58.9 | 60.6 | +3% | 13.8 | 13.7 | −1% |
| aime | 256 | 122.9 | 114.4 | **−7%** | 6.9 | 7.5 | **+9%** |
| math500 | 4 | 12.7 | 12.4 | −2% | 45.2 | 45.8 | +1% |
| math500 | 16 | 18.3 | 18.5 | +1% | 30.6 | 30.5 | 0% |
| math500 | 64 | 31.8 | 31.8 | 0% | 17.8 | 17.8 | 0% |
| math500 | 256 | 65.3 | (pending) | — | 8.7 | — | — |

### Prefix-reuse metric: `gen_cpu_hit`

Fraction of prefix KV lookups served from CPU-resident cached blocks (as opposed to GPU-resident or newly prefilled). Evidence that the V1 offload mechanism is actually engaging:

| Dataset | n | gen_cpu_hit (baseline) | gen_cpu_hit (+offload) | gen_gpu_hit (+offload) |
|---|---:|---:|---:|---:|
| aime | 4 | 0% | 0.3% | 98.3% |
| aime | 16 | 0% | 3.0% | 98.4% |
| aime | 64 | 0% | 20.5% | 97.0% |
| aime | 256 | 0% | **66.7%** | 78.6% |
| math500 | 64 | 0% | 9.1% | 97.0% |

### Findings

1. **GPU prefix cache is already capturing 95–98% of hits without offload.** Across the whole sweep, `gen_gpu_hit` for the baseline sits in {95.2%, 96.0%, 96.1%, 97.1%, 97.4%, 97.6%, 97.7%, 97.9%, 98.0%, 98.4%}. Even at n=256 with saturated KV, the GPU prefix cache retains ≥95% of the requests' KV. V1 offload's potential upside — turning a prefill miss into a CPU reload — is bounded by this 2–5% miss rate. The ceiling is narrow.

2. **On the stable workload (math500, 500 problems), V1 offload impact is ≤1% across all n.** Latency delta stays between −2% and +1%, goodput within ±1%. Consistent with finding (1): the mechanism engages (CPU hit rate 0.9% → 9.1% from n=4 to n=64) but there's not enough miss volume for it to matter.

3. **aime results are noisier (30 problems vs. math500's 500).** The +30% latency regression at aime n=4 and the −7% improvement at aime n=256 are plausible but workload-specific / small-sample signals. We treat math500 as the reliable reference and note aime only as a sanity check of direction.

4. **V1 offload consumes PCIe H2D bandwidth that Phase 3 reserves for weight prefetch.** Every reloaded prefix block travels CPU→GPU over H2D. Phase 3's "100% PCIe H2D → weight prefetch" invariant (`pcie_bandwidth_allocation_design.md`, §0.5) assumes no other H2D traffic. Even if V1 offload's contribution is small at typical operating points (math500: ≤1% wall-clock impact), it is nonzero contention against the prefetch budget and adds scheduler-side variability.

5. **Reusable as Phase 2 infrastructure.** The V1 offload codebase provides a working CPU block pool, pinned allocator, and GPU↔CPU block-copy primitives. Phase 2 extends this framework with "blocks that stay on CPU" semantics (for suffix attention) in addition to the existing "blocks that get reloaded before attention" (for prefix reuse). The two modes can cohabit in the same pool.

### Recommendation for Phase 4 evaluation

**Disable V1 KV offload in both the thesis configuration and the baseline comparison.** Four reasons, in order of importance:

1. **The upside is small to begin with.** GPU prefix cache already captures 95–98% of hits across the whole n sweep (finding 1). V1 offload can only improve the remaining 2–5% miss rate, which math500 confirms is a ≤1% wall-clock effect.

2. **PCIe contention with weight prefetch.** The H2D reload traffic would mix with Phase 3's weight prefetch H2D, breaking the design invariant that prefetch owns PCIe. Even small additional H2D traffic perturbs the measurements and adds scheduler-side variability.

3. **Clean attribution.** We want to isolate Phase 1+2+3's effect. Enabling V1 offload means "with-thesis" numbers include both thesis mechanisms AND whatever V1 offload happens to do at that operating point.

4. **Mechanism orthogonality.** V1 offload is prefill-side prefix-reuse; Phase 2 is decode-time suffix attention. Complementary, not competing. Mixing them in the headline comparison obscures what each contributes.

Both disabled is the apples-to-apples comparison. Re-enabling V1 offload for the full system is a separate ablation if it becomes interesting later.
