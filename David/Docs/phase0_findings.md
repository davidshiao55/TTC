# Phase 0 Benchmark Findings

This document records findings from Phase 0 pre-implementation benchmarking — the **first iteration of the Profiler's output** (see `profiler_design.md`). It establishes baseline numbers for RTX 4090 + Qwen2.5-7B + BF16 and validates the measurement methodology future profile runs will reuse.

Reader note after Phase 1 cleanup: this file remains a Phase 0 profiler record.
It still contains historical references to the Phase 1a/1b/1c milestone docs
because those references explain how specific measurements motivated later
implementation work. For the final production COTS path and current Phase 1
numbers, use `phase1_findings.md`.

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
- §0.7 — CUDA graph impact (Phase 1c scope)
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

**Methodology — cold-cache ring on both CPU and GPU.** `time_cpu` and `time_gpu` (`bench_cpu_gpu_overlap.py`) cycle through `ceil(working_set_mb / weight_mb)` distinct weight tensors (capped at 32) before each iteration repeats. Working-set targets: **192 MB on GPU** (≈2.7× the RTX 4090's 72 MiB L2), **1024 MB on CPU** (≈28× the i9-14900KF's 36 MiB L3). This reflects what the runtime sees during decode — each step touches a different layer's weights and evicts prior layers — and prevents the L2-resident bias that small slices (e.g. WQKV reduced ≈ 31 MB) carry under a single-weight tight loop. Sized by working set rather than a fixed layer count so larger models don't OOM the GPU's ring (28 copies of 14B's WQKV would cost ~3.4 GB).

| Sub-module | Weight (BF16) | GPU time (B=1) | GPU time (B=16) |
|---|---:|---:|---:|
| WQKV | 33.0 MB | 38 µs | 39 µs |
| WO | 25.7 MB | 32 µs | 32 µs |
| MLP1 | 271.6 MB | 287 µs | 294 µs |
| MLP2 | 135.8 MB | 148 µs | 154 µs |
| **Total** | **466.1 MB** | **505 µs** | **519 µs** |

Source: `results/0.3_cpu_gpu_overlap/qwen7b_t16_ring.json`. WQKV and WO are ~2× higher than the original single-weight measurement (19 → 38 µs WQKV; 17 → 32 µs WO) — that bias was the L2-cache effect. MLP1 and MLP2 are unchanged (full weight already exceeds L2).

### 0.3.2: CPU compute path

#### Critical discovery — single-weight tight loops measure L3, not DRAM

A naive CPU-GEMM microbench reuses one weight tensor in a tight loop. For Qwen2.5-7B's MLP slice at f≈0.09, that one tensor is ~35 MiB — fits the i9-14900KF's 36 MiB L3 — so every iteration after the first hits L3 and the timing is **compute-bound**.

Real decode cycles through 28 different layers' weight slices on every forward (~0.96 GiB total), well past L3. Each call is **DRAM-streaming, memory-bound** at ~5× the L3-resident time at small B. The cache effect collapses by B≥8 (compute starts to dominate the per-call FLOPs and amortizes the DRAM read).

| B | MLP1 9% slice — single-tensor loop | MLP1 9% slice — 28-layer ring (real) | Ratio |
|---:|---:|---:|---:|
| 1 | 0.087 ms | 0.442 ms | 5.1× |
| 4 | 0.339 ms | 0.521 ms | 1.5× |
| 8 | 0.674 ms | 0.721 ms | 1.07× |
| 16 | 1.340 ms | 1.378 ms | 1.03× |
| 32 | 2.688 ms | 2.710 ms | 1.01× |

**Design decision: all subsequent CPU GEMM timing in §0.3 cycles through 28 distinct weight tensors** (`bench_cpu_gpu_overlap.py:time_cpu` with `CPU_CYCLE_LAYERS=28`). The Planner's `cpu_gemm_curve` is regenerated from this. All numbers below are the realistic deployment-streaming timings.

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
| **`F.linear` BF16** | **0.442 ms** | `torch._C._nn.linear` → oneDNN BF16 |

`F.linear` BF16 is ~50× faster than `torch.mm` BF16 — the only viable CPU
path for our use case. BF16 weights also halve memory bandwidth vs FP32,
which matters at low B where the GEMM is DRAM-bound.

**Design decision: use `F.linear` with BF16 weights on CPU.** No FP32 weight
duplication needed. The `CpuComputeDispatcher` must use `F.linear`, not `torch.mm`.

#### BF16 `F.linear` scaling with batch size

MLP1 9% slice (3584 × 3410), BF16 `F.linear`:

| B | Time |
|---|---|
| 1 | 0.442 ms |
| 4 | 0.521 ms |
| 8 | 0.721 ms |
| 16 | 1.378 ms |
| 32 | 2.710 ms |

BF16 stays the design choice: no memory doubling vs FP32 (critical for 14B+ where CPU RAM is constrained), and the absolute times still allow useful overlap with GPU at small `f_cpu` (§0.3.3).

### 0.3.3: Per-op overlap

The forward path of one decoder layer dispatches **two** offloaded ops sequentially: WQKV and the fused MLP block (MLP1 → SwiGLU → MLP2 — see `phase1a_findings.md §1.3` for why MLP is fused at block granularity, not per-Linear). For zero overhead at a given `f_cpu`:

- `CPU_WQKV ≤ GPU_WQKV` — WQKV's CPU slice fits within its GPU counterpart.
- `(CPU_MLP1 + CPU_MLP2) ≤ (GPU_MLP1 + GPU_MLP2)` — the MLP block's CPU pipeline fits within its GPU counterpart. SwiGLU runs locally on each device and is small.

WO is **not** offloaded (per `weight_offload_design.md §WO Split Axis Decision` and the §0.4.2 Alt B commitment), so it doesn't enter the overhead check; its GPU time is just baseline. All numbers below are cold-cache (28-layer ring on CPU, working-set ring on GPU; see §0.3.1) at `torch.set_num_threads(16)`, matching the COTS runtime default (`phase1a_findings.md §1.13b`).

#### Per-op GPU budgets (cold-cache)

| B | WQKV op | MLP block (MLP1+MLP2) |
|---:|---:|---:|
| 1 | 38 µs | 287 + 148 = 435 µs |
| 4 | 39 µs | 290 + 156 = 446 µs |
| 8 | 39 µs | 292 + 153 = 445 µs |

**B=1 (single token decode):**

| f_cpu | WQKV CPU vs 38 µs | MLP block CPU vs 435 µs | Layer overhead | GPU freed (×28 layers) |
|---:|---:|---:|---|---:|
| 3% | 16 µs ✓ | 156 + 96 = 252 µs ✓ | **FREE** (0 µs over) | 0.39 GB |
| 5% | 27 µs ✓ | 271 + 153 = 424 µs ✓ | **FREE** (0 µs over, 22 µs total headroom) | 0.65 GB |
| 9% | 54 µs ✗ (+16) | 486 + 246 = 732 µs ✗ (+297) | +313 µs/layer | 1.17 GB |
| 15% | 92 µs ✗ (+54) | 778 + 402 = 1180 µs ✗ (+745) | +799 µs/layer | 1.96 GB |

**B=4 (minimum TTC batch):**

| f_cpu | WQKV CPU vs 39 µs | MLP block CPU vs 446 µs | Layer overhead | GPU freed (×28) |
|---:|---:|---:|---|---:|
| 3% | 28 µs ✓ | 218 + 122 = 340 µs ✓ | **FREE** (0 µs over, 117 µs headroom) | 0.39 GB |
| 5% | 46 µs ✗ (+7) | 367 + 190 = 557 µs ✗ (+111) | +118 µs/layer | 0.65 GB |
| 9% | 78 µs ✗ (+39) | 646 + 328 = 974 µs ✗ (+528) | +567 µs/layer | 1.17 GB |
| 15% | 132 µs ✗ (+93) | 1072 + 544 = 1616 µs ✗ (+1170) | +1263 µs/layer | 1.96 GB |

**B=8:**

| f_cpu | WQKV CPU vs 39 µs | MLP block CPU vs 445 µs | Layer overhead |
|---:|---:|---:|---|
| 3% | 48 µs ✗ (+9) | 343 + 218 = 561 µs ✗ (+116) | +125 µs/layer |
| 5% | 76 µs ✗ (+37) | 568 + 327 = 895 µs ✗ (+450) | +487 µs/layer |
| 9% | 128 µs ✗ (+89) | 1013 + 537 = 1550 µs ✗ (+1105) | +1194 µs/layer |

Key observations:
- **WQKV has a tight budget** (~38–39 µs, almost flat in B because GPU is memory-bandwidth-bound). Overflows at B=4 even at f=5%.
- **MLP block has a generous budget** (~435–446 µs). Fits f=5% at B=1 and f=3% at B=4 with margin.
- **MLP1 dominates the offloaded layer cost.** At f=15% B=1, MLP1 accounts for ~66% of the offloaded sum.

**Visual evidence — `probe_overlap.py`.** A 1-second nsys probe at WQKV, f_cpu=5%, B=1 timeline-confirms CPU+GPU concurrency. Phase A (`A_gpu_only`) and B (`B_cpu_only`) ground-truth the per-side durations; Phase C (`C_concurrent`) shows the GPU bar and CPU `F.linear` bar overlapping in time. Run `nsys profile -o probe_overlap.nsys-rep --trace=cuda,nvtx,osrt --force-overwrite=true python probe_overlap.py` and inspect in Nsight Systems GUI.

#### Throughput tradeoff framing

In the microbench with cold-cache numbers, the free regime is:
- **B=1: f_cpu ≤ 5%** — both WQKV and MLP block fit (22 µs total headroom across the layer at f=5%).
- **B=4: f_cpu ≤ 3%** — both fit with 117 µs headroom; f=5% spills WQKV by 7 µs and MLP by 111 µs.
- **B≥8: no free regime** even at f=3% — WQKV overflows by 9 µs and MLP by 116 µs. The regime becomes a **throughput tradeoff**: per-step latency increase in exchange for more KV cache → more beams per scheduling round → fewer rounds → potentially faster wall-clock TTC.

**Important caveat — microbench-free ≠ e2e-free.** §1.14 of `phase1a_findings.md` measures the e2e gap at f=0.05 B=1 t=16 (decode-heavy) directly: COTS is +2.49 s slower than baseline despite the microbench predicting free. The decomposition (`--cots-dry-run` mode) attributes ~18% to pure host orchestration and ~82% to the *active CPU-work penalty* — extra wall clock from enabling real CPU GEMM, dominated by oneDNN-on-many-threads contending with the main thread's CUDA dispatch path. Neither is visible in this microbench because it measures CPU and GPU in isolation, with no main-thread CUDA dispatch happening on the same cores. The true free regime under Phase 1a runtime is below f=0.05; under Phase 1c (native runner + bucket-aware thread policy, `implementation_roadmap.md`) the floor recovers to roughly the microbench prediction.

The specific numbers (`5%` at B=1, `3%` at B=4) are properties of this (HW, model, dtype, thread-count) combination, not universal constants. The Planner (see `planner_design.md`) derives bucket-specific values from the profile, so the number used in practice varies per `BatchDescriptor`.

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

Measured by `bench_cuda_graph.sh` — `vllm bench latency` with and without `--enforce-eager`, comparing CUDA-graph-captured execution to eager per-kernel launches. Purpose: scope the CUDA-graph-compatibility work that Phase 1c (the `cudaLaunchHostFunc` retrofit that makes the CPU-dispatch and prefetch paths CUDA-graph-compatible) ships.

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

### Implications for Phase 1c scope

1. **Prototyping Phase 1a/1b with `enforce_eager=True` costs at most ~2% throughput** for the main thesis workload. This is well within the noise of FastTTS beam-search measurements. The `CpuComputeDispatcher` abstraction (Phase 1 day-1 choice; `implementation_roadmap.md §1.3`) is viable against eager-mode forward passes through Phase 1b.

2. **Graph capture alone wouldn't justify Phase 1c — but the native runner does.** §0.7 in isolation shows ~2% throughput from graphs, which would normally argue for deferral. But `phase1a_findings.md §1.14` showed the Python `CpuTaskRunner` substrate is the dominant Phase 1a overhead and the native-runner port is the precondition for any graph capture of the COTS path; graph capture falls out for free once the native dispatcher lands. So Phase 1c is gated by the substrate fix, not by §0.7's 2%, and happens before Phase 2.

3. **V1's per-process memory overhead (0.9–1.4 GiB from CUDA graphs + torch.compile, see `vllm_v1_migration.md §12`) is the real cost of enabling graphs.** For our two-process generator+verifier setup, disabling graphs via `enforce_eager=True` recovers ~2 GiB of GPU memory — which then becomes additional KV budget. Phase 1a/1b keep `enforce_eager=True` to harvest that KV; Phase 1c re-evaluates the trade once the dispatcher is graph-compatible.

### Recommendation

Phase 1a/1b prototypes run with `enforce_eager=True` by default. Phase 1c (native runner + graph capture) lands before Phase 2 per `implementation_roadmap.md` and `phase1a_findings.md §1.14`.

### Framing the graph-capture requirement

Graph capture's value is **upstream compatibility, not standalone latency**. The §0.7 measurement shows graph capture gives only ~2% throughput on our thesis workload — too small to be the deciding factor. Phase 1c bundles graph compatibility with the substrate fix because:

1. **vLLM integration.** Production vLLM defaults to `FULL_AND_PIECEWISE` graph capture. A Python-threaded `CpuComputeDispatcher` falls out of graph capture — any downstream user who enables graphs would silently lose the CPU-compute path. For the work to land upstream, the CPU-dispatch and prefetch paths must be `cudaLaunchHostFunc`-based, which the native runner already requires for performance reasons.

2. **Graph benefit grows with system scale.** On 8-GPU TP deployments each layer has many more kernels (all-reduces, MoE routing, etc.), so launch-overhead fraction grows. The 2% we measured on a single 4090 becomes 10%+ on production setups.

3. **Compiler pipelines assume graph capture.** `torch.compile` + piecewise graph capture is becoming the vLLM default; opting out increasingly looks like pre-modern infrastructure.

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

1. **KV cache saturates at n=64 on aime (longer completions, ~800 tok) and n=256 on math500 (shorter, ~520 tok).** This is the goalpost Phase 1 targets: freeing ~1.2 GB of GPU memory via weight offload (predicted from §0.1+0.3 overlap analysis) should raise the saturating n by ~25%, validated in Phase 3 (e2e benchmarking).

2. **Baseline goodput declines sharply from n=1 to n=256** (62 → 7 tok/s on aime, 62 → 9 tok/s on math500). Most of the decline before saturation (n=1 → n=64) is inherent to beam parallelism producing correlated outputs; the decline past saturation (n=64 → n=256) is additional throughput loss from KV-pressure queueing (`gen_frac_steps_queued` jumps from 0.1% to 8.1% on aime).

3. **Prefix caching is highly effective** across all n: `gen_gpu_hit` 95–98% means prefill costs are amortized across beams. This is the "shared prefix" property Phase 1's WQKV K/V-biased split is designed to preserve — our column-parallel weight split must not disrupt prefix caching, which is the FastTTS win.

4. **These are V1 numbers on vLLM 0.19.0**, not V0 on 0.9.2 (the baseline in the original FastTTS paper). V1 has slightly higher per-process memory overhead (~0.9–1.4 GiB from CUDA graphs + torch.compile, see `vllm_v1_migration.md §12`), which slightly raises the GPU memory split for generator vs. verifier (0.74 / 0.16 recommended) and is the reason we can't directly compare to the published FastTTS numbers. Phase 3 e2e benchmarking re-evaluates against these V1 numbers.

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

### Recommendation for Phase 3 e2e evaluation

**Disable V1 KV offload in both the thesis configuration and the baseline comparison.** Four reasons, in order of importance:

1. **The upside is small to begin with.** GPU prefix cache already captures 95–98% of hits across the whole n sweep (finding 1). V1 offload can only improve the remaining 2–5% miss rate, which math500 confirms is a ≤1% wall-clock effect.

2. **PCIe contention with weight prefetch.** The H2D reload traffic would mix with Phase 3's weight prefetch H2D, breaking the design invariant that prefetch owns PCIe. Even small additional H2D traffic perturbs the measurements and adds scheduler-side variability.

3. **Clean attribution.** We want to isolate Phase 1+2+3's effect. Enabling V1 offload means "with-thesis" numbers include both thesis mechanisms AND whatever V1 offload happens to do at that operating point.

4. **Mechanism orthogonality.** V1 offload is prefill-side prefix-reuse; Phase 2 is decode-time suffix attention. Complementary, not competing. Mixing them in the headline comparison obscures what each contributes.

Both disabled is the apples-to-apples comparison. Re-enabling V1 offload for the full system is a separate ablation if it becomes interesting later.

## 0.10: vLLM native weight offloader baseline (decode-step latency)

§0.8/0.9 covered the V1 *KV* offload. This section measures the shipped vLLM *weight* offloaders, and also records the thesis-only native-prefetch variant used as an optimized reference point:

- **Stock `PrefetchOffloader`** (`--offload-backend prefetch`, `vllm/model_executor/offloader/prefetch.py`) — group-granularity, multi-layer-ahead prefetch with a static GPU buffer pool and event-driven CUDA-graph compatibility. Tunable knobs: `offload_group_size` (G), `offload_num_in_group` (N), `offload_prefetch_step` (K). This file and `prefetch_ops.py` are kept factory-clean for shipped-vLLM comparisons.
- **Thesis `PrefetchDeferOffloader`** (`--offload-backend prefetch_defer`, `vllm/model_executor/offloader/prefetch_defer.py`) — a temporary optimized-native baseline that adds the §0.10.3 deferred-wraparound fix, with its custom op isolated in `prefetch_defer_ops.py`. It exists so the optimized baseline can be removed later without unwinding changes from the stock class.
- **`UVAOffloader`** (`vllm/model_executor/offloader/uva.py`) — zero-copy mapping of pinned CPU tensors via `get_accelerator_view_from_cpu_tensor`. Tunable: `cpu_offload_gb` (X).

Each cell shells out to vLLM's native CLI `vllm bench latency` so every measurement runs in a fresh engine. All cells use `--enforce-eager` for clean attribution. We do **not** measure the `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` escape hatch — it's an undocumented compatibility shim, not a deployment vLLM recommends. Setup: Qwen2.5-7B BF16, `(input_len, output_len) = (256, 32)`, batch ∈ {1, 16, 64}, 2 warmup + 3 bench iters.

Decoder-layer arithmetic (used to express prefetch coverage in GiB so it shares UVA's offload-depth axis):

| Sub-module | Shape | Bytes (BF16) |
|---|---|---:|
| WQKV     | 3584 × 4608 | 31.5 MiB |
| WO       | 3584 × 3584 | 24.5 MiB |
| MLP1 (gate+up merged) | 3584 × 37888 | 259 MiB |
| MLP2 (down) | 18944 × 3584 | 129.5 MiB |
| **per layer total** | | **444.5 MiB** |
| 28 layers | | **12.15 GiB** |

**Section structure.** The narrative is built around answering "how good is stock vLLM's prefetch when properly tuned?" first, then layering optimizations on top:

- §0.10.1 — **Stock Native Prefetch Mechanics** (exposition). Picker math, wraparound prefetch, and the implications of the wraparound H2D landing on CE0 / copy_stream.
- §0.10.2 — **Stock Native Parameter Search**. A 60-cell sweep across `(G, N)` at K=1 finds the best stock config per offloaded-layer count. Naturally exposes when placement (avoiding layer 27) wins versus uniform spacing.
- §0.10.3 — **Canonical Stock Sweep** (historical). The four sub-sweeps at G ∈ {1, 2, 4, 7, 14, 28} (divisors of 28). Kept for narrative continuity; not "best stock", just "default-canonical stock".
- §0.10.4 — **UVA Comparison**. Stock prefetch vs UVA at matched offloaded GiB.
- §0.10.5 — **PrefetchDefer Ablation**. The thesis-only `prefetch_defer` backend (deferred-wraparound fix) measured as an optimization on top of stock-best. Includes the eager/graph-mode mechanism analysis and the two-bug isolation experiment.
- §0.10.6 — **nsys overlap probe**. Trace-level visualization of compute/H2D overlap.

Historical arm names like `prefetch_28x1` are retained; rows described as "post-fix" or "fixed" correspond to the thesis `prefetch_defer` backend, while "unfixed" corresponds to stock `prefetch` behavior.

### 0.10.1: Stock Native Prefetch Mechanics

This section explains how the shipped `PrefetchOffloader` behaves before any benchmarks. The mechanics establish *why* the wraparound prefetch is the most fragile part of the design and *what* the parameter search in §0.10.2 is actually exploring.

**Layer picker.** For each decoder-layer index `i ∈ [0, num_layers)`, the layer is offloaded if and only if:

```
i % group_size >= group_size − num_in_group
```

with the user-tunable knobs `group_size` (G), `num_in_group` (N), `offload_prefetch_step` (K). Equivalently: within every G-sized window of consecutive layer indices, the *last* N layers go to CPU. For Qwen2.5-7B (28 layers) this means:

- G=28, N=1 → offloaded set is `{27}` (just the final decoder layer).
- G=14, N=1 → `{13, 27}`.
- G=4, N=1 → `{3, 7, 11, 15, 19, 23, 27}`.
- G=15, N=1 → `{14}` (only the first window has its last position; layers 15–27 fall in a partial window where `i % 15 ∈ [0, 12]` never reaches `G − N = 14`).

When G doesn't divide num_layers, the picker silently drops candidates that would otherwise sit at positions `G − N, G − N + 1, …, G − 1` of an incomplete window. This is the placement freedom §0.10.2 exploits: a non-divisor G with N=1 places its single offloaded layer at the *first* satisfying position (e.g. layer 14 for G=15), not at layer 27.

**Prefetch ordering.** When offloaded module `i` finishes its forward, the offloader fires `start_prefetch((i + K) % num_offloaded)` on a dedicated `copy_stream`. With K=1, each offloaded layer's forward triggers the H2D for the *next* offloaded layer, copying a 466 MB per-layer weight bundle (BF16, all 4 sub-modules) onto a static GPU buffer that the prefetched layer's forward will consume one (or K) layer-stride steps later.

The *wraparound prefetch* is the special case where the offloaded layer with the highest index calls `start_prefetch` for the offloaded layer with the lowest index — wrapping around the circular indexing. For canonical (G, N=1) configs that include layer 27 (the final decoder layer), this wraparound fires at the END of the model's forward, with no remaining iter compute to overlap the H2D against. It then races against the next iter's startup machinery.

**Two sync exposure surfaces from one wraparound H2D.**

The wraparound H2D meets two different sync surfaces depending on execution mode. Only the first is true CE0 FIFO blocking; the second is a separate copy_stream drain caused by vLLM's graph machinery. Both are characterized empirically in §0.10.5.

1. **Eager mode (`--enforce-eager`) — CE0 FIFO contention with input prep.** The wraparound H2D queues on **CE0** (the GPU's H2D copy engine) at end-of-iter-N. Iter N+1's per-step input-prep H2Ds (vLLM's V1 async-scheduling path: `tensor.to('cuda', non_blocking=True)` for `input_ids` / `positions` / block-table indices / attention metadata) ALSO go to CE0, and CE0 is FIFO at the engine level — input prep queues *behind* the wraparound and waits ~19.5 ms (1 wraparound at 24 GB/s PCIe). The per-step `prepare_inputs_event.synchronize()` (`vllm/v1/worker/gpu_model_runner.py:3485`) blocks for this wait, propagating into wall-clock latency.

2. **Graph mode (no `--enforce-eager`) — `sync_prev_onload`'s copy_stream drain.** vLLM captures and replays per-batch-shape graphs. Because the prefetch copy_stream is external to any captured graph, vLLM drains it before each capture/replay via `PrefetchOffloader.sync_prev_onload()` — a `wait_stream(self.copy_stream)` barrier on the compute stream (call sites: `cudagraph_utils.py:259`, `gpu_ubatch_wrapper.py:259/493`). This is NOT CE0 contention; it's a deliberate cross-stream barrier that waits for copy_stream's pending work to finish. Per-event sync time scales as `layers_offloaded × 19.5 ms` (G=1: 561 ms ≈ 28 × 19.5 ms; G=4: 160 ms ≈ 7 × 23 ms; within 3% across the sweep).

**The two surfaces are mutually occluding.** Same wraparound bytes, two different watching points — but only one fires per run. In graph mode, `sync_prev_onload` runs *before* the next replay's input prep gets a chance to queue on CE0, draining the wraparound from copy_stream first. By the time input prep H2Ds queue, the bytes are gone — the eager-mode CE0 contention has nothing to wait on. So a single run pays exactly one wraparound wait, exposed through whichever surface its mode happens to use first. §0.10.5 confirms with a 2x2 isolation experiment (eager × UVA-input-prep flags).

**The placement implication.** Because the wraparound exposure is tied to the *last offloaded layer being layer 27*, a config that places its highest offloaded layer earlier (e.g., layer 14 via G=15 N=1) leaves tail compute (layers 15..27) to overlap with the wraparound H2D — at least partially. This is the placement story §0.10.2 explores empirically.

### 0.10.2: Stock Native Parameter Search

The canonical sweep in §0.10.3 fixes G to a divisor of num_layers (G ∈ {1, 2, 4, 7, 14, 28}). Every one of those configs offloads layer 27 — the worst position for the wraparound. The new question: does varying G away from divisors find better stock configs?

**Methodology.** Smart-subsample sweep across `(G ∈ 1..28, N ∈ 1..G, K=1)`: for each target offloaded-layer count L ∈ {1, 2, 3, 4, 5, 7, 10, 14, 21, 28}, enumerate all `(G, N)` producing exactly L offloaded layers, cap at ~6 per L (stratified on G to cover small/mid/large group sizes), always include the canonical divisor-G config when valid. Total: 60 cells. Workload: Qwen2.5-7B BF16, `input=256 output=32, --enforce-eager`, factory `prefetch` backend (no defer). Top-3 per L re-run at B ∈ {1, 16, 64}; K-sweep at the chosen Pareto-knee config (L=4 G=6 N=1) at B ∈ {1, 16, 64} × K ∈ {1, 2, 3, 4}. All cells run on stock vLLM with no thesis instrumentation patches. Bench script: `bench_prefetch_full_sweep.py`. Results: `results/0.10_full_sweep/`.

**Best stock config per offloaded-layer count.** B=1 latencies (matches §0.10.3 workload). `final?` indicates whether the config's offloaded set includes layer 27.

| L | best (G, N) | final? | last offloaded layer | avg lat (s) | canonical (G, N=1) | canonical lat (s) | Δ best vs canonical |
|---:|:---:|:---:|---:|---:|:---:|---:|---:|
|  1 | **G=15, N=1** | no  | 14 | **0.884** | G=28, N=1 | 1.105 | **−20.0%** |
|  2 | **G=10, N=1** | no  | 19 | **1.438** | G=14, N=1 | 1.507 | **−4.6%** |
|  3 | **G=8, N=1**  | no  | 23 | 2.044 | (no divisor) | — | — |
|  4 | **G=6, N=1**  | no  | 23 | 2.652 | (no divisor) | — | — |
|  5 | **G=5, N=1**  | no  | 24 | 3.276 | (no divisor) | — | — |
|  7 | G=4, N=1      | YES | 27 | 4.537 | G=4, N=1 (same) | 4.537 | tied |
| 10 | G=5, N=2      | no  | 24 | 6.462 | (no divisor) | — | — |
| 14 | G=2, N=1      | YES | 27 | 8.990 | G=2, N=1 (same) | 8.990 | tied |
| 21 | G=4, N=3      | YES | 27 | 13.475 | (no divisor) | — | — |
| 28 | G=12, N=12    | YES | 27 | 17.934 | G=1, N=1 | 17.941 | flat |

**Three regimes.**

1. **Low L (1–5): placement wins by 5–20%.** Non-divisor G with N=1 places the single offloaded layer earlier in the model, leaving 1–13 tail compute layers to hide the wraparound H2D. Headline: at L=1, `G=15, N=1` (offloads only layer 14, leaving 13 tail layers) beats canonical `G=28, N=1` (offloads only layer 27, 0 tail) by 20% with identical 0.43 GiB.

2. **Mid L (7, 14): uniform spacing wins.** With 7+ offloaded layers, the per-layer drain interval shortens enough that *uniform* spacing matters more than any single-layer placement choice. `G=4, N=1` (every 4th layer: {3, 7, 11, 15, 19, 23, 27}) and `G=2, N=1` (every other: {1, 3, …, 27}) win because they maximize per-layer hide budget. Placement-aware alternatives at these L *cluster* the offloaded layers (e.g. `G=18, N=7` = layers {11..17}), losing the uniform-spread advantage despite avoiding layer 27.

3. **High L (21, 28): negligible differences.** At 21+ offloaded layers, every spread is dense enough that placement is irrelevant; intra-forward CE0 saturation between consecutive offloaded layers dominates. All configs converge within <0.5%.

**Top-3 batch validation.** The B=1 winner at every L remains the B=16 and B=64 winner. Selected rows (full table in `results/0.10_full_sweep/summary.json`):

| L | best arm | B=1 (s) | B=16 (s) | B=64 (s) |
|---:|---|---:|---:|---:|
|  1 | L1_G15_N1_K1 | 0.884 | 1.229 | 2.320 |
|  4 | L4_G6_N1_K1  | 2.652 | 2.946 | 4.024 |
|  7 | L7_G4_N1_K1  | 4.537 | 4.849 | 5.952 |
| 14 | L14_G2_N1_K1 | 8.990 | 9.450 | 10.443 |

Placement effect is robust across batches.

**K-sweep at the Pareto-knee config (L=4 G=6 N=1).** The non-divisor `G=6, N=1` (offloads {5, 11, 17, 23}) is the placement-aware best at ~25% offload depth. K ∈ {1, 2, 3, 4} at B ∈ {1, 16, 64}:

| K | B=1 (s) | B=16 (s) | B=64 (s) |
|---:|---:|---:|---:|
| 1 | 2.652 | 2.946 | 4.024 |
| 2 | 2.600 | 2.889 | 3.967 |
| 3 | 2.600 | 2.894 | 3.962 |
| 4 | 2.600 | 2.905 | **OOM** |

Same shape as the canonical `G=4, N=1` K-sweep in §0.10.3d: K=2 captures essentially all the benefit (~2% over K=1), K≥3 flat, K=4 OOMs at B=64. K is orthogonal to placement — K=1 is near-optimal regardless of (G, N).

**Conclusion for stock vLLM tuning.** When picking a stock prefetch config:

- **Low offload depth** (≤ ~17% of weights, ≤ 5 offloaded layers on Qwen2.5-7B): prefer non-divisor G with N=1. Placement matters more than spacing.
- **Mid depth** (~25–50%): prefer divisor G with small N. Uniform spacing dominates.
- **High depth** (≥75%): the choice barely matters; pick the smallest G that fits the buffer pool budget.
- **K=1 is near-optimal everywhere**; K=2 buys ~2% with mild buffer-pool risk.

The §0.10.5 defer fix is an *additional* optimization on top of these stock-best choices. As shown there, defer-at-canonical-G is structurally stronger than placement-at-tuned-G at low L (because defer eliminates the wraparound's CE0 occupancy entirely, while placement only shortens the wait), but the two converge at mid L where stock-best IS the canonical config.

### 0.10.3: Native prefetch knob sweep (canonical, divisor-G)

Measured by `bench_prefetch_knobs.py`. Three sub-sweeps on Qwen2.5-7B BF16 across batch ∈ {1, 16, 64}. The post-fix numbers use `--offload-backend prefetch_defer`; the stock `--offload-backend prefetch` baseline remains available as the unfixed/shipped-vLLM comparison.

#### (a) G at fixed 50% coverage (all 14 layers offloaded, 6.08 GiB)

By picking G ∈ {2, 4, 14, 28} (divisors of 28) and N = G/2, every arm offloads **exactly** 14 layers — true byte-equivalent comparison. The only thing that varies is the *spatial pattern* of offloaded layers.

| Arm | G | N | Pattern | B=1 (s) | B=16 (s) | B=64 (s) |
|---|---:|---:|---|---:|---:|---:|
| `prefetch_2x1`   | 2  | 1  | every 2nd layer (uniform, dense)  | **9.016** | **9.471** | **10.421** |
| `prefetch_4x2`   | 4  | 2  | pairs every 4 layers              | 9.012 | 9.477 | 10.671 |
| `prefetch_14x7`  | 14 | 7  | clusters of 7 in 2 groups         | 9.014 | 9.561 | 10.889 |
| `prefetch_28x14` | 28 | 14 | single cluster of 14 at end       | 9.011 | 9.607 | 10.942 |

**Monotonic — smaller G (denser, more uniform spacing) wins.** G=2 beats G=28 by ~5% at B=64, with monotone progression in between. The mechanism: smaller G means the offloaded layers are interleaved more uniformly with GPU-resident layers, giving each H2D event a fresh "drain" interval. Larger G concentrates offloads into clusters where consecutive offloaded layers must serialize through a single buffer slot (with K=1) — the second-and-later cluster members get only ~1 ms of compute hiding for ~20 ms H2D. The four arms benefit roughly equally from the §0.10.3 `prefetch_defer` wraparound fix (each has exactly one wrap-around per iter at this 50% coverage), so the relative ordering is preserved post-fix.

The earlier `bench_native_weight_offload.py` finding that "G=16 N=4 wins at 25% coverage" was an artifact of unequal byte counts (G=16 N=4 offloads 8 layers / 3.47 GiB; G=4 N=1 offloads 7 layers / 3.04 GiB — 14% more bytes for the supposed winner). With truly fixed bytes the picture inverts: **at fixed coverage, denser uniform spacing dominates clustering**.

#### (b) N=1, G varies — canonical coverage sweep (densest possible spread at each depth)

(a) showed that at fixed 50% coverage, denser uniform spacing wins. (b) generalizes that finding to all depths by holding N=1 (single offloaded layer per group → max spread) and varying G. The arm at each depth has the **densest possible** spatial pattern at that byte count — these are the empirically best prefetch baselines at every offload depth and what every downstream cross-section comparison should use.

| Arm | G | N | Offloaded layers | GiB | B=1 (s) | B=16 (s) | B=64 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prefetch_28x1` | 28 | 1 | 1 | 0.43 | 0.688 | 1.033 | 2.103 |
| `prefetch_14x1` | 14 | 1 | 2 | 0.87 | 1.325 | 1.653 | 2.720 |
| `prefetch_7x1`  | 7  | 1 | 4 | 1.74 | 2.607 | 2.897 | 3.970 |
| `prefetch_4x1`  | 4  | 1 | 7 | 3.04 | 4.527 | 4.850 | 5.892 |
| `prefetch_2x1`  | 2  | 1 | 14 | 6.08 | 9.016 | 9.471 | 10.421 |

The low-coverage cells (G=28, G=14) drop substantially relative to their pre-fix counterparts (`prefetch_28x1` B=1: 1.10 → 0.69 s, −38%; full sweep in §0.10.3 "post-fix sync table" and "end-to-end latency confirmation"). The high-coverage cells (G≤7) are essentially unchanged because intra-forward CE0 saturation between consecutive offloaded layers dominates and the wrap-around defer can't help. `prefetch_28x1` at B=64 is now within ~3.5% of `none` (2.10 s vs 2.03 s baseline) — the prefetch free regime that §0.10.3 set out to find.

Comparison against the legacy clustered baseline (c) (`prefetch_14xN`) at matched depths:

| Depth | (c) clustered | (b) max spread | B=64 (c/b) | gap |
|---|---|---|---|---:|
| 0.87 GiB (2 layers) | `prefetch_14x1` | `prefetch_14x1` (same arm) | 2.720 / 2.720 | 0% |
| 1.74 GiB (4 layers) | `prefetch_14x2` | `prefetch_7x1` | 4.082 / 3.970 | **2.8%** |
| 6.08 GiB (14 layers) | `prefetch_14x7` | `prefetch_2x1` | 10.889 / 10.421 | **4.5%** |

(The 3.04 GiB `prefetch_4x1` and 3.47 GiB `prefetch_14x4` are also close points — 5.892 vs 6.787 = 15.2% — but the byte counts differ by 12% so the apparent gap is dominated by bytes, not pattern.)

**(a)'s finding generalizes: at every byte-equal depth tested, the densest-spread (N=1) arm beats the clustered (G=14) arm by 0–5% at B=64.** The gap is small but consistent post-fix; the slight shrinkage from the pre-fix ~5% at all depths reflects that the fix improves the densest (N=1) arms slightly more — they have the most unhidden wrap-around to recover.

**Why the win shrinks at B=1.** At B=1 decode is bandwidth-bound and per-layer compute time is roughly fixed regardless of how many layers are clustered together — clustering vs spreading affects how much "drain time" each prefetched layer gets, but at B=1 the drain time is short either way. At larger batches, GPU compute per layer grows; clusters of consecutive offloaded layers can't fully hide their H2D behind that compute (the second-and-later cluster member gets only ~1 ms of compute hiding for ~20 ms H2D), while uniformly-spread offloads always have a fresh non-offloaded layer's compute window to hide behind. Hence (b)'s lead expands at higher batches.

**Implication for cross-section comparison:** the right prefetch baseline at any offload depth `d` is the (G, N=1) arm whose `28/G` matches `d / per-layer-bytes`. The §0.10.2 head-to-head uses these N=1 arms directly, anchored at matched offloaded GiB against UVA.

#### (c) N at fixed G=14, K=1 — legacy clustered baseline

Pre-restructure, this was the canonical coverage sweep. Now retained only as the matched-bytes anchor used in (b)'s "uniform vs clustered" comparison above.

| Arm | N | Coverage | Offloaded GiB | B=1 (s) | B=16 (s) | B=64 (s) |
|---|---:|---:|---:|---:|---:|---:|
| `prefetch_14x1`  | 1  | 7.1%  | 0.87 | 1.325 | 1.653 | 2.720 |
| `prefetch_14x2`  | 2  | 14.3% | 1.74 | 2.606 | 2.966 | 4.082 |
| `prefetch_14x4`  | 4  | 28.6% | 3.47 | 5.169 | 5.594 | 6.787 |
| `prefetch_14x7`  | 7  | 50.0% | 6.08 | 9.014 | 9.561 | 10.889 |
| `prefetch_14x10` | 10 | 71.4% | 8.68 | 12.859 | 13.550 | 14.975 |

**Linear in offloaded bytes** at every batch. Slope at B=64: 0.87 → 8.68 GiB ⇒ 2.72 → 14.98 s = **1.57 s/GiB**, essentially unchanged from the pre-fix 1.55 s/GiB (the fix subtracts a roughly constant ~0.27s of low-coverage savings from every arm in this set, leaving the slope intact). At matched bytes the canonical (b) `prefetch_2x1` is ~4–5% faster — see (b)'s comparison table above.

#### (d) K at fixed G=4, N=1 — uniform-spread K sweep

K sweep at the empirically best knob configuration (uniform N=1, ~25% coverage, 7 offloaded layers). K>1's potential upside is structurally smaller under uniform spread than under a clustered config, because each prefetch already gets G−1 = 3 actual layers of compute hiding from K=1 alone. This is therefore a K=1-conservative measurement: if K>1 doesn't help here, it's not going to help at the canonical N=1 configurations downstream consumers actually use.

| Arm | K | B=1 (s) | B=16 (s) | B=64 (s) |
|---|---:|---:|---:|---:|
| `prefetch_4x1`     | 1 | 4.527 | 4.850 | 5.892 |
| `prefetch_4x1_k2`  | 2 | 4.422 | 4.700 | 5.735 |
| `prefetch_4x1_k3`  | 3 | 4.419 | 4.674 | 5.743 |
| `prefetch_4x1_k4`  | 4 | 4.423 | 4.699 | **OOM** |

**K=2 captures essentially all of the available benefit; K≥3 is flat; K=4 OOMs at B=64.** At B=64 K=2 buys **2.7%** over K=1 (5.89 → 5.74 s). K=3 is within run-to-run noise of K=2. K=4 succeeds at B=1 / B=16 with no measurable gain (4.42 / 4.70 s) but fails at B=64 — the K=4 buffer pool (4 × per-layer-buffer slots, ~1.7 GiB extra GPU residency) combined with B=64 activations overruns the 24 GiB budget. The `prefetch_defer` wraparound fix supports K>1 under the `last`-only policy: only the very last wrap-around (zero compute window) is deferred; earlier wrap-around layers at K>1 fire immediately so their H2D overlaps with iter N's tail compute (see §0.10.3 "Fix delivered"). This measurement is post-fix and apples-to-apples with the K=1 cells in (b)/(c).

**The K>1 win is structurally smaller under uniform spread than under clustering.** Each prefetch already gets G−1 = 3 actual layers of compute hiding from K=1 alone, leaving little for K>1 to add. Concretely, the legacy K-sweep at (G=14, N=4) — clustered, where the 2nd–4th members of each cluster benefit most from K>1 — measured a K=2 win of 5.6% at B=64 (data not shown). Under (G=4, N=1) that win shrinks to 2.9%: roughly half the legacy gap, exactly as expected from the spacing argument.

**This empirically validates the COTS commitment to layer-ahead** in `pcie_bandwidth_allocation_design.md` (§Prefetch Distance). The marginal benefit of K>1 is small (≤3% at B=64 under the empirically best spacing) and the OOM risk at large batch + large K is real. Layer-ahead (K=1, the COTS partition) gives essentially all the achievable hiding without the buffer pressure.

### 0.10.4: UVA vs Prefetch — head-to-head

Measured by `bench_uva_vs_prefetch.py`. Prefetch arm: N=1, K=1, varying G ∈ {1, 2, 4, 7, 14, 28} (the empirically best knob configuration per §0.10.1 — uniform spacing dominates clustering). UVA arm: varying `cpu_offload_gb` ∈ {1, 2, 4, 6, 8, 10, 12}. Both curves on the same offloaded-GiB x-axis.

| Arm | Offloaded GiB | B=1 (s) | B=16 (s) | B=64 (s) | tok/s @ B=64 |
|---|---:|---:|---:|---:|---:|
| `none`           | 0.00  | 0.521  | 0.887  |  2.028 | 1010 |
| `prefetch_28x1`  | 0.43  | 0.687  | 1.032  |  2.098 |  976 |
| `prefetch_14x1`  | 0.87  | 1.324  | 1.660  |  2.724 |  752 |
| `prefetch_7x1`   | 1.74  | 2.606  | 2.903  |  3.968 |  516 |
| `prefetch_4x1`   | 3.04  | 4.526  | 4.827  |  5.900 |  347 |
| `prefetch_2x1`   | 6.08  | 9.011  | 9.464  | 10.418 |  197 |
| `prefetch_1x1`   | 12.15 | 17.983 | 18.884 | 20.536 |  100 |
| `uva_1`          | 1.00  | 2.274  | 5.353  | 16.101 |  127 |
| `uva_2`          | 2.00  | 3.565  | 8.365  | 26.531 |   77 |
| `uva_4`          | 4.00  | 6.787  | 16.966 | 52.026 |   39 |
| `uva_6`          | 6.00  | 9.550  | 24.065 | 74.615 |   27 |
| `uva_8`          | 8.00  | 12.592 | 31.600 | 98.037 |   21 |
| `uva_10`         | 10.00 | 15.415 | 38.808 | 121.448 |  17 |
| `uva_12`         | 12.00 | 18.412 | 46.099 | 145.518 |  14 |

#### Findings

1. **Prefetch dominates UVA at B ≥ 16 across the entire offload-depth range.** At B=64, comparing matched-GiB pairs: `prefetch_7x1` (1.74 GiB) at 3.97 s vs `uva_2` (2.00 GiB) at 26.53 s → Prefetch is **6.7×** faster. `prefetch_4x1` (3.04 GiB) at 5.90 s vs `uva_4` (4.00 GiB) at 52.03 s ≈ **8.8×**. `prefetch_2x1` (6.08 GiB) at 10.42 s vs `uva_6` (6.00 GiB) at 74.62 s ≈ **7.2×**. At full coverage `prefetch_1x1` (12.15 GiB) at 20.54 s vs `uva_12` (12.00 GiB) at 145.52 s ≈ **7.1×**. Prefetch numbers here are the optimized-native `prefetch_defer` post-fix numbers; UVA numbers are unaffected by the fix.
2. **At B=1 the two curves nearly meet at low depth, with prefetch always ahead.** UVA is competitive only at the lowest depth and B=1: `uva_1` (2.27 s) vs `prefetch_14x1` (1.32 s, 0.87 GiB) — Prefetch wins by 42% at the lowest depth (was 33% pre-fix; the fix widened the gap because the low-coverage prefetch arms are exactly where the fix helps most). At full 12 GiB, `prefetch_1x1` (17.98 s) beats `uva_12` (18.41 s) by only 2.3% at B=1; the win expands at larger batches because UVA scales with B while prefetch doesn't.
3. **UVA's slowdown is super-linear in batch.** At B=1 → B=16 → B=64, `uva_4` goes 6.79 → 16.97 → 52.03 s — **7.7×** over the batch range. By contrast, `prefetch_4x1`: 4.53 → 4.83 → 5.90 s — only **1.3×**. **UVA is structurally batch-hostile**: each kernel reading UVA-mapped weights pays PCIe per-token, so PCIe traffic scales with batch. Prefetch's H2D is per-call (B-independent).
4. **UVA's latency is roughly linear in offload depth at fixed batch.** At B=64: 1 → 12 GiB ⇒ 16.10 → 145.52 s, slope ≈ **11.8 s/GiB**. Cost of crossing PCIe per-token compounds proportionally with the bytes that have to be crossed.
5. **Prefetch's slope at B=64**: 0.43 → 12.15 GiB ⇒ 2.10 → 20.54 s, slope ≈ **1.57 s/GiB** — within 2% of the clustered-G=14 slope (1.57 s/GiB; §0.10.1c) at the same batch. **Prefetch is ~7.5× more PCIe-efficient than UVA per offloaded GiB** at decode pressure.
6. **The fix matters most at the low-depth end.** `prefetch_28x1` at B=64 (0.43 GiB) is now within ~3.5% of `none` (2.10 s vs 2.03 s baseline) — the prefetch free regime materializes here. UVA's smallest depth (`uva_1`, 1.00 GiB) costs 16.10 s at B=64 — still ~8× over `none`. At low depth the post-fix prefetch advantage over UVA is closer to **15×**, vs ~7× at full depth.

### 0.10.5: PrefetchDefer ablation — defer fix vs stock-best

This section measures the thesis-only `prefetch_defer` backend (`vllm/model_executor/offloader/prefetch_defer.py`) as an *optimization on top of stock-best* (§0.10.2). Defer's mechanism is to take the wraparound prefetch (the H2D from the highest-index offloaded layer to the lowest) out of iter N's tail and issue it from a forward pre-hook on the first decoder layer of iter N+1, AFTER vLLM's per-step input-prep H2Ds have already queued on CE0. This re-orders both wraparound exposure surfaces from §0.10.1: in eager mode the prefetch H2D no longer queues on CE0 ahead of input prep; in graph mode the prefetch is moved out of the captured graph so `sync_prev_onload`'s pre-replay barrier has no wraparound to drain.

**Stock-best vs defer-at-canonical comparison (B=1).** Defer was originally measured against canonical divisor-G configs in this section; that comparison is preserved below for continuity. The new framing: defer-at-canonical-G vs the §0.10.2 stock-best at matched offloaded GiB.

| L | stock-best (tuned G) | latency | defer at canonical G | latency | defer wins by |
|---:|:---:|---:|:---:|---:|---:|
|  1 | G=15, N=1 | 0.884 s | G=28, N=1 + defer | **0.687 s** | **−22.3%** |
|  2 | G=10, N=1 | 1.438 s | G=14, N=1 + defer | **1.325 s** | **−7.9%** |
|  4 | G=6, N=1  | 2.652 s | (no L=4 canonical) | — | — |
|  7 | G=4, N=1  | 4.537 s | G=4, N=1 + defer | 4.525 s | tied |
| 14 | G=2, N=1  | 8.990 s | G=2, N=1 + defer | 9.011 s | tied |
| 28 | G=1, N=1  | 17.941 s | G=1, N=1 + defer | 17.980 s | tied |

**At low L (1, 2), defer beats stock-best despite stock-best already exploiting placement.** Defer eliminates the wraparound's CE0 occupancy entirely by moving it to a stream that runs *after* input prep, while placement only gives the wraparound more compute window to overlap with. The two are not equivalent fixes:

- Placement at `G=15, N=1`: wraparound fires after layer 14, 13 tail layers (~6.5 ms compute) try to hide 19.5 ms H2D → ~13 ms still exposed.
- Defer at `G=28, N=1`: wraparound moved off iter-N tail, fires at iter N+1 start AFTER input prep, then hides behind 27 layers of compute (~13.5 ms) → ~6 ms exposed.

**At mid L (7, 14), they converge.** Stock-best at L=7 is `G=4, N=1` (canonical), and at L=14 it's `G=2, N=1` (canonical). Defer measurement uses the same configs → no placement difference, and defer's marginal contribution is tiny (~0.3% at L=7, slightly *negative* at L=14 because defer's pre-hook adds a small per-iter overhead with negligible benefit when there are many offloaded layers).

**At high L (28), both flat.** Wraparound is irrelevant; intra-forward CE0 saturation dominates either way.

**Untested compound: defer × placement.** No data exists for `prefetch_defer` at non-divisor G (e.g. `G=15, N=1` + defer at L=1). Conceptually they're orthogonal — defer removes the wraparound from CE0; placement gives the wraparound more compute window — and combining could approach the `none` baseline (0.521 s at L=1). Adding a 10-cell follow-up at non-divisor G with defer would round out the ablation.

The remainder of this section retains the original §0.10.3 wraparound-diagnosis content: the "free regime" finding, the per-G `sync_prev_onload` exposure table for graph mode, the UVA-input-prep confirmatory experiment, and the 2x2 isolation table that shows the eager and graph mech surfaces are mutually-occluding manifestations of one wraparound.

### 0.10.5a: Prefetch vs none — finding the prefetch free regime (legacy framing)

Measured by `bench_prefetch_vs_none.py`. Three-arm decomposition mirrors `phase1a_findings.md §1.14`'s COTS-vs-`none` methodology applied to the native prefetch path. The `real` stock-vs-fixed comparisons use `--offload-backend prefetch` for shipped/factory behavior and `--offload-backend prefetch_defer` for the thesis optimized-native behavior; the `dryrun` arm belongs to `prefetch_defer` because stock `prefetch.py` intentionally has no dry-run or deferred-wraparound state.

| Arm | What it isolates |
|---|---|
| `none` | baseline (no offload) |
| `prefetch_dryrun_g{G}` | `prefetch_defer` wrappers/custom ops/event sync installed, H2D copy skipped via `--prefetch-dry-run` |
| `prefetch_real_g{G}` | end-to-end native prefetch (`prefetch` for unfixed stock, `prefetch_defer` for fixed/optimized rows) |

Decomposition:
- `orch = dryrun − none` → pure host orchestration (Python forward hooks + `wait_prefetch` / `start_prefetch` custom op dispatch + `_copy_done_event` recording + stream fork/join)
- `pcie = real − dryrun` → unhidden H2D transfer cost not absorbed by the GPU layer budget
- `total = real − none` → net regression vs no offload; "free regime" = total ≤ ~0 within run-to-run noise

Coverage axis: N=1, K=1 (the empirically best prefetch knob choice per §0.10.1); G ∈ {1, 2, 4, 7, 14, 28} (divisors of 28 → exact uniform coverage = 1/G).

#### Grid A — decode-heavy (matches §1.14 condition)

Workload: Qwen2.5-7B BF16, `--input-len 8 --output-len 128 --enforce-eager`. Same condition as `bench_cots_dryrun_vs_none.py` so the prefetch and COTS gap tables overlay apples-to-apples. `none` baselines: B=1 2.034s, B=4 2.110s, B=16 2.148s, B=64 2.481s.

| B | G | cov | dryrun (s) | real (s) | orch (s) | pcie (s) | total (s) | orch % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 28 |  3.6% |  2.032 |  4.371 | −0.002 |  +2.339 |  +2.337 | −0% |
| 1  | 14 |  7.1% |  2.034 |  6.000 |  0.000 |  +3.966 |  +3.966 |  0% |
| 1  |  7 | 14.3% |  2.033 | 10.656 | −0.001 |  +8.623 |  +8.622 | −0% |
| 1  |  4 | 25.0% |  2.033 | 18.161 | −0.001 | +16.128 | +16.127 | −0% |
| 1  |  2 | 50.0% |  2.039 | 35.941 | +0.005 | +33.902 | +33.906 |  0% |
| 1  |  1 |100.0% |  2.043 | 71.788 | +0.009 | +69.745 | +69.754 |  0% |
| 4  | 28 |  3.6% |  2.106 |  4.446 | −0.004 |  +2.340 |  +2.336 | −0% |
| 4  | 14 |  7.1% |  2.100 |  6.036 | −0.011 |  +3.936 |  +3.926 | −0% |
| 4  |  7 | 14.3% |  2.111 | 10.685 | +0.001 |  +8.574 |  +8.575 |  0% |
| 4  |  4 | 25.0% |  2.113 | 18.171 | +0.003 | +16.057 | +16.060 |  0% |
| 4  |  2 | 50.0% |  2.114 | 36.086 | +0.004 | +33.972 | +33.976 |  0% |
| 4  |  1 |100.0% |  2.115 | 71.855 | +0.004 | +69.741 | +69.745 |  0% |
| 16 | 28 |  3.6% |  2.148 |  4.498 |  0.000 |  +2.350 |  +2.350 |  0% |
| 16 | 14 |  7.1% |  2.147 |  6.100 |  0.000 |  +3.952 |  +3.952 |  0% |
| 16 |  7 | 14.3% |  2.147 | 10.779 |  0.000 |  +8.632 |  +8.631 |  0% |
| 16 |  4 | 25.0% |  2.149 | 18.326 | +0.002 | +16.176 | +16.178 |  0% |
| 16 |  2 | 50.0% |  2.155 | 36.293 | +0.007 | +34.138 | +34.145 |  0% |
| 16 |  1 |100.0% |  2.157 | 72.454 | +0.009 | +70.297 | +70.307 |  0% |
| 64 | 28 |  3.6% |  2.480 |  4.796 | −0.001 |  +2.316 |  +2.315 | −0% |
| 64 | 14 |  7.1% |  2.479 |  6.261 | −0.002 |  +3.783 |  +3.781 | −0% |
| 64 |  7 | 14.3% |  2.477 | 10.888 | −0.004 |  +8.411 |  +8.408 | −0% |
| 64 |  4 | 25.0% |  2.483 | 18.429 | +0.003 | +15.945 | +15.948 |  0% |
| 64 |  2 | 50.0% |  2.482 | 36.458 | +0.001 | +33.976 | +33.977 |  0% |
| 64 |  1 |100.0% |  2.486 | 72.718 | +0.006 | +70.231 | +70.237 |  0% |

#### Grid B — §0.10-matched workload

Workload: Qwen2.5-7B BF16, `--input-len 256 --output-len 32 --enforce-eager`. Cross-validates the `prefetch_real_g{G}` numbers against the existing knob-sweep results in `0.10_prefetch_knobs/summary.json` (where `prefetch_14x1` at B=1 measured 1.505 s — within 0% of this grid's 1.506 s, confirming the `--prefetch-dry-run` plumbing did not perturb the non-dryrun path). `none` baselines: B=1 0.521s, B=16 0.889s, B=64 2.035s.

| B | G | cov | dryrun (s) | real (s) | orch (s) | pcie (s) | total (s) | orch % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 28 |  3.6% | 0.521 |  1.103 |  0.000 |  +0.581 |  +0.581 |  0% |
| 1  | 14 |  7.1% | 0.522 |  1.506 |  0.000 |  +0.984 |  +0.985 |  0% |
| 1  |  7 | 14.3% | 0.522 |  2.668 | +0.001 |  +2.147 |  +2.147 |  0% |
| 1  |  4 | 25.0% | 0.522 |  4.539 | +0.001 |  +4.017 |  +4.018 |  0% |
| 1  |  2 | 50.0% | 0.523 |  8.991 | +0.002 |  +8.468 |  +8.469 |  0% |
| 1  |  1 |100.0% | 0.523 | 17.936 | +0.002 | +17.413 | +17.415 |  0% |
| 16 | 28 |  3.6% | 0.887 |  1.485 | −0.002 |  +0.598 |  +0.595 | −0% |
| 16 | 14 |  7.1% | 0.885 |  1.865 | −0.005 |  +0.981 |  +0.976 | −1% |
| 16 |  7 | 14.3% | 0.884 |  2.988 | −0.005 |  +2.104 |  +2.099 | −0% |
| 16 |  4 | 25.0% | 0.880 |  4.868 | −0.010 |  +3.989 |  +3.979 | −0% |
| 16 |  2 | 50.0% | 0.875 |  9.451 | −0.014 |  +8.576 |  +8.562 | −0% |
| 16 |  1 |100.0% | 0.875 | 18.825 | −0.015 | +17.951 | +17.936 | −0% |
| 64 | 28 |  3.6% | 2.028 |  2.635 | −0.007 |  +0.607 |  +0.600 | −1% |
| 64 | 14 |  7.1% | 2.023 |  2.978 | −0.012 |  +0.954 |  +0.942 | −1% |
| 64 |  7 | 14.3% | 2.016 |  4.101 | −0.019 |  +2.085 |  +2.066 | −1% |
| 64 |  4 | 25.0% | 2.009 |  5.965 | −0.026 |  +3.955 |  +3.930 | −1% |
| 64 |  2 | 50.0% | 1.983 | 10.429 | −0.053 |  +8.446 |  +8.394 | −1% |
| 64 |  1 |100.0% | 1.967 | 20.480 | −0.068 | +18.513 | +18.445 | −0% |

#### Four observations

1. **Prefetch's free-regime topology is opposite to the CPU-compute path's — and that's what motivated the two-grid design.** The CPU-compute path (one of COTS's two offload mechanisms; characterized in `phase1a_findings.md §1.14`) finds its free regime at *small* workload: small `f_cpu × num_tokens` fits in the GPU compute window, and workload growth kills it because CPU GEMM scales with `num_tokens` faster than the hiding window. Prefetch (the other COTS mechanism, and what the stock `PrefetchOffloader` exclusively uses) should be the *opposite*: per-layer PCIe time is `bytes / BW = 466 MB / 24 GB/s ≈ 19.5 ms`, **fixed, independent of `num_tokens` and B**, while the per-forward GPU compute window grows monotonically with `num_tokens` (15.8 ms at decode_heavy B=1 → 62 ms at pf_match B=64, a 4× window growth from the same 28-layer model). By theory, prefetch's free regime should appear at the **high-compute** end of the sweep. The two-grid design (decode_heavy with 8 input / 128 output tokens, pf_match with 32× more `num_tokens` per forward at 256 input / 32 output) probes exactly this axis. Together the two mechanisms are intended to cover complementary regions of the workload space.

2. **Empirically, prefetch's free regime never materializes — per-forward unhidden PCIe is ~18 ms across every cell, regardless of forward duration. The mechanism is verified across all G ∈ {1, 2, 4, 7, 14, 28} by nsys probe (`probe_native_offload_overlap.py`).** At G=28 (1 prefetched layer per forward), per-forward unhidden is 18.1, 18.1, 18.2, 18.0 ms across decode_heavy B ∈ {1, 4, 16, 64} and 17.6, 18.0, 18.2 ms across pf_match B ∈ {1, 16, 64} — within run-to-run noise despite forward times spanning 15.8 → 62.0 ms. The cross-grid total-pcie ratio (4×: pf_match has 33 forwards/generate vs decode_heavy's 129) is *purely from fewer forwards*, not from better per-forward hiding.

   **Root cause: the prefetch H2D and vLLM's per-step input-prep H2Ds contend for the same hardware copy engine (CE0), and CE0 is FIFO at the engine level — not the stream level.** §0.5.1 already established this property: *"once bg chunks are queued on CE0, fg's `cudaMemcpyAsync` goes to the end of the queue — its wait scales with total queued bg, not just the in-flight chunk's remainder."* That finding was framed as a concern for activation returns; the same hardware property bites prefetch even harder.

   The full chain, verified at the trace level for the G=28 decode_heavy probe (`probe_native_offload_overlap.py` arm `prefetch_g28_decode`):

   1. At end of iter N's offloaded layer compute, `start_prefetch` queues a 466 MB H2D on `prefetch_copy_stream`. The copy gets dispatched onto **CE0**, the sole H2D copy engine on the RTX 4090.
   2. Iter N+1's input prep (vLLM v1 async-scheduling path) does many small `tensor.to('cuda', non_blocking=True)` calls for `input_ids`, `positions`, block-table indices, attention metadata, etc. — typically ~20 tiny H2Ds each a few bytes. These are also dispatched onto CE0 (regardless of which CUDA stream they belong to) and **queue FIFO behind the in-flight 466 MB prefetch H2D**.
   3. The prefetch H2D occupies CE0 for ~19.5 ms. Input prep H2Ds wait the full 19.5 ms before they get serviced. Once they drain (microseconds), `prepare_inputs_event.record()` (`vllm/v1/worker/gpu_model_runner.py:3489`, recorded on main_stream after input prep) fires.
   4. Iter N+2's `synchronize_input_prep()` calls `prepare_inputs_event.synchronize()` (`vllm/v1/worker/gpu_model_runner.py:3485`) — Python blocks until that event fires. Block duration ≈ prefetch H2D duration ≈ 18 ms. (In the non-async-scheduling path, the analogous block is `transfer_event.synchronize()` for the sampled-token D2H at `vllm/v1/worker/gpu_model_runner.py:6943`; same shape of bug, different code site.)
   5. While Python is blocked, main_stream is idle (next forward's compute kernels haven't been dispatched yet). When the sync returns, Python rapidly dispatches iter N+2's forward and main_stream resumes.

   **Direct trace evidence (window from the G=28 probe, sequential events on CE0):**

   ```
   19.62055   stream 21   271.6 MB   prefetch H2D running on CE0  ┐
   19.63194   stream 21   135.8 MB   prefetch H2D continues       │ FIFO on CE0
   19.63766   stream 21       7 B    last byte of prefetch H2D    ┘
   19.63766   stream  7       8 B    input prep H2D STARTS  ← exact same instant
   19.63767   stream  7       1 B    more input prep H2Ds drain
   ...                               (~20 tiny H2Ds in <1 ms)
   ```

   At the precise instant the prefetch H2D bundle finishes on CE0, the queued input prep H2Ds on main_stream drain. The two streams are software-independent (no `wait_event` between them) but the hardware FIFO at CE0 serializes them anyway.

   **There is no software-level sync bug.** vLLM didn't accidentally chain the prefetch into the per-step sync. The per-step sync is necessary and correct (it ensures the previous step's input prep is done before reusing the CPU pinned-memory tensors). The serialization is implicit in the hardware: any H2D issued anywhere on the device after the prefetch's H2D has to wait for it to drain CE0.

   Per-G nsys evidence (decode_heavy workload, input=8 output=32 B=1, 2 iters; sync events filtered to syncType=2, duration > 1ms, post-warmup):

| G | layers offloaded /forward | sync events | sync_avg_ms | sync_total_s | mean inter-bundle gap (ms) | min gap (ms) | H2D total (GB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 28 |  1 | 125 |  **25.88** |  3.24 | 59.25 | 14.73 |  31.2 |
| 14 |  2 | 125 |  **32.28** |  4.04 | 25.88 |  0.53 |  62.0 |
|  7 |  4 | 125 |  **50.55** |  6.32 | 11.91 |  0.53 | 123.5 |
|  4 |  7 | 125 |  **80.02** | 10.00 |  6.71 |  0.53 | 215.8 |
|  2 | 14 | 125 | **149.96** | 18.75 |  3.19 |  0.53 | 431.1 |
|  1 | 28 | 125 | **290.94** | 36.37 |  1.83 |  0.53 | 861.8 |

   What this shows:

   - **Sync count is fixed at 125 (~2 per forward) regardless of G** — the sync is per-step Python control flow, not per-prefetch.
   - **Sync duration scales monotonically with the number of prefetched layers per forward** — each added offloaded layer adds another 19.5 ms of CE0 occupancy that input prep H2Ds (and intra-forward `wait_prefetch` events on main_stream) must wait through.
   - **`min_gap_ms = 0.53 ms` for every G ≥ 14** — copy_stream's FIFO is fully back-to-back saturated; consecutive prefetches serialize at 19.5 ms cadence on CE0. Only G=28's `min_gap = 14.73 ms` reflects the inter-forward gap (the only G with a single bundle per forward).
   - **The G=28 sync time (~26 ms) ≈ 1 H2D**; the G=1 sync time (~291 ms) ≈ chain of ~15 of the 28 H2Ds (the rest gets ~10 ms intra-forward hide each via the `(G−1)·layer_time` window in observation 4).

   **Confirmatory experiment.** To rule out a wrong diagnosis before committing to a fix, route input prep through a UVA-style mapped pinned buffer instead of going through CE0 (per §0.5.3, UVA reads use an SM-issued PCIe path that bypasses CE0). If the per-step sync block disappears under this configuration — without touching the offloader at all — the CE0 FIFO contention is unambiguously confirmed as the mechanism. A short-lived `VLLM_INPUT_PREP_UVA=1` shim was added to `vllm/v1/utils.py` (since reverted) that monkey-patches `torch.Tensor.{to, copy_}` to redirect pinned-CPU→CUDA non_blocking copies smaller than 1 MiB through a UVA-mapped Triton kernel (`_copy_k` from `David/Benchmarks/phase0/probe_uva_bypass.py`). The 1 MiB threshold excludes prefetch's 466 MB H2Ds; only the small input-prep H2Ds get diverted off CE0. Re-running the G=28 decode_heavy probe (`input=8 output=32 B=1, 2 iters, 0 warmup`) with the shim:

   | Metric | Unfixed (CE0) | UVA input prep | Δ |
   |---|---:|---:|---:|
   | sync events (>1 ms) | 125 | 63 | **−50%** (per-step `prepare_inputs_event` sync eliminated; `transfer_event` sync remains) |
   | `sync_avg_ms` | 25.88 | 19.67 | −24% |
   | `sync_total_s` | 3.235 | 1.239 | **−62%** |
   | end-to-end latency (avg) | 1.115 s | 0.685 s | **−38.5%** |

   The sync block disappears on the input-prep side without touching prefetch — diagnosis confirmed. Traces: `results/0.10_uva_validation/prefetch_g28_decode_{unfixed,uva_inputprep_v3}.{nsys-rep,sqlite}`.

   **Fix delivered as a separate thesis backend.** With the diagnosis confirmed, the offloader-localized fix is now isolated in `PrefetchDeferOffloader` (`--offload-backend prefetch_defer`) rather than mixed into the stock `PrefetchOffloader`. Stock `prefetch.py` / `prefetch_ops.py` stay byte-for-byte factory-clean; `prefetch_defer.py` implements the scheduling policy and `prefetch_defer_ops.py` registers the deferred custom op. In the fixed backend, **only the last wrap-around `start_prefetch`** (the one fired by `index == n_offloaders - 1`, whose post-hook has zero remaining iter-N compute window) is *skipped* at the end of iter N and instead issued from a forward pre-hook on the first decoder layer at the start of iter N+1 — *after* vLLM's per-step input-prep H2Ds have already queued on CE0. The CE0 FIFO then services the tiny input prep H2Ds first (microseconds) and the deferred prefetch H2D second; the per-step `prepare_inputs_event.synchronize()` no longer transitively waits for the prefetch, and the first offloaded layer's `wait_prefetch` stalls main_stream by `max(0, PCIe − L·layer_time)` instead of by the full PCIe time. At K > 1 the *earlier* wrap-around layers (those whose post-hook still has remaining iter-N compute) fire immediately so their H2D overlaps with iter N's tail compute — empirically Pareto-dominant vs defer-all and defer-none across G ∈ {1..14} K=2 (`David/Benchmarks/phase1b/results/native_defer_policy/`: defer-only-last beats defer-all by +0.3% at G=1 to +5.1% at G=14, and beats defer-none by +0.1% at G=1 to +22% at G=14). Implementation: static `deferred_wraparound_index` identifies the single deferred target; the first-decoder pre-hook starts it through `torch.ops.vllm.start_deferred_prefetch`. Token output is byte-identical to stock prefetch at K=1 and K=2 (verified empirically).

   **Post-fix sync table (same per-G probe condition as the unfixed table above):**

| G | layers offloaded /forward | unfixed sync_avg_ms | fixed sync_avg_ms | unfixed sync_total_s | fixed sync_total_s | Δ total | unfixed n_sync | fixed n_sync |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 28 |  1 |  25.88 |  21.05 |  3.235 |  1.326 | **−59%** | 125 | 63 |
| 14 |  2 |  32.28 |  40.93 |  4.040 |  2.579 | **−36%** | 125 | 63 |
|  7 |  4 |  50.55 |  80.89 |  6.320 |  5.096 | **−19%** | 125 | 63 |
|  4 |  7 |  80.02 | 140.96 | 10.000 |  8.880 | **−11%** | 125 | 63 |
|  2 | 14 | 149.96 | 281.00 | 18.750 | 17.703 |  **−6%** | 125 | 63 |
|  1 | 28 | 290.94 | 561.21 | 36.370 | 35.356 |  **−3%** | 125 | 63 |

   What this shows:
   - **Per-step `prepare_inputs_event` sync is eliminated at every G** — sync count drops from 125 → 63 universally (one fewer sync per forward), exactly matching the UVA-input-prep validation above. The remaining 63 syncs are the `transfer_event` (sampled-token D2H on CE1) waits, which the fix does not affect.
   - **Per-event sync_avg_ms goes UP at G ≤ 14** because the deferred prefetches now queue ahead of the next iter's `transfer_event` window, but the *count* halving dominates so `sync_total_s` still drops. At G=28 even the per-event sync drops because there's only one deferred prefetch per iter — well-hidden by the (G−1)·layer_time compute window.
   - **Effect shrinks with coverage** as the diagnosis predicted: ~59% sync-time saving at G=28 (1 layer offloaded, 27 GPU-resident layers of compute window) but only ~3% at G=1 (no GPU-resident layers to hide behind).

   **End-to-end latency confirmation (`bench_prefetch_vs_none.py` Grid B `pf_match`, `input=256 output=32`).** Re-runs include a third arm `prefetch_real_unfixed_g{G}` — same workload, same hour, stock `--offload-backend prefetch` instead of optimized `--offload-backend prefetch_defer` — for matched-condition before/after.

| G | B=1 unfixed (s) | B=1 fixed (s) | Δ % | B=64 unfixed (s) | B=64 fixed (s) | Δ % | "Free" at B=64? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 28 |  1.104 |  0.687 | **−38%** |  2.630 |  2.102 | **−20%** | **YES** (total = +0.07 s vs `none` 2.03 s) |
| 14 |  1.506 |  1.325 | −12% |  2.974 |  2.722 |  −9% | near (+0.69 s vs `none`) |
|  7 |  2.668 |  2.605 |  −2% |  4.081 |  3.983 |  −2% | no |
|  4 |  4.538 |  4.525 |  −0% |  5.942 |  5.887 |  −1% | no |
|  2 |  8.988 |  9.011 |  +0% | 10.427 | 10.422 |  −0% | no |
|  1 | 17.938 | 17.980 |  +0% | 20.469 | 20.519 |  +0% | no |

   **The free regime appears for the first time at G=28 pf_match B=64** (total = +0.07 s vs `none` = 2.03 s, within run-to-run noise) — closing the question this whole subsection opened with. The G=28 B=1 result (−38%) matches the UVA-input-prep validation (−38.5%) within 0.5 percentage points: two independent measurements of the same mechanism converging on the same number.

   At full coverage (G=1, G=2) the fix delivers ~0% improvement because every layer's prefetch must drain on CE0 anyway; the wrap-around is no longer the rate limiter (intra-forward CE0 serialization between consecutive offloaded layers dominates). The decode_heavy grid (`input=8 output=128`) shows a similar shape with smaller absolute fix-Δ in the high-coverage cells; full numbers in `results/0.10_prefetch_vs_none/summary.json`. Pre-fix snapshots are preserved unmodified under `results/0.10_*-unfixed/` for the four affected sweeps.

   **Graph-mode wraparound (Codex follow-up).** The eager-mode CE0 FIFO bug above is one face of the wraparound problem; CUDA-graph mode has its own face. Outside `--enforce-eager`, vLLM captures and replays per-batch-shape graphs (`vllm/v1/worker/gpu/cudagraph_utils.py`, `gpu_ubatch_wrapper.py`). Because the prefetch copy_stream is external to any graph, vLLM must drain it before each capture/replay via `PrefetchOffloader.sync_prev_onload()` — a `wait_stream(self.copy_stream)` barrier on the compute stream. Pre-fix, that barrier waits exactly one wraparound H2D worth of work each time it fires.

   Per-G probes at the §0.10.3 condition (`input=8 output=32 B=1, 3 warmup + 2 bench iters`), comparing `--enforce-eager` to graph mode (`--enforce-eager` removed). Filter syncType=2, duration > 1 ms (matches the §0.10.3 unfixed table's filter):

   | G | layers offloaded /forward | eager_unfixed sync_ms (n=317) | graph_unfixed sync_ms (n=160) | graph_fixed sync_ms (n=160) | predicted PCIe × layers |
   |---:|---:|---:|---:|---:|---:|
   | 28 |  1 |  25.90 |  35.23 | **25.00** | ~19.5 ms |
   | 14 |  2 |  32.27 |  48.55 |  46.91 | ~39.0 ms |
   |  4 |  7 |  79.84 | 159.82 | 158.95 | ~136.5 ms |
   |  1 | 28 | 289.82 | 561.47 | 561.95 | ~546.0 ms |

   The graph_unfixed per-event sync time **scales exactly with `layers_offloaded × per-layer-PCIe-time`** at every G (G=1: 561 ms vs predicted 546 ms; G=4: 160 ms vs predicted 156 ms — within 3% across the sweep). This confirms Codex's diagnosis: in graph mode, `sync_prev_onload()` exposes the entire pre-replay wraparound state in one barrier, instead of the per-step input-prep CE0 exposure pattern of eager mode. The two mechanisms have different sync surfaces but the bytes that have to drain are the same.

   Sync count drops from 317 (eager) to 160 (graph) because the graph-mode path issues fewer host-side syncs per generate; total sync_total_s is similar in unfixed cells. End-to-end avg latency in unfixed mode is also similar (G=28: eager 1.10 s vs graph 1.13 s; G=4: eager 4.54 s vs graph 5.12 s — graph is slightly slower at high coverage because of capture overhead, no benefit at low coverage because the pathology dominates). Probe traces: `results/0.10_overlap_probe/prefetch_g{G}_{eager,graph}_{unfixed,fixed}.{nsys-rep,sqlite}`.

   **Same bytes, two mode-exclusive sync surfaces (mutually occluding).** A direct 2x2 isolation experiment confirms the two sync exposures are different surfaces of the *same underlying wraparound H2D bytes*, gated by execution mode. Two binary axes:

   - `--enforce-eager` ON → mech 2 structurally impossible (graph code path never runs; `sync_prev_onload` never called).
   - `VLLM_INPUT_PREP_UVA=1` (a temporary diagnostic shim that monkey-patches small `Tensor.{to, copy_}` H2Ds to use an SM-issued Triton kernel reading UVA-mapped pinned memory) → mech 1 structurally impossible (input-prep H2Ds bypass CE0 entirely; nothing for the wraparound to queue ahead of).

   Four cells per G, factory `prefetch` backend, G=28 N=1 K=1, `input=8 output=32 B=1`, 3 warmup + 2 bench iters:

   | cell | `--enforce-eager` | UVA shim | mech 1 can fire? | mech 2 can fire? | avg latency (s) | Δ vs δ baseline |
   |---|:---:|:---:|:---:|:---:|---:|---:|
   | **α** eager + noUVA | ✓ | ✗ | ✓ | ✗ | 1.097 | +0.455 s |
   | **β** graph + UVA   | ✗ | ✓ | ✗ | ✓ | 1.136 | +0.494 s |
   | **γ** graph + noUVA | ✗ | ✗ | ✓ | ✓ | 1.134 | +0.492 s |
   | **δ** eager + UVA   | ✓ | ✓ | ✗ | ✗ | 0.642 | baseline |

   Two facts visible immediately:

   1. **The two single-mechanism waits are equal in magnitude** (α: +0.455 s, β: +0.494 s, both ≈ ~one wraparound H2D worth of PCIe per ~33 forwards). Same bytes, same total wait — just exposed through different sync surfaces.

   2. **γ ≈ β, NOT α + β.** The "both active" cell adds *zero* sync penalty over the "mech 2 only" cell. So the two surfaces don't compound — they're mutually occluding.

   The reason mech 2 occludes mech 1: `sync_prev_onload` runs *before* the next graph replay starts. Its `wait_stream(copy_stream)` drains the wraparound H2D from copy_stream entirely. By the time the next replay's input-prep H2Ds queue on CE0, the wraparound bytes have already finished — there's nothing for input prep to queue behind, so mech 1 has nothing to wait on.

   ```text
   eager mode timeline:
     iter N tail   → wraparound H2D enters CE0 (running)
     iter N+1     → input prep H2Ds queue on CE0 ← QUEUES BEHIND wraparound
                                                ← prepare_inputs_event waits ~19.5 ms (mech 1)

   graph mode timeline:
     prev replay end → wraparound H2D still pending on copy_stream
     next replay START
       → sync_prev_onload() ← STALLS ~19.5 ms here (mech 2; drains the wraparound)
       → wraparound H2D NOW DONE on CE0
       → graph replay's input-prep H2Ds queue on CE0 ← finds CE0 EMPTY → no mech 1
   ```

   So mech 1 and mech 2 are mode-exclusive **and** mutually-occluding: a single run pays exactly one ~19.5 ms wait per wraparound, exposed through whichever surface its mode happens to use first. Probe data: `results/0.10_two_bug_isolation/`.

   Practical consequence: **the defer fix is necessary in both modes for the same reason** — it removes the wraparound H2D source so neither sync surface has anything to wait on. UVA-routed input prep would also work in eager mode (and was the P1 confirmatory experiment) but doesn't help in graph mode because mech 2 isn't routed through CE0 at all. Conversely, `--enforce-eager` switches the exposure surface but doesn't reduce the wait. Defer is the only single fix that addresses both modes.

   **Graph-compatible defer-wraparound fix.** The first attempt registered `_first_decoder_pre_hook` as a bound method on `PrefetchOffloader`. That ran *inside* the model forward, which Dynamo traces as part of the full graph. Dynamo's AOT guard serialization then walked the hook's closure into `module_offloaders[*]._copy_done_event` (a `torch.cuda.Event`), failing with `TypeError: cannot pickle 'Event' object`. Wrapping the hook with `torch.compiler.disable` reached a different error (`Skip calling torch.compiler.disable()'d function`) because vLLM compiles via `aot_compile_fullgraph` which forbids graph breaks.

   The graph-compatible solution now lives in the separate thesis files: `vllm/model_executor/offloader/prefetch_defer_ops.py` registers `torch.ops.vllm.start_deferred_prefetch(input_tensor)`, and `vllm/model_executor/offloader/prefetch_defer.py` installs a top-level free-function pre-hook that dispatches the op. The stock `prefetch_ops.py` remains limited to `wait_prefetch` / `start_prefetch`. The deferred op reaches into the active offloader from C-level — opaque to Dynamo:

   ```python
   def _start_deferred_prefetch_pre_hook(module, args, kwargs):
       anchor = args[0] if args else kwargs.get("hidden_states")
       if anchor is not None:
           torch.ops.vllm.start_deferred_prefetch(anchor)
   ```

   The empty closure (`del module`, no `self`) means Dynamo never sees offloader state. The deferred target is a static `deferred_wraparound_index` set once at `wrap_modules`, not a per-iter mutable flag. Initial prefetches (`post_init`) skip that index so iter 0 doesn't double-fire it. The wrap-around case at the end of the last offloaded layer's forward simply skips its own `start_prefetch` instead of stashing a Python flag.

   **Post-fix graph-mode results (third column above):**
   - G=28 (1 layer offloaded): per-event sync **35.23 → 25.00 ms (−29%)**, end-to-end **1.135 → 0.812 s (−28.5%)**. The fix transforms graph mode at low coverage from "slightly slower than eager" into "fastest of the three modes" (eager_unfixed 1.097 s vs graph_fixed 0.812 s = **−26%**, because graph capture eliminates the per-step host overhead the deferred eager path still pays).
   - G=14 (2 layers): **48.55 → 46.91 ms (−3%)** — only one of the two wraparounds is deferred, the other still drains in the barrier.
   - G=4 (7 layers): **159.82 → 158.95 ms (−0.5%)** — intra-forward CE0 saturation dominates, the single deferred wraparound is a tiny fraction.
   - G=1 (28 layers): **561.47 → 561.95 ms (~0%)** — at full coverage every layer is offloaded, the one deferred prefetch is rounding error.

   The shape exactly matches the eager-mode defer benefit pattern (§0.10.3 post-fix sync table): big win at low coverage, decaying to zero at full coverage. Graph mode's win is slightly smaller than eager (−29% vs −38%) because graph_unfixed already amortizes input-prep barriers across the captured graph, leaving less for the defer fix to recover. With `prefetch_defer`, **graph mode is now the recommended optimized-native deployment** at low offload depth — it captures the kernel-launch overhead AND avoids the sync_prev_onload wraparound drain.

   **Placement-sensitivity workaround (Codex follow-up).** Independent of the defer fix, the shared stock/deferred picker (`idx % G >= G − N`) always offloads the *last* model layer when G is a divisor of `num_layers`. On Qwen2.5-7B (28 layers), every G ∈ {1, 2, 4, 7, 14, 28} hits this case — the offloaded layer's wraparound prefetch sees zero tail compute to hide behind. A small picker shift (`(idx − offset) % G >= G − N`) moves the last offloaded layer earlier and gives the wraparound H2D some same-iter compute to overlap with.

   Two probes (`input=8 output=32 B=1, --enforce-eager`, stock `prefetch` unless noted):

   *G=4 N=1 (7 offloaded layers, 3.04 GiB):*

   | offset | last offloaded | tail layers | avg latency (s) |
   |---:|---:|---:|---:|
   | 0 (canonical) | 27 | 0 | 4.540 |
   | 1 | 24 | 3 | **4.492** |
   | 2 | 25 | 2 | 4.506 |
   | 3 | 26 | 1 | 4.522 |
   | 0 + defer fix | 27 | 0 | 4.532 |

   At G=4 placement is monotone in tail count (predicted), but the magnitude is small (~1% best-case relief, 4.540 → 4.492). Defer fix is also tiny here (~0.2%) because intra-forward CE0 saturation at 7 offloaded layers dominates either way.

   *G=28 N=1 (1 offloaded layer, 0.43 GiB) — where the wraparound is the rate limiter:*

   | offset | last offloaded | tail layers | avg latency (s) | gap vs `none` (~0.52 s) |
   |---:|---:|---:|---:|---:|
   | 0 (canonical) | 27 | 0 | 1.094 | +0.57 s (109% over) |
   | **1** | 0 | **27** | **0.643** | **+0.12 s (24% over)** |
   | 13 | 12 | 15 | 0.841 | +0.32 s |
   | 0 + defer fix | 27 | 0 | 0.682 | +0.16 s (32% over) |

   At G=28, **placement (offset=1) actually outperforms the defer fix** — 0.643 s vs 0.682 s, a 41% reduction over canonical vs the defer fix's 38%. Mechanism: with one offloaded layer at position 0, the wraparound prefetch fires at end-of-iter-N for iter-N+1's layer 0, has 27 layers of iter-N tail compute window before that prefetch ends up CE0-blocked, and (importantly) finishes well before iter-N+1's input prep. The defer fix achieves a similar but slightly weaker overlap because it puts the prefetch issue *after* iter-N+1's input prep, leaving only ~13 ms of compute to hide ~19.5 ms of H2D.

   **Implication for the thesis recommendation.** Placement is a real, measurable win at low coverage on a model whose layer count is divisible by the chosen G — but it is brittle: it assumes a known model layer count, doesn't help the K>1 case (multiple wraparound prefetches), and silently fails on models or G values where the canonical picker happens not to hit the last layer. The defer fix is general and now graph-compatible, but it is reported as the explicit `prefetch_defer` optimized-native baseline rather than as shipped vLLM stock behavior. Placement (`offload_offset > 0`) remains an additional tunable for future ablations, not part of the factory-clean `prefetch` baseline. Probe data: `results/0.10_placement_sweep/`.

3. **Prefetch's host orchestration is essentially zero** — `prefetch_defer` dryrun − none is in the −68 to +9 ms range across all 42 cells, with magnitude ≤10 ms even at G=1 (28 layers wrapped, full Python wrapper count). Two orders of magnitude smaller than the COTS-CPU-compute-path orch column in `phase1a_findings.md §1.14` (~450 ms). The C++-backed prefetch path has effectively no Python tax compared to the Python `ThreadPoolExecutor` substrate measured for the CPU compute path. **Implication for Phase 1c:** closing the host orchestration gap to prefetch's level (~450 → ~10 ms) is what `cudaLaunchHostFunc` + native CPU runner needs to achieve on the CPU compute path. The remaining ~2 s active-CPU-work penalty in §1.14 is the only structural gap COTS's CPU-compute side will still have post-Phase-1c.

4. **Unhidden PCIe scales linearly with coverage** (Grid A B=64, G=28→14→7→4→2→1 gives pcie = 2.32, 3.78, 8.41, 15.95, 33.98, 70.23 s — a clean power-of-2 sweep) **and is flat across B at fixed G** (Grid A G=2: pcie ≈ 33.9 s for B ∈ {1, 4, 16, 64}). Combined with observation 2, total PCIe penalty per generate ≈ `(~18 ms unhidden per forward per offloaded layer) × forwards × layers_offloaded` — scales with coverage and forward count, but not per-forward duration. The clean linearity in coverage is consistent with each prefetched layer being independently throttled by the same per-forward hiding cap.

Cross-references:
- `phase1a_findings.md §1.14` — sister table for COTS. Anchors the cots-vs-prefetch competition against `none` for both arms.
- `phase1a_findings.md §1.13b` — cots-vs-prefetch numbers; the prefetch column there can now be re-anchored against `none`.

### 0.10.6: nsys overlap probe

Three short nsys traces written by `probe_native_offload_overlap.py` to `results/0.10_overlap_probe/`. Probe uses 25% coverage with the cleanest spatial pattern (uniform single-layer offloads via G=4 N=1). All three traces refreshed post-fix; pre-fix copies preserved under `results/0.10_overlap_probe-unfixed/` for direct visual comparison in `nsys-ui`.

- **`prefetch_4x1.nsys-rep`** — open in `nsys-ui`. Look for: `copy_stream` (separate from the default CUDA stream) showing `Pinned→Device` memcpy events occurring *during* the cuBLAS / SDPA kernels of preceding GPU-resident layers. Stream attribution confirms compute and H2D run on different streams. Quantitatively from `nsys stats`: 3,466 H2D ops totaling 8.26 s of activity (vs 918 ops / 1.14 s for `none` — same baseline traffic, plus ~7 GB of explicit prefetch). Visually contrast stock `prefetch` against `prefetch_defer`: in stock, the wrap-around prefetch lands at the iter boundary ahead of input prep H2Ds; in the optimized trace the same prefetch H2D fires ~one input-prep-window later, after the per-step H2Ds drain.
- **`uva_4.nsys-rep`** — no extra H2D events versus `none` (801 ops / 0.79 s, ≈ baseline). UVA is a memory mapping, not a copy; per-kernel weight reads happen *inside* the cuBLAS/SDPA kernels and show up as inflated kernel durations on the layers whose weights are UVA-offloaded. This is the timeline manifestation of finding §0.10.2 (3).
- **`none.nsys-rep`** — sanity baseline. No offload-related stream activity inside the decode region.

### Implications for COTS

1. **Headline native baselines at B=64 (post-restructure):**
   - **Stock-best (§0.10.2)** at ~25% coverage = `prefetch_4x1` (canonical, 7 layers, 3.04 GiB) at K=1: 5.94 s (B=64). At ~17% coverage = `L4_G6_N1` (placement-aware, 4 layers, 1.74 GiB): 4.02 s.
   - **Optimized-native (§0.10.5)** at ~25% coverage = `prefetch_defer` at G=4 N=1 K=2: 5.74 s (B=64) — within 1% of stock-best because at L=7 the canonical IS the stock-best.
   - At ~3.6% coverage (L=1, 0.43 GiB) the gap between stock-best (`L1_G15_N1`: 2.32 s @ B=64, placement-aware) and `prefetch_defer` at canonical `G=28, N=1` (2.10 s @ B=64): defer wins by ~10% even after stock-best exploits placement, because defer's mechanism is structurally stronger (eliminates the wraparound's CE0 occupancy entirely, not just shortens the wait).
   - **The COTS comparison surface shifts.** At the low-depth end (L=1, 0.43 GiB), the optimized-native baseline is within ~3.5% of `none` (2.10 vs 2.03 s) — essentially free. There is no headroom for COTS to beat it. The meaningful battleground is **medium/high depth** (≥1.7 GiB, L ≥ 4) where prefetch still pays a structural intra-forward CE0 saturation tax that COTS's CPU-compute path can route around.

2. **At fixed bytes, placement vs spacing depends on L** (§0.10.2). At low L (≤5 layers), placement-aware non-divisor G with N=1 wins by 5–20%; at mid L (7–14), uniform divisor-G wins because dense spacing matters more. COTS's tensor-granularity approach (split *within* every layer) is the limit case — no clustering AND no final-layer concentration.

3. **UVA is never the right baseline above B=1.** Even at low offload depth the per-token PCIe cost of UVA-mapped reads is prohibitive. When comparing COTS against shipped vLLM, use stock `prefetch` (`--offload-backend prefetch`) tuned per §0.10.2; when comparing against the thesis optimized-native baseline, use `prefetch_defer` (`--offload-backend prefetch_defer`). `uva_*` is reported only as a sanity floor.

4. **Layer-ahead (K=1) is empirically near-optimal.** §0.10.2's K-sweep at the placement-aware Pareto-knee `(G=6, N=1)` and §0.10.3d's K-sweep at canonical `(G=4, N=1)` both find K=2 buys only ~2–3% over K=1 with K=4 OOMs at B=64. COTS's commitment to one prefetch queue per layer boundary is supported by both the contention analysis (§0.5/§0.9) and these direct latency comparisons.

5. **Prefetch's host orchestration is structurally smaller than COTS's** (§0.10.5 — `prefetch_defer` dryrun − none measured at ~5–10 ms per generate even at full coverage G=1, vs the §1.14 COTS orch column of ~450 ms — a ~50× gap). The prefetch path's C++-backed wrappers are not where any prefetch-vs-none regression comes from; the gap is essentially all unhidden PCIe transfer (`pcie = real − dryrun`). For COTS to compete at the high-coverage end, the Phase 1c native runner has to close most of that 450 ms orch column — the structural Python-substrate gap, not the GEMM path.

### Recommendation for Phase 3 evaluation

The §0.10 numbers form the headline native baselines COTS is compared against in Phase 3:

- Report two native references when possible. **Shipped vLLM stock** is `--offload-backend prefetch` with the best `(G, N, K)` from §0.10.2 (depth-dependent: placement-aware non-divisor G at low L, canonical divisor-G at mid L). **Optimized native thesis baseline** is `--offload-backend prefetch_defer` with the same matched-byte tuning; this captures the deferred-wraparound improvement without modifying stock `prefetch.py`.
- Default vLLM settings (G=8 N=1 K=1) are *not* the best — at L=3 the picker would offload 3 layers including layer 27 (worst-case wraparound). The §0.10.2 best-stock-per-L table is the right reference for "what stock vLLM can do when properly tuned":
  - L=1 (0.43 GiB): `--offload-group-size 15 --offload-num-in-group 1`
  - L=2 (0.87 GiB): `--offload-group-size 10 --offload-num-in-group 1`
  - L=4 (1.74 GiB): `--offload-group-size 6 --offload-num-in-group 1`
  - L=7 (3.04 GiB): `--offload-group-size 4 --offload-num-in-group 1` (canonical divisor-G wins here)
  - L=14 (6.08 GiB): `--offload-group-size 2 --offload-num-in-group 1` (canonical wins here too)
- `uva_*` is reported as a sanity floor (worst-case stock option above B=1) but is not the primary comparison target.
- K=1 is near-optimal at every depth (§0.10.2 K-sweep + §0.10.3d K-sweep agree). K=2 is at most a ~2–3% improvement and risks OOM at B=64 large batch.
- `none` is a valid comparison only at zero offload — at non-trivial offload depths the baseline simply OOMs without an offloader.
