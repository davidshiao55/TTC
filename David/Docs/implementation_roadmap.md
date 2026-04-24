# Implementation Roadmap

This document is the authoritative implementation plan for the thesis offloading
work. It supersedes the roadmap in `thesis_proposal.md` and the implementation
phases in `CLAUDE.md`.

**Prerequisites completed:**
- FastTTS migrated to vLLM V1 (see `vllm_v1_migration.md`)
- Both conda environments (`baseline`, `thesis`) working
- vLLM fork full CUDA build done

---

## Architecture Overview

The thesis system has three components (see `thesis_proposal.md` §3):

- **Profiler** (offline) — measures HW/model behavior; produces cached tables. See `profiler_design.md`.
- **Planner** (load-time) — solves for placement + per-bucket dispatch; primary contribution. See `planner_design.md`.
- **Scheduler** (runtime) — executes the plan; tier-aware admission + KV migration. See `scheduler_design.md`.

And three offloading mechanisms the components orchestrate (`thesis_proposal.md` §3.2):

- Two-tier weight storage + three-way compute dispatch
- Two-tier KV pool
- PCIe allocation (100% to weight prefetch — design invariant)

### Phase-to-Component Map

| Phase | Profiler | Planner | Scheduler | Mechanism scope |
|---|---|---|---|---|
| **0** — Benchmarks | First profile run (schema validation) | — | — | — |
| **1** — Resident Hybrid Weight Split | + CPU GEMM curves (all sub-modules) | First Planner output: `f_cpu_store` only, single-entry dispatch | — | Weight storage; CPU-compute dispatch path (all sub-modules incl. WQKV) |
| **2** — Attention Offloading | + CPU attention curve | Planner gains `KV_gpu_bytes` / `KV_cpu_bytes` variables | Tier-aware KV admission | KV two-pool storage; CPU suffix attention |
| **3** — Tensor Prefetch | + PCIe BW curve refinement | Dispatch table gains `f_prefetch_compute` axis per bucket | — | Prefetch dispatch path added |
| **4** — E2E | — | — | — | All mechanisms active |
| **5** — CUDA Graph | — | — | `cudaLaunchHostFunc` retrofit | Graph-compatible dispatch |

Phases deliver the system incrementally: each extends Profiler coverage, Planner variable set, or Scheduler capabilities while preserving the three-component architecture.

---

## Phase 0 — Pre-Implementation Benchmarking

Validate the quantitative assumptions before writing any offloading code. If any
of these numbers are significantly off, the approach changes.

### 0.1 CPU GEMM Throughput (CRITICAL — gates Phase 1)

**What:** Measure CPU matmul throughput for representative column-slice sizes.

**Why:** The f_cpu=9% "free" argument assumes CPU finishes its portion within
GPU layer time (~0.5 ms for 7B). If CPU BLAS achieves only 40 GB/s instead of
80 GB/s, f_cpu must drop to ~4.5%, halving memory savings.

**How:** Benchmark `torch.mm` on CPU pinned tensors for each sub-module's CPU
slice at batch sizes B=1,4,8,16,32. Test both BF16 and FP32 — CPU BF16 BLAS may
be slower than FP32 depending on hardware (AMX vs no-AMX).

Representative shapes (7B, f_cpu=9% uniform column choice):
| Sub-module | GPU shape | CPU shape |
|---|---|---|
| WQKV | [B, 3584] × [3584, 4194] | [B, 3584] × [3584, 414] |
| WO | [B, 3584] × [3584, 3262] | [B, 3584] × [3584, 322] |
| MLP1 | [B, 3584] × [3584, 34478] | [B, 3584] × [3584, 3410] |
| MLP2 | [B, 18944] × [18944, 3262] | [B, 18944] × [18944, 322] |

And for the WQKV K/V-biased slice at `f_cpu_store_WQKV = 22%` (strict Q | K | V boundary; see `weight_offload_design.md`):
| Split | GPU (Q) | CPU (K+V) |
|---|---|---|
| WQKV | [B, 3584] × [3584, 3584] | [B, 3584] × [3584, 1024] |

### 0.2 GPU Layer Time Breakdown (CRITICAL — gates Phase 1)

**What:** Profile per-layer decode time for 7B on V1, broken down by sub-module
(WQKV, attention, WO, MLP1, MLP2).

**Why:** Confirms the ~0.5 ms/layer assumption and gives the exact GPU idle time
that CPU compute must fit within. Also validates the memory-BW-bound assumption.

**How:** `torch.cuda.Event` based timing around each sub-module, or vLLM's
layerwise profiler. Run at B=1,4,8,16,32,64.

### 0.3 PCIe Effective Bandwidth (validates Phase 1b prefetch estimates)

**What:** Extend `David/Benchmarks/pcie_bandwidth_test.py` to sweep over
transfer sizes: 256 KB, 1 MB, 4 MB, 10 MB, 50 MB, 100 MB, 500 MB.

**Why:** The analysis assumes 22 GB/s H2D. Effective bandwidth varies with
transfer size — small transfers (activation results ~300 KB) may have much lower
effective bandwidth due to launch overhead.

**How:** Measure both H2D and D2H with pinned memory at each size.

### 0.4 Column-Split Correctness (quick validation)

**What:** Split a real Qwen2.5-7B layer's weight matrix, compute both halves,
concat, compare to unsplit result.

**Why:** Sanity check that column-parallel split produces bit-identical results.

**How:** Load a real layer, split WQKV/MLP columns, compute both paths, assert
`torch.allclose(atol=0)`.

### 0.5 CPU Attention Latency (gates Phase 2)

**What:** Measure `cpu_attention_with_kv_cache` latency at various (B, S)
combinations: B=4,8,16,32 × S=100,500,1000,2000.

**Why:** Determines the practical batch size ceiling for attention offloading.

**How:** Use `vllm/benchmarks/kernels/cpu/benchmark_cpu_attn.py` with target
configurations.

### 0.6 CUDA Graph Impact (informs Phase 4 priority)

**What:** Measure decode throughput and per-step latency with CUDA Graphs
enabled vs. disabled (`enforce_eager=True`) on 7B at various batch sizes.

**Why:** Quantifies the actual performance cost of disabling CUDA Graphs. If
the gap is small (e.g., <10%), Phase 4 is low priority and we can prototype
comfortably with `enforce_eager=True`. If the gap is large (>20%), Phase 4
becomes more urgent and we should consider building CUDA Graph compatibility
earlier.

**How:** Run `vllm bench latency` (or `benchmark_latency.py`) on 7B with:
- `--enforce-eager` vs. default (CUDA Graphs enabled)
- Batch sizes: B=1,4,8,16,32,64,128
- Measure: per-step decode latency (ms), throughput (tok/s)
- Also measure graph capture time to understand startup cost

### 0.7 KV Cache CPU Offload Impact (informs Phase 2)

**What:** Measure the performance impact of vLLM's existing KV cache offloading
to CPU. This tests how well the system handles KV data on CPU and gives a
baseline for attention offloading overhead.

**Why:** vLLM V1 has a KV offload subsystem (`vllm/v1/kv_offload/`). Testing
it reveals: (1) how much batch capacity increases when KV spills to CPU,
(2) the actual CPU↔GPU transfer overhead for KV data, and (3) whether the
existing infrastructure is usable as a building block for our suffix-on-CPU
design. If existing KV offload already gives significant batch capacity gains,
our attention offloading (Phase 2) builds on proven infrastructure. If it
causes severe slowdowns, we know the bottleneck to target.

**How:** Run 7B decode with KV offloading enabled at various configurations:
- Compare: KV offload disabled (baseline) vs. enabled
- Vary `gpu_memory_utilization` to force different amounts of KV spill
- Measure: throughput, decode latency, max concurrent sequences
- Monitor: PCIe transfer volume, KV cache hit/miss rates if available

### 0.8 vLLM V1 Baseline with FastTTS

**What:** Run FastTTS-thesis end-to-end on 7B (beam search, MATH-500 subset).
Record throughput, latency, accuracy.

**Why:** The V1 migration may have changed performance vs. the V0 numbers in
`vllm_benchmarking_findings.md`. Need a fresh baseline.

---

## Phase 1 — Collaborative Weight Offload

Mixed col/row tensor-granularity split across three sub-modules (WQKV, MLP1, MLP2), with uniform per-bucket dispatch and layer-ahead prefetch. WO is not offloaded in Phase 1 (fully GPU-resident — see `weight_offload_design.md §WO Split Axis Decision`). The phase is split into two sub-milestones: 1a ships the static compute path (no prefetch, `f_cpu` only) as an early checkpoint; 1b adds layer-ahead prefetch to complete the Planner's per-bucket dispatch story.

**Why a single phase with sub-milestones?** Sub-milestones 1a and 1b share the same mechanism: the `CpuComputeDispatcher`, the col/row split machinery, and the per-bucket dispatch table. 1b extends 1a's dispatch table with a second scalar (`f_prefetch`). Treating them as sub-milestones of one phase (instead of separate phases) reflects that engineering cohesion — no context-switch between "weight offload" and "something else" in the middle of building the mechanism.

### Phase 1a — Static Weight Offload (no prefetch)

CPU-resident weights are CPU-computed each forward; `f_prefetch = 0`. Ships as a standalone checkpoint that validates the split mechanism and CPU-compute path.

**Scope:**

- Extend `MergedColumnParallelLinear` (MLP1 / gate_up) for a col-parallel CPU slice on the output dim.
- Extend `RowParallelLinear` (MLP2 / down) for a row-parallel CPU slice on the input dim; assembly is `add_` (partial-sum reduce), not `concat`.
- Extend `QKVParallelLinear` (WQKV) for a col-parallel CPU slice with the K/V-biased column picker. Implement the K/V-pin guard inside `CpuComputeDispatcher` (see `weight_offload_design.md §Implementation Note`).
- WO is untouched — stays fully GPU-resident, no CPU path, no prefetch path.
- At model load: split each sub-module's weights into `W_gpu` and `W_cpu` along its assigned axis per the Planner's single `f_cpu_store` output (applied to WQKV/MLP1/MLP2 uniformly).
- At forward: CPU and GPU compute their slices in parallel; assembly is `concat` for col-parallel and `add_` for row-parallel. For the MLP block, SwiGLU runs locally on each device's intermediate slice between MLP1 and MLP2 — no intermediate transfer (matched-index invariant is automatic under uniform dispatch).

**Planner output at this sub-milestone:** `f_cpu_store` (load-time scalar) and `f_cpu` per bucket (with `f_prefetch = 0`). Dispatch table collapses to a 1-D bucket → `f_cpu` lookup.

### Phase 1b — Layer-Ahead Weight Prefetch

Add `f_prefetch` to the per-bucket dispatch: CPU-stored bytes not covered by `f_cpu` are streamed to GPU via the prefetch path during the previous layer's compute.

**Scope:**

- Add layer-ahead prefetch queue: one prefetch operation per layer boundary covering `Σ_m (f_prefetch × W_m)` across WQKV/MLP1/MLP2. One `cudaStreamWaitEvent` per layer.
- Extend `CpuComputeDispatcher` to accept the `f_prefetch` share — prefetched weights land in the circular buffer and are consumed by the layer's GEMMs from GPU memory. `f_cpu_compute + f_prefetch_compute = f_cpu_store` exactly.
- K/V-pin guard on WQKV (from 1a) ensures prefetch on WQKV only applies to Q columns above the K/V-bias boundary.
- Planner's per-bucket output becomes a single `(f_cpu, f_prefetch)` pair — see `planner_design.md §4.2` and §7.3.

**Expected result (7B at sub-milestone 1b):** layer-ahead prefetch enables meaningful `f_cpu_store` values without CPU-compute latency dominating at large batch. Free up GPU memory for larger KV pool without per-step latency regression at decode buckets where `f_prefetch` absorbs most of the offload.

### WQKV Column Choice (K/V-Biased)

CPU columns for WQKV are assigned in priority order: KV-head groups (K+V together per head) first, then Q heads. For Qwen2.5-7B GQA, K+V together is 22% of WQKV output; at `f_cpu_store = 22%`, the strict Q | K | V split emerges naturally. See `weight_offload_design.md` for the full rationale (volume savings + H2D contention avoidance) and the K/V-pin guard as an implementation detail.

**Relevance to Phase 2**: above the 22% K/V-biased boundary, the K/V-pin guard keeps all K/V columns CPU-computed — K/V output lands directly in the suffix cache, no D2H. Below 22%, K/V output for the GPU-resident portion still requires D2H to CPU cache each step; K/V-group bias only *reduces* this transfer vs. uniform, not eliminates it.

### Size Budget (Qwen2.5-7B, GQA: 28 Q heads, 4 KV heads, head_dim=128)

| Component | Output dim | Size (BF16) | % of WQKV |
|---|---|---|---|
| W_Q | 3584 | 25.6 MB | 78% |
| W_K | 512 | 3.7 MB | 11% |
| W_V | 512 | 3.7 MB | 11% |

K + V = 22% of WQKV; WQKV is 8.8% of the layer → at most ~2% of total layer weight from a pure K/V-on-CPU WQKV slice.

### CpuComputeDispatcher Abstraction

Design this from day 1 to make CUDA Graph retrofit (Phase 4) a localized change:

```python
class CpuComputeDispatcher:
    """Prototype: Python threading. Production: C++ CPUInfer."""
    def start(self, x_cpu, W_cpu, out_cpu): ...
    def wait(self) -> Tensor: ...
```

Prototype uses `ThreadPoolExecutor` + `enforce_eager=True`. Production swaps internals for KTransformers-style C++ `CPUInfer` + `cudaLaunchHostFunc`. Forward pass code doesn't change.

### What to Measure

**At 1a:** per-layer decode time (should be ≤ baseline); GPU memory freed; KV cache capacity increase; per-sub-module CPU-slice latency vs. GPU-slice budget (validates Planner's dispatch-table entries).

**At 1b:** layer-ahead prefetch critical-path behavior; buffer size vs. layer-time budget; per-bucket `(f_cpu, f_prefetch)` entries emitted by the Planner; throughput change on FastTTS workload.

---

## Phase 2 — Attention Offloading

Move suffix KV to CPU and compute suffix attention on CPU in parallel with GPU prefix attention. Independent of Phase 1's WQKV choice: if Phase 1's WQKV slice produces K/V on CPU, Phase 2 consumes them directly; otherwise a small CPU↔GPU transfer bridges the two phases.

### Prefill Handling

During prefill, K/V must go to the **GPU prefix** KV cache. Simplest approach: always compute K/V on whichever device holds those WQKV columns, transferring to GPU during prefill only. Prefill is compute-bound and processes many tokens — the extra transfer is negligible. During decode (latency-critical path), K/V stays on CPU.

### Engineering Gaps

1. **CPU attention must return LSE** (CRITICAL): Modify the C++ kernel `cpu_attention_with_kv_cache` to output per-head LSE alongside the attention output. The softmax denominator is already computed internally. Requires C++ kernel + Python binding changes.
2. **Separate block tables for GPU prefix / CPU suffix**: Two-pass attention with separate block tables (extends existing cascade pattern). GPU pass over prefix blocks, CPU pass over suffix blocks, merge via `merge_attn_states`.

### What to Measure

- CPU attention latency vs. GPU prefix attention time at target batch sizes
- End-to-end decode latency with attention offloading enabled
- Maximum batch size (beams) achievable vs. baseline
- FastTTS wall-clock TTC improvement from increased batch capacity

---

## Phase 3 — End-to-End Benchmarking

Full FastTTS runs on RTX 4090 with all offloading features.

### Configurations

| Config | Model | Offloading | Purpose |
|---|---|---|---|
| Baseline | 7B | None | Reference throughput |
| Static offload | 7B | Phase 1a (CPU-compute only, no prefetch) | Validate split mechanism + CPU-compute path |
| Full weight offload | 7B | Phase 1 (1a + 1b prefetch) | Exercise full weight-offload mechanism |
| Full offload | 7B | Phase 1 + 2 (attention) | Max batch capacity with attention offload |
| 14B minimal | 14B | Phase 1 + 2 | Demonstrate 14B feasibility |

### Metrics

- Throughput (tokens/sec, problems/hour)
- Per-step decode latency
- GPU memory utilization (weights vs KV cache)
- Accuracy on MATH-500 (should be unchanged)
- Comparison against baseline (FastTTS-AE + vLLM 0.9.2, V0)

---

## Phase 4 — CUDA Graph Integration (If Time Permits)

Replace the Python `CpuComputeDispatcher` prototype with C++ `CPUInfer` +
`cudaLaunchHostFunc`, enabling CUDA Graph compatibility.

### What to Port from KTransformers

| Component | Source | Lines | Notes |
|---|---|---|---|
| `CPUInfer` class | `kt-kernel/cpu_backend/cpuinfer.h` | ~80 | `submit_with_cuda_stream`, `sync_with_cuda_stream` |
| `TaskQueue` | `kt-kernel/cpu_backend/task_queue.{h,cpp}` | ~85 | Lock-free SPSC queue, direct port |
| Python bindings | `kt-kernel/ext_bindings.cpp` | ~30 | pybind11 wrapper |

### What We Skip

| Component | Why |
|---|---|
| `WorkerPool` (~500 lines) | MKL manages its own threads for BLAS |
| NUMA awareness (~200 lines) | Single-socket consumer system |
| MoE task wrappers (~300 lines) | Replace with single `cblas_gemm_bf16bf16f32` call |

### What We Write

| Component | Lines | Notes |
|---|---|---|
| CPU matmul task (C++) | ~50 | Wrapper calling MKL `cblas_gemm_bf16bf16f32` |
| Pre-allocated buffer manager | ~80 | Pinned I/O buffers indexed by `cuda_graph_idx` |
| Build integration | ~30 | CMake for the C++ extension |

### CUDA Graph Compatible Forward Pass

```
CUDA stream:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ D2H copy x │ submit │ x @ W_gpu (91%) │ sync │ H2D │ concat │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  ↓                         ↑
           TaskQueue ══ x @ W_cpu (9%) ═════╝
```

`submit` is a `cudaLaunchHostFunc` callback that enqueues to TaskQueue and
returns immediately. GPU proceeds to its matmul. `sync` is another callback
that blocks the stream until the CPU task completes. Between submit and sync,
GPU and CPU run in parallel — giving `max(GPU, CPU)` timing naturally.

**Total new/ported code:** ~400 lines for full CUDA Graph compatibility.

---

## Dependencies and Critical Path

```
Phase 0 (benchmarks)
  └─→ Phase 1 — Collaborative Weight Offload
        ├── 1a: Static col/row split (no prefetch)   ← Python-only, earliest checkpoint
        └── 1b: Layer-ahead prefetch                 ← adds f_prefetch on top of 1a
              │
              └─→ Phase 2: Attention Offload         ← requires C++ kernel (LSE)
                    │
                    └─→ Phase 3: End-to-End Benchmarking
                          │
                          └─→ Phase 4: CUDA Graphs (if time)
```

**Critical path:** Phase 0 → Phase 1a → Phase 1b → Phase 2 → Phase 3.

Phase 1a and 1b are sequential sub-milestones of the same mechanism (1b extends 1a's dispatch with `f_prefetch`). Phase 2 work lives in the attention backend + CPU attention kernel and is orthogonal to Phase 1's linear-layer code; if the CPU-attention LSE kernel is ready, Phase 2 can be developed in parallel with Phase 1b.

---

## Key Design Decisions

1. **`CpuComputeDispatcher` abstraction from day 1.** Prototype with Python
   threading + `enforce_eager=True`. Swap internals for C++ CPUInfer later.
   Forward pass code never changes.

2. **Mixed col/row split per TP convention (Phase 1).** WQKV and MLP1 are
   column-parallel (shard output dim); MLP2 is row-parallel (shard input dim,
   pairs with MLP1 to eliminate intermediate activation round-trip). WO is
   not offloaded. CPU plays the role of an additional TP rank along each
   sub-module's native shard axis.

3. **Uniform per-bucket dispatch across WQKV/MLP1/MLP2.** The Planner emits a
   single `(f_cpu, f_prefetch)` pair per bucket applied uniformly to the
   three offloaded sub-modules. Justified empirically by uniform CPU μs/MB
   at decode (`phase0_findings.md §0.2`). The MLP1↔MLP2 matched-index
   invariant is automatic under uniform dispatch.

4. **Layer-ahead prefetch (Phase 1b).** One prefetch queue per layer
   boundary, one sync per layer. Rejected tensor-ahead because its
   topological constraint would force per-sub-module f tuning to avoid
   prefetch starvation — needless given uniform CPU throughput and
   resolved PCIe contention (`phase0_findings.md §0.9`).

5. **K/V-biased WQKV column choice + K/V-pin guard.** Picker assigns CPU
   columns in priority order (K+V head pairs first, then Q tail). Runtime
   K/V-pin guard in `CpuComputeDispatcher` keeps K/V CPU-computed regardless
   of the uniform `f_cpu`, avoiding K/V round-trip. See
   `weight_offload_design.md §Implementation Note`.

6. **All PCIe to weight prefetch.** No KV prefetch. Suffix attention → CPU
   only. Batch size is the control variable for CPU attention bottleneck.

7. **Prototype with `enforce_eager=True`.** Defer CUDA Graph compatibility to
   Phase 4. Correctness and performance validation come first.

8. **Prefill: transfer K/V to GPU prefix cache once.** Prefill overhead is
   negligible (compute-bound, amortized over many tokens); decode keeps K/V
   on CPU for the suffix cache.
