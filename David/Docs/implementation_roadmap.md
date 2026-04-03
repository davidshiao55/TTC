# Implementation Roadmap

This document is the authoritative implementation plan for the thesis offloading
work. It supersedes the roadmap in `thesis_proposal.md` (Section 8.3) and the
implementation phases in `CLAUDE.md`.

**Prerequisites completed:**
- FastTTS migrated to vLLM V1 (see `vllm_v1_migration.md`)
- Both conda environments (`baseline`, `thesis`) working
- vLLM fork full CUDA build done

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

Representative shapes (7B, f_cpu=9%):
| Sub-module | GPU shape | CPU shape |
|---|---|---|
| WQKV | [B, 3584] × [3584, 4194] | [B, 3584] × [3584, 414] |
| WO | [B, 3584] × [3584, 3262] | [B, 3584] × [3584, 322] |
| MLP1 | [B, 3584] × [3584, 34478] | [B, 3584] × [3584, 3410] |
| MLP2 | [B, 18944] × [18944, 3262] | [B, 18944] × [18944, 322] |

And for the WQKV Q|K|V split (Phase 1b+2):
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

### 0.3 PCIe Effective Bandwidth (validates Phase 3 estimates)

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

### 0.5 CPU Attention Latency (gates Phase 1b+2)

**What:** Measure `cpu_attention_with_kv_cache` latency at various (B, S)
combinations: B=4,8,16,32 × S=100,500,1000,2000.

**Why:** Determines the practical batch size ceiling for attention offloading.

**How:** Use `vllm/benchmarks/kernels/cpu/benchmark_cpu_attn.py` with target
configurations.

### 0.6 CUDA Graph Impact (informs Phase 5 priority)

**What:** Measure decode throughput and per-step latency with CUDA Graphs
enabled vs. disabled (`enforce_eager=True`) on 7B at various batch sizes.

**Why:** Quantifies the actual performance cost of disabling CUDA Graphs. If
the gap is small (e.g., <10%), Phase 5 is low priority and we can prototype
comfortably with `enforce_eager=True`. If the gap is large (>20%), Phase 5
becomes more urgent and we should consider building CUDA Graph compatibility
earlier.

**How:** Run `vllm bench latency` (or `benchmark_latency.py`) on 7B with:
- `--enforce-eager` vs. default (CUDA Graphs enabled)
- Batch sizes: B=1,4,8,16,32,64,128
- Measure: per-step decode latency (ms), throughput (tok/s)
- Also measure graph capture time to understand startup cost

### 0.7 KV Cache CPU Offload Impact (informs Phase 1b+2)

**What:** Measure the performance impact of vLLM's existing KV cache offloading
to CPU. This tests how well the system handles KV data on CPU and gives a
baseline for attention offloading overhead.

**Why:** vLLM V1 has a KV offload subsystem (`vllm/v1/kv_offload/`). Testing
it reveals: (1) how much batch capacity increases when KV spills to CPU,
(2) the actual CPU↔GPU transfer overhead for KV data, and (3) whether the
existing infrastructure is usable as a building block for our suffix-on-CPU
design. If existing KV offload already gives significant batch capacity gains,
our attention offloading (Phase 1b+2) builds on proven infrastructure. If it
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

## Phase 1a — Resident Hybrid (WO, MLP1, MLP2)

Column-parallel split on all sub-modules **except** WQKV. This validates the
core mechanics without attention coupling.

### Scope

- Modify `MergedColumnParallelLinear` (gate_up / MLP1) and
  `RowParallelLinear` (down / MLP2) in `vllm/model_executor/layers/linear.py`
- Modify `QKVParallelLinear` for WO projection
- At model load: split weight columns into W_gpu (91%) and W_cpu (9%)
- At forward: CPU computes `x @ W_cpu` in parallel with GPU `x @ W_gpu`, concat

### CpuComputeDispatcher Abstraction

Design this from day 1 to make CUDA Graph retrofit (Phase 5) a localized change:

```python
class CpuComputeDispatcher:
    """Prototype: Python threading. Production: C++ CPUInfer."""
    def start(self, x_cpu, W_cpu, out_cpu): ...
    def wait(self) -> Tensor: ...
```

Prototype uses `ThreadPoolExecutor` + `enforce_eager=True`.
Production swaps internals for KTransformers-style C++ `CPUInfer` +
`cudaLaunchHostFunc`. Forward pass code doesn't change.

### Expected Result (7B)

~1.2 GB freed across 28 layers with near-zero latency cost. Validate by
measuring KV cache capacity before/after.

### What to Measure

- Per-layer decode time (should be ≤ baseline)
- GPU memory freed (target: ~1.2 GB)
- KV cache capacity increase
- Throughput change on FastTTS workload

---

## Phase 1b+2 — WQKV Split + Attention Offloading (Coupled)

WQKV is split along the **Q|K|V dimension**, not arbitrary columns:
- **Q → GPU**: needed for GPU prefix attention
- **K, V → CPU**: new K/V go directly into CPU suffix KV cache

This couples weight splitting and attention offloading by design. New K/V values
are produced where they're consumed — no GPU→CPU transfer for KV cache
population during decode.

### Data Flow (Decode)

```
GPU                                    CPU
 │                                      │
 x ── x @ W_Q ──→ q                    │
 │                                      │
 x ── copy (async) ──────────→ x @ W_K ──→ k ──→ suffix KV cache
 │                              x @ W_V ──→ v ──→ suffix KV cache
 │                                      │
 q @ K_prefix ──→ out_gpu, lse_gpu      │
 │                    q (small) ──→ q @ K_suffix ──→ out_cpu, lse_cpu
 │                                      │
 merge(out_gpu, lse_gpu, out_cpu, lse_cpu)
```

### Size Budget (Qwen2.5-7B)

| Component | Output dim | Size (BF16) | % of WQKV | Device |
|---|---|---|---|---|
| W_Q | 3584 → 3584 | 25.6 MB | 78% | GPU |
| W_K | 3584 → 512 | 3.7 MB | 11% | CPU |
| W_V | 3584 → 512 | 3.7 MB | 11% | CPU |

K+V = 22% of WQKV, but WQKV is 8.8% of the layer → **~2% of total layer
weight on CPU** from the QKV split. Well within f_cpu budget; leaves ~7% for
WO/MLP (from Phase 1a).

### Prefill Handling

During prefill, K/V must go to the **GPU prefix** KV cache. Approach: always
compute K/V on CPU, transfer to GPU during prefill only. Prefill is
compute-bound and processes many tokens — the extra CPU→GPU transfer is
negligible. During decode (the latency-critical path), K/V stays on CPU.

### Engineering Gaps

1. **CPU attention must return LSE** (CRITICAL): Modify the C++ kernel
   `cpu_attention_with_kv_cache` to output per-head LSE alongside attention
   output. The softmax denominator is already computed internally. Requires C++
   kernel + Python binding changes.

2. **Separate block tables for GPU prefix / CPU suffix**: Two-pass attention
   with separate block tables (extends existing cascade pattern). GPU pass over
   prefix blocks, CPU pass over suffix blocks, merge via `merge_attn_states`.

### What to Measure

- CPU attention latency vs. GPU prefix attention time at target batch sizes
- End-to-end decode latency with attention offloading enabled
- Maximum batch size (beams) achievable vs. baseline
- FastTTS wall-clock TTC improvement from increased batch capacity

---

## Phase 3 — Tensor-Granularity Weight Prefetch

Per-sub-module three-way split: `W = [W_gpu_permanent | W_gpu_prefetched | W_cpu]`.
All PCIe H2D bandwidth dedicated to weight prefetch (no KV prefetch).

### When This Matters

- **7B (fits on GPU):** Elective offloading — prefetch replaces f_gpu to free
  more memory for KV cache, at the cost of per-step latency.
- **14B+ (doesn't fit):** Mandatory offloading — prefetch replaces f_cpu to
  reduce latency on offloaded layers.

### Sub-Layer Pipeline

Each sub-module has independently tuned f_gpu/f_prefetch/f_cpu. The prefetch
for sub-module N+1 runs during sub-module N's compute:

```
GPU:   WQKV compute → Attn → WO compute → MLP1 compute → MLP2 compute
PCIe:  (KV/idle)    → WO   → MLP1       → MLP2         → WQKV(next)
CPU:   WQKV(K+V)    → Attn → WO         → MLP1         → MLP2
```

### Prefetch Distance Options

| Distance | Buffer size | MLP1 bottleneck? |
|---|---|---|
| Tensor-ahead | max(f_prefetch_i × W_i) — smallest | Yes (WO phase too short) |
| Layer-ahead | sum(f_prefetch_i × W_i) | No (global budget) |

Decision deferred to implementation — benchmark both.

### Implementation

Extend vLLM's `PrefetchOffloader` pattern (`StaticBufferPool`, async copy stream
+ `cudaStreamWaitEvent`) to operate at tensor granularity instead of layer
granularity.

---

## Phase 4 — End-to-End Benchmarking

Full FastTTS runs on RTX 4090 with all offloading features.

### Configurations

| Config | Model | Offloading | Purpose |
|---|---|---|---|
| Baseline | 7B | None | Reference throughput |
| Hybrid-only | 7B | Phase 1a (WO/MLP split) | Validate "free" 1.2 GB claim |
| Full offload | 7B | Phase 1a + 1b+2 (+ attention) | Max batch capacity |
| Prefetch | 7B | Phase 1a + 1b+2 + 3 (elective) | Latency-throughput tradeoff |
| 14B minimal | 14B | Phase 1a + 1b+2 + 3 (mandatory) | Demonstrate 14B feasibility |

### Metrics

- Throughput (tokens/sec, problems/hour)
- Per-step decode latency
- GPU memory utilization (weights vs KV cache)
- Accuracy on MATH-500 (should be unchanged)
- Comparison against baseline (FastTTS-AE + vLLM 0.9.2, V0)

---

## Phase 5 — CUDA Graph Integration (If Time Permits)

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
  ├─→ Phase 1a (WO/MLP split)  ← Python-only, fastest win
  │     └─→ Phase 3 (prefetch)  ← builds on column splits
  └─→ Phase 1b+2 (WQKV + attention)  ← requires C++ kernel (LSE)
        └─→ Phase 3 (prefetch)

Phase 1a + 1b+2 + 3 → Phase 4 (benchmarking)
Phase 4 → Phase 5 (CUDA Graphs, if time)
```

**Critical path:** Phase 0 → Phase 1a → Phase 1b+2 → Phase 4

Phase 1a and Phase 1b+2 can be developed in parallel after Phase 0, since
Phase 1a touches WO/MLP and Phase 1b+2 touches WQKV/attention — different
sub-modules.

---

## Key Design Decisions

1. **`CpuComputeDispatcher` abstraction from day 1.** Prototype with Python
   threading + `enforce_eager=True`. Swap internals for C++ CPUInfer later.
   Forward pass code never changes.

2. **WQKV split along Q|K|V dimension.** Q on GPU (prefix attention), K/V on
   CPU (suffix KV cache). Eliminates GPU→CPU KV transfer during decode.

3. **All PCIe to weight prefetch.** No KV prefetch. Suffix attention → CPU
   only. Batch size is the control variable for CPU attention bottleneck.

4. **Prototype with `enforce_eager=True`.** Defer CUDA Graph compatibility to
   Phase 5. Correctness and performance validation come first.

5. **Prefill: always compute K/V on CPU, transfer to GPU.** One code path.
   Prefill overhead is negligible (compute-bound, amortized over many tokens).
