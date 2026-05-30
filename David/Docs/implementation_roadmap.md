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
| **0** — Benchmarks | First profile run (schema validation) + vLLM native offloader baselines (§0.10) | — | — | — |
| **1** — Collaborative Weight Offload | + CPU GEMM curves + PCIe BW curve | Per-bucket `(f_cpu, f_prefetch)` dispatch (1a: `f_cpu` only; 1b: adds `f_prefetch`) | — | Weight storage; CPU-compute + layer-ahead prefetch dispatch paths (WQKV, MLP1, MLP2; WO not offloaded) |
| **1c** — Native CPU Runner | — | — | `cudaLaunchHostFunc` + native CPU worker; bucket-aware thread policy | Graph-compatible dispatch; eliminates Python `executor.submit` / `future.result` host critical path (`phase1a_findings.md §1.14`) |
| **2** — Attention Offloading | + CPU attention curve | Planner gains `KV_gpu_bytes` / `KV_cpu_bytes` variables | Tier-aware KV admission | KV two-pool storage; CPU suffix attention |
| **3** — E2E | — | — | — | All mechanisms active |

Phases deliver the system incrementally: each extends Profiler coverage, Planner variable set, or Scheduler capabilities while preserving the three-component architecture.

---

## Phase 0 — Pre-Implementation Benchmarking

Validate the quantitative assumptions before writing any offloading code. If any
of these numbers are significantly off, the approach changes.

Section numbering aligns with `phase0_findings.md`. Each entry below is a forward-looking gate (what to measure, why it matters); detailed numbers and analysis live in the corresponding findings section.

### 0.1 Dispatch axis — num_tokens unification

**What:** Validate that GPU layer time is governed by `num_tokens` alone (collapses uniform-decode and mixed-prefill-decode onto one axis).

**Why:** The Planner's dispatch table is keyed on `BatchDescriptor` whose discriminating axis is `num_tokens`. If layer time depends meaningfully on the prefill/decode mix at fixed `num_tokens`, the dispatch design needs an extra axis.

**How:** `bench_num_tokens_axis.py`. See `phase0_findings.md §0.1`.

### 0.2 Split mechanism correctness (CRITICAL — gates Phase 1)

**What:** Split a real Qwen2.5-7B layer along col-parallel (WQKV, MLP1) or row-parallel (MLP2) axes, compute partial paths, assemble, compare to unsplit reference.

**Why:** Sanity check that mixed col/row partitioning produces bit-identical results. Includes the K/V-biased WQKV picker.

**How:** `bench_split_correctness.py`. See `phase0_findings.md §0.2`.

### 0.3 CPU/GPU compute characterization (CRITICAL — gates Phase 1)

**What:** Per-sub-module GPU layer-time, CPU GEMM curve, and reduced-GPU timing (when slice f_cpu of weights is on CPU). Quantifies the µs/MB curves the Planner uses and the bucket-specific microbench-free `f_cpu` (cold-cache numbers in `phase0_findings.md §0.3.3` show f_cpu ≈ 5% B=1, ≈ 3% B=4, no free regime at B≥8 on this hardware). Note that microbench-free is not e2e-free under the Phase 1a runtime — see `phase1a_findings.md §1.14`.

**Why:** Phase 1's "free" argument assumes CPU finishes within GPU compute window. If CPU µs/MB is too slow, f_cpu drops and memory savings shrink. Also validates uniform CPU µs/MB across sub-modules at decode B≥16 — the empirical foundation for the Planner emitting a single uniform `(f_cpu, f_prefetch)` per bucket.

**How:** `bench_cpu_gpu_overlap.py`. See `phase0_findings.md §0.3` (subsections 0.3.1 GPU baseline, 0.3.2 CPU compute path, 0.3.3 per-sub-module overlap, 0.3.4 CPU µs/MB uniformity).

### 0.4 Per-sub-module split-axis design

**What:** Empirical wall-clock comparison of MLP1→MLP2 pipelining under uniform col vs col→row splits, plus WO offload Alt A vs Alt B decision.

**Why:** Pins down (i) which axis each sub-module should split along, and (ii) whether WO is offloaded at all in Phase 1. Both decisions are inputs to the `CpuComputeDispatcher` design.

**How:** `bench_mlp_pipeline.py`, `bench_wo_offload_tradeoff.py`. See `phase0_findings.md §0.4` (0.4.1 MLP pipelining, 0.4.2 WO Alt A vs Alt B, 0.4.3 Phase 1/2 commitments).

### 0.5 PCIe behavior — bandwidth, contention, UVA bypass (validates Phase 1b prefetch)

**What:** PCIe effective bandwidth across transfer sizes; same-direction H2D contention on CE0; bidirectional H2D + D2H; DMA vs SM-issued UVA copy under contention.

**Why:** Phase 1b's prefetch budget (`layer_time × PCIe_BW`) and Phase 2's spill writes (D2H) require knowing both link bandwidth and the engine-level scheduling behavior. The DMA vs UVA distinction lets fg activation returns bypass bg weight prefetch (different PCIe paths).

**How:** `bench_pcie_sweep.py`, `bench_contention.py`, `probe_uva_bypass.py`. See `phase0_findings.md §0.5`.

### 0.6 CPU attention latency (gates Phase 2)

**What:** Measure `cpu_attention_with_kv_cache` latency at representative (B, S) combinations.

**Why:** Determines the practical batch size ceiling for suffix attention offloading. CPU attention is an intrinsically B-bound operation; this measurement bounds the regime where Phase 2 keeps pace with GPU.

**How:** Use `vllm/benchmarks/kernels/cpu/benchmark_cpu_attn.py`. See `phase0_findings.md §0.6`.

### 0.7 CUDA graph impact (informs Phase 1c scope)

**What:** Decode throughput and per-step latency with CUDA Graphs enabled vs. `--enforce-eager` on 7B across batch sizes.

**Why:** Quantifies the cost of prototyping Phase 1a/1b with eager mode. The §0.7 measurement found the eager-vs-graph gap is small for the thesis workload (~2%), so the original plan deferred CUDA Graph work to a tail phase. After §1.14 of `phase1a_findings.md` showed the Python `CpuTaskRunner` substrate is the dominant Phase 1a overhead (and that the native runner port is the precondition for graph capture anyway), the work moved into Phase 1c — graph compatibility falls out for free once the native dispatcher lands.

**How:** `vllm bench latency` with and without `--enforce-eager`. See `phase0_findings.md §0.7`.

### 0.8 vLLM V1 FastTTS baseline

**What:** End-to-end FastTTS-thesis run on 7B (beam search, MATH-500 subset). Record throughput, latency, accuracy.

**Why:** The V1 migration may have changed performance vs. the V0 numbers in `vllm_benchmarking_findings.md`. Need a fresh reference for the headline COTS comparison.

**How:** `bench_kv_offload.py` (baseline arm). See `phase0_findings.md §0.8`.

### 0.9 vLLM V1 KV offload — impact (informs Phase 2)

**What:** A/B test of vLLM V1's built-in CPU KV offload on the FastTTS workload (`vllm/v1/kv_offload/`). Measures prefix-reuse hit rate and end-to-end latency/goodput delta.

**Why:** V1 KV offload is a *prefill-side* prefix-reuse mechanism (orthogonal to Phase 2's *decode-time* suffix attention). Two questions: (i) does it help meaningfully on the FastTTS workload, (ii) is it reusable as Phase 2 infrastructure (CPU block pool, allocator).

**How:** `bench_kv_offload.py`. See `phase0_findings.md §0.9`.

### 0.10 vLLM native weight offloader baseline (Prefetch + UVA)

**What:** Decode-step latency under vLLM's stock weight offloaders on 7B BF16. Six sub-experiments:
- **Stock Native Prefetch Mechanics** (§0.10.1): picker math + wraparound H2D theory; identifies the two sync surfaces (eager-mode CE0 contention, graph-mode `sync_prev_onload` drain).
- **Stock Native Parameter Search** (§0.10.2): broad `(G, N)` sweep at K=1 finds the best stock config per offloaded-layer count. Headline: at low offload depth, non-divisor G with N=1 (placement-aware, avoiding layer 27) wins by 5–20% over canonical divisor-G.
- **Canonical knob sweep** (§0.10.3): G ∈ {2, 4, 14, 28} (divisors of 28) at fixed 50% coverage, N at fixed G=14, K ∈ {1, 2, 3, 4} at fixed (G=14, N=4), and a densest-spread sub-sweep (N=1, varying G). Historical reference; superseded as "best stock" by §0.10.2.
- **Head-to-head** (§0.10.4): Prefetch vs UVA at matched offloaded GiB across batch ∈ {1, 16, 64}, using prefetch's best config from §0.10.2.
- **PrefetchDefer ablation** (§0.10.5): defer fix as an optimization on top of stock-best. Includes the 3-arm `none / dryrun / real` decomposition, the eager/graph mechanism analysis, and the 2x2 isolation experiment showing the two sync surfaces are mutually-occluding manifestations of one wraparound.
- **nsys overlap probe** (§0.10.6): visual confirmation of H2D ↔ compute overlap on the best prefetch config.

**Why:** The COTS thesis claims to outperform stock vLLM offloading. Without on-hardware baselines for both stock options, "COTS is faster" has no anchor point. Three empirical findings also feed back into the COTS design:
1. **At fixed offload bytes, placement vs spacing depends on depth** — at low L, placement-aware non-divisor G wins; at mid L, dense uniform divisor-G wins. COTS's tensor-granularity approach is the limit case.
2. **K=1 (layer-ahead) within ≤3% of optimal K**, K=4 OOMs at B=64 — empirically validates the layer-ahead commitment in `pcie_bandwidth_allocation_design.md`.
3. **The defer fix recovers ~22% on top of stock-best at low L** (§0.10.5) — anchors the cots-vs-prefetch competition. The optimized-native baseline (`prefetch_defer`) is within ~3.5% of `none` at L=1; meaningful COTS headroom exists only at mid/high depth.

**How:** `bench_prefetch_full_sweep.py` + `bench_prefetch_knobs.py` + `bench_uva_vs_prefetch.py` + `bench_prefetch_vs_none.py` + `probe_native_offload_overlap.py`. See `phase0_findings.md §0.10`.

---

## Phase 1 — Collaborative Weight Offload

Mixed col/row tensor-granularity split across three sub-modules (WQKV, MLP1, MLP2), with uniform per-bucket dispatch and layer-ahead prefetch. WO is not offloaded in Phase 1 (fully GPU-resident — see `weight_offload_design.md §WO Split Axis Decision`). The phase is split into two sub-milestones: 1a ships the static compute path (no prefetch, `f_cpu` only) as an early checkpoint; 1b adds layer-ahead prefetch to complete the Planner's per-bucket dispatch story.

**Why a single phase with sub-milestones?** Sub-milestones 1a and 1b share the same mechanism: the `CpuComputeDispatcher`, the col/row split machinery, and the per-bucket dispatch table. 1b extends 1a's dispatch table with a second scalar (`f_prefetch`). Treating them as sub-milestones of one phase (instead of separate phases) reflects that engineering cohesion — no context-switch between "weight offload" and "something else" in the middle of building the mechanism.

### Phase 1a — Static Weight Offload (no prefetch)

CPU-resident weights are CPU-computed each forward; `f_prefetch = 0`. Ships as a standalone checkpoint that validates the split mechanism and CPU-compute path.

**Scope:**

- Extend `MergedColumnParallelLinear` (MLP1 / gate_up) for a col-parallel CPU slice on the output dim.
- Extend `RowParallelLinear` (MLP2 / down) for a row-parallel CPU slice on the input dim; assembly is `add_` (partial-sum reduce), not `concat`.
- Extend `QKVParallelLinear` (WQKV) for a col-parallel CPU slice with the K/V-biased column picker.
- WO is untouched — stays fully GPU-resident, no CPU path, no prefetch path.
- At model load: split each sub-module's weights into `W_gpu` and `W_cpu` along its assigned axis per the Planner's single `f_cpu_store` output (applied to WQKV/MLP1/MLP2 uniformly).
- At forward: CPU and GPU compute their slices in parallel; assembly is `concat` for col-parallel and `add_` for row-parallel. For the MLP block, SwiGLU runs locally on each device's intermediate slice between MLP1 and MLP2 — no intermediate transfer (matched-index invariant is automatic under uniform dispatch).
- **Activation return path: SM-issued UVA copy kernel** — CPU's GEMM output lands in pinned memory; a one-shot Triton kernel reads it via UVA mapping and writes to a GPU buffer for the assembly step. This bypasses CE0 (the H2D copy engine), so fg returns don't queue behind bg weight prefetches once Phase 1b ships. fg_s2c stays at ~30 μs/event regardless of bg state (`phase0_findings.md §0.5`). Downstream consumers always read from the GPU buffer, never from UVA — the one-shot access pattern is a load-bearing contract.

**Planner output at this sub-milestone:** `f_cpu_store` (load-time scalar) and `f_cpu` per bucket (with `f_prefetch = 0`). Dispatch table collapses to a 1-D bucket → `f_cpu` lookup.

### Phase 1b — Layer-Ahead Weight Prefetch

Add `f_prefetch` to the per-bucket dispatch: CPU-stored bytes not covered by `f_cpu` are streamed to GPU via the prefetch path during the previous layer's compute.

**Scope:**

- Add layer-ahead prefetch queue: one `cudaMemcpyAsync` (on CE0, the H2D copy engine) per layer boundary covering `Σ_m (f_prefetch × W_m)` across WQKV/MLP1/MLP2. One `cudaStreamWaitEvent` per layer. Bg prefetch and fg activation return use different PCIe paths (CE0 vs SM-issued UVA), so they share link BW but don't serialize on each other (`phase0_findings.md §0.5`).
- Extend `CpuComputeDispatcher` to accept the `f_prefetch` share — prefetched weights land in the circular buffer and are consumed by the layer's GEMMs from GPU memory. `f_cpu_compute + f_prefetch_compute = f_cpu_store` exactly.
- Planner's per-bucket output becomes a single `(f_cpu, f_prefetch)` pair — see `planner_design.md §4.2` and §7.3.

**Expected result (7B at sub-milestone 1b):** layer-ahead prefetch enables meaningful `f_cpu_store` values without CPU-compute latency dominating at large batch. Free up GPU memory for larger KV pool without per-step latency regression at decode buckets where `f_prefetch` absorbs most of the offload.

### WQKV Column Choice (K/V-Biased)

CPU columns for WQKV are assigned in priority order: KV-head groups (K+V together per head) first, then Q heads. For Qwen2.5-7B GQA, K+V together is 22% of WQKV output; at `f_cpu_store = 22%`, the strict Q | K | V split emerges naturally. See `weight_offload_design.md` for the full rationale (volume savings + H2D contention avoidance).

**Relevance to Phase 2**: at `f_cpu_compute ≥ 22%` all K/V columns are CPU-computed by the dispatch table — K/V output lands directly in the suffix cache, no D2H. At `f_cpu_compute < 22%`, K/V output for the GPU-resident / prefetched portion still requires D2H to CPU cache each step. The Planner's cost model (`planner_design.md §7.3`) charges this round-trip and biases toward `f_cpu_compute ≥ K/V fraction` when the budget allows.

### Size Budget (Qwen2.5-7B, GQA: 28 Q heads, 4 KV heads, head_dim=128)

| Component | Output dim | Size (BF16) | % of WQKV |
|---|---|---|---|
| W_Q | 3584 | 25.6 MB | 78% |
| W_K | 512 | 3.7 MB | 11% |
| W_V | 512 | 3.7 MB | 11% |

K + V = 22% of WQKV; WQKV is 8.8% of the layer → at most ~2% of total layer weight from a pure K/V-on-CPU WQKV slice.

### CpuComputeDispatcher Abstraction

Design this from day 1 to make CUDA Graph retrofit (Phase 1c) a localized change:

```python
class CpuComputeDispatcher:
    """Prototype: Python threading. Production: C++ CPUInfer."""
    def start(self, x_cpu, W_cpu, out_cpu): ...
    def wait(self) -> Tensor: ...
```

Prototype uses `ThreadPoolExecutor` + `enforce_eager=True`. Production swaps internals for KTransformers-style C++ `CPUInfer` + `cudaLaunchHostFunc`. Forward pass code doesn't change.

### What to Measure

**At 1a:** Phase 1a is structural and partial throughput, not "free overlap". Gates are: (i) split-mechanism correctness (token-level parity with baseline at the smoke prompt, see `phase1a_findings.md §1.8`), (ii) GPU memory freed and KV-cache capacity increase (§1.7), (iii) head-to-head against vLLM's PrefetchOffloader at matched offload depth (§1.13), (iv) attribution of any e2e gap vs `none` into orchestration vs active CPU-work penalty (§1.14). The "≤ baseline" gate is deferred to Phase 1c — `phase1a_findings.md §1.14` shows the Python `CpuTaskRunner` substrate makes that gate unattainable at any non-trivial f_cpu in Phase 1a; it becomes the right gate once the native runner lands.

**At 1b:** layer-ahead prefetch critical-path behavior; buffer size vs. layer-time budget; per-bucket `(f_cpu, f_prefetch)` entries emitted by the Planner; throughput change on FastTTS workload.

---

## Phase 1c — Native CPU Runner [LANDED]

**Status:** Stages 1-5 landed; Stage 6 (this section + `phase1c_findings.md`)
documents the implementation and locks the gates. Stage 7 (transposed-storage
unification) is optional and deferred. See `phase1c_findings.md` for the full
design + measured numbers.

Replace the Python `CpuTaskRunner` (Python `ThreadPoolExecutor` + `future.result`)
with a `cudaLaunchHostFunc`-based handoff backed by a native CPU worker. Phase
1c is what was previously called Phase 4: the work is the same, but the
sequencing is moved up before Phase 2 because Phase 1a's postmortem
(`phase1a_findings.md §1.14`) showed the host critical path — not CPU GEMM
throughput — is the dominant overhead at the f=0.05 B=1 free-regime cell. Any
Phase 2 (attention offload) measurement built on the Python prototype would
mix the runtime gap into the attention numbers.

### Why Phase 1c precedes Phase 2

`phase1a_findings.md §1.14` measured the COTS-vs-none gap at f=0.05 B=1
(decode-heavy, input=8, output=128) by inserting a `--cots-dry-run` mode
that installs all wrappers but skips the worker GEMM. The decomposition (see
§1.14 for the full t × B table):

- **Pure host orchestration** (`dryrun − none`) — what `cudaLaunchHostFunc` +
  CUDA Graph capture eliminate at runtime. Roughly flat in `t` and `B`.
- **Active CPU-work penalty** (`real − dryrun`) — extra wall-clock from
  enabling real CPU GEMM. Includes the GEMM time itself, oneDNN-vs-main-
  thread interference, reduced CUDA launch runahead, and any cache/BW
  contention. Upper bound on the post-1c floor; strongly `t`-dependent
  because oneDNN-on-many-threads contends with the main thread's CUDA
  dispatch path. Bucket-aware thread policy is what shrinks this.

Both contributions are addressed by Phase 1c (one in the dispatcher port,
the other in the thread policy). Building Phase 2 on the Python prototype
would conflate the runtime gap with the attention numbers.

### What to Port from KTransformers

| Component | Source | Lines | Notes |
|---|---|---|---|
| `CPUInfer` class | `kt-kernel/cpu_backend/cpuinfer.h` | ~80 | `submit_with_cuda_stream`, `sync_with_cuda_stream` |
| `TaskQueue` | `kt-kernel/cpu_backend/task_queue.{h,cpp}` | ~85 | Michael-Scott MPSC queue + condvar sync; one worker thread; direct port |
| Python bindings | `kt-kernel/ext_bindings.cpp` | ~30 | pybind11 wrapper |

### What We Skip

| Component | Why |
|---|---|
| `WorkerPool` (~500 lines) | oneDNN manages its own threads for BF16 GEMM |
| NUMA awareness (~200 lines) | Single-socket consumer system |
| MoE task wrappers (~300 lines) | Replaced with `at::linear` per-op (qkv / mlp_block / dryrun_noop) |

### What We Wrote (as landed)

| Component | LOC (actual) | Notes |
|---|---|---|
| CPU matmul task (C++ slab dispatcher) | ~210 (`cots_cpu_infer.cpp`) | `at::linear` (NOT `cblas_gemm_bf16bf16f32` — phase0 §0.3.2 shows oneDNN BF16 hits the AVX2 fast path while MKL `cblas_gemm_bf16bf16f32` requires AVX512_BF16 and falls to the scalar path on i9-14900KF; Stage 1 hard-gate microbench confirmed parity with Python `F.linear`) |
| Slab pool + worker-local scratch | included above | `unique_ptr<TaskSlab[]>`, NOT `std::vector` (`std::atomic<int32_t>` makes TaskSlab non-MoveConstructible). One shared `scratch_silu_up_` tensor sized at install. |
| Custom-op registration (Python) | ~120 (`cots_ops.py`) | `vllm.cots_submit_gemm` and `vllm.cots_sync_then_uva`; barrier-installing `mutates_args` declarations + module-private weak runner registry for multi-engine safety |
| Build integration | ~30 (`CMakeLists.txt`) | `_cots_C` extension gated on `VLLM_GPU_LANG STREQUAL "CUDA"`; LANGUAGE CXX (no CUDA kernels — only Runtime API for `cudaLaunchHostFunc`); standard Python ABI; `torch_python` linked for `at::Tensor` pybind type caster |
| Bucket-aware thread policy | ~80 (in `cots.py` `_n_threads_for` + `_validate_thread_policy`) | Per-`BatchDescriptor` `cpu_num_threads` via slab field; cache-guarded worker-side `at::set_num_threads`; main-thread `at::get_num_threads` isolation confirmed (omp pragma contingency NOT needed) |
| Worker affinity | ~40 (in `cots_cpu_infer.cpp::set_worker_affinity`) | `uint64_t` mask intersected with `sched_getaffinity`; warns-and-skips on empty intersection |

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

CUDA Graph compatibility is a free side effect: replacing the Python dispatcher
with `cudaLaunchHostFunc` callbacks is the precondition for capturing the
forward pass into a graph. `enforce_eager=True` (Phase 1a/1b prototype mode) is
no longer required.

### Stage 1-5 Gates [as landed]

Every stage gated on green tests + a measured invariant before the next started:

| Stage | Gate | Result |
|---|---|---|
| 1 | C++ `at::linear` parity vs Python `F.linear` (BF16, B ∈ {1, 4, 16, 32}, contiguous + strided down-proj) | All within 5%; oneDNN BF16 fast path on AVX2; no scalar-fallback regression |
| 2 | Substrate: native runner round-trip ≤ python runner round-trip (eager, dry_run=True) | ratio 0.974 (native is 3% faster) |
| 3 | Native-vs-python parity: bit-equivalent QKV + MLP outputs at f_prefetch ∈ {0, 0.10, 0.15, 0.20, 0.25} | All parity assertions pass; FX-positional ordering preserved |
| 4 | Bucket-aware n_threads observed by worker; main-thread `at::get_num_threads` isolation | All assertions pass; risk #3 GREEN (omp pragma contingency unused) |
| 5 | CUDA graph capture + 50× replay deterministic + parity; collapse ratio capture/eager < 0.7; FX-positional submit < GEMMs < sync via `torch._dynamo.export` | Collapse ratio 0.477 PASS; FX ordering proven; risk #4 GREEN |

§1.14 ABSOLUTE on Qwen2.5-7B + FastTTS is now closed by the final
Phase 1c production validation:

- **Default Phase 1 weight path:** `cpu_runner="native"`,
  `auto_graph_split=True`, piecewise graph split at the COTS weight
  submit/sync custom ops, and `weight_capture_sync_mode="wait_kernel"`
  in graph mode.
- **Retained validation harnesses:**
  `bench_capture_gap_qwen.py`, `bench_capture_gap_qwen_grid.py`, and
  `check_capture_piecewise_parity_qwen.py`.
- **Resolved blockers:** `_bucket_for` is Dynamo-traceable, dispatch
  bucket/task selection is published out of graph through
  `CotsOffloader.on_dispatch`, and native slabs cap CPU work to live
  token rows rather than graph-bucket capacity.
- **Removed probes:** the dryrun-native bring-up harnesses, fused
  wait+UVA prototype, and ablation-only tests/results were deleted.
  Their causal conclusions are preserved in `phase1c_findings.md`.

See `David/Docs/phase1c_findings.md` and
`David/Benchmarks/phase1c/results/phase1c_final_summary.json` for the
production numbers and cleanup policy.

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
| Baseline (no offload) | 7B | None | Reference throughput |
| **Native prefetch baseline** | 7B | Stock vLLM `PrefetchOffloader`, best config from `phase0_findings.md §0.10.1` (knob sweep) anchored against `none` via `phase0_findings.md §0.10.3` (free-regime decomposition) | Headline native baseline COTS competes against |
| Static offload | 7B | Phase 1a (CPU-compute only, no prefetch) | Validate split mechanism + CPU-compute path |
| Full weight offload | 7B | Phase 1 (1a + 1b prefetch) | Exercise full weight-offload mechanism |
| Full offload | 7B | Phase 1 + 2 (attention) | Max batch capacity with attention offload |
| 14B minimal | 14B | Phase 1 + 2 | Demonstrate 14B feasibility |

### Metrics

- Throughput (tokens/sec, problems/hour)
- Per-step decode latency
- GPU memory utilization (weights vs KV cache)
- Accuracy on MATH-500 (should be unchanged)
- Comparison against the FastTTS V0 reference (FastTTS-AE + vLLM 0.9.2)
- **Comparison against vLLM native prefetch baseline** at each (model, batch, offload depth) point — speedup ratio at matched depth is the headline COTS metric. Use the best `prefetch_*` config from `phase0_findings.md §0.10.1` (knob sweep, post-restructure), not the default `prefetch_8x2`. Anchor every (cots, prefetch) pair against `none` using `phase0_findings.md §0.10.3` so regressions are attributed to the right column.

---

## Dependencies and Critical Path

```
Phase 0 (benchmarks)
  └─→ Phase 1 — Collaborative Weight Offload
        ├── 1a: Static col/row split (no prefetch)   ← Python prototype, [DONE]
        ├── 1b: Layer-ahead prefetch                 ← adds f_prefetch on top of 1a
        └── 1c: Native CPU runner (was Phase 4)       ← removes Python dispatcher tax
              │                                          + bucket-aware thread policy
              │                                          + CUDA Graph compatibility
              │
              └─→ Phase 2: Attention Offload         ← requires C++ kernel (LSE);
                    │                                    runs on the 1c substrate
                    │
                    └─→ Phase 3: End-to-End Benchmarking
```

**Critical path:** Phase 0 → Phase 1a → Phase 1b → Phase 1c → Phase 2 → Phase 3.

Phase 1a / 1b / 1c are sequential sub-milestones of the same mechanism: 1a ships the Python prototype, 1b extends the dispatch with `f_prefetch`, and 1c replaces the Python `CpuTaskRunner` with a native runner so the substrate is performance-faithful before attention offload depends on it. The reorder (vs the original Phase 4-after-Phase 3 plan) is driven by `phase1a_findings.md §1.14`: the Python dispatcher and oneDNN-vs-main-thread interference together account for the bulk of the COTS-vs-none gap at the supposed free regime, and Phase 2 measurements built on that substrate would confound runtime overhead with attention-offload regression.

Phase 2 work lives in the attention backend + CPU attention kernel and is orthogonal to Phase 1's linear-layer code; the C++ kernel work for CPU attention with per-head LSE can be developed in parallel with Phase 1b/1c.

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
   at decode (`phase0_findings.md §0.3.4`). The MLP1↔MLP2 matched-index
   invariant is automatic under uniform dispatch.

4. **Layer-ahead prefetch (Phase 1b).** One prefetch queue per layer
   boundary, one sync per layer. Rejected tensor-ahead because its
   topological constraint would force per-sub-module f tuning to avoid
   prefetch starvation — needless given uniform CPU throughput and
   layer-grain fg-wait bypass via SM-issued UVA copy kernel
   (`phase0_findings.md §0.5`: fg activation returns use a separate
   PCIe path from CE0, so they don't queue behind bg DMA prefetches).

5. **K/V-biased WQKV column choice.** Picker assigns CPU columns in
   priority order (K+V head pairs first, then Q tail) so the K/V slice
   sits at the start of cpu_indices. The Planner's cost model charges
   the K/V round-trip when `f_cpu_compute < K/V fraction` and biases the
   optimum toward avoiding it. See `weight_offload_design.md §Implementation
   Note: WQKV K/V Positioning` and `planner_design.md §7.3`.

6. **All PCIe to weight prefetch.** No KV prefetch. Suffix attention → CPU
   only. Batch size is the control variable for CPU attention bottleneck.

7. **Prototype with `enforce_eager=True` through Phase 1b; CUDA Graph
   compatibility lands in Phase 1c.** Phase 1a/1b ship the Python
   `CpuTaskRunner` prototype with `enforce_eager=True`; Phase 1c (was
   Phase 4) replaces the Python dispatcher with `cudaLaunchHostFunc` +
   native worker, which both removes the Python dispatcher tax and is the
   precondition for capturing the forward pass into a CUDA Graph.
   Correctness and decomposition come first; performance-faithful measurement
   comes with Phase 1c.

8. **Prefill: transfer K/V to GPU prefix cache once.** Prefill overhead is
   negligible (compute-bound, amortized over many tokens); decode keeps K/V
   on CPU for the suffix cache.
