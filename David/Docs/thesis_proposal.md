# Thesis Proposal

**Working title**: *COTS: Collaborative CPU-GPU Offloading for Test-Time Scaling on a Consumer GPU*

## 1. Problem Statement

**Workload**: Test-time compute (TTC) inference — beam search, tree search, best-of-N sampling.
**Computing Platform**: Consumer GPU (NVIDIA RTX 4090, 24 GB GDDR6X)

**Existing Methods**: Locality-Aware Beam Scheduling, FastTTS — these improve TTC scheduling and runtime efficiency, but they assume model weights stay GPU-resident throughout inference. That limits the deployment of larger models on a consumer GPU.

**Objective**: Use the full local system — GPU compute/memory, CPU compute/memory, and PCIe bandwidth — to run larger TTC model pairs efficiently on a consumer GPU.

---

## 2. Background & Motivation

### Test-Time Compute

Rather than relying on ever-larger pretraining budgets, test-time methods use dynamic inference strategies that allow models to "think longer" on harder problems. The common paradigm is *search against a verifier*: generate multiple candidate answers, then use a verifier to select the best one. Search strategies include:

- Best-of-N
- Beam search
- Diverse verifier tree search (DVTS)

However, this new computing paradigm also introduces new challenges for edge deployment.


### Edge Deployment for TTC

Existing work has attempted to improve test-time scaling at the edge through:

- speculative execution
- request scheduling
- memory offloading
- memory partitioning

But these systems still tend to assume that model weights stay GPU-resident throughout inference. That assumption limits deployable model size on a consumer GPU.

#### Model Size Limitation

A consumer GPU like the RTX 4090 has only 24 GB of memory, which can fit models under ~12B parameters. TTC pipelines typically require two models (a generator and a verifier), further limiting deployable model size.


### Challenges

#### Multiple Reasoning Path Throughput

Previous local LLM inference work usually optimizes a single stream: a request produces one sequence, and latency is the only concern.

TTC changes the workload shape. One request expands into many candidate reasoning paths. To keep end-to-end latency low, the system needs enough throughput to advance these paths efficiently.

In modern LLM serving, batching throughput is largely determined by how many active sequences the KV cache can hold. This makes TTC offloading harder than standard weight offloading: instead of focusing only on weight placement, the system has to decide how memory is split between weights and KV.

#### Two Asymmetric Models

A TTC pipeline requires a generator and a verifier to be resident simultaneously, doubling the resource pools that must be budgeted: model weights and KV state.
The two models also have different compute patterns, which further amplifies the variation already present in standard inference. Three concurrent axes contribute:

- **Intra-request** — prefill vs. decode within a single request.
- **Inter-request** — continuous batching mixes prefill and decode tokens across requests.
- **Inter-model** — generator runs autoregressive decode (one token per beam); verifier runs step scoring (many tokens per step).

Each axis produces forward calls with different arithmetic intensity, and therefore different optimal CPU/GPU/PCIe splits. A single static configuration cannot cover the full TTC workload well.

---

## 3. System Overview

### 3.1 Architecture

Three components, each on a different timescale:

- **Profiler** (offline) — measures hardware and model behavior; produces cached tables. See `profiler_design.md`.
- **Planner** (load-time) — at engine launch, consumes the profile + budgets + workload contract (`n`, search strategy, `max_context`) and emits a placement plan (per-model scalars) and a per-bucket dispatch table. **Primary thesis contribution.** See `planner_design.md`.
- **Scheduler** (runtime) — executes the plan per step: tier-aware KV admission, KV migration, dispatch lookup. Thin extension over vLLM's existing scheduler. See `scheduler_design.md`.

```
           ┌──────────────┐                           ┌───────────────┐
 HW, model │   Profiler   │  profile tables           │    Planner    │
 ─────────▶│   (offline)  │──────────────────────────▶│  (load-time)  │
           └──────────────┘                           └───────┬───────┘
                                                              │ placement +
                                   SearchConfig ──────────────┤ dispatch table
                                   VRAM / RAM budgets ────────┤
                                                              ▼
                                                   ┌────────────────────┐
                                    request flow ─▶│     Scheduler      │
                                                   │ (runtime, per step)│
                                                   └──────────┬─────────┘
                                                              ▼
                                                     vLLM forward pass
                                                     (gen or ver model)
```

Each component writes data the next one reads; runtime never changes the plan.

### 3.2 Offloading Strategy

Prior weight-offloading strategies mainly target model fit under single-stream latency constraints: keep as many weights as possible on GPU, and use CPU memory as overflow storage for the rest. That is enough when offloading is primarily a memory-capacity problem. TTC breaks this assumption. It is compute-intensive and jointly weight/KV-memory-intensive, so sustaining this workload requires using all available resources in a consumer system. COTS therefore partitions all computation three ways so GPU compute, CPU compute, and PCIe bandwidth all contribute to each forward pass.

Three fundamental partitions follow:

**Weight computation: 3-way split.** Every production matmul sub-module (WQKV, WO, MLP1, MLP2) has three concurrent computation paths so that GPU compute, CPU compute, and PCIe can all contribute to the same operation:

- GPU path — weights resident on GPU, computed on GPU
- Prefetch path — weights streamed CPU→GPU on demand, computed on GPU
- CPU path — weights resident on CPU, computed on CPU, result returned over PCIe

A 2-way split (no prefetch) leaves PCIe idle. A pure-prefetch split wastes CPU compute. 3-way is the minimum that engages all three resources concurrently.

The split axis is **per-sub-module**, chosen to match vLLM's tensor-parallel conventions so the same mechanism applies across the device boundary as it does across GPU ranks: **WQKV, WO, and MLP1 are column-parallel** (shard the output dim); **MLP2 is row-parallel** (shard the input dim). The col→row pairing between MLP1 and MLP2 keeps the intermediate activation device-local automatically — under the Planner's uniform per-bucket `(f_cpu, f_prefetch)` applied to both, MLP1's CPU output-column set matches MLP2's CPU input-column set by construction, so each device holds its own intermediate slice and SwiGLU runs locally. **WO uses the same production dispatch policy as the rest of the module set**, but with a coarser dense-output snap quantum so tiny WO slices do not pay a fixed sync/activation-return cliff. For **WQKV**, columns are assigned K/V-biased (KV-head groups placed on the CPU slice before Q heads), aligning weight offload with attention offload: the new K/V produced by the CPU slice lands directly on the CPU side where the suffix cache lives, and PCIe H2D (reserved for weight prefetch) is not competed for. For the other sub-modules, any slice within the chosen axis is mathematically equivalent. See `weight_offload_design.md`.

**Attention (KV cache): 2-way split.** KV partitions naturally by sharing pattern, enabling prefix and suffix attention to run concurrently on different devices:

- Prefix KV (shared across beams) → GPU — GPU runs prefix attention
- Suffix KV (per-beam, large aggregate) → CPU — CPU runs suffix attention in parallel

Results merge exactly via online softmax (`merge_attn_states`).

**PCIe H2D: 100% weight prefetch.** PCIe H2D is the only contended direction (D2H is otherwise idle). Weight prefetch strictly dominates KV prefetch at every scale — see `pcie_bandwidth_allocation_design.md` for the full analysis. Writing new suffix KV to CPU uses PCIe D2H, so it does not compete.

These three partitions define *what* can be split and *where* the boundaries lie. *How much* goes on each side — the actual partition fractions, per model and per batch shape — is what the Planner decides (§5).

---

## 4. Profiler

The Profiler characterizes hardware and model behavior once per `(hardware, model, dtype)` tuple. Its output — GPU per-layer timing, CPU GEMM curves, PCIe bandwidth, CPU attention latency — is cached and consumed by the Planner at every engine launch.

See `profiler_design.md` for schema, methodology, and caching.

---

## 5. Planner

The Planner is the primary technical contribution. At engine launch, it consumes the Profiler's tables, the VRAM/RAM budgets, and the workload contract (`n`, search strategy, `max_context`) and emits two kinds of output: load-time placement scalars and a per-bucket compute dispatch table.

### 5.1 Why a per-bucket dispatch table

TTC serving produces forward calls spanning a wide range of shapes. Three sources of variation contribute:

- **Continuous batching** — vLLM v1 admits requests based on KV memory pressure, so batch composition fluctuates across forward calls.
- **Prefill/decode mixing** — chunked prefill and decode tokens share the same batch.
- **Generator/verifier asymmetry** — the generator runs autoregressive decode (one new token per beam → small forward calls); the verifier runs step scoring (many tokens per step → medium forward calls).

Each produces calls with different arithmetic intensity: small calls have GPU idle time that CPU compute can hide in, while large calls are GPU compute-bound and must lean on prefetch. A single static offloading strategy wastes resources on one end of the spectrum.

The v0 Planner uses the forward call's `num_tokens` as the dispatch key. vLLM
already captures one CUDA graph per `BatchDescriptor` keyed on this bucket, and
the Planner emits one dispatch entry per captured bucket. Variation from
batching, prefill/decode, and gen/ver therefore becomes variation along the
same bucket axis. If profiling shows that prefill/decode mix still matters at
fixed `num_tokens`, the dispatch key can grow; the first design keeps the
runtime surface minimal.

### 5.2 Load-time vs Runtime Decisions

The Planner's output structure follows directly from *when* each decision is best made.

**Storage (load-time).** The fraction of each model's weights on CPU (`f_cpu_store_m`) and the per-tier KV pool sizes (`KV_gpu_bytes_m`, `KV_cpu_bytes_m`) set VRAM/RAM allocations for the entire serving session. Changing them at runtime would trigger reallocation stalls and cache thrashing, so they are decided once at engine launch. `f_cpu_store_m` is a single scalar applied uniformly to the production module set: WQKV, WO, MLP1, and MLP2.

**Compute dispatch (pre-computed at plan time, looked up at runtime).** The compute split `(f_cpu_compute, f_prefetch_compute)` depends on the bucket's `num_tokens`, and the bucket distribution is not under our control (vLLM decides batch composition). Rather than re-solve at runtime, the Planner pre-computes the full table `(f_cpu_compute, f_prefetch_compute)[BatchDescriptor]` at plan time — a single pair per bucket, applied uniformly across WQKV/WO/MLP1/MLP2. Runtime dispatch is a table lookup, not an optimization.

The two levels are tied by the invariant `f_cpu_compute + f_prefetch_compute = f_cpu_store_m`: storage decides which weights are CPU-resident; dispatch decides how those bytes are used each forward — CPU-computed or streamed to GPU via the prefetch path. The matched-index invariant between MLP1 (col-parallel) and MLP2 (row-parallel) is satisfied automatically under uniform dispatch, so the intermediate activation stays device-local through the MLP block.

### 5.3 Decision variables

Per model `m ∈ {generator, verifier}`:

- `f_cpu_store_m ∈ [0, 1]` — single scalar applied uniformly to {WQKV, WO, MLP1, MLP2}; WO uses a coarser runtime snap quantum
- `KV_gpu_bytes_m`, `KV_cpu_bytes_m` — per-tier KV pool sizes

Plus a derived dispatch table: one `(f_cpu_compute, f_prefetch_compute)` pair per `BatchDescriptor` per model, applied uniformly across WQKV/WO/MLP1/MLP2.

### 5.4 Constraints

```
Σ_m (W_gpu_m + KV_gpu_m + |B_prefetch_m|) ≤ VRAM_budget
Σ_m (W_cpu_m + KV_cpu_m)                  ≤ Host_RAM_budget
f_cpu_compute + f_prefetch_compute = f_cpu_store_m     (per bucket)
KV_gpu_m + KV_cpu_m ≥ KV_needed_m(strategy, n, max_context)
```

### 5.5 Performance model

Per scheduled round, the Planner estimates one layer's critical path from two
overlapping terms:

- **Current-layer compute**: a sequential sum over `{WQKV, attention, WO,
  MLP1, MLP2}`. Within each operation, CPU and GPU paths run concurrently and
  the slower path gates the operation.
- **Layer-ahead prefetch**: PCIe H2D of the next layer's prefetched weight
  bundle, using `pcie_h2d_bw`.

```text
t_m,b = num_layers_m × max(C_layer_m,b, P_layer_m,b)
C_layer_m,b = Σ_{op ∈ layer_ops} max(T_gpu_op,m,b, T_cpu_op,m,b) + O_layer_m,b
```

Weight terms come from `gpu_layer_timing` and `cpu_gemm_curve`, attention terms
come from GPU prefix timing plus `cpu_attn_curve`, and `O_layer` captures
partial-result merges, CPU/GPU sync, task submission, and runtime bookkeeping.

### 5.6 Objective and solution method

The Planner optimizes single-request TTC latency to `n` completed paths for the
given workload contract:

```text
T_TTC(plan) ≈ Σ_m N_rounds_m(plan, workload) × L_round_m(plan)
```

Weight residency mostly changes per-round latency and model-fit feasibility.
KV placement determines whether the multi-path workload can be admitted under
the remaining memory budget, and secondarily how many scheduler/admission waves
are needed. A plan wins when its memory-placement benefit — fitting a larger
model pair, avoiding an otherwise bad weight/KV budget split, or producing a
useful scheduling gain — outweighs the per-round latency cost of CPU compute,
prefetch, and CPU suffix attention.

The implementation follows a frontier-based hierarchy: `ModelMemoryPartitioner`
enumerates generator/verifier budget splits, `WeightKVPartitioner` scores
engine-local weight/KV placement candidates, and `DispatchCompiler`
materializes the per-bucket dispatch table for the chosen placement. The
current prototype implements the weight-placement subset and leaves final
hybrid-KV policy selection profile-gated. The thesis contribution is the
constrained performance model, not a particular continuous optimizer. A
closed-form idle-budget dispatch heuristic can become a fast path after it
matches the snapped reference search on the profiled bucket set. See
`planner_design.md` for the full formulation.

---

## 6. Scheduler

The Scheduler executes the Planner's output. Its responsibilities:

- **Tier-aware admission** — respects both `KV_gpu` and `KV_cpu` pools.
- **KV migration** — spills suffix blocks to CPU on beam growth, reclaims on pruning, promotes shared-prefix blocks.
- **Dispatch lookup** — maps runtime `BatchDescriptor` to the Planner's dispatch entry.
- **Back-pressure** — signals FastTTS to throttle beam expansion when `KV_cpu` is near full.

Runtime adds no new decisions — batch composition and graph-bucket selection remain with vLLM's existing scheduler and `CudagraphDispatcher`.

See `scheduler_design.md`.

---

## 7. Reference Architecture

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

The ~40× ratio is the fundamental constraint: **pure prefetch cannot hide latency for BF16 decode.** The Planner's CPU-compute path exists because prefetch alone is insufficient.

---

## 8. Implementation

### 8.1 Existing vLLM Building Blocks

| Component | File |
|---|---|
| Cascade attention + online softmax merge | `flash_attn.py:1038`, `merge_attn_states.py` |
| CPU attention + KV offload | `cpu_attn.py`, `kv_offload/` |
| Prefetch offloader + circular buffer | `offloader/prefetch.py` |
| CUDA graph dispatcher | `cudagraph_dispatcher.py` |
| QKVParallelLinear, MergedColumnParallelLinear | `layers/linear.py` |

### 8.2 Implementation Status

The core mechanism work has largely landed in the thesis vLLM fork:

1. Mixed col/row tensor-granularity weight split: WQKV/WO/MLP1 use output-column splits, MLP2 uses an input-row split, and the MLP1↔MLP2 index match keeps the intermediate activation device-local.
2. Three-way weight dispatch: GPU-resident compute, layer-ahead prefetch-to-GPU, and native CPU compute share one per-bucket `(f_cpu_compute, f_prefetch_compute)` table.
3. Native COTS runner: CPU work uses C++ task runners and `cudaLaunchHostFunc` submit/sync glue; the Python runner remains an eager-only diagnostic path.
4. Hybrid KV: GPU prefix attention and CPU suffix attention run in parallel and merge via online softmax using output plus per-head LSE.
5. FastTTS planner wiring: manual/config-driven plans and the current weight-placement prototype can emit COTS weight, dispatch, thread-policy, and hybrid-KV runtime kwargs.

Remaining thesis work is primarily policy and evaluation: calibrated Profiler tables, Planner selection of weight/KV placement under realistic workload contracts, and end-to-end RTX 4090 experiments for the target model matrix.

### 8.3 CUDA Graph Integration

vLLM's default `FULL_AND_PIECEWISE` mode captures per-bucket graphs. COTS integrates with that policy through piecewise graph boundaries and CUDA/host handoff points:

- Non-attention regions (QKV/WO/MLP projections) — native COTS weight submit/sync operations become split points; graph-mode weight sync uses the `wait_kernel` path, while eager uses the host-callback path.
- Attention regions — hybrid KV uses the normal piecewise attention boundaries; `merge_attn_states` on GPU joins prefix and suffix attention outputs.

The current production path is native COTS with piecewise graphs. Phase 2 hybrid KV does not add Phase 1 weight split points unless weight offload is also active.

### 8.4 Roadmap

See `implementation_roadmap.md`, with `phase1_findings.md` and `phase2_findings.md` as the current source of truth for landed weight and hybrid-KV behavior.

---

## 9. Evaluation Plan

### 9.1 Target model matrix

Use pure instruction-tuned generators for the main evaluation matrix. Math
variants and QwQ-style reasoning models are excluded from the primary Planner
evaluation because they change the workload semantics and would blur the
systems claim.

| Tier | Generator | Verifier | Purpose |
|---|---|---|---|
| Smoke only | `Qwen/Qwen2.5-1.5B-Instruct` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | Fast correctness/debug run; not part of the main size claim. |
| Fit-model policy probe | `Qwen/Qwen2.5-7B-Instruct` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | Tests whether the Planner avoids harmful offload when the models already fit; any throughput win from extra KV capacity is secondary upside. |
| Higher-KV-pressure probe | `meta-llama/Llama-3.1-8B-Instruct` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | Same broad size class as Qwen 7B, but higher KV pressure; use if we need a stronger stress test for joint weight/KV placement. |
| Main capacity | `Qwen/Qwen2.5-14B-Instruct` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | First clear forced-fit / enlarged-KV regime on a 24 GB GPU. |
| Upper stress | `Qwen/Qwen2.5-32B-Instruct` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | Stretch run for the largest-generator claim if memory, load time, and experiment budget allow it. |
| Large-verifier secondary | `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen2.5-14B-Instruct` | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B` | Isolates whether the Planner still behaves well when the verifier also becomes a meaningful memory consumer. |

The default verifier should be the 1.5B PRM because it keeps the main story
focused on generator scaling and TTC batch/KV pressure. The 7B PRM is a
secondary axis, not a full cross product with every generator. Similarly,
`1.5B + 1.5B` is a smoke/debug tier only; it is too small to support the thesis'
larger-model offloading claim.

For already-fit models, run Qwen 7B first as a policy sanity check. If it shows
little or no benefit, keep that result and optionally run Llama 8B as the
higher-KV-pressure comparison. The thesis claim should not require a fit-model
speedup: offloading helps primarily when it enables a model pair or avoids a
bad weight/KV memory split, with throughput gains treated as profile-gated
upside when the measured pressure regime supports them.

### 9.2 Ablations

Attribute gains by enabling one component at a time:

1. **Baseline** — FastTTS + vLLM, no offloading.
2. **Mechanism-only, manual tuning** — weight + attention offload enabled, placement and dispatch hand-tuned per model.
3. **+ Planner** — Planner sets placement and dispatch. Expected: matches or beats manual tuning.
4. **+ Scheduler (tiered KV admission + migration)** — full system.

### 9.3 Workload sweeps

- **`n` sweep**: `n ∈ {1, 4, 16, 64, 256}`. Report throughput, latency, Planner output (`f_cpu_store`, `KV_*_bytes`) per `n`.
- **Model size sweep**: Qwen 7B generator (fits), Llama 8B fallback for higher KV pressure, Qwen 14B generator (mandatory offload), and Qwen 32B stretch. Verifier is 1.5B by default, with one 7B-verifier secondary run.
- **Search strategy**: beam search vs best-of-N, to exercise the prefix-sharing asymmetry.

### 9.4 System-paper specifics

- **Planner runtime**: measured wall-clock at launch; target ≤ 1 s.
- **Dispatch heuristic vs snapped per-bucket search**: compare the closed-form idle-budget rule against the snapped reference search before using it as a fast path.
- **What we did NOT make dynamic**: explicit statement (placement + dispatch table are static; runtime scheduling is vLLM's scheduler + our tiered KV policy).
- **VRAM accounting**: breakdown per ablation — weights, KV pools, prefetch buffer, vLLM overhead.

### 9.5 Accuracy

MATH-500 accuracy across all configurations. Expected: unchanged at the benchmark level. The col/row decompositions and online-softmax merge are mathematically equivalent to the GPU-only computation, while BF16 CPU/GPU kernel-order differences are handled through numerical-parity tests rather than exact greedy-token identity.

---

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| CPU GEMM below theoretical bandwidth | Profile measures directly; Planner adapts. |
| CPU attention bottleneck at long contexts | Batch size clamping via Scheduler; Planner can reduce `KV_cpu_bytes` to force shorter effective suffix. |
| Column/row split numerical differences | The decomposition is mathematically exact; tests assert numerical parity within the expected BF16 CPU/GPU tolerance. |
| CUDA Graph incompatibility with CPU compute | Native COTS uses piecewise graphs, `cudaLaunchHostFunc` submit points, and `wait_kernel` graph sync; the Python runner remains eager-only. |
| Planner solver runtime dominates launch | Bounded snapped candidate search; measured runtime reported. |

---

## 11. Detailed Design Documents

| Document | Scope |
|---|---|
| `profiler_design.md` | Profile schema, methodology, caching |
| `planner_design.md` | Inputs/outputs, constraints, objective, solution method (primary contribution) |
| `scheduler_design.md` | Tier-aware admission, KV migration, dispatch lookup |
| `weight_offload_design.md` | Storage-vs-compute separation, tensor granularity, buffer sizing |
| `attention_offload_design.md` | CPU suffix attention, KV pool sizing |
| `pcie_bandwidth_allocation_design.md` | Why 100% PCIe → weight prefetch |
| `implementation_roadmap.md` | Phased implementation plan |
| `phase0_findings.md` | First-iteration Profiler output on RTX 4090 + Qwen2.5-7B |
| `phase1_findings.md` | Production COTS weight-offload path and Phase 1 results |
| `phase2_findings.md` | Hybrid CPU/GPU KV implementation and profile-gated policy results |
