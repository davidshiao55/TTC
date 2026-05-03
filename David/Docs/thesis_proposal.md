# Thesis Proposal

**Working title**: *COTS: Collaborative CPU-GPU Offloading for Test-Time Scaling on a Consumer GPU*

## 1. Problem Statement

**Workload**: Test-time compute (TTC) inference — beam search, tree search, best-of-N sampling.
**Computing Platform**: Consumer GPU (NVIDIA RTX 4090, 24 GB GDDR6X)

**Existing Methods**: Locality-Aware Beam Scheduling, FastTTS — these optimize KV-cache handling (movement, scheduling, reuse), but assume model weights and KV cache of current computing batch stay GPU-resident throughout inference.

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

However, the new computing paradigm also introduce new challenge for edge deployment.


### Edge Deployment for TTC

Existing work has attempted to improve test-time scaling at the edge through:

- speculative execution
- request scheduling
- kv cache offloading
- memory partitioning

But they assume model weights and active KV cache (i.e. the KV cache of the current computing batch) stay GPU-resident throughout inference.

#### Model Size Limitation

A consumer GPU like the RTX 4090 has only 24 GB of memory, which can fit models under ~12B parameters. TTC pipelines typically require two models (a generator and a verifier), further limiting deployable model size.

#### Throughput Limitation

With vLLM's continuous batching, maximum batch size is constrained by GPU memory available for KV cache. FastTTS submits all beams in one `generate()` call; vLLM's scheduler (`scheduler.py:541`) allocates KV blocks per request and processes excess beams round by round:

```
wall_time ≈ ⌈total_beams / batch_capacity⌉ × time_per_step
```

### Challenges

The standard mechanisms for enabling larger models and higher throughput on a single device — **weight offloading** and **attention offloading** — do not transfer directly to TTC.

#### Existing Weight Offloading Methods

Existing edge weight offloading methods typically target online interactive serving — small batches, low-frequency requests, with per-token latency as the binding constraint — and do not exploit the multi-path parallelism that TTC produces.

While there are also methods aimed at offline batch inference that exploit large batches at the cost of latency, they fit TTC no better: TTC is still a single-request workload — its batch comes from multi-path exploration, not independent requests — so end-to-end latency remains the user-facing metric.

#### Existing Attention Offloading Methods

Existing attention offloading methods assume the model fits in GPU memory and overlap CPU/GPU compute by splitting the batch in two. This does not transfer to TTC:

1. Batch splitting conflicts with our goal of maximizing batch size.
2. Once weights are partitioned across CPU and GPU, every sub-module already engages both devices concurrently — a batch-split scheme on top is strictly inferior.

#### The TTC Workload

TTC compounds the offloading problem in two ways:

1. **Two models, not one.** A TTC pipeline requires a generator and a verifier to be resident simultaneously, doubling the resources — model weights and KV pools — that must be partitioned across CPU, GPU, and PCIe.
2. **Complex computing pattern.** 
Computing pattern are already diverse in standard inference; TTC amplifies this further. Three concurrent axes contribute:
   - **Intra-request** — prefill vs. decode within a single request.
   - **Inter-request** — continuous batching mixes prefill and decode tokens across requests.
   - **Inter-model** — generator runs autoregressive decode (one token per beam); verifier runs step scoring (many tokens per step).

   Each axis produces forward calls with different arithmetic intensity, and therefore different optimal CPU/GPU/PCIe splits. A single static configuration cannot cover them.

---

## 3. System Overview

### 3.1 Architecture

Three components, each on a different timescale:

- **Profiler** (offline) — measures hardware and model behavior; produces cached tables. See `profiler_design.md`.
- **Planner** (load-time) — at engine launch, consumes the profile + budgets + workload target (`n`, search strategy, `max_context`) and emits a placement plan (per-model scalars) and a per-bucket dispatch table. **Primary thesis contribution.** See `planner_design.md`.
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

Transformer inference is fundamentally sequential — each layer feeds the next, and within a layer each sub-module (WQKV → attention → WO → MLP1 → MLP2) feeds the next. There are no independent branches that could be naively scheduled onto separate devices. For GPU + CPU + PCIe to run concurrently, each sequential operation must be split internally.

Three fundamental partitions follow:

**Weight computation: 3-way split.** Every matmul sub-module (WQKV, WO, MLP1, MLP2) has three concurrent computation paths so that GPU compute, CPU compute, and PCIe can all contribute to the same operation:

- GPU path — weights resident on GPU, computed on GPU
- Prefetch path — weights streamed CPU→GPU on demand, computed on GPU
- CPU path — weights resident on CPU, computed on CPU, result returned over PCIe

A 2-way split (no prefetch) leaves PCIe idle. A pure-prefetch split wastes CPU compute. 3-way is the minimum that engages all three resources concurrently.

The split axis is **per-sub-module**, chosen to match vLLM's tensor-parallel conventions so the same mechanism applies across the device boundary as it does across GPU ranks: **WQKV and MLP1 are column-parallel** (shard the output dim); **MLP2 is row-parallel** (shard the input dim). The col→row pairing between MLP1 and MLP2 keeps the intermediate activation device-local automatically — under the Planner's uniform per-bucket `(f_cpu, f_prefetch)` applied to both, MLP1's CPU output-column set matches MLP2's CPU input-column set by construction, so each device holds its own intermediate slice and SwiGLU runs locally. **WO is not offloaded in Phase 1/2** — fully GPU-resident (see `weight_offload_design.md §WO Split Axis Decision`). For **WQKV**, columns are assigned K/V-biased (KV-head groups placed on the CPU slice before Q heads), aligning weight offload with attention offload: the new K/V produced by the CPU slice lands directly on the CPU side where the suffix cache lives, and PCIe H2D (reserved for weight prefetch) is not competed for. For the other sub-modules, any slice within the chosen axis is mathematically equivalent. See `weight_offload_design.md`.

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

The Planner is the primary technical contribution. At engine launch, it consumes the Profiler's tables, the VRAM/RAM budgets, and the workload target (`n`, search strategy, `max_context`) and emits two kinds of output: load-time placement scalars and a per-bucket compute dispatch table.

### 5.1 Why a per-bucket dispatch table

TTC serving produces forward calls spanning a wide range of shapes. Three sources of variation contribute:

- **Continuous batching** — vLLM v1 admits requests based on KV memory pressure, so batch composition fluctuates across forward calls.
- **Prefill/decode mixing** — chunked prefill and decode tokens share the same batch.
- **Generator/verifier asymmetry** — the generator runs autoregressive decode (one new token per beam → small forward calls); the verifier runs step scoring (many tokens per step → medium forward calls).

Each produces calls with different arithmetic intensity: small calls have GPU idle time that CPU compute can hide in, while large calls are GPU compute-bound and must lean on prefetch. A single static offloading strategy wastes resources on one end of the spectrum.

All three axes collapse into a single scalar — the forward call's `num_tokens`. Arithmetic intensity, GPU idle time, and therefore the optimal compute split all depend on `num_tokens` alone. vLLM already captures one CUDA graph per `BatchDescriptor` (keyed on `num_tokens`), and the Planner emits one dispatch entry per captured bucket. Variation from batching, prefill/decode, and gen/ver becomes variation along the same `num_tokens` axis — handled uniformly by indexing the dispatch table.

### 5.2 Load-time vs Runtime Decisions

The Planner's output structure follows directly from *when* each decision is best made.

**Storage (load-time).** The fraction of each model's weights on CPU (`f_cpu_store_m`) and the per-tier KV pool sizes (`KV_gpu_bytes_m`, `KV_cpu_bytes_m`) set VRAM/RAM allocations for the entire serving session. Changing them at runtime would trigger reallocation stalls and cache thrashing, so they are decided once at engine launch. `f_cpu_store_m` is a single scalar applied uniformly to WQKV, MLP1, and MLP2 (WO is fully GPU-resident in Phase 1/2).

**Compute dispatch (pre-computed at plan time, looked up at runtime).** The compute split `(f_cpu_compute, f_prefetch_compute)` depends on the bucket's `num_tokens`, and the bucket distribution is not under our control (vLLM decides batch composition). Rather than re-solve at runtime, the Planner pre-computes the full table `(f_cpu_compute, f_prefetch_compute)[BatchDescriptor]` at plan time — a single pair per bucket, applied uniformly across WQKV/MLP1/MLP2. Runtime dispatch is a table lookup, not an optimization.

The two levels are tied by the invariant `f_cpu_compute + f_prefetch_compute = f_cpu_store_m`: storage decides which weights are CPU-resident; dispatch decides how those bytes are used each forward — CPU-computed or streamed to GPU via the prefetch path. The matched-index invariant between MLP1 (col-parallel) and MLP2 (row-parallel) is satisfied automatically under uniform dispatch, so the intermediate activation stays device-local through the MLP block.

### 5.3 Decision variables

Per model `m ∈ {generator, verifier}`:

- `f_cpu_store_m ∈ [0, 1]` — single scalar applied uniformly to {WQKV, MLP1, MLP2}; WO is fixed at 0 (not offloaded in Phase 1/2)
- `KV_gpu_bytes_m`, `KV_cpu_bytes_m` — per-tier KV pool sizes

Plus a derived dispatch table: one `(f_cpu_compute, f_prefetch_compute)` pair per `BatchDescriptor` per model, applied uniformly across WQKV/MLP1/MLP2.

### 5.4 Constraints

```
Σ_m (W_gpu_m + KV_gpu_m + |B_prefetch_m|) ≤ VRAM_budget
Σ_m (W_cpu_m + KV_cpu_m)                  ≤ Host_RAM_budget
f_cpu_compute + f_prefetch_compute = f_cpu_store_m     (per bucket)
KV_gpu_m + KV_cpu_m ≥ n × max_context × kv_bytes_per_token_m
```

### 5.5 Performance model

Per step, per sub-module (WQKV, WO, MLP1, MLP2), three paths run concurrently:

- **GPU path**: GPU compute on `(1 − f_cpu_compute − f_prefetch_compute)` of the weight, from `gpu_layer_timing`.
- **Prefetch path**: PCIe H2D of `f_prefetch_compute × weight_bytes` (from `pcie_h2d_bw`), then GPU compute on the prefetched slice.
- **CPU path**: CPU GEMM on `f_cpu_compute × weight_bytes` (from `cpu_gemm_curve`), plus activation round-trip.

Per-sub-module critical path: `t = max(t_gpu, t_prefetch, t_cpu_compute)`. Plus attention-path cost when attention offload is active (GPU prefix attention + CPU suffix attention + merge via online softmax). Layer time is the sum over sub-modules; decode wall-clock is layer time × num_layers × num_decode_tokens.

### 5.6 Objective and solution method

**Two-stage optimization**:

1. *Placement stage* — per model, minimize representative-bucket decode wall-clock subject to constraints. Representative bucket for generator is small `num_tokens` (≈ `n`); for verifier is medium (≈ `n × tokens_per_step`).
2. *Dispatch stage* — for each captured bucket, closed-form single-scalar solve over `f_cpu` (with `f_prefetch = f_cpu_store − f_cpu`). Small `num_tokens` lean on CPU-compute (hides in GPU idle); medium/large shift toward prefetch (GPU compute-bound). The dispatch is uniform across WQKV/MLP1/MLP2, enabled empirically by uniform CPU μs/MB at decode (`phase0_findings.md §0.3.4`).

Grid search over the 3–4 placement scalars is tractable (sub-second). See `planner_design.md` for the full formulation.

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

### 8.2 Engineering Gaps

1. Mixed col/row weight split at tensor granularity: col-parallel for WQKV/MLP1 (partial-result concat), row-parallel for MLP2 (add-reduce), CPU matmul on both axes, with MLP1↔MLP2 index-set matching for the col→row pipelining save
2. CPU attention kernel returning per-head LSE (for online softmax merge)
3. `cudaLaunchHostFunc` glue for CUDA Graph + CPU task co-scheduling (Phase 1c)
4. Per-model, per-bucket `(f_cpu_compute, f_prefetch_compute)` configuration and dispatch
5. Profiler implementation (schema → cached tables)
6. Planner implementation (two-stage optimization)

### 8.3 CUDA Graph Integration

vLLM's default `FULL_AND_PIECEWISE` mode captures per-bucket graphs. Our mechanisms integrate via `cudaLaunchHostFunc`:

- Non-attention regions (MLP, QKV/O projections) — CPU-compute and prefetch paths captured inline.
- Attention regions — same technique for CPU suffix attention; `merge_attn_states` on GPU joins prefix + suffix.

Prototype Phase 1a/1b with `enforce_eager=True`; `cudaLaunchHostFunc` retrofit is a localized swap in `CpuComputeDispatcher` (Phase 1c, gating Phase 2 per `phase1a_findings.md §1.14`).

### 8.4 Roadmap

See `implementation_roadmap.md`.

---

## 9. Evaluation Plan

### 9.1 Ablations

Attribute gains by enabling one component at a time:

1. **Baseline** — FastTTS + vLLM, no offloading.
2. **Mechanism-only, manual tuning** — weight + attention offload enabled, placement and dispatch hand-tuned per model.
3. **+ Planner** — Planner sets placement and dispatch. Expected: matches or beats manual tuning.
4. **+ Scheduler (tiered KV admission + migration)** — full system.

### 9.2 Workload sweeps

- **`n` sweep**: `n ∈ {1, 4, 16, 64, 256}`. Report throughput, latency, Planner output (`f_cpu_store`, `KV_*_bytes`) per `n`.
- **Model size sweep**: 7B generator (fits), 14B generator (mandatory offload). Verifier (1.5B) fixed.
- **Search strategy**: beam search vs best-of-N, to exercise the prefix-sharing asymmetry.

### 9.3 System-paper specifics

- **Planner runtime**: measured wall-clock at launch; target ≤ 1 s.
- **Dispatch-heuristic vs exhaustive**: on a small bucket set, exhaustive per-bucket optimization vs closed-form heuristic. Verify near-optimality.
- **What we did NOT make dynamic**: explicit statement (placement + dispatch table are static; runtime scheduling is vLLM's scheduler + our tiered KV policy).
- **VRAM accounting**: breakdown per ablation — weights, KV pools, prefetch buffer, vLLM overhead.

### 9.4 Accuracy

MATH-500 accuracy across all configurations. Expected: unchanged (mathematical exactness of col-parallel and row-parallel splits, and of the online-softmax merge).

---

## 10. Key Risks

| Risk | Mitigation |
|---|---|
| CPU GEMM below theoretical bandwidth | Profile measures directly; Planner adapts. |
| CPU attention bottleneck at long contexts | Batch size clamping via Scheduler; Planner can reduce `KV_cpu_bytes` to force shorter effective suffix. |
| Column-parallel numerical differences | Mathematically exact; unit tests assert bit-identical. |
| CUDA Graph incompatibility with CPU compute | `cudaLaunchHostFunc` (KTransformers pattern) is graph-capturable. Prototype Phase 1a/1b with `enforce_eager=True`, retrofit in Phase 1c. |
| Planner solver runtime dominates launch | Closed-form dispatch + small grid outer loop; measured runtime reported. |

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
