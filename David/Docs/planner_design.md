# Planner Design

The Planner is the primary contribution of this thesis. It takes profile tables (from the Profiler), resource budgets, and a workload target, and emits a concrete placement and dispatch plan that the Scheduler executes at runtime.

- **Timescale**: load-time, runs once per engine launch
- **Inputs**: profile tables, VRAM/RAM budgets, workload target (`n`, beam width, `max_context`)
- **Outputs**: load-time scalars (placement) + per-bucket dispatch table

Prior offloading systems hardcode placement decisions or expose them as manual tunables. The Planner automates this: given a profile and a workload, it picks where weights live, how big KV pools should be, and per-CUDA-graph-bucket compute dispatch — all from the perf model.

---

## 1. Inputs

### 1.1 Profile tables

From the Profiler (see `profiler_design.md`):

- `gpu_layer_timing[sub_module, bucket]`
- `cpu_gemm_curve[sub_module, batch_size, slice_cols]`
- `pcie_h2d_bw[transfer_bytes]`
- `cpu_attn_curve[batch_size, suffix_context_len]`

### 1.2 Resource budgets

Derived from the hardware and engine configuration:

```
VRAM_budget = total_VRAM − vLLM_overhead − |B_prefetch|
Host_RAM_budget = total_RAM − OS_overhead
```

- `|B_prefetch| = 2 × max_layer_bytes_m` per model (pipeline-depth sized; see §5)
- `vLLM_overhead` ≈ 1–2 GB from engine process state (see `vllm_v1_migration.md`)

FastTTS runs two engines (generator + verifier) sharing the GPU. The Planner inherits FastTTS's existing `gpu_memory_utilization` split between them rather than re-deciding the split — one less knob.

### 1.3 Workload target

From FastTTS `SearchConfig`:

- `n` — maximum concurrent reasoning paths for this engine instance (see §2)
- `max_context` — longest expected suffix
- **Search strategy** — `{beam_search, best_of_n}`. The two representative TTS strategies the thesis supports. Variants (DVTS, dynamic branching) fall under `beam_search` for planning purposes.

Strategy affects KV pool sizing because prefix-sharing patterns differ. **Beam search has substantial prefix sharing** (beams share the prompt and common-ancestry tokens), so the same physical GPU KV capacity supports more logical KV than BoN. **Best-of-N has minimal sharing beyond the prompt** — each rollout's generation is independent.

The Planner consumes the strategy and adjusts `KV_gpu_bytes` / `KV_cpu_bytes` / `f_cpu_store` accordingly. Concrete sharing assumptions (e.g., expected shared-prefix fraction for beam search) are parameters of the Planner's workload model — pinned down in implementation.

---

## 2. `n` as "max for this engine launch"

`n` is treated as a launch-time parameter analogous to vLLM's `max_num_seqs` or `max_model_len`: the maximum the engine instance must support, fixed for the lifetime of that launch. The user requesting fewer reasoning paths at runtime just underutilizes capacity — this is normal and not an error.

Consequences:

- **Per-launch planning.** The Planner re-runs each time the engine launches. If the user wants to sweep `n`, they launch per `n` — same UX as any other engine parameter.
- **No live `n` changes.** Weights are placed at launch; the KV pool is allocated at launch. Changing `n` mid-run is out of scope for v1.
- **FastTTS UX compatibility.** FastTTS already re-launches per experiment; the Planner adds a few seconds to that launch (solver runtime). No workflow change.

---

## 3. Per-Model, Per-Bucket Planning

### Why per-bucket

Modern LLM serving engines (vLLM v1) use **continuous batching with variable `num_tokens` per forward call**. Batch size is not a fixed knob — it's a runtime consequence of KV memory pressure: the engine admits as many requests as GPU KV can hold, and as requests complete or new ones arrive, the effective `num_tokens` per forward call fluctuates. Prefill chunks and decode tokens are also mixed into the same batch, further widening the range of bucket shapes a single captured graph must handle.

Arithmetic intensity varies materially across these buckets:

- Small `num_tokens` (pure decode, memory-BW-bound): GPU has idle time that CPU compute can hide in.
- Large `num_tokens` (mixed prefill-decode, trending compute-bound): GPU saturates, no idle time — CPU compute adds to the critical path.

A single fixed offloading strategy optimal for one regime is suboptimal for the other. The Planner therefore produces a **table indexed by `BatchDescriptor`** so each captured graph bakes in the configuration suited to its bucket. The table is computed once at launch and consumed at O(1) per forward call — no runtime planning overhead.

### Why per-model

TTS pipelines run two models whose per-forward-call shapes differ:

- **Generator** — autoregressive decode. One new token per beam per forward call → small `num_tokens` (≈ `n`).
- **Verifier (PRM)** — step scoring. A full step's worth of tokens per forward call → medium `num_tokens` (≈ `n × tokens_per_step`).

The two models hit different bucket distributions. Because the dispatch table is per-model, per-bucket, each model's plan adapts to its own pattern without any strategy-specific code. One Planner invocation produces two plans (gen, ver), coupled only through the shared VRAM/RAM budgets.

(Motivation for why prior offloading systems handle this suboptimally — and how this addresses the thesis problem statement — lives in `thesis_proposal.md`.)

---

## 4. Outputs

### 4.1 Load-time scalars

Per model `m ∈ {generator, verifier}`:

- `f_cpu_store_m ∈ [0, 1]` — single scalar applied uniformly to **WQKV, MLP1, MLP2**. WQKV's CPU-stored bytes are ordered by the K/V-biased picker (K+V head groups first, then Q tail). WO is not offloaded in Phase 1/2 (`f_cpu_store_WO = 0`, fixed — see `weight_offload_design.md §WO Split Axis Decision`).
- `KV_gpu_bytes_m` — GPU KV pool size
- `KV_cpu_bytes_m` — CPU KV pool size (the "extension")

Six scalars total. The verifier often degenerates to `f_cpu_store_ver = 0` and `KV_cpu_bytes_ver = 0` on a 24 GB RTX 4090 with a 1.5B PRM — the Planner discovers this rather than assuming it.

### 4.2 Dispatch table

Per model, a table keyed by `BatchDescriptor`:

```
dispatch[model][BatchDescriptor] → (f_cpu_compute, f_prefetch_compute)
```

One entry per captured CUDA graph bucket, emitting a **single `(f_cpu, f_prefetch)` pair applied uniformly to WQKV, MLP1, and MLP2** at that bucket. WO has no per-bucket dispatch (not offloaded in Phase 1/2). Constraint per entry:

```
f_cpu_compute + f_prefetch_compute = f_cpu_store_m
```

Every CPU-stored byte is dispatched each forward — either CPU-computed or prefetched back to GPU. In practice the Planner picks `f_cpu_compute` and `f_prefetch_compute` falls out as the remainder (single-scalar solve per bucket; see §7.3).

**MLP1↔MLP2 matched-index invariant is automatic.** Because the same `(f_cpu_compute, f_prefetch_compute)` pair applies to both MLP1 and MLP2, and `f_cpu_store_m` is uniform across them, the col→row pipelining's index-matching requirement (MLP1's CPU output columns must equal MLP2's CPU input columns) is satisfied by construction — not an enforced Planner constraint, just a consequence of uniform dispatch.

**WQKV K/V-pin is an implementation detail.** The runtime `CpuComputeDispatcher` pins WQKV's K/V portion to the CPU-compute path (prefetch on WQKV only covers Q columns above the K/V boundary). This avoids K/V-output PCIe round-trips when the per-bucket `f_cpu_compute` lands below the K/V fraction. The Planner's cost model in §7.3 accounts for the round-trip nudge; from the dispatch-table perspective the Planner still emits a single uniform pair and the K/V-pin guard lives inside the dispatcher. See `weight_offload_design.md §Implementation Note: WQKV K/V-Pin Optimization`.

### 4.3 Eager-fallback entry

For `num_tokens > max_cudagraph_capture_size`, vLLM falls back to eager execution. The Planner emits one extra entry for this case (simplest: reuse the largest captured bucket's dispatch). The Scheduler looks up this entry for out-of-bucket batches.

### 4.4 Fixed constants (documented outputs, not tuned)

- `|B_prefetch_m| = layer-ahead buffer` — sized to hold `Σ_m (f_prefetch × W_m)` across WQKV/MLP1/MLP2 within one layer (see `pcie_bandwidth_allocation_design.md §Prefetch Distance`).
- **Prefetch distance = layer-ahead** — one prefetch queue per layer, one sync per layer boundary. Committed (not an option). Empirically validated in `phase0_findings.md §0.10.2c`: K>1 buys ≤6% over K=1 on Qwen2.5-7B at decode B=64; K=4 OOMs because the buffer pool grows linearly with K.
- KV placement policy — see `attention_offload_design.md §Two-Pool KV Model` for the two candidate mechanisms (position-split vs head-split); one is selected globally and baked in before Phase 2 locks.
- Per-sub-module split axis — `{WQKV: col, MLP1: col, MLP2: row}` is hardcoded across all models and buckets (see `weight_offload_design.md §Per-Sub-Module Split Axis`). WO is not offloaded in Phase 1/2 (`f_gpu` only, no dispatch).
- PCIe allocation — 100% weight prefetch (see `pcie_bandwidth_allocation_design.md`).

These appear in the output for completeness so the Scheduler has one place to read the plan, but the Planner does not optimize over them.

### 4.5 Runtime dispatch lookup — graph-enabled vs graph-disabled

The dispatch table is keyed on `cudagraph_capture_sizes`, independent of whether CUDA graphs are actually enabled at runtime. This keeps the Planner output identical across Phase 1a/1b (graph-disabled prototypes) and Phase 1c (graph-enabled native runner; was Phase 4). The difference is only in how `num_tokens` maps to a bucket at runtime:

| Regime | Forward-pass `num_tokens` | Dispatch-lookup `num_tokens` |
|---|---|---|
| **Graph enabled (Phase 1c)** | Padded up to the nearest bucket by vLLM's `CudagraphDispatcher._bs_to_padded_graph_size` | Same padded value — the lookup is exact. |
| **Graph disabled (Phase 1a/1b, `enforce_eager=True`)** | Exact runtime value (no padding) | **Rounded up** to the nearest bucket for the lookup only. Rounding up (rather than nearest or down) matches what graph-enabled mode would pad to, so Phase 1a/1b measurements predict Phase 1c dispatch exactly. |

Runtime pseudocode:

```python
BUCKETS = cudagraph_capture_sizes   # same list for both regimes

def runtime_forward(model, num_tokens):
    bucket = bisect_left(BUCKETS, num_tokens)            # round up
    if bucket >= len(BUCKETS):
        entry = dispatch[model]["__eager_fallback__"]    # §4.3
    else:
        entry = dispatch[model][BUCKETS[bucket]]
    (f_cpu, f_prefetch) = entry
    # Actual forward pass executes at:
    #   num_tokens_exec = BUCKETS[bucket]  if graphs enabled (padded)
    #   num_tokens_exec = num_tokens       if graphs disabled (no pad)
    run_layer(num_tokens_exec, f_cpu, f_prefetch)
```

Why this decouples correctly:

1. **Dispatch choice depends on `num_tokens_lookup`, not `num_tokens_exec`.** The Planner's cost model `(t_gpu, t_cpu, t_prefetch)` is measured per-bucket; the dispatch entry for a bucket represents the optimal split for that bucket's workload. Looking up at the bucket position is the right decision in both regimes — the bucket represents a "size class" of arithmetic intensity.

2. **Graph-disabled mode wastes less compute.** With graphs off, the forward pass runs at exact `num_tokens`, saving the 0–20% compute-padding overhead that graph-enabled mode pays. This is measured in §0.7 of `phase0_findings.md`.

3. **Conservative dispatch under mismatch.** When `num_tokens < BUCKETS[bucket]` in graph-disabled mode (the common case — rounded up), the dispatch entry was computed for a larger workload, so `f_cpu_compute` is slightly too low relative to optimal. This underuses the CPU path by a small margin but never causes stall. Rounding down would pick an entry computed for a smaller workload, risking too-aggressive CPU dispatch and actual stall — which is why we round up.

The Scheduler implementation (see `scheduler_design.md`) is a single `bisect` call plus a dict lookup — no branching on whether graphs are enabled.

---

## 5. Constraints

### 5.1 VRAM

```
Σ_{m ∈ {gen, ver}} (W_gpu_m + KV_gpu_m + |B_prefetch_m|) ≤ VRAM_budget
```

Where `W_gpu_m = (1 − f_cpu_store_m) × total_weight_bytes_m`.

### 5.2 Host RAM

```
Σ_{m ∈ {gen, ver}} (W_cpu_m + KV_cpu_m) ≤ Host_RAM_budget
```

Where `W_cpu_m = f_cpu_store_m × total_weight_bytes_m`.

### 5.3 KV pool sizing floor

The KV pool must at least hold `n` beams' worth of KV at max context:

```
KV_gpu_m + KV_cpu_m ≥ n × max_context × kv_bytes_per_token_m
```

Otherwise the engine cannot host the target workload. If infeasible, the Planner reports infeasibility (caller must reduce `n` or `max_context`).

### 5.4 Compute-dispatch invariant

Per model, per `BatchDescriptor`:

```
f_cpu_compute + f_prefetch_compute ≤ f_cpu_store_m
```

Can't compute on a path that doesn't have storage backing it.

---

## 6. Objective

The Planner emits one dispatch entry per captured bucket; each entry should be locally optimal for that bucket. The only objective that couples buckets is the placement decision (`f_cpu_store`, `KV_gpu`, `KV_cpu`), which applies across all buckets.

**Split the objective by stage:**

- **Placement stage** (per model): minimize decode wall-clock on a model-specific *representative bucket*:
  - Generator → small `num_tokens` (≈ `n`) — matches its autoregressive decode pattern.
  - Verifier → medium `num_tokens` (≈ `n × tokens_per_step`) — matches its step-scoring pattern.
  
  Subject to the shared VRAM/RAM/KV constraints that couple the two plans.
- **Dispatch stage**: given placement, optimize each bucket's `(f_cpu_compute, f_prefetch_compute)` independently — closed-form per bucket from the idle-budget heuristic (§7.3).

The perf model (used by both stages) composes per-step time from:

- GPU compute time on `(1 − f_cpu_compute − f_prefetch_compute)` of weights (from `gpu_layer_timing`)
- Prefetch transfer time on `f_prefetch_compute` of weights (from `pcie_h2d_bw`)
- CPU compute time on `f_cpu_compute` of weights (from `cpu_gemm_curve`)
- CPU attention time (from `cpu_attn_curve`, only when attention offload is enabled)
- Merge time (constant)

The critical path per sub-module phase is `max(GPU, prefetch, CPU-compute)` plus attention-path cost. See the perf model section of `thesis_proposal.md` for the full formulation.

**What we do not model**: fine-grained prefill/decode mix over time. The Scheduler picks whichever bucket fires; our plan covers every captured bucket. Placement is optimized against the representative bucket for each model; dispatch is locally optimal per-bucket. This keeps the Planner tractable and avoids introducing workload-distribution assumptions we can't justify.

---

## 7. Solution Method

### 7.1 Two-stage: placement, then dispatch

Placement (`f_cpu_store_m`, `KV_gpu_m`, `KV_cpu_m`) is global and couples VRAM and RAM budgets. Dispatch (`f_cpu_compute`, `f_prefetch_compute` per bucket) is per-bucket given placement. Split the problem:

1. **Outer loop**: search over placement scalars (4–6 values total).
2. **Inner pass**: for each placement candidate, derive the dispatch table in closed form (see §7.3) and compute the objective.

### 7.2 Outer loop: grid search

With 4–6 scalars and a cheap objective (one evaluation takes ≤ 10 ms — profile lookups + arithmetic), grid search over reasonable discretization (say 11 values per scalar) is tractable: on the order of `11^4 ≈ 15K` evaluations, sub-second total. Brute force is justified at this scale; no need for a convex solver.

If the search space grows (more models, more flags), switch to a simple local search starting from a heuristic initialization.

### 7.3 Inner pass: closed-form dispatch table

Under layer-ahead prefetch and uniform `(f_cpu, f_prefetch)` across WQKV/MLP1/MLP2, each bucket's dispatch reduces to **a single scalar solve** (`f_cpu`), with `f_prefetch = f_cpu_store − f_cpu` fixed as the remainder.

Per bucket:

- **Small `num_tokens` (memory-BW-bound GPU)**: GPU has substantial idle time during weight fetches. CPU compute hides in that idle → maximize `f_cpu` up to the amount that fits the idle window.
- **Large `num_tokens` (compute-bound GPU)**: GPU is saturated, no idle time → CPU compute adds to critical path. Shift to `f_prefetch` to keep weights on GPU at compute time.

Closed-form:

```
idle_budget   = gpu_layer_time(bucket) − gpu_compute_time(bucket, 1 − f_cpu_store)
cpu_fit       = largest f for which (f × W_offloaded × cpu_μs_per_MB) ≤ idle_budget
f_cpu         = min(f_cpu_store, cpu_fit)
f_prefetch    = f_cpu_store − f_cpu
```

Because CPU μs/MB is uniform across WQKV/MLP1/MLP2 (`phase0_findings.md §0.3.4`), the CPU-fit computation uses a single throughput constant, not per-sub-module lookups.

**Layer-ahead prefetch feasibility check**: the total prefetch per layer is `f_prefetch × Σ_m W_m` (sum over {WQKV, MLP1, MLP2}). Cap `f_prefetch` if it exceeds `layer_time × pcie_h2d_bw`; excess falls back to `f_cpu`.

**WQKV K/V round-trip cost**: the Planner's objective must account for the round-trip when `f_cpu` falls below WQKV's K/V-fraction (see `weight_offload_design.md §Implementation Note`). Concretely, subtract `max(0, K/V-fraction − f_cpu) × W_WQKV` bytes from the effective prefetch budget for the MLP block — this nudges the optimum toward `f_cpu ≥ K/V-fraction` when PCIe budget is tight, since sitting below wastes budget on a round-trip the K/V-pin guard would otherwise avoid.

This is a dispatch heuristic, not a full optimization — sufficient given the Planner's objective smoothness and the uniform-CPU finding that eliminates per-sub-module variation. Ablation: compare the heuristic against per-bucket search on a small problem to verify it's near-optimal.

---

## 8. Dispatch Heuristic Intuition

Restated informally:

- If the GPU has nothing to do while waiting for a weight fetch, **let the CPU compute some of those weights** — that work was going to be idle anyway.
- If the GPU is already pegged, **don't pile CPU compute onto the critical path** — stream the weights into GPU instead (prefetch) so the GPU gets to them as part of its regular work.
- The cutover between these regimes is determined by the workload's `num_tokens` distribution, which is why the dispatch is bucket-indexed.

---

## 9. Verifier Degenerate Cases

Two separate reasons the verifier's plan can degenerate:

**Small verifier** (size-driven): with FastTTS defaults (Qwen2.5-Math-1.5B generator + Skywork-PRM-1.5B verifier on RTX 4090), the verifier is 1.5B × 2 bytes ≈ 3 GB and fits easily alongside the generator. Planner typically finds `f_cpu_store_ver = 0`, `KV_cpu_bytes_ver = 0`.

**Different bucket pattern** (pattern-driven, §3): the verifier's medium-`num_tokens` buckets have less GPU idle time than the generator's, so even when offloading the verifier is feasible, `f_cpu_compute` is typically smaller and `f_prefetch_compute` is preferred. This is expected — the Planner's bucket-indexed dispatch handles it automatically.

Both are emergent properties of the optimization, not hardcoded assumptions. When the generator scales up (7B, 14B), the Planner may start offloading the verifier too — let it decide.

---

## 10. Infeasibility Handling

The Planner can report infeasibility for three reasons:

1. **VRAM infeasible**: even at `f_cpu_store_m = 1`, weights + KV_gpu + overhead exceeds VRAM. Caller must reduce `n` or `max_context`.
2. **RAM infeasible**: even at `f_cpu_store_m = 0` for both models, `KV_cpu` exceeds host RAM. Same remedy.
3. **PCIe infeasible at all dispatch splits**: cannot hide any `f_prefetch_compute`, and `f_cpu_compute` slows decode beyond a user-specified threshold. Caller must reduce `n` or accept slower decode.

The Planner returns a structured error identifying which constraint is tight. The Scheduler never runs without a feasible plan.

---

## 11. Non-Goals

- **Online re-planning.** The Planner runs once at launch. Runtime adaptation to workload drift is the Scheduler's concern (and a future-work item).
- **Beyond two models.** Generator + verifier is hard-coded. Multi-expert or MoE-style extensions are out of scope.
- **Pareto output (latency vs throughput).** The Planner returns one plan per launch. Users sweep launches to explore Pareto curves.
- **Automatic selection of gen/ver VRAM split.** Inherited from FastTTS's `gpu_memory_utilization` setting. Sweeping this is future work.

---

## 12. Evaluation

How to validate the Planner in the thesis:

- **Ablation**: mechanism-only (manual constant placement) vs Planner-chosen. Expected: Planner matches or beats manual tuning without human effort.
- **`n` sweep**: for each `n ∈ {1, 4, 16, 64, 256}`, run Planner, measure throughput/latency. Expected: `f_cpu_store` and `KV_cpu_bytes` scale monotonically with `n`; throughput improvement is bucket-dependent.
- **Solver-cost report**: measured Planner runtime at launch. Expected: ≤ 1 s for typical problem sizes.
- **Dispatch heuristic vs per-bucket search**: on a small bucket set, run exhaustive per-bucket optimization; compare to closed-form heuristic. Expected: heuristic within a few percent of optimal, justifying the simplification.

---

## References

- `profiler_design.md` — source of profile tables
- `scheduler_design.md` — consumer of Planner output
- `weight_offload_design.md` — storage-vs-compute separation, tensor-granularity mechanism
- `attention_offload_design.md` — CPU suffix attention mechanism (affects `cpu_attn_curve` usage)
- `pcie_bandwidth_allocation_design.md` — PCIe invariant (fixed, not optimized)
- `thesis_proposal.md` — perf model formulation
- `vllm/v1/cudagraph_dispatcher.py` — `BatchDescriptor` and padding (`_compute_bs_to_padded_graph_size`); Planner must emit entries for `cudagraph_capture_sizes`
- `FastTTS-thesis/config.py` `SearchConfig` — workload-target source
