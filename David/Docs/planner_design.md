# Planner Design

The Planner is the primary contribution of this thesis. It takes profile tables
(from the Profiler), resource budgets, and a minimal TTC workload contract, and
emits a concrete placement and dispatch plan that the Scheduler executes at
runtime.

- **Timescale**: load-time, runs once per engine launch
- **Inputs**: profile tables, VRAM/RAM budgets, workload contract (`strategy`, `n`, `max_context`)
- **Outputs**: load-time scalars (placement) + per-bucket dispatch table

Prior offloading systems hardcode placement decisions or expose them as manual
tunables. The Planner automates this: given a profile and a workload contract,
it solves a constrained TTC performance model that decides where weights live,
how large each model's KV pools should be, and how each CUDA-graph bucket should
split CPU compute versus weight prefetch.

---

## 0. Planner v0 Summary

The v0 Planner is a **constrained discrete optimizer**. It makes two classes of
decisions:

- **Placement** per model: CPU weight storage fraction `x_m`, GPU KV bytes
  `G_m`, and CPU KV bytes `C_m`.
- **Dispatch** per model and bucket: CPU-compute fraction `u_m,b` and
  prefetch-to-GPU fraction `p_m,b`.

The central tradeoff is:

```text
T_TTC(plan) ≈ Σ_m N_rounds_m(plan, workload) × L_round_m(plan)
```

where `L_round_m` is one scheduled forward round's expected latency and
`N_rounds_m` is how many scheduled rounds the TTC workload needs. Weight
residency mostly affects `L_round_m`; KV capacity mostly affects
`N_rounds_m`. A plan is good only if its KV/admission gain outweighs its
per-round latency cost.

The per-bucket forward latency is modeled at layer granularity:

```text
t_m,b = num_layers_m × max(C_layer_m,b, P_layer_m,b)
C_layer_m,b = Σ_{op ∈ layer_ops} max(T_gpu_op,m,b, T_cpu_op,m,b) + O_layer_m,b
```

`P_layer` is the layer-ahead H2D prefetch time. `layer_ops = {WQKV, attention,
WO, MLP1, MLP2}`; the operations are sequential, while the CPU/GPU paths inside
each operation overlap. `O_layer` is a calibrated overhead term for partial
result merges, CPU/GPU synchronization, task submission, and runtime
bookkeeping.

Legal plans are discrete after snapping to tensor split axes, KV block sizes,
CPU-thread policy buckets, and CUDA graph capture sizes. The implementation can
therefore use deterministic bounded search over snapped candidates; the thesis
contribution is the modeled objective and constraints, not a particular
continuous optimizer.

### 0.1 Model calibration rule

This document records the first Planner model, not a permanent law of the
system. Planner development is empirical: write the model, emit a plan, run
representative FastTTS workloads, compare predicted latency and memory behavior
against measurements, and revise the model when residuals show a systematic
mismatch.

The equations are useful because they make assumptions explicit. If profiling
shows a better abstraction, a missing term, or a simpler equivalent model,
update the design and implementation together rather than preserving an
equation that no longer predicts the system.

---

## 1. Inputs

### 1.1 Profile tables

From the Profiler (see `profiler_design.md`):

- `gpu_layer_timing[sub_module, bucket]`
- `cpu_gemm_curve[sub_module, batch_size, slice_frac]`
- `pcie_h2d_bw[transfer_bytes]`
- `cpu_attn_curve[batch_size, suffix_context_len]`
- optional `overhead_curve[bucket, dispatch_shape]` for `O_layer`; v0 may use a
  constant placeholder and calibrate from validation runs

### 1.2 Resource budgets

Derived from the hardware and engine configuration:

```
VRAM_budget = total_VRAM − vLLM_overhead − |B_prefetch|
Host_RAM_budget = total_RAM − OS_overhead
```

- `|B_prefetch|` is the layer-ahead weight-prefetch buffer reservation per
  model. The engine-local resolver computes the exact bytes for a candidate
  dispatch plan; the Planner may use a conservative upper bound during early
  candidate pruning.
- `vLLM_overhead` ≈ 1–2 GB from engine process state (see `vllm_v1_migration.md`)

FastTTS runs two engines (generator + verifier) sharing the GPU. The
production planner is therefore a two-layer contract:

- **FastTTS/global planner**: owns the shared-memory decision across both
  engines. It chooses each engine's GPU budget, CPU KV budget, and target
  weight CPU-store fraction from the search objective.
- **vLLM/engine-local resolver**: consumes one engine's plan and turns it into
  COTS runtime geometry: snapped weight slices, optional per-bucket dispatch
  table, prefetch buffers, and validation against actual captured buckets.

The older `memory_latency_analysis.py` splitter remains a baseline sweeper: it
can test generator/verifier `gpu_memory_utilization` pairs, but it does not
model weight or KV offload. It should feed/validate the global planner, not
replace it.

### 1.3 Workload contract

From FastTTS `SearchConfig`:

- `strategy` — `{beam_search, best_of_n}`. The two representative TTC
  strategies the thesis supports. Variants (DVTS, dynamic branching) fall under
  `beam_search` for planning purposes.
- `n` — target number of completed reasoning paths for this engine launch
  (see §2)
- `max_context` — longest context the launched engine must support

Optional fields, with defaults inferred from `SearchConfig`, tokenizer limits,
or a short dry trace:

- `expected_prompt_tokens`
- `expected_output_tokens`
- `expected_step_tokens` for verifier scoring calls
- `bucket_distribution_mode` — `{analytic, traced}`

The contract is intentionally small. Users should not have to predict a large
set of runtime statistics manually; the Planner derives the rest either from a
simple analytic model or from a calibration trace.

Strategy affects KV pool sizing because prefix-sharing patterns differ. **Beam search has substantial prefix sharing** (beams share the prompt and common-ancestry tokens), so the same physical GPU KV capacity supports more logical KV than BoN. **Best-of-N has minimal sharing beyond the prompt** — each rollout's generation is independent.

The Planner consumes the contract and derives:

- `q_gen,b`, `q_ver,b` — expected bucket distributions for generator and
  verifier
- `KV_needed_m(strategy, n, max_context)` — minimum logical KV capacity after
  applying strategy-specific prefix sharing
- `S_think`, `D_gen`, `D_ver` — expected reasoning steps and model calls per
  reasoning step
- `R_gen`, `R_ver` — expected KV-limited scheduling rounds per model call
- `N_rounds_gen`, `N_rounds_ver` — expected total scheduled forward rounds per
  model

---

## 2. `n` as "max for this engine launch"

`n` is treated as a launch-time parameter analogous to vLLM's `max_num_seqs` or
`max_model_len`: the target number of completed reasoning paths the launched
engine is optimized to produce. In controlled thesis experiments this equals
the requested search size. A user requesting fewer paths at runtime just
underutilizes capacity — this is normal and not an error.

Consequences:

- **Per-launch planning.** The Planner re-runs each time the engine launches. If the user wants to sweep `n`, they launch per `n` — same UX as any other engine parameter.
- **No live `n` changes.** Weights are placed at launch; the KV pool is allocated at launch. Changing `n` mid-run is out of scope for v1.
- **FastTTS UX compatibility.** FastTTS already re-launches per experiment; the Planner adds a few seconds to that launch (solver runtime). No workflow change.

---

## 3. Per-Model, Per-Bucket Planning

### Why per-bucket

TTC serving produces forward calls spanning a wide range of shapes. Three
sources of variation contribute:

- **Continuous batching** — vLLM v1 admits requests based on KV pressure, so
  batch composition fluctuates across forward calls.
- **Prefill/decode mixing** — chunked prefill and decode tokens can share the
  same batch, widening the arithmetic-intensity range.
- **Generator/verifier asymmetry** — the generator is autoregressive decode
  (often one new token per active path per call), while the verifier scores a
  full reasoning step (many tokens per active path per call).

Arithmetic intensity varies materially across these buckets:

- Small `num_tokens` (pure decode, memory-BW-bound): GPU has idle time that CPU compute can hide in.
- Large `num_tokens` (mixed prefill-decode, trending compute-bound): GPU saturates, no idle time — CPU compute adds to the critical path.

All three axes are visible to the runtime as a forward call's `num_tokens`.
vLLM already captures one CUDA graph per `BatchDescriptor`, and the Planner
emits one dispatch entry per captured bucket. A small generator decode bucket
can lean more on CPU compute; a larger verifier step-scoring bucket may need to
lean more on prefetch. A single fixed offloading strategy would waste one end
of this range. The dispatch table is computed once at launch and consumed at
O(1) per forward call — no runtime planning overhead.

### Why per-model

The bucket mechanism is shared, but generator and verifier still need separate
tables. They have different weights, layer counts, KV sizes, profile tables,
and bucket distributions. Because the dispatch table is per-model and
per-bucket, each model adapts to its own measured cost curve while the global
Planner couples them only through shared VRAM/RAM budgets and the TTC objective.

(Motivation for why prior offloading systems handle this suboptimally — and how this addresses the thesis problem statement — lives in `thesis_proposal.md`.)

---

## 4. Outputs

### 4.1 Global Plan Output

Per model `m ∈ {generator, verifier}`:

- `gpu_memory_utilization_m` or equivalent GPU-byte budget — the global GPU
  budget assigned to this vLLM engine.
- `weight_modules_m ⊆ {qkv, mlp, wo}` — semantic weight modules eligible for
  COTS storage/compute. Default is `{qkv, mlp}`; `wo` is an opt-in forced-fit
  module only.
- `f_cpu_store_m ∈ [0, 1]` — single scalar applied uniformly to the enabled
  module set. WQKV's CPU-stored bytes are ordered by the K/V-biased picker
  (K+V head groups first, then Q tail). WO uses the implemented head-aligned
  dense output split when, and only when, `wo ∈ weight_modules_m`; planner
  policy should keep it disabled unless memory pressure makes the measured
  latency cost worthwhile (see `weight_offload_design.md §WO Split Axis
  Decision`).
- `KV_gpu_bytes_m` — GPU KV pool size
- `KV_cpu_bytes_m` — CPU KV pool size (the "extension")

The verifier often degenerates to `f_cpu_store_ver = 0` and
`KV_cpu_bytes_ver = 0` on a 24 GB RTX 4090 with a 1.5B PRM — the Planner
discovers this rather than assuming it.

### 4.2 Engine-Local Resolved Plan

The FastTTS/global planner emits one engine-local plan per vLLM instance.
The engine-local resolver turns the global byte/fraction plan into concrete
vLLM runtime kwargs:

```
FastTTS planner_config
        -> TTCSystemPlan(generator, verifier)
        -> per-engine vLLM kwargs:
           gpu_memory_utilization
           kv_offloading_size/backend
           offload_backend="cots"
           cots_weight_modules
           cots_f_cpu_store
           cots_f_prefetch
           cots_dispatch_table (optional; complete if set)
           cots_cpu_num_threads_by_bucket (optional; derived/profiled)
```

The resolver is responsible for snapping fractions to legal runtime geometry
and rejecting plans whose dispatch table does not cover the captured bucket
set.

### 4.3 Dispatch table

Per model, a table keyed by `BatchDescriptor`:

```
dispatch[model][BatchDescriptor] → (f_cpu_compute, f_prefetch_compute)
```

One entry per captured CUDA graph bucket, emitting a **single `(f_cpu, f_prefetch)` pair applied uniformly to the enabled module set** at that bucket. In the default plan this means WQKV, MLP1, and MLP2. If the Planner enables WO for a forced-fit case, WO receives the same dispatch pair rather than its own per-bucket tuning. Constraint per entry:

```
f_cpu_compute + f_prefetch_compute = f_cpu_store_m
```

Every CPU-stored byte is dispatched each forward — either CPU-computed or prefetched back to GPU. In practice the Planner picks `f_cpu_compute` and `f_prefetch_compute` falls out as the remainder (single-scalar solve per bucket; see §7.3).

**MLP1↔MLP2 matched-index invariant is automatic.** Because the same `(f_cpu_compute, f_prefetch_compute)` pair applies to both MLP1 and MLP2, and `f_cpu_store_m` is uniform across them, the col→row pipelining's index-matching requirement (MLP1's CPU output columns must equal MLP2's CPU input columns) is satisfied by construction — not an enforced Planner constraint, just a consequence of uniform dispatch.

**WQKV K/V positioning is the Planner's responsibility.** The runtime applies the dispatch table verbatim across all sub-modules; there is no K/V-pin override. To avoid K/V-output PCIe round-trips when the per-bucket `f_cpu_compute` lands below the K/V fraction, the Planner's cost model (§7.3) charges the round-trip and naturally biases toward `f_cpu_compute ≥ K/V fraction` when the budget allows. See `weight_offload_design.md §Implementation Note: WQKV K/V Positioning`.

The current simple interface supports this through vLLM's
`cots_dispatch_table` config. If the table is set, it must contain every
captured bucket used by that engine. For thesis experiments this is acceptable:
the planner config either pins `cudagraph_capture_sizes` or uses a known bucket
set from the controlled launch config. If unset, vLLM preserves today's uniform
fallback:

```
f_cpu_compute = f_cpu_store - f_prefetch
f_prefetch_compute = f_prefetch
```

Do not add a vLLM-side bucket export/partial-policy expansion layer yet. That
is a future simplification only if complete tables become a practical blocker.

### 4.4 Weight CPU Thread Policy

The Planner must be **aware** of CPU GEMM thread policy because the right
thread count depends on the CPU work implied by each dispatch entry. It should
not, however, optimize thread count as an independent search dimension. Thread
count is a deterministic policy derived from the candidate dispatch table:

```text
score(bucket) = bucket * f_cpu_compute(bucket)

score <= 0.08  -> 4 CPU threads
score <= 0.24  -> 16 CPU threads
score >  0.24  -> 24 CPU threads
```

This policy comes from the 2026-05-31 Phase 1 weight-thread experiment in
`/TTC/results/thread_policy_20260531/weight_policy_smoke/summary.md`.
It captures the important interaction between live token bucket and CPU slice
size without adding a `(slice_size x bucket x threads)` grid to the Planner.

The resolver emits this as `cots_cpu_num_threads_by_bucket` when a
`cots_dispatch_table` exists and no explicit thread map is provided. A profile
may still override the derived map explicitly. The Planner's cost model should
consume latency/throughput tables measured with this thread policy already
applied.

### 4.5 Eager-fallback entry

For `num_tokens > max_cudagraph_capture_size`, vLLM falls back to eager execution. The Planner emits one extra entry for this case (simplest: reuse the largest captured bucket's dispatch). The Scheduler looks up this entry for out-of-bucket batches.

### 4.6 Fixed constants (documented outputs, not tuned)

- `|B_prefetch_m| = layer-ahead buffer` — sized by the engine-local resolver to
  hold the enabled COTS prefetched slices for one layer (default
  WQKV/MLP1/MLP2; optional WO) (see `pcie_bandwidth_allocation_design.md
  §Prefetch Distance`).
- **Prefetch distance = layer-ahead** — one prefetch queue per layer, one sync per layer boundary. Committed (not an option). Empirically validated in `phase0_findings.md §0.10.1d`: under uniform spread (G=4 N=1) on Qwen2.5-7B at decode B=64, K=2 buys only 2.9% over K=1 and K=4 OOMs because the buffer pool grows linearly with K.
- KV placement policy — fixed two-pool mechanism: shared prefix KV on GPU,
  per-beam suffix KV on CPU. The Planner sizes the pools; it does not choose a
  different KV split mechanism.
- Per-sub-module split axis — `{WQKV: output-col, MLP1: output-col, MLP2: input-row, WO: output-col}` is hardcoded across all models and buckets (see `weight_offload_design.md §Per-Sub-Module Split Axis`). WO is runtime-supported but planner-disabled by default.
- PCIe allocation — 100% weight prefetch (see `pcie_bandwidth_allocation_design.md`).

These appear in the output for completeness so the Scheduler has one place to read the plan, but the Planner does not optimize over them.

### 4.7 Runtime dispatch lookup — graph-enabled vs graph-disabled

The dispatch table is keyed on `cudagraph_capture_sizes`, independent of
whether CUDA graphs are enabled. Runtime lookup rounds the current `num_tokens`
up to the nearest planned bucket. In graph mode this matches vLLM's padding; in
eager mode it is conservative because it selects the dispatch entry planned for
a slightly larger workload rather than a smaller one.

| Regime | Forward-pass `num_tokens` | Dispatch-lookup `num_tokens` |
|---|---|---|
| Graph enabled | Padded to nearest captured bucket | Same padded bucket |
| Eager | Exact runtime value | Rounded up to nearest planned bucket |

If `num_tokens` exceeds the largest planned bucket, the Scheduler uses the
Planner's eager-fallback entry.

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

The combined KV capacity must cover the target workload's logical KV need:

```
KV_gpu_m + KV_cpu_m ≥ KV_needed_m(strategy, n, max_context)
```

For best-of-N, `KV_needed_m` is close to `n × max_context ×
kv_bytes_per_token_m` because rollouts share little beyond the prompt. For beam
search, shared prefixes reduce the logical-to-physical KV ratio, so
`KV_needed_m` is derived from the workload contract's prefix-sharing model.
Otherwise the engine cannot host the target workload. If infeasible, the
Planner reports infeasibility (caller must reduce `n` or `max_context`).

### 5.4 Compute-dispatch invariant

Per model, per `BatchDescriptor`:

```
f_cpu_compute + f_prefetch_compute ≤ f_cpu_store_m
```

Can't compute on a path that doesn't have storage backing it.

For the default v1 dispatch table, the Planner targets equality:

```
f_cpu_compute + f_prefetch_compute = f_cpu_store_m
```

Every CPU-stored slice is either computed on CPU or streamed back to GPU for
that forward pass. The `≤` form remains the validation rule because snapped
geometry may create small legal remainders that the engine-local resolver must
either assign or reject.

---

## 6. Objective

The Planner optimizes **single-request TTC latency to `n` completed paths** for
the given workload contract. This is the user-facing edge metric: TTC creates
parallel internal work, but the deployed system returns one final answer.

Goodput and throughput remain important evaluation metrics, but they are
derived from the chosen plan rather than optimized directly. Directly
maximizing FastTTS precise goodput can prefer plans that process tokens quickly
without necessarily minimizing time-to-answer at a fixed `n` and answer
selection policy.

The objective couples placement and dispatch:

```text
T_TTC(plan) ≈
  Σ_{m ∈ {gen, ver}} N_rounds_m(plan, workload) × L_round_m(plan)
```

with:

```text
L_round_m(plan) = Σ_b q_m,b × t_m,b(plan)
N_rounds_m(plan, workload) ≈ S_think × D_m × R_m(KV_m, strategy, n)
```

where:

- `L_round_m` is the expected latency of one scheduled forward round for model `m`
- `N_rounds_m` is the expected number of scheduled forward rounds for model `m`
- `t_m,b` is the modeled forward latency for bucket `b`
- `q_m,b` is the bucket distribution per scheduled round
- `S_think` is the expected number of reasoning steps before completion
- `D_m` is the expected model calls per reasoning step
- `R_m` is the expected number of KV-limited scheduling rounds per model call

This is the thesis-facing abstraction, not a cycle-accurate simulator. A
reasoning step consists of generator calls to extend active paths and verifier
calls to score the produced step. Each model call may require multiple
scheduling rounds when KV capacity admits only a subset of active paths. For a
memory-tight launch, `R_m` is where freed VRAM becomes useful: moving some
weight/KV to CPU can increase effective KV capacity, admit more active paths
per scheduling wave, and reduce the number of waves needed to reach `n`. The
plan wins only if that scheduling gain exceeds the per-bucket latency cost from
CPU compute, prefetch, and CPU suffix attention.

Avoid double-counting: `q_m,b` describes the bucket mix within scheduled rounds,
while `N_rounds_m` counts how many rounds occur. If a traced workload contract
provides full-run bucket counts directly, the Planner can compute
`N_rounds_m × q_m,b` from those counts rather than estimating the two terms
separately.

The perf model composes each layer from two overlapping layer-scale terms:

- `C_layer` — current-layer compute, including GPU matmul work on resident and
  prefetched slices, CPU matmul work on CPU-compute slices, attention, and
  required CPU/GPU synchronization.
- `P_layer` — layer-ahead H2D transfer of the next layer's prefetched weight
  bundle, using `pcie_h2d_bw`.

The current-layer compute term is a sequential sum over layer operations:

```text
C_layer_m,b =
    Σ_{op ∈ layer_ops} max(
        T_gpu_op,m,b(...),
        T_cpu_op,m,b(...)
    )
  + O_layer_m,b
```

`layer_ops = {WQKV, attention, WO, MLP1, MLP2}`. For each operation, the CPU
and GPU paths run concurrently and the slower path gates that operation. The
operations themselves remain sequential inside the transformer layer; the
summation is over that sequence.

The split variable differs by operation. Weight terms come from
`gpu_layer_timing` and `cpu_gemm_curve`, and are controlled by the dispatch
fraction. Attention terms come from `cpu_attn_curve` plus GPU prefix-attention
timing, and are controlled by KV placement and context shape. `O_layer` is a
calibrated overhead term covering per-operation partial-result merges, CPU/GPU
sync, task submission, and runtime bookkeeping. The implementation may
decompose `O_layer` per operation if measurements show that a single
layer-level term does not predict well.

**First-attempt simplification.** If no trace is available, `q_m,b` comes from
an analytic workload contract: generator mass near decode buckets, verifier
mass near `active_paths × expected_step_tokens`, and strategy-specific prefix
sharing for KV capacity. If this proves too coarse, the same objective accepts
traced `q_m,b` without changing the Planner interface.

---

## 7. Solution Method

### 7.1 Model first, solver second

The thesis should present the Planner as the constrained performance model in
§5-§6. The implementation solver is an engineering choice. Because legal tensor
splits and KV allocations are snapped to runtime geometry, the practical
problem is discrete even though the equations are written in fractional form.

The first implementation uses a deterministic bounded search over the feasible
snapped plan space:

1. Generate legal placement candidates after snapping weight fractions and KV
   bytes to the engine's alignment constraints.
2. Reject candidates that violate VRAM, host RAM, or KV floor constraints.
3. For each remaining placement, derive the best per-bucket dispatch table
   allowed by that placement.
4. Score the full candidate with the TTC objective.
5. Emit the lowest-latency feasible plan plus diagnostics.

The contribution is the modeled objective and constraints; deterministic
candidate search is just the robust solver for a small snapped space.

### 7.2 Placement candidate generation

Placement is the only part that couples models through shared GPU/RAM budgets.
Candidates should be generated from meaningful breakpoints rather than an
arbitrary dense grid:

- `x_m` breakpoints where tensor slices gain or lose a legal head/channel group
- KV byte breakpoints where vLLM gains or loses one or more KV blocks
- forced-fit breakpoints where weights first fit in VRAM
- small prefetch-fraction breakpoints around the measured throughput crossover
  region

This keeps the solver explainable: each candidate corresponds to a real runtime
geometry change. If the candidate set grows later, use coordinate descent or
dynamic programming over memory budgets, but keep the same objective.

### 7.3 Dispatch solve per bucket

Under layer-ahead prefetch and uniform `(f_cpu, f_prefetch)` across WQKV/MLP1/MLP2, each bucket's dispatch reduces to **a single scalar solve** (`f_cpu`), with `f_prefetch = f_cpu_store − f_cpu` fixed as the remainder.

Per bucket:

- **Small `num_tokens` (memory-BW-bound GPU)**: GPU has substantial idle time during weight fetches. CPU compute hides in that idle → maximize `f_cpu` up to the amount that fits the idle window.
- **Large `num_tokens` (compute-bound GPU)**: GPU is saturated, no idle time → CPU compute adds to critical path. Shift to `f_prefetch` to keep weights on GPU at compute time.

For the first implementation, use a small snapped dispatch search:

```text
for each legal u_m,b in [0, x_m]:
    p_m,b = x_m - u_m,b
    score t_m,b(u_m,b, p_m,b)
pick the minimum-latency entry
```

This handles nonlinear CPU, PCIe, and snapping effects without adding much
runtime. A closed-form idle-budget rule may become the fast path once
validated, but the reference model should stay layer-level:

```
P_layer       = T_h2d(f_prefetch × W_prefetch_bundle_per_layer)
C_layer       = Σ_op max(T_gpu_op(bucket), T_cpu_op(bucket))
                + O_layer
layer_time    = max(C_layer, P_layer)
```

The intuition behind the closed form is still the same: choose the largest CPU
slice that does not extend `C_layer` beyond the work that would already be on
the critical path. Because CPU μs/MB is uniform across WQKV/MLP1/MLP2
(`phase0_findings.md §0.3.4`), the CPU-fit approximation can use a single
throughput constant once validated, not per-sub-module lookups.

**Layer-ahead prefetch feasibility check**: the total prefetch per layer is `f_prefetch × Σ_m W_m` (sum over {WQKV, MLP1, MLP2}). Cap `f_prefetch` if it exceeds `layer_time × pcie_h2d_bw`; excess falls back to `f_cpu`.

**WQKV K/V round-trip cost**: when `f_cpu` falls below WQKV's K/V-fraction, the residual K/V columns flow through prefetch — incurring a Phase 2 D2H back to the CPU suffix cache after K/V is computed on GPU. The Planner's objective charges this round-trip explicitly: subtract `max(0, K/V-fraction − f_cpu) × W_WQKV` bytes from the effective prefetch budget for the MLP block. This nudges the optimum toward `f_cpu ≥ K/V-fraction` when PCIe budget is tight, *via the cost model* — there is no runtime K/V-pin override; the dispatch table is applied verbatim. See `weight_offload_design.md §Implementation Note: WQKV K/V Positioning`.

The snapped dispatch search is the reference solver for validation. The
closed-form rule is accepted only if it matches the reference within a small
error bound on the profiled bucket set.

### 7.4 Future module-group dispatch

The v1 interface emits one `(f_cpu, f_prefetch)` pair per bucket, applied
uniformly across the enabled module set. Phase 1 findings suggest a better
future policy: avoid aggressive WQKV CPU compute near its cliff, while allowing
MLP1/MLP2 to use more CPU work when it overlaps well. Keep the Planner data
model extensible toward:

```text
dispatch[model][bucket][module_group] -> (f_cpu, f_prefetch)
```

but do not block the first Planner on this extra runtime surface.

---

## 8. Dispatch Heuristic Intuition

Restated informally:

- If the GPU has nothing to do while waiting for a weight fetch, **let the CPU compute some of those weights** — that work was going to be idle anyway.
- If the GPU is already pegged, **don't pile CPU compute onto the critical path** — stream the weights into GPU instead (prefetch) so the GPU gets to them as part of its regular work.
- The cutover between these regimes is determined by the bucket's `num_tokens`,
  which is why dispatch is bucket-indexed. Placement is scored by the workload
  contract's bucket distribution, not by a single universal batch size.

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
- **Goodput as the primitive objective.** Goodput is reported, but the v1
  Planner optimizes time-to-`n` completions for a fixed workload contract.
- **Online automatic replanning of gen/ver VRAM split.** The launch-time
  planner may choose the split, but runtime changes still require relaunch.

---

## 12. Evaluation

The target model matrix belongs in `thesis_proposal.md` §9. This section only
describes how to validate the Planner itself once an evaluation target is
chosen:

- **Ablation**: mechanism-only (manual constant placement) vs Planner-chosen. Expected: Planner matches or beats manual tuning without human effort.
- **`n` sweep**: for each `n ∈ {1, 4, 16, 64, 256}`, run Planner, measure throughput/latency. Expected: `f_cpu_store` and `KV_cpu_bytes` scale monotonically with `n`; throughput improvement is bucket-dependent.
- **Model prediction accuracy**: compare predicted `T_TTC`, per-model forward
  latency, and KV admission rounds against measured FastTTS runs.
- **Workload contract ablation**: analytic `q_m,b` vs short traced `q_m,b`.
  Expected: traced distributions improve prediction accuracy without changing
  the solver or output schema.
- **Solver-cost report**: measured Planner runtime at launch. Expected: ≤ 1 s for typical problem sizes.
- **Dispatch heuristic vs snapped per-bucket search**: compare the closed-form
  idle-budget rule to the snapped dispatch search. Expected: heuristic within a
  few percent of the reference before using it as the production fast path.

---

## References

- `profiler_design.md` — source of profile tables
- `scheduler_design.md` — consumer of Planner output
- `weight_offload_design.md` — storage-vs-compute separation, tensor-granularity mechanism
- `attention_offload_design.md` — CPU suffix attention mechanism (affects `cpu_attn_curve` usage)
- `pcie_bandwidth_allocation_design.md` — PCIe invariant (fixed, not optimized)
- `thesis_proposal.md` — perf model formulation and evaluation plan
- `vllm/v1/cudagraph_dispatcher.py` — `BatchDescriptor` and padding (`_compute_bs_to_padded_graph_size`); Planner must emit entries for `cudagraph_capture_sizes`
- `FastTTS-thesis/config.py` `SearchConfig` — workload-contract source
