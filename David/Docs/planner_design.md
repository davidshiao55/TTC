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

### 0.2 Abstraction boundary

The Planner should be framed as three cooperating abstractions, not as a single
"dispatch solver":

- **Static placement**: load-time choices that allocate memory across the two
  engines, weights, and KV pools. This includes `f_cpu_store`, enabled weight
  modules, `gpu_kv_bytes`, `cpu_kv_bytes`, and engine GPU budgets.
- **Resource-lane cost model**: a per-bucket evaluator that predicts the GPU,
  CPU, and PCIe lanes under a static placement and a candidate runtime policy.
  Today the calibrated lane model covers weight dispatch:
  `max(G_B(1-u), C_B u, H_B(s-u))`; future KV terms should enter this layer as
  additional GPU/CPU/PCIe lane costs from the fixed KV placement.
- **Runtime policy compiler**: bounded, deterministic solvers that emit tables
  consumed at runtime. For v1, the only runtime-varying policy is weight
  dispatch, so the implemented `DispatchCompiler` maps
  `(bucket B, f_cpu_store s)` to `(f_cpu_compute u, f_prefetch_compute s-u)`.

This boundary keeps the current implementation honest: the weight solver is a
component of the Planner, not the Planner itself. KV placement can influence
runtime latency through the resource-lane model without becoming another
per-forward dispatch decision.

The planned implementation follows a three-stage load-time decomposition:

```text
Stage 0: ModelMemoryPartitioner
  split shared GPU/CPU memory budgets between generator and verifier

Stage 1: WeightKVPartitioner
  choose feasible per-engine weight/KV placement under an assigned budget
  score each candidate with the placement cost model

Stage 2: DispatchCompiler
  materialize runtime lookup tables for the chosen placement
```

The current prototype implements Stage 0 as exact enumeration over
generator/verifier engine-budget splits, Stage 1's weight-only subset, and
Stage 2.

For the current prototype, the Placement Cost Model scores a candidate CPU
weight storage fraction `s` by optimizing dispatch internally:

```text
Score_weight(s) =
    Σ_B q_B [
        K(B,s)
      + min_u max(G_B(1-u), C_B u, H_B(s-u))
    ]
```

`q_B` is the bucket distribution for the model/workload. After the per-engine
partitioner chooses `s`, `DispatchCompiler` reuses the same per-bucket
`argmin` to emit:

```text
dispatch[B] = (u*_B, s - u*_B)
```

This keeps the thesis-level problem as static memory placement under a
resource-lane cost model, while preserving the runtime invariant that execution
does only a table lookup. Later KV terms extend `Score_weight` into a full
placement score by adding GPU/CPU/PCIe KV costs and KV-capacity effects on
`N_rounds`; they do not require changing the dispatch table abstraction.

The code mirrors this split with visible facades:

```text
ModelMemoryPartitioner  # choose gen/ver engine budget splits
WeightKVPartitioner     # Stage 1: choose per-engine weight/KV placement
DispatchCompiler        # Stage 2: compile dispatch[B] for chosen placement
```

`WeightKVPartitioner` currently implements the weight-only subset of the full
per-engine problem: it scores feasible `f_cpu_store` candidates with
`K(B,s) + min_u lane_cost(B,s,u)`. Given an assigned engine GPU budget, it
derives the current v1 residual:

```text
gpu_kv_bytes
  = engine_gpu_budget_bytes
  - gpu_weight_bytes(s)
  - gpu_buffer_bytes(s)
```

KV memory and KV compute terms will extend that score once the KV profile is
calibrated.

### 0.3 Development strategy

Planner development should proceed **bottom-up**, while the thesis presents the
final system **top-down**.

Bottom-up development is the practical path:

1. Calibrate and validate `DispatchCompiler` first. It solves one local
   variable, `u = f_cpu_compute`, for a fixed bucket and storage fraction.
2. Use that validated dispatch result inside `WeightKVPartitioner`, which
   scores per-engine memory-placement candidates.
3. Let `WeightKVPartitioner` expose engine-local budget breakpoints and solve
   frontiers under assigned budgets, rather than exposing raw dispatch details.
4. Use `ModelMemoryPartitioner` to enumerate generator/verifier budget splits
   under shared memory budgets.

Top-down presentation is still the right thesis story: TTC creates a global
memory-allocation problem, which decomposes into per-engine weight/KV placement,
which in turn compiles into per-bucket dispatch tables.

This also clarifies the role of brute-force experiments. Exhaustive sweeps are
profiling and validation tools: they reveal patterns, fit coefficients, and
check residuals. The final Planner should be described as a
**frontier-based hierarchical optimizer**, not as blind brute force. Each layer
solves or summarizes its local subproblem and passes a compact frontier upward.

For the current two-model FastTTS setting, combining generator and verifier
frontiers by exact pairwise enumeration is sufficient and easier to debug than
a full DP/ILP solver. The same formulation generalizes naturally:

- pairwise frontier enumeration for the current generator/verifier system;
- dynamic programming / multiple-choice knapsack if there are many engines,
  many independent placement groups, or fine memory-budget bins;
- ILP only if the final constraints become irregular enough that a tabular DP
  is awkward.

The dispatch layer should not use DP: after fixing `B` and `s`, it is a
one-variable solve. The DP-like structure belongs at the memory-placement layer,
where candidates compete for shared GPU and CPU budgets.

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
  model. vLLM's runtime pool is sized from the maximum effective prefetch rows
  across dispatch buckets, but the Planner uses an option-A conservative
  full-store reservation so buffer capacity depends on `f_cpu_store`, not on
  the chosen dispatch table.
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
- Large prefill or mixed prefill-decode calls (trending compute-bound): GPU saturates, no idle time — CPU compute often adds to the critical path.

All three axes are visible to the runtime as a forward call's `num_tokens`.
vLLM already captures one CUDA graph per `BatchDescriptor`, and the Planner
emits one dispatch entry per captured bucket. A small generator decode bucket
can lean more on CPU compute; a larger verifier step-scoring bucket may need to
lean more on prefetch. A single fixed offloading strategy would waste one end
of this range. The dispatch table is computed once at launch and consumed at
O(1) per forward call — no runtime planning overhead.

**Validation caveat: bucket is a shape proxy, not a semantic phase label.**
The first dispatch-validation sweep forced one split across every captured
bucket and made CPU compute look much worse than it is for decode: prefill
buckets also used the CPU path, and prefill is where CPU work most easily
becomes the bottleneck. A follow-up "decode-only" validation kept all non-decode
buckets pure prefetch and varied only the measured decode bucket; that exposed
a much larger CPU-compute regime. The Planner should therefore treat bucket
profiles as phase-composition dependent. For v1, this can be approximated by
setting prefill-heavy buckets prefetch-heavy and decode buckets independently.
If prefill and decode routinely collide in the same captured bucket, extend the
dispatch key from `BatchDescriptor` to `(BatchDescriptor, phase_class)`.

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
  COTS storage/compute. Production default is `{qkv, mlp, wo}`; the selector is
  a compatibility/ablation hook, not a Planner policy dimension.
- `f_cpu_store_m ∈ [0, 1]` — single scalar applied uniformly to the enabled
  module set. WQKV's CPU-stored bytes are ordered by the K/V-biased picker
  (K+V head groups first, then Q tail). WO uses the implemented QKVO-aligned
  dense output split with the production coarser snap quantum (see
  `weight_offload_design.md §WO Split Axis
  Decision`).
- `gpu_kv_bytes_m` — GPU KV pool size
- `cpu_kv_bytes_m` — CPU KV pool size (the "extension")

The verifier often degenerates to `f_cpu_store_ver = 0` and
`cpu_kv_bytes_ver = 0` on a 24 GB RTX 4090 with a 1.5B PRM — the Planner
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

One entry per captured CUDA graph bucket, emitting a **single `(f_cpu,
f_prefetch)` pair applied uniformly to the enabled module set** at that bucket.
In the production plan this means WQKV, MLP1, MLP2, and WO. WO receives the
same dispatch pair as the rest of the module set; its coarser snap quantum
decides when the requested fraction is large enough to create WO CPU/prefetch
work. Constraint per entry:

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
Σ_{m ∈ {gen, ver}} (gpu_weight_bytes_m + gpu_kv_bytes_m + gpu_buffer_bytes_m) ≤ VRAM_budget
```

Where `gpu_weight_bytes_m = (1 − f_cpu_store_m) × total_weight_bytes_m`.

### 5.2 Host RAM

```
Σ_{m ∈ {gen, ver}} (cpu_weight_bytes_m + cpu_kv_bytes_m) ≤ Host_RAM_budget
```

Where `cpu_weight_bytes_m = f_cpu_store_m × total_weight_bytes_m`.

### 5.3 KV pool sizing floor

The combined KV capacity must cover the target workload's logical KV need:

```
gpu_kv_bytes_m + cpu_kv_bytes_m ≥ kv_needed_bytes_m(strategy, n, max_context)
```

For best-of-N, `kv_needed_bytes_m` is close to `n × max_context ×
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

### 7.1 Frontier-based hierarchy

The thesis should present the Planner as the constrained performance model in
§5-§6. The implementation solver is an engineering choice. Because legal tensor
splits and KV allocations are snapped to runtime geometry, the practical
problem is discrete even though the equations are written in fractional form.

The implementation should use a frontier-based hierarchy:

1. `DispatchCompiler` solves the local per-bucket dispatch problem for fixed
   placement, returning the best `u`, predicted latency, and bottleneck.
2. `WeightKVPartitioner` exposes snapped per-engine minimum-budget
   breakpoints, then solves placement candidates under an assigned engine
   budget. This is where residual `gpu_kv_bytes` is derived.
3. `ModelMemoryPartitioner` enumerates generator/verifier engine-budget splits
   under shared GPU/CPU budgets and selects the lowest predicted TTC objective.

This is not blind brute force. The candidate generation is bounded by real
runtime breakpoints, and each layer passes only nondominated candidates upward.
The contribution is the modeled objective, constraints, and decomposition;
deterministic frontier enumeration is the robust solver for the current small
snapped space.

For two engines, exact pairwise frontier enumeration is enough:

```text
best = None
for split in candidate_budget_splits:
    gen_frontier = generator_weight_kv.solve(split.generator_budget)
    ver_frontier = verifier_weight_kv.solve(split.verifier_budget)
    for g in gen_frontier:
        for v in ver_frontier:
            if fits_global_budgets(g, v):
                score = TTC_objective(g, v, workload)
                best = min(best, (score, g, v))
```

If the system later adds more engines or many independent memory-placement
groups, this same abstraction becomes a dynamic program over memory budgets:

```text
dp[i][gpu_used][cpu_used] =
    best objective after choosing candidates for the first i groups
```

An ILP formulation is also possible, but it should be treated as a presentation
or future generalization unless the constraints become irregular enough to make
frontier enumeration or DP awkward.

### 7.2 Placement candidate generation

Placement is the only part that couples models through shared GPU/RAM budgets.
Candidates should be generated from meaningful breakpoints rather than an
arbitrary dense grid:

- `x_m` breakpoints where tensor slices gain or lose a legal head/channel group
- KV byte breakpoints where vLLM gains or loses one or more KV blocks
- forced-fit breakpoints where weights first fit in VRAM
- small prefetch-fraction breakpoints around the measured throughput crossover
  region

In the current implementation, `WeightKVPartitioner` first exposes exact
weight+buffer feasibility breakpoints:

```text
min_engine_gpu_budget(s)
  = gpu_weight_bytes(s) + gpu_buffer_bytes(s)
```

`ModelMemoryPartitioner` combines those breakpoints with a coarse
`global.engine_gpu_budget_step_bytes` grid to enumerate generator/verifier
engine-budget splits. Then `WeightKVPartitioner` solves each assigned engine
budget; any budget above `min_engine_gpu_budget(s)` becomes that engine's
derived `gpu_kv_bytes`.

The current conservative buffer model is:

```text
gpu_buffer_bytes(s)
  = round(s * gpu_buffer_bytes_per_store_fraction)
```

where `gpu_buffer_bytes_per_store_fraction` is the full-store COTS GPU
workspace estimate: full prefetch pool plus GPU output scratch. The normal
planner path derives that coefficient from model geometry:

```text
prefetch_pool_full_store_bytes
  = K_slots × dtype_bytes × Σ_unique_enabled_slot_shapes slot_numel

output_scratch_full_store_bytes
  = max_num_batched_tokens × max_enabled_cpu_output_dim × dtype_bytes
```

The prefetch term is intentionally based on unique slot shapes, not total layer
count, because vLLM shares `K=2` prefetch slots across layers with the same
role/shape. A profiler may also emit the already-combined
`gpu_buffer_bytes_per_store_fraction`; a fixed `gpu_buffer_bytes` remains
available as a debug/override path.

This keeps the solver explainable: each candidate corresponds to a real runtime
geometry change. After scoring, prune candidates that are worse in both
resources and objective than another candidate:

```text
drop a if exists b such that:
    b.gpu_bytes <= a.gpu_bytes
    b.cpu_bytes <= a.cpu_bytes
    b.predicted_objective <= a.predicted_objective
    b.gpu_kv_bytes >= a.gpu_kv_bytes
    and at least one comparison is strict
```

For the current weight-only subset, the frontier axes are `gpu_bytes`,
`cpu_bytes`, `gpu_kv_bytes`, and predicted bucket-weighted latency. Once KV is
added, the frontier can convert KV capacity into admitted batch, expected
rounds, or a direct `N_rounds` multiplier.

### 7.3 Dispatch solve per bucket

Under layer-ahead prefetch and uniform `(f_cpu, f_prefetch)` across WQKV/MLP1/MLP2, each bucket's dispatch reduces to **a single scalar solve** (`f_cpu`), with `f_prefetch = f_cpu_store − f_cpu` fixed as the remainder.

Per bucket:

- **Decode-dominated buckets**: CPU compute can hide in the GPU decode window.
  The solve may legitimately choose high `f_cpu`, even at moderate batch.
- **Prefill-heavy or mixed buckets**: GPU compute is denser and CPU work is
  more likely to become critical-path. Bias toward prefetch unless measured
  profiles show a CPU slice still fits.

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

**Validation checkpoint (2026-06-03, Qwen2.5-7B-Instruct, graph mode,
`input_len=8`, `output_len=32`).** After separating CUDA graph capture buckets
from COTS dispatch buckets, the validation harness supports non-uniform routing:
every non-decode bucket is fixed to pure prefetch and only the measured decode
bucket varies. This isolates decode policy from prefill contamination and
supports the three-lane overlap model:

```text
T_decode(B, s, u) =
    K(B,s) + max(
        G_B * (1 - u),
        C_B * u,
        H_B * (s - u),
    )
```

where `s = f_cpu_store`, `u = f_cpu_compute`, and `s - u` is the prefetched
CPU-resident weight fraction. `G_B` is measured with the `--cots-dry-run`
control, while `K(B,s)` is the split-invariant cost for that bucket and storage
fraction: it may change with `B` or `s`, but not with the CPU-vs-prefetch split
`u`. A snapped-grid fit over B=8/16/32/64 matched the measured best split for
every store fraction except one B=8 near-tie, where the predicted cell was one
grid step away and only 5 us slower:

| decode B | G s/frac | C s/frac | H s/frac | C/H | continuous u*/s | rank check |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 0.4440 | 8.3320 | 13.0612 | 0.638 | 0.611 | 4/5 exact, 5/5 +/-1 |
| 16 | 0.4454 | 13.0482 | 13.4518 | 0.970 | 0.508 | 4/4 exact |
| 32 | 0.4675 | 25.5703 | 14.6419 | 1.746 | 0.364 | 5/5 exact |
| 64 | 0.4698 | 49.4604 | 15.0140 | 3.294 | 0.233 | 4/4 exact |

The measured optima move smoothly with batch: roughly 61% of the CPU-resident
slice should compute on CPU at B=8, 51% at B=16, 36% at B=32, and 23% at B=64.
In this decode grid, the GPU lane is measured and included but is not the active
bottleneck; the best split is set by balancing CPU compute against H2D prefetch.
The Planner rule is therefore: keep the three-lane model, solve per bucket, and
profile prefill-heavy buckets separately before applying the same rule there.

**Planner hook (2026-06-03).** `fit_dispatch_cost_model.py` now exports a
`weight_dispatch_profile.json` artifact with schema
`weight_three_lane_v1`, containing `G_B`, `C_B`, `H_B`, optional `K(B,s)`
values per calibrated bucket, and a minimal `weight_resource_model`. This JSON
is the Profiler output and Planner input:

```json
{
  "schema_version": 1,
  "dispatch_model": "weight_three_lane_v1",
  "weight_resource_model": {
    "total_weight_bytes": 123456789,
    "gpu_buffer_bytes_per_store_fraction": 987654321,
    "buffer_model": "cots_option_a_v1"
  },
  "cots_snap": {
    "schema_version": 1,
    "snap_model": "cots_snap_v1",
    "storage_by_store_fraction": {
      "0.15": {
        "cpu_weight_bytes": 1849700000,
        "gpu_buffer_bytes": 224400000
      }
    }
  },
  "buckets": {
    "8": {
      "G_s_per_fraction": 0.444,
      "C_s_per_fraction": 8.332,
      "H_s_per_fraction": 13.061,
      "K_by_store_s": {
        "0.15": 0.592
      }
    }
  }
}
```

`weight_resource_model` is the compact resource model: total weight bytes plus
the option-A full-store GPU-buffer coefficient used for linear fallback. The
`cots_snap` block is the runtime realization profile. It records what vLLM COTS
actually produced after projecting requested fractions onto legal tensor
geometry:

```text
requested placement policy -> COTS snapping -> realized placement geometry
```

vLLM remains the source of truth for snapping because it owns the actual tensor
handles, QKV head grouping, MLP 64-channel granularity, WO dense-output
granularity, and prefetch-slot layout. The Planner consumes the realized
consequences as
calibrated facts. When `cots_snap.storage_by_store_fraction[s]` contains exact
`cpu_weight_bytes` or `gpu_buffer_bytes`, those values override the linear
estimate for that `s`; otherwise the planner falls back to
`total_weight_bytes * s` and `gpu_buffer_bytes_per_store_fraction * s`.

The older direct maps under `weight_resource_model` remain accepted for
compatibility:

```json
{
  "weight_resource_model": {
    "total_weight_bytes": 123456789,
    "gpu_buffer_bytes_per_store_fraction": 987654321,
    "cpu_weight_bytes_by_store_fraction": {
      "0.15": 1849700000
    },
    "gpu_buffer_bytes_by_store_fraction": {
      "0.15": 224400000
    }
  }
}
```

The planner does not need raw model geometry in the normal path; geometry and
slot-shape breakdowns belong in profiler debug artifacts unless a later planner
variable uses them.
The current fit helper accepts `--total-weight-bytes` and
`--gpu-buffer-bytes-per-store-fraction` (or prefetch/scratch component split
flags) to attach the compact resource model. When snapped byte maps are supplied
with `--cpu-weight-bytes-by-store-fraction-json` and
`--gpu-buffer-bytes-by-store-fraction-json`, the helper also emits
`cots_snap_v1`.
The FastTTS manual planner can consume this via
`weight.dispatch_cost_profile_path` plus explicit `weight.dispatch_buckets`.
With a fixed `weight.f_cpu_store`, it derives the runtime
`cots_dispatch_table`; with global model-memory planning, it derives the
candidate storage grid from the common `K(B,s)` support in the profiler artifact
unless `weight.f_cpu_store_candidates` is supplied as an explicit experiment
constraint. It then runs `WeightKVPartitioner` over those feasible storage
candidates and compiles the dispatch table for the selected storage fraction.
The current exact-bucket solver does not interpolate unprofiled buckets; production configs
must either provide a coefficient row for each dispatch bucket or intentionally
restrict the dispatch bucket grid to the calibrated set. The measured-grid
regression is `validate_weight_dispatch_solver.py`: the current B=8/16/32/64
decode profile matches 17/18 best cells exactly and 18/18 within one grid step.

The planner also has a first config-facing Stage-0 path: if `planner_config`
contains a `global` block with shared `gpu_budget_bytes` and `cpu_budget_bytes`,
both generator and verifier provide dispatch cost profiles with
`weight_resource_model` and calibrated dispatch buckets. `ModelMemoryPartitioner`
derives per-engine GPU-budget candidates from the global GPU budget, Stage-1
weight+buffer feasibility breakpoints, and optional
`global.engine_gpu_budget_step_bytes`. Explicit `weight.f_cpu_store_candidates`
and `weight.engine_gpu_budget_candidates` remain debug/override paths. The planner
uses `ModelMemoryPartitioner` to enumerate engine budget splits, asks each
`WeightKVPartitioner` to solve placement under its assigned budget, and then
emits ordinary engine-local vLLM overrides:

Resource-field naming follows the placement perspective and uses device-first
byte fields: `gpu_weight_bytes`, `cpu_weight_bytes`, `gpu_kv_bytes`,
`cpu_kv_bytes`, and `gpu_buffer_bytes`. The last field is reserved COTS GPU
workspace: conservative full-store prefetch slots plus GPU output scratch. In
the full model, both `gpu_buffer_bytes` and `gpu_kv_bytes` are derived:
`gpu_buffer_bytes` from `f_cpu_store`, and `gpu_kv_bytes` inside
`WeightKVPartitioner` from the chosen engine GPU budget after
`gpu_weight_bytes` and `gpu_buffer_bytes` are reserved.
Until the full KV-capacity objective lands, equal-latency model-memory
candidates are tie-broken by maximizing the smaller per-engine `gpu_kv_bytes`
before maximizing total assigned GPU bytes. This keeps the current two-engine
plans from assigning all residual KV to one role merely because weight latency
is identical.

```yaml
planner_config:
  global:
    gpu_budget_bytes: ...
    cpu_budget_bytes: ...
    engine_gpu_budget_step_bytes: ...  # optional
    engine_weights:
      generator: 1.0
      verifier: 1.0
  generator:
    weight:
      dispatch_cost_profile_path: ...
      dispatch_buckets: [8, 16, 32, 64]
  verifier:
    weight:
      dispatch_cost_profile_path: ...
      dispatch_buckets: [8, 16, 32, 64]
```

For planner-only checks, use:

```bash
python David/Benchmarks/planner/plan_from_profiles.py \
  --generator-profile gen_weight_profile.json \
  --verifier-profile ver_weight_profile.json \
  --gpu-budget-gb 22 \
  --cpu-budget-gb 96 \
  --dispatch-buckets 8,16,32,64
```

After launching vLLM/FastTTS with the emitted plan, validate that the runtime
reserved the same resources:

```bash
python David/Benchmarks/planner/validate_runtime_memory_accounting.py \
  --plan-json planner_plan.json \
  --runtime-log run.log \
  --role-order generator,verifier
```

This compares planner `cpu_weight_bytes` against COTS `weights_saved`,
`gpu_buffer_bytes` against runtime `gpu_uva + prefetch_pool`,
`gpu_kv_bytes` against `kv_cache_memory_bytes`, and the dispatch table against
the one-time COTS dispatch policy log. These checks are intentionally about
accounting and route application, not throughput; performance validation still
uses the measured grid and end-to-end benchmark runs.

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

**Small verifier** (size-driven): with FastTTS defaults (Qwen2.5-Math-1.5B generator + Skywork-PRM-1.5B verifier on RTX 4090), the verifier is 1.5B × 2 bytes ≈ 3 GB and fits easily alongside the generator. Planner typically finds `f_cpu_store_ver = 0`, `cpu_kv_bytes_ver = 0`.

**Different bucket pattern** (pattern-driven, §3): the verifier's medium-`num_tokens` buckets have less GPU idle time than the generator's, so even when offloading the verifier is feasible, `f_cpu_compute` is typically smaller and `f_prefetch_compute` is preferred. This is expected — the Planner's bucket-indexed dispatch handles it automatically.

Both are emergent properties of the optimization, not hardcoded assumptions. When the generator scales up (7B, 14B), the Planner may start offloading the verifier too — let it decide.

---

## 10. Infeasibility Handling

The Planner can report infeasibility for three reasons:

1. **VRAM infeasible**: even at `f_cpu_store_m = 1`, weights + `gpu_kv_bytes` + overhead exceeds VRAM. Caller must reduce `n` or `max_context`.
2. **RAM infeasible**: even at `f_cpu_store_m = 0` for both models, `cpu_kv_bytes` exceeds host RAM. Same remedy.
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
- **`n` sweep**: for each `n ∈ {1, 4, 16, 64, 256}`, run Planner, measure throughput/latency. Expected: `f_cpu_store` and `cpu_kv_bytes` scale monotonically with `n`; throughput improvement is bucket-dependent.
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
