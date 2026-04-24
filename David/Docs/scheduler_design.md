# Scheduler Design

The Scheduler executes the Planner's output at runtime. It is the only component that sees live requests and is the bridge between the static plan and the dynamic request flow.

- **Timescale**: runtime, every decode step
- **Inputs**: Planner output (placement + dispatch table), live request stream
- **Outputs**: dispatched forward passes, KV migrations, admission decisions

The Scheduler builds on vLLM's existing scheduler and dispatcher machinery rather than replacing them. Our additions are localized to tier-aware admission, KV migration between the GPU/CPU KV pools, and dispatch lookup.

---

## 1. Responsibilities

### 1.1 Tier-aware request admission

vLLM's default admission checks a single KV pool for available blocks. With the Planner's two-tier KV (`KV_gpu` + `KV_cpu`), admission must ensure:

- The new request's shared-prefix KV fits in `KV_gpu`
- Its per-path suffix KV has room in `KV_cpu`

If either tier is full, the request waits. The combined pool is not a simple sum — each tier has its own placement constraint.

### 1.2 KV migration policy

Suffix KV lives on CPU by default. The Scheduler manages:

- **Spill on beam growth**: when a beam's suffix extends past what can efficiently stay on GPU (if any intermediate buffering exists), spill to CPU.
- **Reclaim on pruning**: when beam search prunes beams, free the CPU blocks for the discarded beams.
- **Promote shared-prefix blocks**: when search trees merge or beams share new ancestry tokens, the blocks in question may need to move from CPU (per-beam) to GPU (shared). Policy for when to promote is part of the Scheduler, not the mechanism.

vLLM's `vllm/v1/kv_offload/` subsystem provides the primitives (CPU-GPU block transfer, pinned memory, async CUDA streams). The *policy* — when to migrate and which blocks — is the Scheduler's responsibility.

### 1.3 Dispatch table lookup

Every decode step, the Scheduler:

1. Identifies the current `BatchDescriptor` (via vLLM's existing code).
2. Looks up the Planner's dispatch entry: `dispatch[model][BatchDescriptor] → (f_cpu_compute, f_prefetch_compute)`.
3. Wires the lookup into the `CpuComputeDispatcher` for that forward pass.

The lookup itself is trivial (table access). The Scheduler's role here is making sure the right entry is applied to the right forward pass — particularly when generator and verifier alternate within a step.

**Eager fallback.** If the batch padding produces `num_tokens > max_cudagraph_capture_size`, vLLM's dispatcher returns `NONE` (eager). The Scheduler looks up the eager-fallback entry from the Planner instead of a captured-bucket entry. Behavior is unchanged; the entry is just pulled from a different slot.

### 1.4 Back-pressure into search

When `KV_cpu` is under pressure (e.g., > 90% full), the Scheduler signals FastTTS to:

- Prune beams more aggressively
- Defer new request admission
- Throttle beam expansion for large-`n` search

The exact signaling mechanism is a FastTTS integration detail (likely a shared flag or callback). The Scheduler produces the signal; FastTTS decides what to do with it.

---

## 2. What Stays with vLLM

Deliberately unchanged:

- **Batch composition**: which requests end up in which forward pass — vLLM's scheduler logic handles this.
- **Graph bucket selection**: `BatchDescriptor` → captured CUDA graph — vLLM's `CudagraphDispatcher`.
- **Block table management**: the core KV block bookkeeping — vLLM primitives.
- **Paging, preemption**: vLLM's paging and preemption paths remain authoritative.

The Scheduler is a *thin* layer on top of these. Our contribution is tier-awareness, not a re-implementation.

---

## 3. Adaptive Re-plan (Stretch)

If the observed workload drifts significantly from the Planner's target (e.g., context lengths double what was expected), the current plan may become suboptimal. Possible responses:

- **v1 scope**: detect drift, log, continue running.
- **v2 (future work)**: trigger a re-plan at the next natural checkpoint (e.g., between search problems). Re-capture CUDA graphs with the new dispatch table.

v1 deliberately does nothing — establishing "we never re-plan during steady-state operation" is part of the thesis's simplicity claim. v2 is called out as a natural extension.

---

## 4. Per-Search-Strategy Policy Differences

The Planner accepts search strategy as input (`beam_search` vs `best_of_n`). The Scheduler's policy differs accordingly:

- **Beam search**: higher KV migration churn. Shared prefix grows as surviving beams extend through common ancestors; pruning creates reclaim events. The Scheduler must track beam lineage to know which CPU blocks are still referenced.
- **Best-of-N**: low KV migration churn. The prompt is the only shared KV; per-rollout suffix grows independently. Migration is limited to reclaim at rollout termination.

The migration mechanisms are the same; the trigger frequency and reference-counting cost differ.

---

## 5. Integration Points

| Component | Path | Role |
|---|---|---|
| vLLM scheduler | `vllm/v1/core/sched/scheduler.py` | Base scheduler we extend |
| CUDA graph dispatcher | `vllm/v1/cudagraph_dispatcher.py` | Unchanged — provides `BatchDescriptor` routing |
| KV offload primitives | `vllm/v1/kv_offload/` | Primitives for CPU-GPU KV transfer |
| KV offload worker | `vllm/v1/kv_offload/worker/cpu_gpu.py` | Pinned-memory async transfers |
| FastTTS search loop | `FastTTS-thesis/search/` | Recipient of back-pressure signals |
| Compute dispatcher | `CpuComputeDispatcher` (new, prototype Python threading) | Consumes dispatch entry per forward pass |

---

## 6. Non-Goals

- **Re-inventing vLLM's scheduler.** Our additions are minimal policy extensions.
- **Dynamic placement.** Weights do not move at runtime. Only KV migrates.
- **Workload-aware rebalancing.** `KV_gpu` / `KV_cpu` sizes are fixed at launch by the Planner.
- **Automatic detection of search strategy changes mid-run.** Strategy is a launch-time parameter.

---

## References

- `planner_design.md` — producer of the plan this Scheduler executes
- `profiler_design.md` — upstream of the plan
- `weight_offload_design.md` — mechanism reference for compute dispatch
- `attention_offload_design.md` — mechanism reference for CPU suffix attention
- `vllm/v1/cudagraph_dispatcher.py` — `BatchDescriptor` dispatch
- `vllm/v1/kv_offload/` — KV migration primitives
- `vllm/docs/design/cuda_graphs.md` — `FULL_AND_PIECEWISE` capture model
