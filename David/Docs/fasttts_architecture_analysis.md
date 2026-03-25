# FastTTS-AE Architecture Analysis

> Analysis of the FastTTS artifact evaluation codebase, its architecture, how it maps to the paper's claims, and paths forward for V1 migration.

---

## 1. Repository Structure

| File/Directory | Purpose |
|---|---|
| `fasttts.py` | Top-level `FastTTS` class. Entry point for search. |
| `core.py` | Alternate `FastTTS` class (async version, uses `fasttts.` package imports). |
| `config.py` | `FastTTSConfig` (model/engine params) and `SearchConfig` (per-request search params). |
| `models/vllm_wrapper.py` | `GeneratorVLLMModelWrapper`, `VerifierVLLMModelWrapper` — multiprocess wrappers around vLLM. |
| `models/tts_llm.py` | `TTSLLM` — extends vLLM's `LLM` class. Selects engine (V0/V1), dispatches scoring. |
| `models/generator_engine.py` | `GeneratorLLMEngine` — extends V0 `LLMEngine` with Speculative Beam Extension. |
| `models/generator_engine_v1.py` | `GeneratorLLMEngineV1` — V1 stub, speculative beam extension is **not implemented** (placeholder). |
| `models/verifier_engine.py` | `VerifierLLMEngine` — minimal V0 `LLMEngine` extension, injects `CustomScheduler`. |
| `models/custom_scheduler.py` | `CustomScheduler` — extends V0 `Scheduler`, aborts spec-finished sequences from waiting/swapped queues. |
| `models/spec_stopchecker.py` | `SpecStopChecker` — suppresses sequence finishing on stop strings to enable speculative continuation. |
| `models/reward_utils.py` | `prepare_input`, `sigmoid` — utilities for Skywork PRM scoring. |
| `search/beam_search.py` | Core beam search algorithm (`_beam_search`, `beam_search`). |
| `search/dvts.py` | DVTS search strategy. |
| `search/best_of_n.py` | Best-of-N search strategy. |
| `search/dynamic_branching.py` | Dynamic branching search strategy. |
| `search/vg_search.py` | Verifier-guided search strategy. |
| `search/beam.py` | `Beam` dataclass — per-beam state (text, scores, future_texts, etc.). |
| `search/utils.py` | Utilities: `build_conversation`, `aggregate_scores`, `split_string_by_separator`, `truncate_sentence_by_tokens`, `assign_prefix_priorities`. |

---

## 2. Execution Flow

### 2.1 Top-Level API

```
FastTTS(config).search(problems)
  -> initialize()  # creates generator + verifier wrappers
  -> _process_batch(problems, search_config)
     -> dispatches to search strategy (beam_search / dvts / best_of_n / etc.)
```

### 2.2 Two-Model System

FastTTS uses two models:
- **Generator**: produces candidate solution steps (e.g., `Qwen2.5-Math-1.5B-Instruct`)
- **Verifier (PRM)**: scores each step to guide search (e.g., `Skywork-o1-Open-PRM-Qwen-2.5-1.5B`)

### 2.3 Process Architecture

```
Main Process
  |-- FastTTS.search()
  |     |-- beam_search() / dvts() / best_of_n() / ...
  |     |     |-- generator.generate() --> Pipe --> Generator Process (CUDA ctx 1)
  |     |     |                                      |-- TTSLLM + GeneratorLLMEngine (V0)
  |     |     |-- verifier.score()    --> Pipe --> Verifier Process (CUDA ctx 2)
  |     |                                          |-- TTSLLM + VerifierLLMEngine (V0/V1)
```

Both models run in **separate processes** via `multiprocessing.Process` for CUDA context isolation. Communication uses `mp.Pipe` with a request/response protocol (actions: `generate`, `score`, `apply_chat_template`, `encode`, `decode`, `shutdown`).

### 2.4 Engine Selection

- **Generator**: Always uses **V0** (`VLLM_USE_V1=0`, set in `vllm_wrapper.py:110`)
- **Verifier**: Uses **V1** (`VLLM_USE_V1=1`, set in `vllm_wrapper.py:112`)

The generator uses V0 because Speculative Beam Extension depends on V0-specific internals (see Section 5).

### 2.5 Sleep Mode (Model Swapping)

When `offload_enabled=True`:
```
generator.generate():
    model.wake_up()    # restore weights from CPU -> GPU
    model.generate_text(prompts)
    model.sleep()      # offload weights GPU -> CPU, discard KV cache
```

This allows both models to time-share the same GPU. Each model gets `gpu_memory_utilization=0.95` when active.

---

## 3. Beam Search Algorithm

The core algorithm (`search/beam_search.py:_beam_search`):

```
Initialize n beams per problem
For each iteration (up to num_iterations):
  1. Duplicate active beams to fill n slots (with random truncation for diversity)
  2. Handle speculative beam extension (consume future_texts if available)
  3. Build chat conversations -> apply chat template
  4. generate_beam() -> get next step text + optional lookahead steps
  5. score_beam() -> get PRM scores from verifier
  6. Aggregate scores (last/min/prod/mean strategy)
  7. Prune: keep top n/beam_width beams by aggregated score
  8. Move completed beams to completed_beams list
  9. Early exit if n completions collected
Return completed beams sorted by score
```

Key details:
- Stop token is `\n\n` (step boundary in math reasoning)
- `generate_beam()` generates one step per beam, plus optional greedy lookahead steps
- `score_beam()` calls `verifier.score(prompts, completions)` for per-step PRM scores
- Score aggregation strategies: `last` (default), `min`, `prod`, `mean`

---

## 4. Paper Claims vs. Implementation

The paper (FastTTS, ASPLOS '26) claims three optimizations:

### 4.1 Speculative Beam Extension

**Paper**: Generates speculatively to hide latency of irregular workloads (stragglers).

**Implementation**: **Implemented** in `generator_engine.py:_process_model_outputs_spec`. When a sequence hits `\n\n`, instead of finishing it, the engine keeps it alive for speculative continuation by:
- Setting `stop_reason` but NOT changing `status` to `FINISHED_STOPPED` (in `SpecStopChecker`)
- Assigning `SPEC_BEAM_CANDIDATE_PRIORITY` to speculative sequences
- Only truly finishing when all sequences in the batch are done or queue exceeds 256

### 4.2 Dynamic Prefix-Aware Scheduling

**Paper**: Reorders execution to maximize KV cache reuse from dynamic prefix sharing.

**Implementation**: **Partially implemented / disabled**. The code exists but is commented out:
- `assign_prefix_priorities()` in `search/utils.py` is implemented but never called (commented out in `beam_search.py:111-116` and `custom_scheduler.py:38-55`)
- The beam duplication logic in `_beam_search` has a prefix-aware path (`if getattr(generator.config, 'prefix_aware_scheduling', False)`) that places duplicate beams adjacent for better prefix locality
- Config flag `enable_prefix_aware_scheduling` exists but the scheduler-level implementation is commented out

### 4.3 Asymmetric Multi-Model Memory Allocation

**Paper Section 4.3.1**: Roofline-guided KV allocation — dynamically partition KV cache between generator and verifier to minimize total execution time.

**Implementation**: **Not implemented as described**. The implementation uses:
- **Baseline mode**: Static split — generator gets 19%, verifier gets 71% GPU memory (for 1.5B+7B)
- **Offload mode**: Each model gets 95% when active (time-sharing via sleep/wake)
- There is no roofline model, no dynamic allocation searcher, no runtime partition adjustment

**Paper Section 4.3.2**: "Extended Search Space with Offloading" — "the KV cache of the inactive model is offloaded to CPU memory, enabling a single model to fully utilize the GPU cache space."

**Implementation**: **Does NOT match the paper's claim**. What actually happens:

| Component | Paper claims | Code does |
|---|---|---|
| **What's offloaded** | KV cache of inactive model | **Model weights** of inactive model |
| **KV cache** | Preserved on CPU | **Discarded** (freed, not saved) |
| **Mechanism** | Custom FastTTS optimization | **Stock vLLM sleep mode** (`CuMemAllocator.sleep/wake_up`) |
| **On wake_up** | Reload KV from CPU (skip recomputation) | Reallocate empty KV cache (must recompute) |

The sleep mode implementation chain:
```
FastTTSConfig.offload_enabled
  -> enable_sleep_mode=True (passed to model wrappers)
  -> model.sleep()  (in child process, after each inference call)
  -> vLLM gpu_worker.sleep(level=1)
  -> CuMemAllocator.sleep(offload_tags=("weights",))
     -> Copies weight tensors to CPU pinned memory (cudaMemcpy GPU->CPU)
     -> Unmaps and releases ALL GPU memory (weights backed up, KV cache discarded)
```

This is entirely stock vLLM functionality. FastTTS adds no custom offloading logic.

---

## 5. Why the Generator Uses V0 (Not V1)

Speculative Beam Extension requires fine-grained control over sequence lifecycle that only V0 provides:

### V0 hooks exploited by FastTTS

1. **`_process_model_outputs()` override** — `GeneratorLLMEngine` overrides this 200+ line method to inspect every sequence after each decode step and selectively keep sequences alive past stop strings.

2. **`SpecStopChecker`** — Custom `StopChecker` that sets `seq.stop_reason` but deliberately does NOT set `seq.status = FINISHED_STOPPED` (lines 86, 101 commented out). This keeps sequences alive in the scheduler for speculative continuation.

3. **Direct scheduler queue access** — `SpecStopChecker` checks `len(self.scheduler.waiting) == 0` to decide whether to use speculative mode. `CustomScheduler.schedule()` iterates `self.waiting` and `self.swapped` to abort spec-finished sequences.

4. **Per-sequence status/priority manipulation** — Sets `seq_group.priority = SPEC_BEAM_CANDIDATE_PRIORITY` for speculative sequences, directly sets `seq.status = SequenceStatus.FINISHED_STOPPED` when done, calls `scheduler.free_seq()`.

5. **`SingleStepOutputProcessor` replacement** — Replaces the default output processor with one using the custom `SpecStopChecker`.

### V1 architectural differences that prevent direct porting

| V0 | V1 |
|---|---|
| `LLMEngine` directly owns `Scheduler` and `OutputProcessor` | Scheduler runs inside `EngineCore` (potentially separate process) |
| Can access `scheduler.waiting`, `scheduler.running` queues directly | Scheduler is behind `EngineCoreClient` API boundary |
| `_process_model_outputs()` gives per-sequence-group control | `OutputProcessor.process_outputs()` processes flat `EngineCoreOutput` structs |
| `Sequence` / `SequenceGroup` objects with mutable `status`, `stop_reason` | `Request` objects with `priority` field, but finishing is via `finish_requests()` / `abort_requests()` |
| Custom `StopChecker` can suppress finishing | Stop detection is in detokenizer, automatically triggers abort |

### `GeneratorLLMEngineV1` — the V1 stub

`models/generator_engine_v1.py` exists but is a **placeholder**:
- `enable_spec_beam_extension()` just sets a flag (line 134)
- `_apply_spec_beam_extension_v1()` returns outputs unchanged (lines 182-192)
- Comments say "This is a placeholder implementation"

---

## 6. vLLM KV Cache Offloading Capabilities

vLLM has multiple KV cache management systems. FastTTS uses almost none of them intentionally.

### 6.1 V0 Swap Space (Preemption)

- **What**: When GPU KV cache is full, scheduler preempts low-priority sequences by swapping KV blocks to CPU pinned memory.
- **Config**: `swap_space=4` GiB (default in `TTSLLM.__init__`, never overridden by FastTTS)
- **How**: Reactive — only triggers on GPU KV cache pressure. Not proactive capacity extension.
- **FastTTS usage**: Passively available via default. Never configured or tuned.

### 6.2 V1 KV Offload System (`vllm/v1/kv_offload/`)

- **What**: Proactive CPU<->GPU KV cache offloading framework. Uses CPU memory as secondary KV cache storage with managed eviction.
- **Key components**:
  - `OffloadingManager` — scheduler-side tracking (lookup, prepare_load, prepare_store, complete_load, complete_store)
  - `CPUOffloadingSpec` — configures CPU memory pool with `cpu_bytes_to_use`, LRU or ARC eviction
  - `CpuGpuOffloadingHandlers` — allocates pinned CPU tensors, async block transfers via `ops.swap_blocks()` on dedicated CUDA streams
  - `LRUOffloadingManager` / `ARCOffloadingManager` — eviction policies
- **Config**: Activated via `kv_transfer_config` with `spec_name="CPUOffloadingSpec"`
- **Marked as**: Experimental
- **FastTTS usage**: **Not used**. Generator forces V0 (`VLLM_USE_V1=0`), and `kv_transfer_config` is never set.

### 6.3 Distributed KV Transfer (`vllm/distributed/kv_transfer/`)

- **What**: Transfer KV caches between vLLM instances (e.g., disaggregated prefill/decode across machines).
- **Not relevant** to single-GPU scenarios.

### 6.4 Weight Offloaders (`vllm/model_executor/offloader/`)

- `PrefetchOffloader` — streams model weights layer-by-layer CPU->GPU with double-buffering
- `UVAOffloader` — keeps weights in pinned CPU memory, GPU accesses via Unified Virtual Addressing
- **These offload weights, not KV cache.**
- **FastTTS usage**: Not used (`cpu_offload_gb=0` default, never changed).

### 6.5 Sleep Mode (`vllm/device_allocator/cumem.py`)

- **What**: Offloads model weights to CPU, discards KV cache. For time-sharing GPU between models or RLHF weight updates.
- **FastTTS usage**: **This is what `offload_enabled` triggers.** The only offloading mechanism FastTTS actually uses.

### Summary

| System | FastTTS uses? | Would benefit FastTTS? |
|---|---|---|
| V0 Swap Space | Passively (default 4GB) | Minimal — reactive only |
| V1 KV Offload | No (generator is V0) | **Yes** — extend KV cache capacity to CPU |
| KV Transfer | No | No (single GPU) |
| Weight Offloaders | No | Potentially (for thesis offloading work) |
| Sleep Mode | Yes (`offload_enabled`) | Already used, but discards KV cache |

---

## 7. V1 Migration Analysis

### 7.1 Would V1 KV Offload benefit FastTTS?

**Yes, significantly**, in two scenarios:

**Baseline (no offload) mode** — generator gets ~19% GPU memory (for 1.5B+7B). KV cache space is tiny. With n=8+ beams each needing up to 4096 tokens of KV, beams get preempted frequently. V1 KV offload would spill evicted blocks to CPU and reload instead of recompute.

**Offload (sleep) mode** — currently all KV cache is destroyed on every model swap. The ideal flow (matching the paper's Section 4.3.2 claim):
1. Generator finishes -> offload KV cache to CPU (not discard)
2. Verifier wakes -> uses full GPU
3. Verifier finishes -> offload verifier KV to CPU
4. Generator wakes -> reload KV from CPU (skip prefix recomputation)

This would preserve prefix cache across model swaps, which is a huge win for beam search with heavy prefix sharing.

### 7.2 What V1 offers for Speculative Beam Extension

| Requirement | V1 support |
|---|---|
| Priority-based scheduling | **Supported** — `Request.priority` field, `SchedulingPolicy.PRIORITY` for preemption |
| Request abortion | **Supported** — `abort_requests(request_ids)`, can be queued during execution |
| Pause/resume scheduling | **Supported** — `pause_scheduler()` with `abort`/`keep`/`wait` modes |
| Stop string suppression | **Requires workaround** — don't use stop strings in SamplingParams; detect `\n\n` externally |
| Scheduler queue inspection | **Limited** — `get_num_unfinished_requests()` available, but no direct queue access |
| Per-sequence mid-batch manipulation | **Not supported** — can only abort/finish between `step()` calls |
| Custom output processing | **Possible** — subclass `OutputProcessor` |

### 7.3 Recommended V1 migration approach

Rather than porting V0 hacks, **reimplement using V1's paradigm**:

1. **Don't use stop strings at vLLM level.** Generate with `max_tokens` only. Detect `\n\n` step boundaries in a custom `OutputProcessor` subclass or in the FastTTS search loop.

2. **Use `abort_requests()` for beam management.** After each `step()`, inspect returned `RequestOutput`s. Track which requests hit step boundaries. Abort speculative sequences that should be pruned.

3. **Use `request.priority` for scheduling.** Set priority when adding requests. Speculative continuations get lower priority.

4. **Use V1 KV Offload.** Configure `CPUOffloadingSpec` with `cpu_bytes_to_use` to extend KV cache capacity to CPU memory.

5. **Move beam coordination to search layer.** The search loop (`beam_search.py`) already manages beam state (current_text, scores, pruning). Push more control there instead of embedding it in the engine.

### 7.4 Blockers and risks

- **V1 KV Offload is experimental** — API may change
- **No mid-step intervention** — cannot abort a sequence during a forward pass, only between steps
- **Separate process boundary** — if `EngineCore` runs in a separate process (multiprocess mode), all communication is via serialized messages
- **V1 may not support reward models** — `vllm_wrapper.py:18` sets `VLLM_USE_V1=0` globally with comment "Force V0 since reward models are not supported in V1" (though the verifier override to V1 at line 112 contradicts this)

---

## 8. Key Configurations

### Default models
- Generator: `Qwen/Qwen2.5-Math-1.5B-Instruct`
- Verifier: `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B`

### Benchmark configurations (1.5B+7B, AIME)

| Config | Generator GPU% | Verifier GPU% | offload | spec_beam | prefix_sched |
|---|---|---|---|---|---|
| `baseline/` | 0.19 | 0.71 | No | No | No |
| `offload/` | 0.95 | 0.95 | Yes | No | No |
| `spec_offload_prefix/` | 0.95 | 0.95 | Yes | Yes | Yes |

### Search defaults
- `beam_width=4`, `n=8`, `num_iterations=10` (benchmarks); `num_iterations=40` (default)
- `temperature=0.8`, `top_p=1.0`, `max_tokens=2048`
- `stop="\n\n"`, `agg_strategy="last"`
- `batch_size=1` (one problem at a time for beam search)
