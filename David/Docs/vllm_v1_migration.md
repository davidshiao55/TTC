# FastTTS vLLM V1 Migration (0.9.2 -> 0.18.1)

## 1. Introduction & Motivation

FastTTS-thesis was written against the vLLM 0.9.2 API and the V0 engine
(synchronous scheduler, direct queue access). The thesis requires vLLM V1
(`VLLM_USE_V1=1`) because all offloading work -- attention offloading, weight
offloading, KV-cache management -- targets `vllm/v1/`. Running FastTTS on V0
would mean none of those optimisations are exercised at test-time, making any
comparison against the baseline meaningless.

The thesis vLLM fork is pinned to **v0.18.1** (tag `a26e8dc7f`) for
reproducibility. The `thesis` branch is created from this tag for all
thesis-specific modifications.

### Migration scope

| Component | Status |
|---|---|
| `models/tts_llm.py` | Fully migrated to V1 APIs |
| `models/generator_engine_v1.py` | New file -- V1 SBE implementation |
| `models/vllm_wrapper.py` | Updated (`task`->`runner`, `PoolerConfig`) |
| `models/reward_utils.py` | Rewritten (tokenizer boundary fix) |
| `search/beam_search.py` | Fully rewritten (decomposed, StepChunk, step-hash propagation) |
| `search/beam.py` | Restructured (renamed fields, StepChunk, beam identity) |
| `search/common.py` | New -- shared infrastructure (SearchState, phase functions, generate/score/parse) |
| `search/dvts.py` | Fully migrated -- uses shared infrastructure + DVTS-specific subtree pruning |
| `search/best_of_n.py` | Fully migrated -- standalone single-shot generation + scoring |
| `search/dynamic_branching.py` | Fully migrated -- uses shared infrastructure + score-proportional duplication |
| `search/vg_search.py` | Fully migrated -- uses shared infrastructure + 3-stage sampling params |
| PRM plugin (`prm_model.py`) | Migrated to V1 TokenPooler |
| `accuracy_evaluation/evaluation/evaluate.py` | Rewritten (multi-metric, no test-set tuning) |
| `run_all_experiments.py` | Updated (removed `top_n` sweep, scaling curve plots) |

The non-migrated search strategies construct `Beam` objects with field names
that no longer exist (`index`, `next_texts`, `lookahead_texts`, `best_scores`,
`all_scores`, `previous_text`, `history`, `completion_tokens`,
`total_completion_tokens`, `completion_time`, `future_texts`). See
[Section 11](#11-known-limitations--open-issues) for the full field mapping.

---

## 2. PRM Plugin Migration

**File:** `modified-skywork-o1-prm-inference/vllm_add_dummy_model/prm_model.py`

### What broke

| Old import | Status in v0.18.1 |
|---|---|
| `vllm.model_executor.pooling_metadata.PoolingMetadata` | Moved to `vllm.v1.pool.metadata.PoolingMetadata` |
| `vllm.model_executor.pooling_metadata.PoolingTensors` | Removed |
| `vllm.sequence.PoolingSequenceGroupOutput` | Removed |
| `vllm.sequence.PoolerOutput` | Moved to `vllm.v1.outputs.PoolerOutput` (TypeAlias) |
| `vllm.model_executor.layers.pooler.PoolingType` | Removed |
| `vllm.model_executor.layers.pooler.Pooler` | Removed -- entire API redesigned |

### Fix: vLLM's composable TokenPooler

`Qwen2ForPrmModel.forward()` runs a `ValueHead` on Qwen2 hidden states to
produce per-token scalar scores `[total_tokens, 1]`. The pooler slices this
flat tensor back into one tensor per request.

The old `Pooler.from_config_with_defaults(PoolingType.ALL, ...)` was replaced
with vLLM's composable system:

```python
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.heads import TokenClassifierPoolerHead

self.pooler = TokenPooler(
    pooling=AllPool(),                    # handles chunked-prefill accumulation
    head=TokenClassifierPoolerHead(),     # passthrough (no classifier)
)
```

`AllPool` accumulates hidden states across chunked-prefill steps via
`PoolingStates.hidden_states_cache`, returning the concatenated result only
when the request finishes.

### Other V1 interface changes

| Change | Why |
|---|---|
| `get_input_embeddings()` -> `embed_input_ids()` | V1 `VllmModel` protocol |
| `load_weights()` now returns `set[str]` | V1 weight validation contract |
| Added `is_pooling_model = True` class variable | V1 pooling model detection |

### Bug 1: PrmPooler must be an `nn.Module`

`gpu_model_runner.py` calls `model.pooler.get_supported_tasks()` during
warmup. If `pooler()` was a method, Python returns a bound method object
with no `get_supported_tasks` attribute.

```
AttributeError: 'function' object has no attribute 'get_supported_tasks'
```

Fixed by making the pooler an `nn.Module` subclass assigned as `self.pooler`.

### Bug 2: Chunked prefill produces truncated pooling output

V1 defaults to `enable_chunked_prefill=True` with
`max_num_batched_tokens=8192`. When a batch exceeds this budget, the
scheduler splits work across multiple steps. The original `PrmPooler`
sliced `hidden_states[first:last+1]` per step with no cross-step
accumulation -- hidden states from earlier steps were discarded.

**Symptom:** `IndexError` in `_score_outputs_skywork` when
`reward_embedding` has fewer rows than `reward_flag` entries.

**Example:** 16 sequences (~1750 tokens each, ~28k total) exceed the
8192-token budget:
- Step 1: first ~8192 tokens processed, sliced -> discarded
- Step 2: remaining tokens processed -> only these returned
- Result: output shape `[750, 1]` instead of `[1750, 1]`

V0 was unaffected because it defaults to `enable_chunked_prefill=False`.

Fixed by switching to `TokenPooler(pooling=AllPool(), ...)` which
accumulates across steps.

---

## 3. Engine Stack Migration

### Files removed (V0 code)

These imported V0 APIs that no longer exist in v0.18.1. Originals preserved
in `FastTTS-AE/models/`.

| File | Purpose (V0) | V1 Replacement |
|---|---|---|
| `generator_engine.py` | V0 `LLMEngine` subclass with SBE via `SingleStepOutputProcessor` + `SpecStopChecker` | `generator_engine_v1.py` |
| `verifier_engine.py` | V0 `LLMEngine` subclass injecting `CustomScheduler` | Plain V1 `LLMEngine` alias |
| `spec_stopchecker.py` | `StopChecker` subclass with conditional stop based on `scheduler.waiting` | `_step_sbe()` in `generator_engine_v1.py` |
| `custom_scheduler.py` | V0 `Scheduler` subclass for speculative beam preemption | V1 native `PRIORITY` policy + `_cleanup_preempted_speculative()` |

### `tts_llm.py` API changes

| Old | New |
|---|---|
| `task: TaskOption` | `runner: RunnerOption` |
| `task="reward"` | `runner="pooling"` |
| `override_pooler_config: dict` | `pooler_config: PoolerConfig` |
| `swap_space` param | Removed (deprecated) |
| `max_seq_len_to_capture` param | Removed |
| `disable_async_output_proc` param | Removed |
| `from vllm.utils import Counter` | `from vllm.utils.counter import Counter` |
| `from vllm.config import TokenizerMode` | `from vllm.config.model import TokenizerMode` |
| `from vllm.engine.arg_utils import TaskOption` | `from vllm.engine.arg_utils import RunnerOption` |
| `_validate_truncation_size(...)` | Removed; tokenizer-level `truncation=True` used directly |
| `_validate_and_add_requests` override | Removed; parent no longer has this method |
| `_run_engine()` | Renamed to `_run_completion()` |
| `PoolerConfig(task=...)` | `task` field removed; only `pooling_type`, `step_tag_id`, `returned_token_ids` |

**Engine selection:**
```python
if runner in ('generate', 'auto'):
    engine_cls = GeneratorLLMEngineV1   # always V1
else:
    engine_cls = VerifierLLMEngine      # plain V1 LLMEngine alias
```

**Attribute bootstrap** -- `TTSLLM.__init__` bypasses `LLM.__init__()` to use
custom engine classes. After engine creation, attributes that `LLM.generate()`
and `LLM.encode()` expect must be set manually:

```python
self.model_config = self.llm_engine.model_config
self.renderer = getattr(self.llm_engine, 'renderer', None)
self.runner_type = self.model_config.runner_type
self.supported_tasks = self.llm_engine.get_supported_tasks()
self.pooling_io_processors = {}
self.io_processor = getattr(self.llm_engine, 'io_processor', None)
self.input_processor = getattr(self.llm_engine, 'input_processor', None)
```

### `vllm_wrapper.py` verifier kwargs

```python
# Before (V0)
return {
    **self.config.verifier_vllm_config,
    "task": "reward",
    "override_pooler_config": {"pooling_type": "STEP", "step_tag_id": 12902, ...},
}

# After (V1)
from vllm.engine.arg_utils import PoolerConfig
pooler_config = PoolerConfig(pooling_type="STEP", step_tag_id=12902,
                             returned_token_ids=[648, 387])
return {
    **self.config.verifier_vllm_config,
    "runner": "pooling",
    "pooler_config": pooler_config,
}
```

### GPU memory allocation

v0.18.1 has higher memory overhead than v0.9.2 (CUDA graphs, torch.compile,
per-process CUDA contexts). Each spawned process adds ~0.9–1.4 GiB overhead
beyond what `gpu_memory_utilization` budgets for. With two processes
(generator + verifier), the combined overhead is ~2.3 GiB.

#### Root cause: PyTorch memory fragmentation

The original AE splits (baseline 0.68/0.22=0.90, spec-prefix 0.75/0.15=0.90)
OOM in v0.18.1 — but not from absolute memory exhaustion. The OOM occurs on
**activation allocations** (MLP gate_up projection: `max_num_batched_tokens ×
18944 × 2` bytes = 296 MiB at batch 8192) inside torch.compile/inductor
compiled code. PyTorch's default caching allocator fragments into "slivers"
of unusable memory (reported as "reserved but unallocated" in OOM messages,
observed 187–869 MiB). Two engines on one GPU compounds this: each process
independently fragments its allocator pool, and the combined waste exceeds
the ~2.4 GiB headroom at 0.90 total utilization.

This fragmentation is worse in v0.18.1 because CUDA graphs + torch.compile
create more allocation/deallocation cycles with varying tensor sizes. The
`memory_latency_analysis.py` profiling (1 problem / 1 iteration) didn't
reproduce it because fewer cycles = less fragmentation. The full evaluation
pipeline (30 problems / 10 iterations) fragments far more.

#### Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Set in `models/vllm_wrapper.py` at module level. This tells PyTorch's default
caching allocator to use growable memory segments instead of fixed-size blocks,
eliminating fragmentation-induced OOM. This is the documented PyTorch solution
for workloads with varying allocation sizes (see PyTorch CUDA Memory
Management docs).

**Compatibility:**
- **Safe** with kv_offload, prefetch offloader, UVA offloader (they use
  standard `torch.zeros()`/`torch.empty_strided()`, not custom allocators)
- **Incompatible** with vLLM's `CuMemAllocator` (sleep mode). Sleep mode
  replaces PyTorch's default allocator via `cuMemCreate`/`cuMemMap` for
  tagged memory management. The two allocators cannot coexist. When
  `offload_enabled=True` enables sleep mode, `expandable_segments` must be
  removed — but at that point only one model's weights are GPU-resident
  (the other is sleeping), so fragmentation headroom is not an issue.

#### Updated splits

With `expandable_segments:True`, the original AE total=0.90 is restored:

- Verifier minimum: 0.15 -> **0.16** (0.15 has insufficient KV cache capacity
  for `max_model_len=4096` due to higher per-process overhead in v0.18.1;
  this is a capacity constraint, not fragmentation)
- Baseline split: generator 0.68 / verifier **0.22** (total 0.90)
- Spec-prefix split: generator **0.74** / verifier 0.16 (total 0.90;
  original AE was 0.75/0.15, adjusted for verifier minimum 0.16)

---

## 4. Speculative Beam Extension (SBE) for V1

**File:** `models/generator_engine_v1.py`

### Architecture

SBE in V1 uses two mechanisms:

1. **Stop-string stripping:** `SamplingParams.stop` set to `None` before
   request submission. Neither the EngineCore nor the detokenizer detect
   stop strings. Requests generate freely past `\n\n` boundaries.

2. **Direct scheduler access:** `VLLM_ENABLE_V1_MULTIPROCESSING=0` puts
   everything in one process. `GeneratorLLMEngineV1` holds a direct
   reference to the V1 scheduler via
   `self.engine_core.engine_core.scheduler`, enabling Phase 1/Phase 2
   decisions based on `len(scheduler.waiting)`.

### Phase 1 / Phase 2 decision (paper SS4.1.2)

In `_step_sbe()`, after each engine step:
1. Read `detokenizer.output_text` for every active request
2. If a stop string is found:
   - **`scheduler.waiting` non-empty -> Phase 1:** truncate text at stop
     boundary, create finished `RequestOutput`, abort request (frees KV
     cache for queued beams)
   - **`scheduler.waiting` empty -> Phase 2:** mark as speculative, set
     `request.priority = SPEC_BEAM_CANDIDATE_PRIORITY` (1e9), keep
     generating

### Text truncation policy

Only Phase 1 truncates at the stop-string boundary. All other paths
preserve full text:

| Path | Truncate? | Rationale |
|---|---|---|
| Phase 1 finish | Yes | Beam legitimately stopped; matches AE's `StopChecker` |
| Phase 2 (speculative) | N/A | Beam keeps generating |
| Force-finish | No | Full text returned for `split_string_by_separator` |
| Overflow (>256) | No | Matches AE: `FINISHED_STOPPED` with full text |
| Preemption cleanup | No | Matches AE: aborted with accumulated text intact |

### Preemption and overflow

- **Preemption:** speculative beams (`priority=1e9`) preempted first by
  V1's `max(running, key=lambda r: (r.priority, r.arrival_time))`.
  `_cleanup_preempted_speculative()` detects and aborts them each step.
- **Overflow:** if active beams exceed 256, oldest speculative beams
  force-finished first.
- **Force-finish:** when `len(finished) + len(speculative) >= total_requests`,
  all speculative beams terminated.

### Files modified

| File | Change |
|---|---|
| `models/generator_engine_v1.py` | Full SBE: `SBETracker`, `enable_spec_beam_extension()`, `add_request()` (strip stops), `step()` with `_step_sbe()`, Phase 1/2, preemption, overflow, force-finish |
| `models/tts_llm.py` | Inject `scheduling_policy="priority"` when `spec_beam_extension=True` |
| `models/vllm_wrapper.py` | Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` at module level and in child process |

---

## 5. PRM Prefix Caching Optimization

### Background

Both V0 and V1 disable prefix caching for ALL-token pooling models:
- V0: disabled at `config.py:4594`
- V1: sets `skip_reading_prefix_cache=True` at `pooling_params.py:125`

Verified empirically: `enable_prefix_caching=True/False` gives 1.00x
speedup on both V0 and V1 -- caching is disabled internally.

### Why it matters

In beam search, the PRM scores N solutions per question per iteration. All
N share the same question prefix (200-800 tokens). Without caching, the
prefix is recomputed N times. With n=128: ~25,600 redundant tokens per
scoring call.

### Implementation

No vLLM changes required. `PoolingParams.skip_reading_prefix_cache` already
supports explicit `False`.

#### 1. Enable prefix caching for PRM (`tts_llm.py`)

```python
rewards = self.encode(
    prompts,
    pooling_params=PoolingParams(skip_reading_prefix_cache=False),
)
```

#### 2. Offset-based reward_flags indexing (`tts_llm.py`)

With prefix caching, `reward_embedding` is shorter than `reward_flags`
(cached prefix tokens produce no hidden states). The offset computation
handles both cold cache (offset=0) and warm cache (offset>0):

```python
offset = len(reward_flag) - len(reward_embedding)
for i, flag in enumerate(reward_flag):
    if flag == 1:
        local_idx = i - offset
        if local_idx < 0:
            step_reward.append(None)  # cached -- fill in search layer
        elif local_idx >= len(reward_embedding):
            break
        else:
            step_reward.append(sigmoid(reward_embedding[local_idx][0]))
```

All question prefix flags are 0, so within-batch prefix sharing produces
zero Nones in the common case.

#### 3. Score propagation via step-hash matching (`beam_search.py`)

Score propagation runs in `_score_and_assign()` in the search layer, not
in the PRM subprocess. This avoids tokenization mismatches from
`prepare_input`'s truncation logic. Three layers:

**Layer 1 -- lock prev_scores (cross-iteration):** Each beam's existing
`scores` fills Nones from its own history. Score locking: once set, a
score is never recomputed. Prevents BF16 noise (~2%) from producing
different scores for identical text.

```python
for j in range(min(len(score[0]), len(beam.scores))):
    if beam.scores[j] is not None:
        score[0][j] = beam.scores[j]
```

**Layer 2 -- step_hash propagation (persistent bank + within-iteration):**
`state.step_hash_bank` accumulates all scored hash->score pairs across
iterations (survives pruning). Fresh PRM scores are added to the bank
between Layer 1 and Layer 2; `_propagate_by_step_hash()` then fills
remaining Nones from it. Matches by cumulative text hash (encodes full
prefix). Only runs when Nones exist.

```
Within a single score_outputs batch:
  beam X: [q + step1 + step2 + spec3_X]  -> cold -> [s1, s2, s3_X]
  beam Y: [q + step1 + step2 + spec3_Y]  -> KV hits from X
    -> only spec3_Y computed -> [None, None, s3_Y]
    -> Layer 1: prev [s1_Y] fills step1
    -> Layer 2: bank has step2 hash -> s2 from X -> fills step2
    -> result: [s1_Y, s2, s3_Y]

Cross-iteration (pruned beam donor):
  Iter N: beam A scored -> step_hash_bank[h2] = s2 -> A pruned
  Iter N+1: beam B generates step2 (same text, KV cache hit -> None)
    -> Layer 2: persistent bank has h2 -> s2 -> fills step2
```

**Layer 3 -- RuntimeError safety net:** If Nones remain after all layers,
raises RuntimeError.

#### Performance

128 PRM requests with shared question prefix:

| Config | Time | Speedup |
|---|---|---|
| Prefix caching OFF | 0.437s | -- |
| ON, default params (skip=True internally) | 0.440s | 1.00x |
| ON, skip_reading_prefix_cache=False | 0.038s | **18.5x** |

---

## 6. Correctness Fixes

### 6a. Tokenizer boundary overflow (`reward_utils.py`)

`prepare_input()` could produce `input_ids` exceeding `max_model_len` (4096).

**Root cause:** The old code tokenized response text multiple times in
different ways: whole-string for budget check (lower count), per-step for
output (higher count). BPE tokenizers produce different counts at `\n\n`
split points.

```
Whole-string tokenization: 4037 tokens  (budget says: FITS)
Per-step tokenization:     4083 tokens  (output: OVERFLOW)
Difference:                +46 tokens across 131 steps
```

AE masked this with a hardcoded budget of 3072 (1024 tokens of headroom).
We raised it to 4096 (full model context), exposing the bug.

**Fix:** Tokenize each step exactly once. Same token IDs used for both
budget accounting and output building -- correct by construction. For
responses near the limit, truncation is slightly more aggressive (drops one
more step) but the output never overflows.

### 6b. Beam duplication (paper-correct)

**Problem (pre-existing in AE):** AE's duplication only truncated the LAST
`future_texts` entry, keeping all earlier speculative steps intact.
Duplicates consumed parent's speculative steps unchanged -- no divergence.
This contradicts the FastTTS paper's Algorithm 1 line 19
(`DuplicateThenTruncate`).

**Fix:** Truncate the FIRST speculative step (R=0.85), clear all subsequent
steps. Trim `scores[:i]` to remove speculative scores:

```python
if beam.pending_steps:
    first_text = truncate_sentence_by_tokens(beam.pending_steps[0].text, tokenizer)
    duplicate.pending_steps = [
        StepChunk(text=first_text, is_complete_step=False, terminal=False)
    ]
    duplicate.scores = beam.scores[:i]
    duplicate.step_hashes = beam.step_hashes[:i]
```

Parent beam keeps full scores and pending_steps unchanged. Duplicate gets
truncated first step + cleared rest for immediate divergence.

### 6c. All-step lookahead scoring (paper SS4.1.3)

**Problem:** AE scored only one step ahead (`current_text + future_texts[0]`).
The PRM scores every `\n\n` boundary in a single forward pass -- cost is
identical whether 1 or 10 finished steps.

**Fix:** Score all finished pending_steps at once:

```python
scoring_text = beam.current_text
for chunk in beam.pending_steps:
    if not chunk.is_complete_step and not chunk.terminal:
        break
    scoring_text += chunk.text
    beam.step_hashes.append(step_hash(scoring_text))
completions.append([scoring_text])
```

### 6d. Delayed completion semantics + StepChunk

**Problem:** When SBE speculation reaches EOS several steps ahead,
`beam_is_completed()` at the raw generator output's `stop_reasons[0]` --
which describes the *last* speculative step, not the step being committed --
marks the beam `completed=True` immediately. The beam leaves `active_beams`,
removing its `step_hashes` from the propagation bank. Siblings can no
longer resolve None scores.

**Fix:** `beam.completed` fires only when the step actually consumed at the
current iteration is terminal. The beam stays in `active_beams` until its
terminal chunk is consumed.

**StepChunk abstraction:**

```python
@dataclass
class StepChunk:
    text: str
    is_complete_step: bool   # ends at "\n\n" -> scorable
    terminal: bool           # consuming this completes the beam
```

`Beam.future_texts` renamed to `Beam.pending_steps: List[StepChunk]`.

Four shapes at parse time:

| is_complete_step | terminal | Semantics |
|---|---|---|
| True | False | Regular speculative step -- skippable, scorable |
| True | True | Final step (EOS at `\n\n`) -- completes beam |
| False | True | Terminal partial tail (EOS mid-step) |
| False | False | Continuation prefix -- consumed next iteration |

**Parser: `parse_generation_into_chunks`** replaces `beam_is_completed` +
`split_string_by_separator` with a terminal/non-terminal waterfall:

1. `token_length >= max_model_len` -> terminal (context exhaustion)
2. `stop_reason == "\n\n"` and `text != ""` -> not terminal (clean boundary)
3. `finish_reason == "length"` and `text.count("\n\n") >= 1` -> not terminal
   (length-cap recovery: budget shared across multiple steps)
4. Otherwise -> terminal (EOS, single mega-step length cap, empty text)

**Length-cap recovery:** `SamplingParams.max_tokens` (default 2048) is a
per-call budget. In SBE mode, hitting it means the budget was shared across
N steps -- no single step violated the limit. The trailing partial becomes a
`(False, False)` continuation prefix consumed next iteration.

### 6e. Lost-beam fix (pre-existing in AE)

When a beam was marked `completed=True` but still had `future_texts`, it was
removed from `active_beams` but NOT added to `completed_beams` (guarded by
`if not beam.future_texts`). The beam was lost.

Fix: removed the guard. Completed beams always added to `completed_beams`.

---

## 7. Accuracy Evaluation Redesign

### Issues found

Five methodological flaws in the original evaluation pipeline, all
pre-existing in AE and carried through to the initial thesis migration.

| # | Issue | Severity | Location |
|---|---|---|---|
| 1 | **Test-set tuning of `top_n`** | Critical | `run_all_experiments.py:186,257-272` |
| 2 | **Completion count exceeds n** | Critical | `beam_search.py:_check_n_completion()` |
| 3 | **Inconsistent latency/accuracy pairing** | Critical | latency at n, accuracy over >n |
| 4 | **Hardcoded `np.prod` aggregation** | Moderate | `evaluate.py:34` |
| 5 | **Non-deterministic tie-breaking** | Minor | `evaluate.py:41` |

**Issue 1 -- test-set tuning:** `run_all_experiments.py` swept `top_n`
over `[8, 16, 32, ..., 512]` and reported the maximum accuracy -- classic
hyperparameter tuning on the test set. Neither compute-optimal-tts (Liu
et al.) nor search-and-learn (HuggingFace) do this.

**Issue 2 -- completion overshoot:** `_check_n_completion()` only recorded
latency metrics when `completed_beams >= n` but never signalled the main
loop to stop. Search continued until all beams exhausted or max iterations.
Empirically confirmed: n=8 produced 12 completions.

**Issue 3 -- inconsistent pairing:** `n_gen_latency` was snapshot at the
moment n completions arrived, but `_finalize()` returned ALL completed
beams (often >n). `evaluate.py` scored all of them. Result: low latency
(at n) paired with inflated accuracy (from >n completions).

**Issue 4 -- hardcoded aggregation:** `evaluate.py` used `np.prod(score)`
regardless of `SearchConfig.agg_strategy`. Both search-and-learn and
compute-optimal-tts default to `"last"` (last step score), not product.
This created a mismatch: beam search pruned by `"last"` score but
evaluation ranked by product.

**Issue 5 -- tie-breaking:** `max(pred, key=lambda x: pred.count(x))`
returns the first element achieving the max count -- order-dependent.

### Fixes

**Beam search early exit** (`beam_search.py`):
- `_check_n_completion()` returns `bool` -- `True` when `completed >= n`
  or no active beams remain
- `_filter_completed_and_prune()` simplified: no longer returns stop signal
  (redundant -- `_check_n_completion` covers both conditions)
- Main loop breaks on `reached_n`
- `_finalize()` always sorts by aggregate PRM score (descending) and
  truncates to `[:config.n]`. Handles burst completions from the final
  iteration fairly (multiple beams complete simultaneously -> keep top-n
  by score since they share the same compute budget).

**Evaluation pipeline rewrite** (`evaluate.py`):
- Replaced single-metric pipeline with multi-metric evaluation
- Four metrics computed at each N:
  - **Pass@N**: unbiased estimate via OpenAI Codex formula
  - **Majority Vote**: most common extracted answer (no PRM)
  - **PRM-Max**: best single completion by aggregate PRM score
  - **PRM-Vote**: group answers by `math_equal`, sum PRM scores per group,
    pick highest-sum group (matches Liu et al. `_agg_prm_last_vote`)
- Ordered subsampling: `completions[:n_eval]` for multi-N curves from a
  single max-N run (following search-and-learn methodology)
- Configurable `agg_strategy` (default: `"last"`, matching both reference
  frameworks)
- Deterministic tie-breaking (lexicographic on canonical answer form)

**Orchestration** (`run_all_experiments.py`):
- Removed `TOP_N_VALUES` sweep entirely
- `evaluate_accuracy()` calls `evaluate.py` once per result file
- `plot_accuracy()` generates scaling curves (accuracy vs N, log2 x-axis)

### Reference framework alignment

| Aspect | compute-optimal-tts | search-and-learn | FastTTS-thesis |
|---|---|---|---|
| Stops at exactly N | Yes | Yes (pad if fewer) | Yes (early exit + truncate) |
| Score aggregation | Configurable (min/last/avg) | Hardcoded `"last"` | Configurable, default `"last"` |
| Answer selection | 7 methods, fixed per run | 3 methods, all reported | 4 methods, all reported |
| Multi-N evaluation | Separate runs per N | Subsample from max-N | Subsample from max-N |
| Pass@N metric | Not implemented | OpenAI formula | OpenAI formula |
| Test-set tuning prevention | Config locked before eval | Ordered subsampling | No sweeping |

### Empirical findings: TTC accuracy scaling

After fixing the pipeline, the original FastTTS setup
(`Qwen2.5-Math-7B-Instruct` + Skywork-PRM on AIME) still showed Pass@N
scaling but **flat selection metrics**. Confirmed pre-existing in AE
(not a regression). Three independent root causes:

1. **Context exhaustion** -- `Qwen2.5-Math-*-Instruct` has
   `max_position_embeddings=4096`. On hard AIME problems, 25-88% of
   completions hit the limit and degenerate before producing
   `\boxed{}`. Non-Math `Qwen2.5-*-Instruct` has 32K context.
2. **PRM miscalibration** -- Skywork-PRM-1.5B systematically ranks
   wrong answers above correct ones on hard problems (verified: on
   problem 69 with 75% correct rate, incorrect completions still have
   higher mean PRM score).
3. **AIME is too small** -- 30 problems means each flip = 3.3% noise;
   any scaling signal under ±5% is invisible. Both reference frameworks
   use **MATH-500** (500 problems) for scaling demonstrations.

**Sweep results** (MajVote, N=8 → N=128, agg_strategy="last"):

| Generator | Dataset | N=8 | N=128 | Δ |
|---|---|---|---|---|
| Math-1.5B | MATH-500 | 73.2% | 76.4% | +3.2% (saturated) |
| Math-1.5B | AIME | 16.7% | 10.0% | noise |
| Math-7B | AIME | 13.3% | 13.3% | 0% (context exhaustion) |
| Instruct-1.5B | MATH-500 | 60.8% | 68.8% | **+8.0%** |
| Instruct-1.5B | AIME | 3.3% | 13.3% | **+10.0%** |
| Instruct-7B | MATH-500 | 73.0% | 76.4% | +3.4% |
| Instruct-7B | AIME | 13.3% | 20.0% | **+6.7%** |

**Key insights**:

- **Math-specialized models saturate** (1.5B-Math hits 73.2% MajVote at
  N=8 on MATH-500 -- close to ceiling).
- **General-purpose Instruct models scale clearly** on both AIME and
  MATH-500.
- **Larger generators help most on hard problems**: 7B-Instruct
  outperforms 1.5B-Instruct by +6.7% on AIME at N=128, justifying the
  thesis offloading work.
- **Beam search makes MajVote ≈ PRM-Vote** -- surviving beams already
  concentrate on the same answer regardless of PRM weighting. PRM-Max
  remains the weakest selection method (consistent with Liu et al.'s
  finding for off-policy PRMs).

**Recommended thesis setup**: `Qwen2.5-7B-Instruct` generator +
Skywork-PRM-1.5B verifier, `max_model_len=8192`, MATH-500 (primary) +
AIME 2024 (hard subset). Report all four selection methods (MajVote,
PRM-Vote, PRM-Max, Pass@N) -- no cherry-picking.

---

## 8. Code Restructuring

### Decomposed beam search into named phases (`beam_search.py`)

The monolithic 336-line `_beam_search()` loop was split into 11 functions:

| Function | Responsibility |
|---|---|
| `_init_state()` | Create beams, sampling params, compute prompt token length |
| `_filter_active()` | Remove pruned beams |
| `_check_n_completion()` | Record metrics when n completions first reached |
| `_duplicate_beams()` | Expand active beams to `config.n` via duplication |
| `_prepare_step_source()` | Decide skip vs generate per beam |
| `_generate()` | Build conversations, call generator |
| `_process_results()` | Parse chunks, commit step, build scoring batch |
| `_score_and_assign()` | Call verifier, propagate scores, assign to beams |
| `_filter_completed_and_prune()` | Remove completed, prune lowest scores |
| `_log_iteration()` | Structured logging (INFO summary + DEBUG per-beam) |
| `_finalize()` | Post-loop metrics, sort completed beams |

New dataclasses: `BeamSearchState` (shared mutable state), `ScoringBatch`
(typed container for verifier inputs).

### Beam identity tracking (`beam.py`)

New fields: `beam_id` (unique, auto-incrementing), `parent_id` (set on
duplication), `born_at_iteration`. Module-level counter
(`_next_beam_id()`, `reset_beam_id_counter()`).

Enables tracing beam lineage: "beam 14 (parent=3, born@2)".

### Beam field renaming

| Old | New | Reason |
|---|---|---|
| `completion_tokens` | `step_tokens` | per-step, not cumulative |
| `total_completion_tokens` | `total_tokens_generated` | cumulative |
| `completion_time` | `time_to_complete` | wall-clock at completion |
| `all_scores` | `scores` | only score field after removing `best_scores` |
| `next_texts` | `gen_text` | raw generation output |
| `history` | `gen_history` | distinguish from score history |
| `future_texts` | `pending_steps` | type changed to `List[StepChunk]` |

Removed dead fields: `index`, `best_scores`, `previous_text`.

Added: `step_hashes` (cumulative text hash per step, aligned 1:1 with
`scores`), `finish_reasons`, `prompt_token_lens`, `stop_label` property.

### Structured logging

- **INFO** (always): one-line summary -- beam counts, latencies
- **DEBUG** (opt-in): per-beam table with ID lineage, token counts, scores,
  pending_steps count, stop reasons

### Dead code removal

- Removed multi-step lookahead generation loop (`config.lookahead` always 0)
- Deleted `core.py` (imported non-existent async wrappers)
- Fixed `__init__.py` (removed async exports)
- Removed `_score_call_counter`, `score_debug_dump.txt` writer

---

## 9. Paper Discrepancies (Pre-existing in AE)

These discrepancies exist in the **original AE codebase** -- they are NOT
migration errors. The V1 migration faithfully preserves AE's actual behavior.

### Score-based speculative priority tiering (SS4.1.1) -- NOT in AE

**Paper claim:** "System policy partitions scores into B discrete bins.
For beam b_i with score s_i, if s_i in C_j, then M_i = B - j + 1."

**AE code:** All speculative beams receive the same flat priority:
```python
SPEC_BEAM_CANDIDATE_PRIORITY = 1e9  # uniform -- no score binning
```

No discrete-bin partitioning, no M_i calculation exists anywhere.

### Dynamic prefix-aware scheduling (SS4.2) -- commented out in AE

**Paper claim:** Describes reordering beams by shared prefix using a radix
tree to minimize KV cache evictions.

**AE code:** Entire implementation commented out:
- `models/custom_scheduler.py` -- scheduling functions commented out
- `search/beam_search.py:45-57` -- priority assignment commented out
- `config.py:121` -- `prefix_aware_scheduling: bool = False` (default disabled)
- `search/utils.py:85-127` -- `assign_prefix_priorities()` exists but never called

The V1 migration carries the config flag through but does not implement
scheduling logic, matching AE.

### Features that ARE implemented

| Feature | Paper section | V1 status |
|---|---|---|
| Speculative Beam Extension (core SBE) | SS4.1, Algorithm 1 | Migrated |
| Two-phase scheduling (Phase 1/2) | SS4.1.2 | Migrated |
| Preemption of speculative beams | SS4.1.2 | Migrated |
| Force-finish when standard beams done | SS4.1.2 | Migrated |
| 256-beam overflow cap | SS4.1 | Migrated |
| LookAhead Verification | SS4.1.3 | Unchanged (search layer) |
| `split_string_by_separator` for SBE output | SS4.1.3 | Unchanged |
| Asymmetric Multi-Model Memory Allocation | SS4.3 | Unchanged |

These unimplemented features (score-based priority tiering, prefix-aware
scheduling) are orthogonal to the thesis offloading work. They operate at
the search/scheduling layer, not the engine/model layer where offloading
changes are made.

---

## 10. Verification

Test scripts live in `migration_verification/`. Run from the container with
`conda activate thesis` and `cd /TTC/FastTTS-thesis`.

### Standalone tests

```bash
python migration_verification/verify_prm.py    # PRM plugin + prefix caching (13 tests)
python migration_verification/verify_sbe.py    # SBE behavioral equivalence (8 tests)
```

**`verify_prm.py`** (13 tests):
- Tests 1-8: Migration correctness (plugin, model, scoring, pooler interface)
- Test 9: Question prefix sharing -- within-batch, all scores correct
- Test 10: Cross-iteration merge -- prev_scores preserves old step scores
- Test 11: Within-batch propagation -- siblings share solution prefix
- Test 12: Edge case -- RuntimeError fires when no donor available
- Test 13: Performance -- 18.82x speedup

**`verify_sbe.py`** (8 tests):
- Baseline stop behavior, speculative continuation past stop strings,
  multi-step speculation, force-finish, priority scheduling, two-phase
  scheduling, text preservation on force-finish, `split_string_by_separator`
  lossless reconstruction

### Direct AE <-> Thesis comparison

```bash
python migration_verification/compare.py    # Cross-env comparison (5 tests)
```

Runs identical inputs through both `baseline` (AE + vLLM 0.9.2) and
`thesis` (V1) via subprocess:

| Test | What it checks | Result |
|---|---|---|
| 1. Tokenization | `input_ids` identical | Exact match |
| 2. Reward flags | Step boundary positions | Exact match |
| 3. Raw per-token rewards | Per-token scalar scores | max diff 4.7e-2 (BF16 noise) |
| 4. Per-step PRM scores | Sigmoid-transformed scores | max diff 0.0e+0 (exact) |
| 5. Baseline generation | temp=0, no SBE, text output | Byte-for-byte identical |

Raw per-token reward differences (Test 3) are BF16 mantissa noise from
different attention backends between V0 and V1. After sigmoid, final
per-step scores are exactly identical (Test 4).

---

## 11. Known Limitations & Open Issues

### Non-migrated search strategies (CRITICAL)

`dvts.py`, `best_of_n.py`, `dynamic_branching.py`, `vg_search.py` use old
Beam field names. They will crash with `TypeError` on any call. Full
field mapping:

| Old field | New field / status |
|---|---|
| `index` | Removed |
| `next_texts` | Renamed to `gen_text` |
| `lookahead_texts` | Removed |
| `best_scores` | Removed |
| `all_scores` | Renamed to `scores` |
| `previous_text` | Removed |
| `history` | Renamed to `gen_history` |
| `completion_tokens` | Renamed to `step_tokens` |
| `total_completion_tokens` | Renamed to `total_tokens_generated` |
| `completion_time` | Renamed to `time_to_complete` |
| `future_texts` | Renamed to `pending_steps` (type: `List[StepChunk]`) |

These strategies also pass dead `prev_scores`/`skipped_beam_context` kwargs
through the verifier pipe -- score propagation was moved to the search
layer (`beam_search.py:_score_and_assign()`), so these kwargs are silently
absorbed.

Additionally, `best_of_n.py` sorts by `completion_time` instead of score
when `sort_completed=True` (pre-existing bug from AE).

### SBE detokenizer text/token_ids desync

**Status:** Root cause identified, fix not yet implemented.

For ~1 in 263 SBE generation outputs, `output.outputs[0].text` contains
significantly more content than `tokenizer.decode(output.outputs[0].token_ids)`.

**Root cause:** `FastIncrementalDetokenizer` primes its `DecodeStream` with
`request.prompt_token_ids`. For specific token sequences at the
prompt/generation boundary, `DecodeStream.step()` flushes the prompt text
into `output_text`:
- `output_text` = system prompt (~8001 chars) + generation (~5631 chars)
- `token_ids` = generation tokens only (1108 tokens -> 5631 chars)

The inflated `output_text` gets split into ~27 pending_steps (instead of
~10), creating phantom steps from prompt text.

### Context length limit (4096)

Qwen2.5-Math-7B-Instruct has `max_position_embeddings=4096` -- the model's
hard limit. With 7B generators, beams hit this as early as iteration 5/10
without producing `\boxed{}` answers. The reference paper uses 8192 total
with Qwen2.5-7B-Instruct (128K context), not the Math variant. Raising the
limit would require `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and risks NaN from
RoPE extrapolation.
