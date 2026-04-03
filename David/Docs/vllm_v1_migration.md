# FastTTS → vLLM V1 Migration (0.9.2 → 0.18.1)

## Why We Did This

FastTTS-thesis was written against the vLLM 0.9.2 API and the V0 engine
(synchronous scheduler, direct queue access).  The thesis requires vLLM V1
(`VLLM_USE_V1=1`) because all offloading work — attention offloading, weight
offloading, KV-cache management — targets `vllm/v1/`.  Running FastTTS on V0
would mean **none of those optimisations are exercised at test-time**, making
any comparison against the baseline meaningless.

The thesis vLLM fork is pinned to **v0.18.1** (tag `a26e8dc7f`) for
reproducibility.  The `thesis` branch is created from this tag for all
thesis-specific modifications.

Two independent problems required fixing:

1. **PRM plugin broken** — `Qwen2ForPrmModel` (Skywork verifier) used
   internal vLLM APIs that were removed or relocated.
2. **FastTTS engine stack broken** — `tts_llm.py`, `generator_engine.py`,
   `verifier_engine.py`, and `spec_stopchecker.py` all imported V0
   internals that no longer exist.

---

## Part 1 — Fix the PRM Plugin

**File:** `modified-skywork-o1-prm-inference/vllm_add_dummy_model/prm_model.py`

### What broke

| Old import | What happened |
|---|---|
| `vllm.model_executor.pooling_metadata.PoolingMetadata` | Moved → `vllm.v1.pool.metadata.PoolingMetadata` |
| `vllm.model_executor.pooling_metadata.PoolingTensors` | Removed |
| `vllm.sequence.PoolingSequenceGroupOutput` | Removed |
| `vllm.sequence.PoolerOutput` | Moved → `vllm.v1.outputs.PoolerOutput` (now a `TypeAlias`) |
| `vllm.model_executor.layers.pooler.PoolingType` | Removed |
| `vllm.model_executor.layers.pooler.Pooler` | Removed — entire Pooler API redesigned |

### What the pooler does (and why we could simplify it)

`Qwen2ForPrmModel.forward()` runs a `ValueHead` on top of the Qwen2 hidden
states to produce a **per-token scalar score** — shape `[total_tokens, 1]`
for the whole batch.  The `pooler()` method's only job is to slice that flat
tensor back into one tensor per request.

The old code used `Pooler.from_config_with_defaults(PoolingType.ALL, ...)`,
which was a heavyweight wrapper that handled many pooling strategies.  V1
replaced this with a composable `TokenPooler(method, head)` system.  For our
use case — raw per-token scores already produced by `ValueHead` — we don't
need any of that machinery.

### New pooler: vLLM's built-in `TokenPooler`

The initial migration used a custom `PrmPooler(Pooler)` that simply sliced
`hidden_states[first:last+1]` per request.  This worked but lacked
chunked-prefill accumulation logic — see Bug #2 below.

The final implementation uses vLLM's composable pooler system:

```python
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.heads import TokenClassifierPoolerHead

self.pooler = TokenPooler(
    pooling=AllPool(),               # handles chunked-prefill accumulation
    head=TokenClassifierPoolerHead(),  # no classifier → passthrough
)
```

`AllPool` accumulates hidden states across chunked-prefill steps via
`PoolingStates.hidden_states_cache` and returns the concatenated result only
when the request finishes.  `TokenClassifierPoolerHead` with no arguments
passes data through unchanged.

### Other V1 interface updates

| Change | Why |
|---|---|
| `get_input_embeddings()` → `embed_input_ids()` | V1 `VllmModel` protocol requires `embed_input_ids` |
| `load_weights()` now returns `set[str]` | V1 weight validation contract |
| Added `is_pooling_model = True` class variable | V1 pooling model detection |
| Removed unused imports, updated type annotations to 3.10+ | Cleanup |

### Bugs discovered during PRM plugin migration

#### Bug 1: `PrmPooler` must be an `nn.Module`, not a bound method

`gpu_model_runner.py` calls `model.pooler.get_supported_tasks()` during the
warmup/memory-profile run at engine startup.  The original code defined
`pooler()` as a method on `Qwen2ForPrmModel`.  When `gpu_model_runner` accessed
`model.pooler`, Python returned a bound method object — which has no
`get_supported_tasks` attribute.

```
AttributeError: 'function' object has no attribute 'get_supported_tasks'
```

Fixed by creating a `PrmPooler(Pooler)` class and assigning `self.pooler = PrmPooler()` in `__init__`.

#### Bug 2: Chunked prefill produces truncated pooling output

**Root cause:** V1 defaults to `enable_chunked_prefill=True` with
`max_num_batched_tokens=8192`.  When a batch of pooling requests exceeds this
budget, the scheduler splits the work across multiple engine steps.  The
original `PrmPooler` simply sliced `hidden_states[first:last+1]` per request
per step — with no accumulation across steps.  When a sequence spanned two
steps, the hidden states from the first step were discarded, producing a
truncated output tensor.

**Symptom:** `IndexError: list index out of range` in `_score_outputs_skywork`
when `reward_embedding` has fewer rows than `reward_flag` entries.

**Example:** A batch of 16 sequences (~1750 tokens each, ~28k total) exceeds
the 8192-token budget:
- Step 1: first ~8192 tokens processed; PrmPooler slices them → discarded
- Step 2: remaining tokens processed; PrmPooler returns only these
- Result: output shape `[750, 1]` instead of `[1750, 1]`

The bug is caused by chunked prefill alone.  Prefix caching exacerbates it
because cached prefixes let the scheduler fit more requests per step,
increasing the chance of sequences being split.  Prefix caching alone is safe
— vLLM V1 sets `skip_reading_prefix_cache=True` for token-level pooling
tasks (`pooling_params.py:130-135`), so `get_computed_blocks()` always returns
0 cached tokens for pooling requests.

V0 was not affected because it defaults to `enable_chunked_prefill=False`.

**Fix:** Replaced `PrmPooler` with vLLM's built-in
`TokenPooler(pooling=AllPool(), head=TokenClassifierPoolerHead())`.  `AllPool`
accumulates hidden states across chunked-prefill steps via
`PoolingStates.hidden_states_cache`, returning the concatenated result only
when the request finishes.  `TokenClassifierPoolerHead` with no classifier is
a passthrough.  This works correctly regardless of `enable_chunked_prefill`.

---

## Part 2 — Port the FastTTS Engine Stack

### Files removed from `models/`

These V0 files were removed from `models/__init__.py` exports during the
initial migration, then deleted entirely after confirming they are dead code
(only reference each other; no live code imports them; they import V0 APIs
that no longer exist in v0.18.1).  Originals are preserved in `FastTTS-AE/models/`.

| File | Purpose (V0) | V1 Replacement | Why removed |
|---|---|---|---|
| `generator_engine.py` | V0 `LLMEngine` subclass with SBE via `SingleStepOutputProcessor` + `SpecStopChecker` | `generator_engine_v1.py` | V0 imports (`LLMEngine`, `SequenceGroupOutput`, `SingleStepOutputProcessor`) no longer exist |
| `verifier_engine.py` | Thin V0 `LLMEngine` subclass that injected `CustomScheduler` | Plain V1 `LLMEngine` alias | V1 has native priority scheduling; `CustomScheduler` is V0-only |
| `spec_stopchecker.py` | `StopChecker` subclass: conditional stop behaviour based on `scheduler.waiting` state (Phase 1 vs Phase 2) | `_step_sbe()` in `generator_engine_v1.py` | V1 stop checking moved to output processor; SBE now strips stop strings entirely and does its own detection |
| `custom_scheduler.py` | V0 `Scheduler` subclass: preemption cleanup for speculative beams (scan `waiting`/`swapped` for beams with `is_finished_stopped_with_stop`) | V1 native `PRIORITY` policy + `_cleanup_preempted_speculative()` | V1 scheduler has built-in priority-based preemption; custom cleanup in `generator_engine_v1.py` |

### `models/tts_llm.py` — rewrite

The old `TTSLLM` was a copy of the old `LLM` class internals and used ~30
vLLM-internal symbols that have since moved or been deleted.

Key API changes from old vLLM to new:

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
| `GuidedDecodingRequest`, `GuidedDecodingParams` | Removed from API |
| `_validate_and_add_requests` override | Removed; parent no longer has this method |
| `_run_engine()` | Renamed to `_run_completion()` with different signature |
| `self.pooling_task = model_config.get_pooling_task(...)` | Removed — `get_pooling_task()` deleted in v0.18.1; `pooling_task` is now a per-call param on `LLM.encode()` |
| `PoolerConfig(task=...)` | `task` field removed in v0.18.1; only `pooling_type`, `step_tag_id`, `returned_token_ids` used |

**Engine selection simplified:**
```python
if runner in ('generate', 'auto'):
    engine_cls = GeneratorLLMEngineV1   # always V1 now
else:
    engine_cls = VerifierLLMEngine      # plain V1LLMEngine alias
```

**Attribute bootstrap** — `TTSLLM.__init__` bypasses `LLM.__init__()` to use
custom engine classes.  After engine creation, the attributes that `LLM`'s
`generate()` and `encode()` methods read must be set manually:
```python
self.model_config = self.llm_engine.model_config
self.renderer = getattr(self.llm_engine, 'renderer', None)
self.runner_type = self.model_config.runner_type
self.supported_tasks = self.llm_engine.get_supported_tasks()
self.pooling_io_processors = {}
self.io_processor = getattr(self.llm_engine, 'io_processor', None)
self.input_processor = getattr(self.llm_engine, 'input_processor', None)
```

**`encode()` simplified:**
```python
def encode(self, prompts, pooling_params=None, *, use_tqdm=True,
           lora_request=None, priority=None, **_kwargs):
    if pooling_params is None:
        pooling_params = PoolingParams()
    return self._run_completion(
        prompts=prompts, params=pooling_params,
        output_type=PoolingRequestOutput,
        use_tqdm=use_tqdm, lora_request=lora_request, priority=priority,
    )
```

### `models/vllm_wrapper.py` — verifier kwargs update

```python
# Before
return {
    **self.config.verifier_vllm_config,
    "task": "reward",
    "override_pooler_config": {"pooling_type": "STEP", "step_tag_id": 12902, ...},
}

# After
from vllm.engine.arg_utils import PoolerConfig
pooler_config = PoolerConfig(pooling_type="STEP", step_tag_id=12902,
                             returned_token_ids=[648, 387])
return {
    **self.config.verifier_vllm_config,
    "runner": "pooling",
    "pooler_config": pooler_config,
}
```

### `models/generator_engine_v1.py`

Full SBE implementation — see Part 3 below.

---

## Part 3 — Speculative Beam Extension (SBE) for V1

### Architecture

SBE in V1 uses two key mechanisms:

1. **Stop-string stripping**: `SamplingParams.stop` is set to `None` before
   request submission. Neither the EngineCore (`check_stop()`) nor the
   detokenizer will detect stop strings. Requests generate freely past
   `\n\n` boundaries.

2. **Direct scheduler access**: `VLLM_ENABLE_V1_MULTIPROCESSING=0` puts
   everything in one process. `GeneratorLLMEngineV1` holds a direct
   reference to the V1 scheduler via
   `self.engine_core.engine_core.scheduler`, enabling Phase 1/Phase 2
   decisions based on `len(scheduler.waiting)`.

### Files modified

| File | Change |
|---|---|
| `models/generator_engine_v1.py` | Full SBE implementation: `SBETracker`, `enable_spec_beam_extension()`, `add_request()` override (strip stop strings), `step()` override with `_step_sbe()`, Phase 1/2 logic, preemption cleanup, overflow cap (256), force-finish |
| `models/tts_llm.py` | Inject `scheduling_policy="priority"` into EngineArgs when `spec_beam_extension=True` |
| `models/vllm_wrapper.py` | Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` at module level and in child process |
| `models/__init__.py` | Export `SBETracker` |

### Phase 1 / Phase 2 decision (maps to paper §4.1.2)

In `_step_sbe()`, after each engine step:
- Read `detokenizer.output_text` for every active request
- If a stop string is found:
  - **`scheduler.waiting` non-empty → Phase 1**: truncate text at stop boundary,
    create finished `RequestOutput`, abort request (frees KV cache for queued beams)
  - **`scheduler.waiting` empty → Phase 2**: mark as speculative, set
    `request.priority = SPEC_BEAM_CANDIDATE_PRIORITY` (1e9), keep generating

### Text truncation policy

Only **Phase 1** (waiting queue non-empty) truncates text at the stop-string
boundary.  All other termination paths preserve full untruncated text,
matching AE:

| Path | Truncate? | Rationale |
|---|---|---|
| Phase 1 finish | Yes | Beam legitimately stopped; matches AE's normal `StopChecker` |
| Phase 2 (speculative) | N/A | Beam keeps generating |
| Force-finish | No | Full speculative text returned for `split_string_by_separator` |
| Overflow (>256) | No | Matches AE: `FINISHED_STOPPED` with full text |
| Preemption cleanup | No | Matches AE: aborted with accumulated text intact |

### Preemption and overflow

- **Preemption**: speculative beams with `priority=1e9` are preempted first
  by V1's `max(running, key=lambda r: (r.priority, r.arrival_time))`.
  `_cleanup_preempted_speculative()` detects and aborts them each step.
- **Overflow**: if active beams exceed 256, oldest speculative beams are
  force-finished first.
- **Force-finish**: when `len(finished) + len(speculative) >= total_requests`,
  all speculative beams are terminated.

---

## Part 4 — GPU Memory Allocation for v0.18.1

v0.18.1 has higher memory overhead than v0.9.2 (CUDA graphs, torch.compile).
Re-profiled with `memory_latency_analysis.py` and updated benchmark configs:

- Total memory lowered from 0.90 to **0.89** to prevent OOM.
- Verifier minimum raised from 0.15 to **0.16** — at 0.15, the verifier
  fails to initialize due to insufficient KV cache memory (0.07 GiB
  available, 0.11 GiB needed for `max_model_len=4096`).

Both AE and thesis configs updated to the same splits for fair comparison.

---

## Part 5 — PRM Prefix Caching Optimization

### Background

Both V0 (0.9.2) and V1 (0.18.1) disable prefix caching for ALL-token
pooling models.  V0 disables it at `config.py:4594` ("Only 'last' pooling
supports chunked prefill and prefix caching; disabling both").  V1 sets
`skip_reading_prefix_cache=True` at `pooling_params.py:125` for
`token_classify` tasks.

Verified empirically: `enable_prefix_caching=True/False` gave 1.00x speedup
on both V0 and V1 — caching is truly disabled internally.

### Why it matters

In beam search, the PRM scores N solutions per question per iteration.  All
N share the same question prefix (200-800 tokens).  Without caching, the
prefix is recomputed N times.  With n=128 beams, ~25,600 redundant tokens
per scoring call.

### Implementation

**No vLLM changes required.**  `PoolingParams.skip_reading_prefix_cache`
already supports explicit `False` — the default override at
`pooling_params.py:121` only triggers when the value is `None`.

#### 1. Enable prefix caching for PRM (`tts_llm.py`)

```python
rewards = self.encode(
    prompts,
    pooling_params=PoolingParams(skip_reading_prefix_cache=False),
)
```

#### 2. Offset-based reward_flags indexing (`tts_llm.py`)

With prefix caching, `reward_embedding` is shorter than `reward_flags`
(cached prefix tokens produce no hidden states).  The offset computation
handles both cold cache (offset=0) and warm cache (offset>0):

```python
offset = len(reward_flag) - len(reward_embedding)
for i, flag in enumerate(reward_flag):
    if flag == 1:
        local_idx = i - offset
        if local_idx < 0:
            step_reward.append(None)  # cached — fill later
        elif local_idx >= len(reward_embedding):
            break
        else:
            step_reward.append(sigmoid(reward_embedding[local_idx][0]))
```

All question prefix flags are 0 (no step boundaries in the question), so
within-batch prefix sharing (the common case) produces zero Nones.

#### 3. Score bookkeeping — prev_scores merge + within-batch propagation

**Problem:** With prefix caching, sibling beams sharing solution prefixes
(not just question prefixes) within the same scoring batch lose earlier step
scores.  The overwrite at `beam_search.py:422` (`beam.all_scores = score[0]`)
replaces a complete score list with a potentially shorter one.  This breaks:
- SBE skip logic (`len(all_scores) >= i+1`) — fails, beam re-scored every
  iteration
- Non-`last` aggregation strategies (`min`, `prod`, `mean`) — wrong results

**Two-layer fix:**

**Layer 1 — prev_scores merge (cross-iteration):**  Each scored beam's
existing `all_scores` is passed as `prev_scores` to `_score_outputs_skywork`.
None entries from cached step boundaries are filled from the beam's own
history.  Zero overhead — direct index alignment.

Score locking: Layer 1 always prefers prev_scores over fresh computation.
Once a step is scored, its value is locked — never changes.  This prevents
BF16 noise (~2%) from producing different scores for identical text across
duplicates.

```python
# Layer 1: scores locked once set
for j in range(min(len(step_reward), len(prev))):
    if prev[j] is not None:
        step_reward[j] = prev[j]
```

**Layer 2 — within-batch + skipped beam propagation (within-iteration):**
If Nones remain after Layer 1, copy scores from batch neighbors AND
skipped beams with matching token prefixes.  Same tokens → same hidden
state → same score.  Only runs when Nones exist (not on the critical path).

Skipped beams (SBE skip: `len(all_scores) >= i+1`) are not in the scoring
batch but have valid scores.  Their context `(prompt, current_text,
all_scores)` is collected in `beam_search.py` and passed to propagation.
The bank is seeded with their scores (tokenized lazily).

This fix is necessary because of the interaction between `all_scores[:i]`
trim and KV cache persistence.  When a parent beam's LookAhead scores a
step and the parent is later skipped, the `[:i]` trim removes that score
from the duplicate.  But the parent's KV cache persists.  With R=0.85
(paper's truncation ratio), the duplicate keeps 85% of the first speculative
step's tokens.  For math problems with constrained derivations (e.g.,
Heron's formula, quadratic equations), the remaining 15% is often
regenerated identically — causing KV cache hits at the step boundary.

```
Iteration N:
  Parent: LookAhead scores [step0 + step1 + step2]
    → all_scores = [s0, s1, s2], KV cache stores all tokens

  Duplication at i=N+1:
    duplicate.all_scores = [:N+1] → trims s2 (LookAhead score)
    duplicate.future_texts = [(truncated_85%_of_step1, False)]

Iteration N+1:
  Parent: len([s0,s1,s2]) >= N+2 → SKIPPED
  Duplicate: generates step1 (85% kept + 15% regenerated)
    → math constrained: regenerated 15% matches parent's tokens
    → KV cache hits through step2 boundary → step2 score = None
    → prev_scores has no step2 (trimmed)
    → Layer 2: skipped beam bank has s2 from parent → fills None ✓
```

Overhead: zero in common case (propagation only runs when Nones exist).
When triggered: ~0.5ms to tokenize ~4 skipped beams (vs ~250ms scoring).

Example — two beams from the same parent, scoring new steps:

```
Within a single score_outputs batch:
  beam X: [q + step1 + step2 + spec3_X]  → first in batch → cold → [s1, s2, s3_X]
  beam Y: [q + step1 + step2 + spec3_Y]  → KV hits [q + step1 + step2] from X
    → only spec3_Y computed → [None, None, s3_Y]
    → Layer 1: prev_scores = [s1_Y] (from prior iteration) → fills step1
    → result after Layer 1: [s1_Y, None, s3_Y]
    → Layer 2: bank has (prefix_at_step2 → s2) from X
    → token prefix matches (identical step1+step2) → copies s2
    → result: [s1_Y, s2, s3_Y] ✓
```

**Layer 3 — RuntimeError safety net:**  If Nones remain after both layers
(pruned beam's KV cache reused — astronomically unlikely with temp > 0),
raise `RuntimeError` to alert rather than silently producing wrong scores.

### Performance

Measured on 128 PRM requests with shared question prefix:

| Config | Time | Speedup |
|---|---|---|
| Prefix caching OFF | 0.437s | — |
| ON, default params (skip=True internally) | 0.440s | 1.00x |
| ON, skip_reading_prefix_cache=False | 0.038s | **18.5x** |

### Files modified

| File | Change |
|---|---|
| `models/tts_llm.py` | `skip_reading_prefix_cache=False`, None placeholders, prev_scores merge, propagation with skipped beam bank, score locking, RuntimeError |
| `search/beam_search.py` | `score_beam` accepts `prev_scores` + `skipped_beam_context`; collects both |
| `search/dvts.py` | Same pattern |
| `search/dynamic_branching.py` | Same pattern |
| `search/vg_search.py` | Same pattern |
| `search/best_of_n.py` | No change (single-pass, no iterations) |

---

## Part 6 — Beam Duplication + Score Consistency Fixes

### 6a. Stale speculative score inheritance (pre-existing in AE)

When a beam with SBE speculative scores is duplicated (`deepcopy`) and its
last `future_text` is truncated, the duplicate inherits the parent's full
`all_scores` including speculative scores.  If the duplicate's text later
diverges from the parent's speculative text, the SBE skip logic
(`len(all_scores) >= i+1`) skips scoring and uses stale speculative scores.

Verified empirically: step2 score error ~0.017, step3 score error ~0.279
between stale (parent's speculative) and fresh (duplicate's actual text).

### 6b. Duplication fix: truncate first speculative step (paper-correct)

The AE's duplication only truncates the LAST `future_texts` entry, keeping
all earlier speculative steps intact.  Duplicates consume the parent's
speculative steps unchanged — producing identical text (no divergence).
This contradicts the FastTTS paper's Algorithm 1 line 19
(`DuplicateThenTruncate`) which says "speculative tokens truncated to
simulate divergence."

**Fix:** Truncate the FIRST speculative step (R=0.85), clear all
subsequent speculative steps.  Trim `all_scores[:i]` to remove speculative
scores (which are for text the duplicate won't have).

```python
# Paper-correct (our fix):
if beam.future_texts:
    first_text = truncate_sentence_by_tokens(beam.future_texts[0][0], tokenizer)
    duplicate.future_texts = [(first_text, False)]
    duplicate.all_scores = beam.all_scores[:i]
```

- **Parent beam**: keeps full `all_scores` and `future_texts` — unchanged
- **Duplicate**: truncated first step + cleared rest → immediate divergence.
  `all_scores[:i]` keeps only verified scores.

### Files modified

| File | Change |
|---|---|
| `search/beam_search.py` | Truncate first spec step, clear rest; collect prev_scores |
| `search/dvts.py` | Same duplication fix |
| `search/dynamic_branching.py` | Same fix |
| `search/vg_search.py` | Same fix |
| `models/tts_llm.py` | Score locking (Layer 1 prefers prev); RuntimeError diagnostics |

---

## Part 7 — PRM Input Tokenizer Boundary Overflow Fix

### The Bug

`prepare_input()` in `models/reward_utils.py` could produce `input_ids`
exceeding `max_model_len` (4096), causing:

```
VLLMValidationError: This model's maximum context length is 4096 tokens.
However, you requested 0 output tokens and your prompt contains at least
4097 input tokens.
```

### Root Cause

The old code tokenized the response text **multiple times in different ways**:

1. `tokenizer.encode(full_response)` — whole-string tokenization for budget check
2. `tokenizer.encode(step + step_token)` — per-step tokenization for truncation
3. `tokenizer.encode(step_text)` — per-step tokenization again for output

BPE tokenizers produce **different token counts** when the same text is
tokenized as a whole vs. split at `\n\n` boundaries and tokenized per-part.
The budget check used whole-string count (lower), but the output used
per-step count (higher). When the response was near the limit, the output
could silently exceed `max_model_len`.

Verified empirically with Qwen2.5 tokenizer:
```
Whole-string tokenization: 4037 tokens  (budget check says: FITS)
Per-step tokenization:     4083 tokens  (output actually: OVERFLOW)
Difference:                +46 tokens across 131 steps
Old prepare_input produces: 4110 > 4096
```

The effect accumulates — **not** a +1 off-by-one but up to +46 tokens
across many steps, because each `\n\n` split point can shift BPE merges.

### How AE and Other Frameworks Avoid This

| Framework | Approach |
|---|---|
| **FastTTS-AE** | Hardcoded budget of 3072 (vs 4096 limit) — 1024 tokens of headroom absorbs the mismatch |
| **compute-optimal-tts** | PRM runs through raw HuggingFace (not vLLM) — no `max_model_len` validation |
| **search-and-learn** | No truncation for Skywork PRM — assumes inputs fit |

We raised the budget from 3072 → 4096 (to use the full model context),
which eliminated all headroom and exposed the bug.

### The Fix

**Tokenize each step exactly once.** Use the same token IDs for both
budget accounting and output building — correct by construction.

```python
# Split into steps and tokenize each exactly once
raw_steps = response.split(step_token)
step_entries = []  # list of (text, token_ids)
for i, step in enumerate(raw_steps):
    if step == "" and not step_entries:
        continue
    is_last = (i == len(raw_steps) - 1)
    text = step if is_last else step + step_token
    ids = tokenizer.encode(text)
    step_entries.append((text, ids))

# Budget check uses the SAME ids that will be in the output
total_tokens = sum(len(ids) for _, ids in step_entries)
if total_tokens > max_response_tokens:
    # Keep complete steps from the end (reversed scan)
    ...

# Build output from the pre-tokenized steps
for text, ids in step_entries:
    response_ids.extend(ids)  # ← same ids object used for budget
```

The output `input_ids` are **identical** to the old code's output (both
tokenize per-step). The only change is the budget check now uses the
true per-step count instead of the misleading whole-string count. For
responses near the limit, truncation is slightly more aggressive (drops
one more step) but the output never overflows.

---

## Paper Claims vs AE Code: What Was Never Implemented

Several features described in the FastTTS paper (ASPLOS '26) are either
commented out or absent in the AE artifact.  These discrepancies are
**pre-existing in the original codebase** — they are NOT migration errors.
The V1 migration faithfully preserves the AE's actual behaviour.

### 1. Score-based speculative priority tiering (§4.1.1) — NOT in AE

**Paper claim:** "Our system policy partitions these scores into *B* discrete
bins, {*C*₁, …, *C_B*}, where *C*₁ is the highest-score bin and *B* is the
search's branching factor.  For a beam *bᵢ* with score *sᵢ*, our policy
determines its *speculative potential*: If *sᵢ* ∈ *Cⱼ*, then *Mᵢ* = *B* − *j* + 1."

**AE code:** All speculative beams receive the **same flat priority**:

```python
# models/numbers.py
SPEC_BEAM_CANDIDATE_PRIORITY = 1e9     # every speculative beam gets this
WAITING_DEFAULT_PRIORITY = 1e8

# models/generator_engine.py:238
if is_finished_stopped_with_stop(seq_group.first_seq):
    seq_group.priority = SPEC_BEAM_CANDIDATE_PRIORITY   # uniform — no score binning
```

No discrete-bin partitioning, no *Mᵢ* calculation, no score-aware tiering
exists anywhere in the codebase.  The V1 migration preserves this flat
priority design.

### 2. Dynamic Prefix-Aware Scheduling (§4.2) — commented out in AE

**Paper claim:** "We solve this optimization problem using a greedy approach.
Given the set *Q* of CoTs to be scheduled, the following scheduling invariant
is maintained: *T*_{*k*+1} = argmax *P*(*c_k*, *cᵢ*)."  Describes reordering
beams by shared prefix using a radix tree (Trie) to minimize KV cache evictions.

**AE code:** The entire implementation is commented out:

- `models/custom_scheduler.py` — `enable_prefix_aware_scheduling()`,
  `update_running_priorities_with_prefix()` are fully commented out (~40 lines).
- `search/beam_search.py:45-57` — prefix-aware priority assignment commented out.
- `search/dynamic_branching.py:45-57` — same.
- `config.py:121` — `prefix_aware_scheduling: bool = False` (flag exists,
  default disabled, never enabled in any config file).
- `search/utils.py:85-127` — `assign_prefix_priorities()` function exists
  but is never called by any live code.

The V1 migration carries the config flag through (`prefix_aware_scheduling`
parameter in `TTSLLM.__init__` and `vllm_wrapper.py`) but does not implement
scheduling logic, matching AE.

### 3. Features that ARE implemented

| Feature | Paper section | AE status | V1 status |
|---|---|---|---|
| Speculative Beam Extension (core SBE loop) | §4.1, Algorithm 1 | Implemented | Migrated ✓ |
| Two-phase scheduling (Phase 1 / Phase 2) | §4.1.2 | Implemented | Migrated ✓ |
| Preemption of speculative beams | §4.1.2 | Implemented | Migrated ✓ |
| Force-finish when standard beams done | §4.1.2 | Implemented | Migrated ✓ |
| 256-beam overflow cap | §4.1 (impl detail) | Implemented | Migrated ✓ |
| LookAhead Verification | §4.1.3 | Implemented (search layer) | Unchanged — search layer not modified |
| `split_string_by_separator` for SBE output | §4.1.3 | Implemented (`search/utils.py`) | Unchanged |
| Asymmetric Multi-Model Memory Allocation | §4.3 | Implemented (`memory_latency_analysis.py`) | Unchanged — orthogonal to engine migration |

**Note:** The unimplemented features above (score-based priority tiering,
dynamic prefix-aware scheduling) are orthogonal to our thesis offloading
work.  They operate at the search/scheduling layer, not the engine/model
layer where offloading changes are made.  If implemented in the future,
they would compose with the V1 engine without conflict.

---

## Verification

Test scripts live in `migration_verification/`.  Run from the container with
`conda activate thesis` and `cd /TTC/FastTTS-thesis`.

### Standalone tests

```bash
python migration_verification/verify_prm.py            # PRM plugin + prefix caching (13 tests)
python migration_verification/verify_sbe.py             # SBE (8 tests)
python migration_verification/verify_prepare_input.py   # Tokenizer overflow fix (3 tests)
```

**`verify_prm.py`** — tests plugin registration, model loading, output shape,
`prepare_input` reward flags, per-step scoring, score range [0,1],
multi-question batching, pooler nn.Module interface, and chunked prefill +
prefix caching correctness (single-sequence shape, batch of 16 with shared
prefix, `score_outputs` end-to-end, repeated scoring for cache consistency).

```
Tests 1-8:   Migration correctness (plugin, model, scoring, pooler)
Test 9:      Question prefix sharing — within-batch, all scores correct
Test 10:     Cross-iteration merge — prev_scores preserves old step scores
Test 11:     Within-batch propagation — siblings share solution prefix
Test 12:     Edge case — RuntimeError fires when no donor available
Test 13:     Performance — 18.82x speedup
```

**`verify_sbe.py`** — tests organized by paper section (§4.1–§4.1.3):
baseline stop behaviour, speculative continuation past stop strings,
multi-step speculation, force-finish, priority scheduling, two-phase
scheduling (Phase 1 / Phase 2), text preservation on force-finish,
and `split_string_by_separator` lossless reconstruction.

```
Tests 1-8:  SBE behavioral equivalence
```

**`verify_prepare_input.py`** — reproduces the tokenizer boundary overflow
bug with the old code, then verifies the fix.

```
Part 1: Bug reproduced — old code produces 4110 > 4096
Part 2: Old code overflows 169/200, new code overflows 0/200
Part 3: Stress test — 0 overflows across 1000 random inputs, max 4095
```

### Direct AE ↔ Thesis comparison

```bash
python migration_verification/compare.py      # Cross-env comparison (5 tests)
```

Runs identical inputs through both `baseline` (AE + vLLM 0.9.2, V0) and
`thesis` (V1) environments via subprocess, then compares:

| Test | What it checks | Result |
|---|---|---|
| 1. Tokenization | `input_ids` identical | Exact match |
| 2. Reward flags | Step boundary positions identical | Exact match |
| 3. Raw per-token rewards | Per-token scalar scores | max diff 4.7e-2 (BF16 noise) |
| 4. Per-step PRM scores | Sigmoid-transformed step scores | max diff 0.0e+0 (exact after sigmoid) |
| 5. Baseline generation | temp=0, no SBE, text output | All 4 beams identical |

The raw per-token reward differences (Test 3) are BF16 mantissa noise from
different attention backends / kernel paths between V0 and V1.  After sigmoid
transformation the final per-step scores — which the search algorithm actually
uses — are **exactly identical** (Test 4).  Generator output is also
**byte-for-byte identical** (Test 5).

---

## Part 8 — Codebase Restructuring

### Motivation

Debugging migration issues (score propagation, generator overflow, SBE
interactions) was increasingly difficult because `_beam_search()` was a
336-line monolithic function with 8+ interleaved concerns, score
propagation logic was entangled with debug artifacts in `tts_llm.py`, and
dead code accumulated during the migration.

### Changes

#### 1. Decomposed `_beam_search()` into named phases (`search/beam_search.py`)

The monolithic loop body was split into 11 functions, each handling one
concern:

| Function | Responsibility |
|---|---|
| `_init_state()` | Create beams, sampling params, compute prompt token length |
| `_filter_active()` | Remove pruned beams |
| `_check_n_completion()` | Record metrics when n completions first reached |
| `_duplicate_beams()` | Expand active beams to `config.n` via duplication |
| `_consume_future_texts()` | Pop SBE future_texts into current_text |
| `_generate()` | Build conversations, call generator |
| `_process_results()` | Process gen results, build `ScoringBatch` |
| `_score_and_assign()` | Call verifier, assign scores to beams |
| `_filter_completed_and_prune()` | Remove completed, prune lowest scores |
| `_log_iteration()` | Structured logging (INFO summary + DEBUG per-beam) |
| `_finalize()` | Post-loop metrics, sort completed beams |

New dataclasses: `BeamSearchState` (shared mutable state across phases),
`ScoringBatch` (typed container for verifier inputs).

`beam_is_completed()` rewritten for clarity.  The old expression
`stop == "EOS" or ... or (stop != "\n\n")` was a tautology for all
non-`\n\n` stops.  New version inverts the logic: returns False only when
`stop == "\n\n"` and text is non-empty (normal step boundary, keep going).
Logically equivalent.

#### 2. Added beam identity tracking (`search/beam.py`)

New fields on `Beam`: `beam_id` (unique, auto-incrementing), `parent_id`
(set on duplication), `born_at_iteration`.  Module-level counter
(`_next_beam_id()`, `reset_beam_id_counter()`).

Enables tracing beam lineage through logs: "beam 14 (parent=3, born@2)".

Removed dead methods `add_generation()`, `clone()`, `get_score()` — never
called by any code.

#### 3. Structured iteration logging

Replaced the ad-hoc debug dump (old lines 498–521, which ran expensive
per-beam tokenization every iteration at INFO level) with `_log_iteration()`:

- **INFO** (always): one-line summary — beam counts, latencies
- **DEBUG** (opt-in): per-beam table with ID lineage, token counts, scores,
  future_texts count, stop reasons.  Expensive tokenization only at DEBUG.

#### 4. Extracted score propagation (`search/score_propagation.py`)

Three-layer score propagation (prev_scores locking, within-batch
propagation, RuntimeError safety net) extracted from
`tts_llm.py:_score_outputs_skywork()` into standalone functions:

- `lock_prev_scores()` — Layer 1: always needed with prefix caching
- `propagate_within_batch()` — Layer 2: only triggered with SBE active
- `validate_no_missing()` — Layer 3: only triggered with SBE active

Without SBE, Layer 1 fills all Nones (earlier step boundaries are always
in prev_scores because they were scored when they were the "last" step).
Layers 2–3 naturally no-op via the `has_nones` check.

Removed from `tts_llm.py`: `_score_call_counter` class variable, 45-line
debug file dump (`score_debug_dump.txt`), `_propagate_scores_within_batch`
static method.

#### 5. Removed lookahead dead code (`search/beam_search.py`)

`config.lookahead` defaults to 0 and is never overridden.  The multi-step
generation loop in `generate_beam` (`for i in range(lookahead_steps + 1)`)
always ran exactly once.  Collapsed to a single generation call.

Removed: `lookahead_steps` parameter, `lookahead_sampling_params`,
`lookahead_text` accumulator, `Beam.lookahead_texts` field (only written,
never read by search logic).

This is unrelated to SBE's `future_texts` (from `split_string_by_separator`),
which remains unchanged.  The paper's "LookAhead Verification" (§4.1.3)
refers to scoring `current_text + future_texts[0]` when the next speculative
step is complete — that works via SBE, not via multi-call generation.

#### 6. Removed dead code

- Deleted `core.py` — imported `AsyncGeneratorVLLMModelWrapper`,
  `beam_search_async`, neither of which exist.  The live entry point is
  `fasttts.py`.
- Fixed `__init__.py` — removed async wrapper exports, imports from
  `fasttts.py` instead.

### Verification

All changes are pure restructuring — no behavioral changes.  The same
`migration_verification/` test suite validates correctness.

---

## TODOs / Open Issues

### Pruned beam KV cache persistence (unsolvable)

A pruned beam's KV blocks remain in vLLM's global cache but its scores
are discarded.  No donor exists anywhere — the scores are permanently lost.
Caught by the RuntimeError safety net (Layer 3).

```
Iteration N:
  beam A: [q + step1 + spec2 + spec3]
    → scored → all_scores = [s1, s2, s3]
  beam B: [q + step1]
    → scored → all_scores = [s1']
  (identical step1 → identical s1 → same aggregated score)
  → tie-broken by array position: A pruned, B survives

Iteration N+1:
  beam B: generates step2 (== spec2 tokens) + spec3'
    → scoring: [q + step1 + step2 + spec3']
    → KV cache hits [q + step1 + step2] from A (blocks persist after prune)
    → step2 score = None
    → prev_scores = [s1'] (no step2)
    → A pruned, scores gone → no donor anywhere
    → RuntimeError
```

### Generator prompt overflow — SBE future_texts dump (pre-existing in AE)

**Status:** Root cause identified, fix not yet implemented.

**Bug:** At the last beam search iteration, all remaining `future_texts` are
dumped into `current_text` at once (line 328-334 in `beam_search.py`).  If a
beam accumulated many speculative steps via SBE, this dump can push
`current_text` far past `max_model_len`.  The subsequent `generate_beam` call
builds a chat-templated prompt from `current_text` and vLLM rejects it.

**Confirmed identical in AE** — same code at `FastTTS-AE/search/beam_search.py:298-305`.
Both V0 and V1 reject prompts > `max_model_len` (`verify_gen_overflow.py`).
The AE doesn't crash in practice because the 1.5B-generator config and typical
AIME problems rarely accumulate enough text.  With the 7B generator, certain
problems trigger it reliably (`compare_max_model_len.py` crashes on AIME
problem 17 at iteration 9).

**Root cause trace** (from debug logging in `compare_max_model_len.py`):

```
Iteration 0:
  beam 6: ct_tokens=13, templated_prompt=236, future_texts=23 (788 tokens)
  ← SBE generated 2048 tokens in one call, split into 23 steps by \n\n

Iteration 6:
  beam 5: ct_tokens=2288, templated_prompt=2511, future_texts=27 (3666 tokens)
  ← beam duplicated + generated, accumulated more speculative steps

Iteration 7 (generate_beam):
  beams 0,1,3,5: prompt=3748, output=348, total=4096 → length cap → completed
  beams 2,4,6,7: found \n\n before cap → NOT completed, continue

Iteration 8:
  beam 0: ct_tokens=2323, templated_prompt=2546, future_texts=25 (3631 tokens)

Iteration 9 (LAST — dump-all fires):
  while beam.future_texts:           ← adds 3631 tokens to current_text
      beam.current_text += next_text
  prompt = chat_template(question + 2323 + 3631 tokens) = 6174 tokens
  → VLLMValidationError: 6174 > 4096
```

**Why `beam_is_completed` doesn't catch it:**
- `prompt_token_length` is stale (computed once at start with empty response = ~200)
- `completion_tokens` only counts the primary generation step, not accumulated text
- The real prompt is 3748 tokens but the check sees `200 + 348 = 548 < 4096`
- The `stop_reasons[0] != "\n\n"` condition only catches non-`\n\n` stops
- Beams that found `\n\n` (normal step boundary) pass all checks and continue

**Why skipped beams don't trigger it:**
- Skipped beams consume one `future_text` per iteration and ARE scored by the PRM
- This per-step consumption is correct — each step gets individually scored
- The bug is only the **excess** future_texts that can't be consumed within
  `num_iterations` — they're dumped all at once at the last iteration

**Proposed fix:** If a beam's total speculated text (`current_text` + all
remaining `future_texts`) would exceed `max_model_len` when chat-templated,
move it to `completed_beams` with its current scores instead of trying to
dump and generate.  Alternatively, cap `future_texts` so the dump-all at the
last iteration stays within `max_model_len`.

### Context length limit (4096)

Qwen2.5-Math-7B-Instruct has `max_position_embeddings=4096` — this is the
model's hard limit, not a conservative config choice.  With 7B generators,
beams hit this limit as early as iteration 5/10 without producing `\boxed{}`
answers.  The reference paper ("Can 1B LLM Surpass 405B LLM?") uses 8192
total, but with Qwen2.5-7B-Instruct (128K context), not the Math variant.
Raising the limit for Math-7B would require `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
and risks NaN from RoPE extrapolation.  Alternative: switch to
Qwen2.5-7B-Instruct (128K context, no Math fine-tuning) for longer
reasoning chains.

### Debug logging cleanup — DONE (Part 8)

Removed `_score_call_counter` and `score_debug_dump.txt` writer from
`tts_llm.py`.  Replaced ad-hoc beam state dumps in `beam_search.py` with
structured `_log_iteration()` (INFO summary always, DEBUG per-beam detail
opt-in).  `generate_beam` token length warnings gated on DEBUG level.
