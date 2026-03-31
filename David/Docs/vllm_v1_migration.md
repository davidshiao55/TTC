# FastTTS → vLLM V1 Migration

## Why We Did This

FastTTS-thesis was written against the vLLM 0.9.x API and the V0 engine
(synchronous scheduler, direct queue access).  Our vLLM fork has moved to the
V1 architecture (`VLLM_USE_V1=1`), and all thesis offloading work — attention
offloading, weight offloading, KV-cache management — lives under
`vllm/v1/`.  Running FastTTS on V0 would mean **none of those optimisations
are exercised at test-time**, making any comparison against the baseline
meaningless.

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

---

## Part 2 — Port the FastTTS Engine Stack

### Files removed from `models/__init__.py`

| File | Reason removed |
|---|---|
| `generator_engine.py` | V0 engine, broken imports, replaced by `generator_engine_v1.py` |
| `verifier_engine.py` | V0 engine, broken imports, custom scheduler replaced by V1 native priority |
| `spec_stopchecker.py` | V0 SBE mechanism, not needed for base migration |

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
supported_tasks = self.llm_engine.get_supported_tasks()
self.supported_tasks = supported_tasks
self.pooling_task = self.model_config.get_pooling_task(supported_tasks)
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

## Verification

Test scripts live in `migration_verification/`.  Run from the container with
`conda activate thesis` and `cd /TTC/FastTTS-thesis`.

### Standalone tests (thesis env only)

```bash
python migration_verification/verify_prm.py   # PRM plugin (12 tests)
python migration_verification/verify_sbe.py   # SBE (8 tests)
```

**`verify_prm.py`** — tests plugin registration, model loading, output shape,
`prepare_input` reward flags, per-step scoring, score range [0,1],
multi-question batching, pooler nn.Module interface, and chunked prefill +
prefix caching correctness (single-sequence shape, batch of 16 with shared
prefix, `score_outputs` end-to-end, repeated scoring for cache consistency).

**`verify_sbe.py`** — tests organized by paper section (§4.1–§4.1.3):
baseline stop behaviour, speculative continuation past stop strings,
multi-step speculation, force-finish, priority scheduling, two-phase
scheduling (Phase 1 / Phase 2), text preservation on force-finish,
and `split_string_by_separator` lossless reconstruction.

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

## Bugs Fixed

### 1. `PrmPooler` must be an `nn.Module`, not a bound method

`gpu_model_runner.py` calls `model.pooler.get_supported_tasks()` during the
warmup/memory-profile run at engine startup.  The original code defined
`pooler()` as a method on `Qwen2ForPrmModel`.  When `gpu_model_runner` accessed
`model.pooler`, Python returned a bound method object — which has no
`get_supported_tasks` attribute.

```
AttributeError: 'function' object has no attribute 'get_supported_tasks'
```

Fixed by creating a `PrmPooler(Pooler)` class and assigning `self.pooler = PrmPooler()` in `__init__`.

### 2. Chunked prefill produces truncated pooling output

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

**Future optimization — prefix caching for PRM:** Currently
`skip_reading_prefix_cache=True` forces full recomputation of all tokens.
Since step boundaries are only in the solution suffix (not the shared question
prefix), prefix caching could safely skip the question KV computation.  This
would require: (1) overriding `skip_reading_prefix_cache`, (2) adjusting
`reward_flags` indexing to account for the prefix offset.

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