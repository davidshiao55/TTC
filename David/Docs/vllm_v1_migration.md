# FastTTS vLLM V1 Migration (0.9.2 -> 0.19.0)

## 1. Introduction & Motivation

FastTTS-thesis was written against the vLLM 0.9.2 API and the V0 engine
(synchronous scheduler, direct queue access). The thesis requires vLLM V1
(`VLLM_USE_V1=1`) because all offloading work -- attention offloading, weight
offloading, KV-cache management -- targets `vllm/v1/`. Running FastTTS on V0
would mean none of those optimisations are exercised at test-time, making any
comparison against the baseline meaningless.

The thesis vLLM fork is pinned to **v0.19.0** (tag `2a69949bd`) for
reproducibility. The `thesis` branch is created from this tag for all
thesis-specific modifications.

### Migration scope

| Component | Status |
|---|---|
| `models/tts_llm.py` | Fully migrated to V1 APIs; `__init__` decomposed, score dispatch via registry, `_score_outputs_skywork` split into 3 phases (§8) |
| `models/generator_engine_v1.py` | New file -- V1 SBE implementation |
| `models/vllm_wrapper.py` | Updated (`task`->`runner`, `PoolerConfig`); worker now dispatches via `_WORKER_HANDLERS` registry, `ProcessTokenizerWrapper` collapsed via `_rpc` (§8); new `get_run_stats` action + public method for per-engine run totals (prefix cache, PCIe transfers, batch-size histogram) (§12c) |
| `models/reward_utils.py` | Rewritten (tokenizer boundary fix, §6a; tail-truncate runaway newest step, §6f); `DEFAULT_STEP_TOKEN` constant |
| `benchmarks/benchmark_config.py` | Passes `kv_offloading_size` / `kv_offloading_backend` / `disable_hybrid_kv_cache_manager` through to engine kwargs (§12b) |
| `config.py` | Extracted `DEFAULT_SYSTEM_PROMPT`, `DEFAULT_GENERATOR_VLLM_CONFIG`, `DEFAULT_VERIFIER_VLLM_CONFIG` module constants; removed duplicate `FastTTSConfig` generation-param fields (§8); `DEFAULT_SYSTEM_PROMPT` replaced with the OpenR/PRM800K short standard prompt (§13); added `SearchConfig.spec_truncation_ratio` (R) with default 0.0 for vanilla equivalence (§6b) |
| `fasttts.py` | `_SEARCH_STRATEGIES` registry replaces if/elif; `search()` decomposed; `SearchResults` in/out; dead `search_single` / `create_search_config` proxy removed (§8) |
| `search/beam_search.py` | Fully rewritten (decomposed, StepChunk, step-hash propagation) |
| `search/beam.py` | Restructured (renamed fields, StepChunk, beam identity) |
| `search/common.py` | New -- shared infrastructure (SearchState, phase functions, generate/score/parse); `package_results` returns `SearchResults`; duplicate blocks (lines 359-, 377-) gated by `spec_truncation_ratio` (§6b) |
| `search/results.py` | New -- `SearchResults` dataclass, canonical return type (§8) |
| `search/dvts.py` | Fully migrated -- uses shared infrastructure + DVTS-specific subtree pruning; duplicate block gated by `spec_truncation_ratio` (§6b) |
| `search/best_of_n.py` | Fully migrated -- now routes through `common.package_results` (dedupe); added comment documenting that SBE and prefix-aware scheduling are no-ops for single-iteration BoN (§14) |
| `search/dynamic_branching.py` | Fully migrated -- uses shared infrastructure + score-proportional duplication; duplicate block gated by `spec_truncation_ratio` (§6b) |
| `search/vg_search.py` | Fully migrated -- uses shared infrastructure + 3-stage sampling params |
| PRM plugin (`prm_model.py`) | Migrated to V1 TokenPooler |
| `accuracy_evaluation/evaluation/evaluate.py` | Rewritten; headline metrics `pass@n` + `pass@1` (PRM-Vote), no test-set tuning; per-problem entries now carry `reference_answer` / `selected_answer` / `selected_idx` for error inspection (§8, §14) |
| `accuracy_evaluation/evaluation/evaluate_full.py` | **New** — full 4-metric × 4-aggregation-strategy utility for appendix/ablation tables |
| `run_all_experiments.py` | Updated (removed `top_n` sweep, new N sweep `{1, 4, 16, 64, 256}`, scaling curve plots); refactored to two-axis `(search_strategy, optimization)` sweep when BoN joined the grid (§14) |
| `benchmarks/run_benchmarks.py` | Calls `results.to_dict()` before JSONL serialization; dumps `{jsonl_stem}.runstats.json` sidecar before shutdown (§12c) |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py` | Fork patch: store transfer records as dicts rather than `OffloadingOperationMetrics` dataclass to match `reduce`/`observe` invariants in single-process mode (§12a) |
| `migration_verification/worker_e2e_thesis.py` | Uses `SearchResults` attribute access |

All 5 search strategies (`beam_search`, `dvts`, `best_of_n`,
`dynamic_branching`, `vg_search`) use the new `Beam` field names and return
`SearchResults`. The old `example.py` / `run_aime_fasttts.py` entry points
were deleted; single-run testing now goes through `benchmarks/run_benchmarks.py`.

---

## 2. PRM Plugin Migration

**File:** `modified-skywork-o1-prm-inference/vllm_add_dummy_model/prm_model.py`

### What broke

| Old import | Status in v0.19.0 |
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

These imported V0 APIs that no longer exist in v0.19.0. Originals preserved
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

v0.19.0 has higher memory overhead than v0.9.2 (CUDA graphs, torch.compile,
per-process CUDA contexts). Each spawned process adds ~0.9–1.4 GiB overhead
beyond what `gpu_memory_utilization` budgets for. With two processes
(generator + verifier), the combined overhead is ~2.3 GiB.

#### Root cause: PyTorch memory fragmentation

The original AE splits (baseline 0.68/0.22=0.90, spec-prefix 0.75/0.15=0.90)
OOM in v0.19.0 — but not from absolute exhaustion. The failure is on
activation allocations inside torch.compile-emitted code (e.g. MLP
`gate_up`, ~296 MiB at `max_num_batched_tokens=8192`). PyTorch's default
caching allocator fragments into unusable slivers (reported as "reserved
but unallocated", observed 187–869 MiB). Two engines share one GPU, so
the combined waste exceeds the ~2.4 GiB headroom at 0.90 total utilization.
Fragmentation scales with compile/graph cycles, so the full eval pipeline
reproduces it while single-iteration profiling does not.

#### Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Set in `models/vllm_wrapper.py` at module level. Growable segments replace
fixed-size blocks, which is PyTorch's documented fix for workloads with
varying allocation sizes.

**Compatibility:**
- **Safe** with kv_offload, prefetch offloader, UVA offloader (they use
  standard `torch.zeros()` / `torch.empty_strided()`).
- **Incompatible** with vLLM's `CuMemAllocator` (sleep mode), which
  replaces the default allocator. When `offload_enabled=True` turns on
  sleep mode, drop `expandable_segments` — only one model's weights are
  GPU-resident at a time, so fragmentation headroom is not needed.

#### Updated splits

With `expandable_segments:True`, the original AE total=0.90 is restored:

- Verifier minimum: 0.15 -> **0.16** (0.15 has insufficient KV cache capacity
  for `max_model_len=4096` due to higher per-process overhead in v0.19.0;
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
    pooling_params=PoolingParams(
        skip_reading_prefix_cache=skip_reading_prefix_cache,
    ),
)
```

The flag is threaded from `score_beam` through `VerifierVLLMModelWrapper.score`
→ `_handle_score` → `score_outputs` → `_score_outputs_skywork`. Default is
`False` (caching on) so the iterative strategies keep the 18× speedup and
rely on the step-hash propagation layer to fill cached-step `None`s.
`best_of_n` opts in to `True`: it's single-shot, has no step-hash bank to
propagate through, and its completions diverge from the first answer token
so the cached prefix (question only) buys little.

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

#### 3. Score propagation via step-hash matching (`search/common.py`)

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

**Fix:** Controlled by `SearchConfig.spec_truncation_ratio` (R). The
duplicate's `pending_steps` is set based on R, and `scores` /
`step_hashes` are trimmed to remove speculative entries:

```python
if beam.pending_steps:
    if config.spec_truncation_ratio <= 0.0:
        # R = 0  → true vanilla beam search equivalence. Duplicate starts
        # with no speculative seed; regenerates from current_text.
        duplicate.pending_steps = []
    else:
        # R > 0  → paper-SBE behavior: inherit ~R fraction of parent's
        # speculative first step as a divergence seed. Clears subsequent
        # speculative steps.
        first_text = truncate_sentence_by_tokens(
            beam.pending_steps[0].text, tokenizer,
            mean_ratio=config.spec_truncation_ratio,
        )
        duplicate.pending_steps = [
            StepChunk(text=first_text, is_complete_step=False, terminal=False)
        ]
    duplicate.scores = beam.scores[:i]
    duplicate.step_hashes = beam.step_hashes[:i]
```

Parent beam keeps full scores and pending_steps unchanged; duplicates
diverge immediately.

**Default is R = 0.0** (vanilla equivalence) — gives the cleanest baseline
for comparing orthogonal optimizations like kv_offload. Every SBE-enabled
yaml under `benchmarks/configs/*/*/beam_search/fasttts*/` pins
`spec_truncation_ratio: 0.0` explicitly so the record is unambiguous.
Switching to paper-SBE behavior is a single yaml edit:

```yaml
search_config:
  ...
  spec_truncation_ratio: 0.85    # paper §4.1.1 recommended value
```

Why R = 0 needs the empty-list branch rather than relying on the
`min_words=1` default inside `truncate_sentence_by_tokens`: the minimum
was designed to keep the duplicate's `pending_steps` non-empty (avoiding
a degenerate `StepChunk(text="", ...)`), but a single-token seed still
biases the duplicate's next generation slightly away from vanilla. The
explicit branch produces true algorithmic equivalence — the child
regenerates from `current_text` with no carryover content.

The same R-gate is applied to the three duplicate call sites:
`search/common.py:359-` (two blocks — reverse-iteration and forward-
iteration paths), `search/dvts.py:73-`, `search/dynamic_branching.py:93-`.
All four take `config.spec_truncation_ratio` from the `SearchState`'s
`SearchConfig`, which `benchmark_config.py:42` unpacks from the yaml
via `SearchConfig(**search_cfg)` — no manual plumbing required.

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
| 2 | **Completion count exceeds n** | Critical | `search/common.py:_check_n_completion()` |
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

**Beam search early exit** (`search/common.py`):
- `_check_n_completion()` returns `bool` -- `True` when `completed >= n`
  or no active beams remain; main loop breaks on that signal
- `_finalize()` always sorts completed beams by aggregate PRM score
  (descending) and truncates to `[:config.n]`. Handles burst completions
  fairly when several beams finish in the same iteration.
- `n_completion_tokens` is computed in `_finalize()` **after** the top-n
  sort+truncate, so it references the same beam set that `evaluate.py`
  scores for accuracy. Previously it was set in `_check_n_completion()`
  before truncation, which over-counted by the `M - n` burst-completion
  excess -- the latency/goodput plot was paying for beams the accuracy
  plot never saw.
- `n_generator_latency_s` / `n_verifier_latency_s` were removed from
  the JSONL schema and `SearchResults`. After early-exit the loop
  breaks the moment `n` is reached, so they always equalled
  `total_generator_latency_s` / `total_verifier_latency_s`. Consumers
  (`run_all_experiments.py`, `compare_e2e.py`) now read `total_*`.
  The AE-vs-thesis comparison worker still emits `n_*_latency_s` JSON
  keys sourced from `total_*` for schema parity with the untouched AE
  worker.

**Evaluation pipeline rewrite** (`evaluate.py`):
- Headline metrics (two curves per plot):
  - **pass@n**: at least one of N completions correct (OpenAI Codex
    unbiased formula). Measures search coverage.
  - **pass@1**: PRM-Vote — group answers by exact string equality, sum
    aggregate PRM scores per group, pick highest-sum group (matches
    Liu et al. `_agg_prm_last_vote`). Represents the deployed
    single-answer accuracy.
- Answer grouping uses **exact string match** (matches Liu et al.
  `_agg_orm_vote`) after `extract_answer` + `strip_string` normalization.
  `math_equal`-based grouping was abandoned due to O(n²) symbolic-math
  cost; it produced identical numbers on our data (>18× slower with no
  measurable gain).
- Configurable `agg_strategy` (default: `"last"`, matching both reference
  frameworks).
- Deterministic tie-breaking (lexicographic on canonical answer form).
- Invalid/empty extracted answers filtered before selection (matches
  Liu et al. `judge_ans`).

**Full-metric utility** (`evaluate_full.py`):
- Separate script reporting all four metrics (Pass@N, MajVote, PRM-Max,
  PRM-Vote) across all four aggregation strategies (last, min, prod,
  mean) — 16 numbers per result file. Used for appendix/ablation tables;
  not driven by `run_all_experiments.py`.
- Shares extract-and-grade work across strategies via a per-problem
  cache; only the aggregation step is recomputed per strategy.

**Orchestration** (`run_all_experiments.py`):
- Removed `TOP_N_VALUES` sweep entirely.
- `evaluate_accuracy()` calls `evaluate.py` once per result file.
- `plot_accuracy()` generates pass@n + pass@1 scaling curves per
  (dataset, generator), overlaying fasttts and baseline as separate
  line styles.

### Reference framework alignment

| Aspect | compute-optimal-tts | search-and-learn | FastTTS-thesis |
|---|---|---|---|
| Stops at exactly N | Yes | Yes (pad if fewer) | Yes (early exit + truncate) |
| Score aggregation | Configurable (min/last/avg) | Hardcoded `"last"` | Configurable, default `"last"` |
| Answer selection (headline) | 7 methods, fixed per run | 3 methods, all reported | pass@1 (= PRM-Vote) |
| Answer grouping | Exact string | Exact string | Exact string (switched from `math_equal`) |
| Multi-N evaluation | Separate runs per N | Subsample from max-N | Separate runs per N |
| Pass@N metric | Not implemented | Helper only | Headline metric (pass@n) |
| Test-set tuning prevention | Config locked before eval | Ordered subsampling | No sweeping |
| N sweep | {4, 16, 64, 256} | Powers of 2 up to max-N | **{1, 4, 16, 64, 256}** (matches Liu et al. + N=1 reference) |

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

**Recommended thesis setup** (adopted):
- Generators: `Qwen2.5-7B-Instruct` (primary) + `Qwen2.5-1.5B-Instruct`
  (secondary). Both non-Math variants — 32K context avoids the Math-7B
  4096 exhaustion.
- Verifier: `Skywork-o1-Open-PRM-Qwen-2.5-1.5B` (fixed).
- `max_model_len=8192` for all configs (matches compute-optimal-tts).
- Datasets: MATH-500 (primary, 500 problems) + AIME 2024 (hard subset).
- N sweep: `{1, 4, 16, 64, 256}` — matches Liu et al.'s `{4, 16, 64, 256}`
  plus N=1 as the "no TTC" reference point (`beam_width=1`, no search).
- Methods: `fasttts` (FastTTS with all optimizations enabled) and
  `baseline` (vanilla beam search). Both swept across every N.
- SBE truncation ratio **R = 0.85** (the FastTTS paper's default — keep
  85% of speculative tokens for duplicates). Matches the published
  FastTTS configuration.
- Headline metrics: **pass@n** (search coverage) and **pass@1** (deployed
  accuracy via PRM-Vote). Full 4-metric × 4-aggregation-strategy table
  in the appendix via `evaluate_full.py`.

---

## 8. Code Restructuring

### Decomposed beam search into named phases (`search/common.py`)

The monolithic 336-line `_beam_search()` loop was split into 11 phase
functions. They live in `search/common.py` so every strategy
(`beam_search`, `dvts`, `dynamic_branching`, `vg_search`, `best_of_n`) can
compose them; `search/beam_search.py` is now a ~110-line orchestrator.

| Function | Responsibility |
|---|---|
| `_init_state()` | Create beams, sampling params, compute prompt token length |
| `_filter_active()` | Remove pruned beams (SBE: sort low→high by score) |
| `_check_n_completion()` | Record metrics, signal early exit when `n` reached |
| `_duplicate_beams()` | Expand active beams to `config.n` via duplication |
| `_prepare_step_source()` | Decide skip vs generate per beam |
| `_generate()` | Build conversations, call generator |
| `_process_results()` | Parse chunks, commit step, build scoring batch |
| `_score_and_assign()` | Call verifier, propagate scores, assign to beams |
| `_filter_completed_and_prune()` | Remove completed, prune lowest scores |
| `_log_iteration()` | Structured logging (INFO summary + DEBUG per-beam) |
| `_finalize()` | Post-loop metrics, sort completed beams |

New dataclasses: `SearchState` (shared mutable state across phases) and
`ScoringBatch` (typed container for verifier inputs).

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

### Dead code removal

- Multi-step lookahead generation loop (`config.lookahead` always 0)
- `core.py` (imported non-existent async wrappers) and its `__init__.py`
  re-exports
- `_score_call_counter` / `score_debug_dump.txt` writer
- Math-Shepherd PRM path (`_score_outputs_math_shepherd`,
  `MATH_SHEPHERD_POOLER_CONFIG`, `_SCORE_DISPATCH` registry). Thesis
  benchmarks run Skywork-PRM exclusively, so the dispatch collapsed to
  a direct call. Default verifier in `benchmark_config.py` and
  `memory_latency_analysis.py` updated to Skywork-PRM-1.5B.

### Post-migration restructuring (cross-layer)

After the search-layer decomposition landed, the same pattern (typed
dataclasses, registry dispatch, shared helpers, module-level constants)
was applied to the rest of the codebase. All changes are behaviour-preserving
-- the migration verification suite still passes.

**`SearchResults` dataclass (`search/results.py`).** Replaces the 15-key
dict returned by every strategy and `FastTTS.search()`. `append_batch()`
sums scalars + extends per-problem lists for multi-batch merging;
`to_dict()` preserves the JSONL schema so benchmark artifacts are
byte-compatible. Callers now use attribute access (`results.completions`)
instead of string keys -- typos caught at import time, and fixed a
pre-existing drift between `"completion_tokens"` and `"effective_num_tokens"`.

**Registries replace if/elif chains.**
| Location | Registry | Branches removed |
|---|---|---|
| `fasttts.py` | `_SEARCH_STRATEGIES` | 5 (approach -> strategy fn) |
| `models/tts_llm.py` | `_SCORE_DISPATCH` | 3 (PRM model -> scorer) |
| `models/vllm_wrapper.py` | `_WORKER_HANDLERS` | 7 (action -> handler) |

**`models/tts_llm.py` decomposition.** `__init__` split via
`_build_compilation_config`, `_resolve_worker_cls`, `_bootstrap_llm_attributes`.
`_score_outputs_skywork` (61 lines) split into `_build_skywork_scoring_prompts`,
`_extract_step_rewards`, `_rebuild_nested_scores`. Dead `lengths` list
removed from `_score_outputs_math_shepherd`.

**`models/vllm_wrapper.py` decomposition.** `_model_process_worker`'s
9-branch if/elif replaced with `_WORKER_HANDLERS` dict + `WorkerContext`
dataclass. `@_with_sleep_wake` decorator dedupes 4× repeated sleep/wake
wrapping around `generate`/`score`. `ProcessTokenizerWrapper`'s 4 methods
collapsed to one-liners via `_rpc(action, response_key, **payload)`.
`MATH_SHEPHERD_POOLER_CONFIG` lifted to module-level constant (was inlined
with magic numbers `step_tag_id=12902`, `returned_token_ids=[648, 387]`).
`_ensure_v1_env()` helper dedupes the env-var setup (module load + worker).

**`config.py` constants.** `DEFAULT_SYSTEM_PROMPT`,
`DEFAULT_GENERATOR_VLLM_CONFIG`, `DEFAULT_VERIFIER_VLLM_CONFIG` lifted out
of `field(default_factory=...)` inline dicts. Removed duplicate
`FastTTSConfig` fields (`system_prompt`, `temperature`, `top_p`,
`max_tokens`, `custom_chat_template`) and the `__post_init__` sync --
these fields live only on `SearchConfig` now. `custom_chat_template` was
subsequently dropped from `SearchConfig` too (no YAML or caller ever set
it, and the process-based tokenizer path never forwarded it across the
RPC boundary anyway). `num_samples` was removed for the same reason:
`best_of_n` reads `n`, and `num_samples` was never consulted.

*Latent bug fixed:* `VerifierVLLMModelWrapper.score` previously read
`self.config.system_prompt`, which existed at the `FastTTSConfig` level
only by the now-removed `__post_init__` sync. Now reads
`self.config.search_config.system_prompt` directly.

**`run_all_experiments.py`.** Extracted plot-styling constants
(`PLOT_STYLE_{GOODPUT,LATENCY,ACCURACY}`) and shared plot helpers
(`_build_records_df`, `_combos_per_dataset`, `_draw_dataset_separators`,
`_save_figure`, `_lighten`). `parse_jsonl_folder` (81 lines) split into
`_extract_n_from_filename` / `_load_jsonl_records` / `_compute_folder_metrics`.
`plot_accuracy` extracted `_collect_accuracy_panels` + `_draw_accuracy_panel`.
Subsequently refactored to a **two-axis sweep** over `(search_strategy,
optimization)` once Best-of-N joined the benchmark grid — see Section 14.

**`accuracy_evaluation/evaluation/evaluate.py`.** Headline pipeline
keeps only `_extract_and_check` + `_select_prm_vote`. `ProblemMetrics` /
`AggregatedResult` dataclasses carry `pass_at_n` + `pass_at_1_correct`.
Per-problem entries additionally carry `reference_answer`,
`selected_answer`, and `selected_idx` so a reader can diff model vs
ground truth for a wrong problem without re-parsing the source JSONL.
Full completion text is intentionally *not* stored (bloats the file across
batched configs); `selected_idx` points into
`solutions.completions[0]` in the corresponding `*_results.jsonl` for
on-demand lookup. The richer 4-metric × 4-aggregation-strategy ablation
variant lives in `evaluate_full.py` and caches extract-and-grade across
strategies.

**Dead file removal.**
- `example.py`, `run_aime_fasttts.py` -- standalone single-run entry
  points. Redundant with `benchmarks/run_benchmarks.py` and the YAML
  config grid. `experiment_utils.py` (their shared helpers) removed
  along with them.

---

## 9. Paper Discrepancies (Pre-existing in AE)

These discrepancies exist in the **original AE codebase** -- they are NOT
migration errors. The V1 migration faithfully preserves AE's actual behavior.

### Score-based speculative priority tiering (SS4.1.1) -- approximated in V1

**Paper claim:** "System policy partitions scores into B discrete bins.
For beam b_i with score s_i, if s_i in C_j, then M_i = B - j + 1."

**AE code:** All speculative beams receive the same flat priority:
```python
SPEC_BEAM_CANDIDATE_PRIORITY = 1e9  # uniform -- no score binning
```

No discrete-bin partitioning, no M_i calculation exists anywhere.

AE had a commented-out `sorted(active_beams, key=aggregate_scores)` (e.g.
`FastTTS-AE/search/dynamic_branching.py:249`) that, if enabled, selects
highest-score beams to speculate: with the waiting queue ordered low→high
by score, low-score beams drain first and hit their stops while waiting is
still non-empty (no speculation -- Phase 1); high-score beams are admitted
last and hit stops after waiting empties (speculation -- Phase 2).

**V1 status:** re-enabled in the thesis branch and consolidated into
`search/common.py:_filter_active()` so all 5 strategies pick it up. Gated on
`SearchConfig.spec_beam_extension` (mirrored from `FastTTSConfig` in
`FastTTS.search()`), so it is active only when SBE itself is on. This is not
the paper's discrete-bin scheme, but produces the same qualitative effect:
speculation skewed toward high-score beams, without per-request priority
tiering.

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
without producing `\boxed{}` answers. Raising the limit would require
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and risks NaN from RoPE extrapolation.

**Resolved:** current benchmarks use `Qwen2.5-{1.5B,7B}-Instruct` (non-Math
variants, 32K native context) at `max_model_len=8192`. Every YAML config
under `benchmarks/configs/*/` pins this value. The Math variants are no
longer referenced in the benchmark grid.

---

## 12. KV Offload Integration (v0.19.0)

Enables stock V1 `OffloadingConnector` for the Phase 0.7 POC comparing
FastTTS latency with and without CPU KV offload on the memory-tightest
7B+1.5B config. Four coordinated patches bridge the thesis experiment
harness to vLLM's kv_offload feature. The upstream bugs are pre-existing
and reproduce on both v0.18.1 and v0.19.0.

### 12a. vLLM fork: dict-vs-dataclass fix

**File:** `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py`
(was `offloading_connector.py` in v0.18.1 — same bug, file moved in the
v0.19.0 refactor)

`OffloadingConnectorStats.record_transfer` stored each operation as an
`OffloadingOperationMetrics` dataclass instance, but both `reduce()` and
`OffloadPromMetrics.observe()` assert `isinstance(op, dict)`. The
discrepancy hides in multi-process mode because IPC serialization
converts the dataclass into a dict; in single-process mode (required by
SBE via `VLLM_ENABLE_V1_MULTIPROCESSING=0`) no serialization happens and
the assertion fires on the first KV transfer.

```python
# Bug (upstream):
op = OffloadingOperationMetrics(num_bytes, time)     # dataclass
# ...later...
assert isinstance(op, dict)                           # consumer expects dict

# Fix:
op = {"op_size": num_bytes, "op_time": time}         # dict matches consumer
```

Dataclass definition left intact for API compat. Both consumer call
sites (`reduce` at the local stats logger, `observe` at the Prometheus
logger) now produce correct aggregates.

### 12b. HMA auto-disable workaround

**File:** `FastTTS-thesis/benchmarks/benchmark_config.py`

`VllmConfig.__post_init__` has an ordering bug: the Hybrid KV Cache
Manager auto-disable check runs **before** `_post_init_kv_transfer_config`
materialises `kv_transfer_config` from `cache_config.kv_offloading_size`.
When yamls use the `kv_offloading_size` shortcut (cleaner than writing a
full `kv_transfer_config`), HMA stays on and `OffloadingConnector`'s
factory rejects it:

```
ValueError: Connector OffloadingConnector does not support HMA but HMA is
enabled. Please set `--disable-hybrid-kv-cache-manager`.
```

`OffloadingConnector` in v0.19.0 still does not implement the
`SupportsHMA` mixin despite the "multiple KV groups" / "hybrid model
support" release notes, so the workaround is still required.

**Fix:** inject `disable_hybrid_kv_cache_manager=True` on **every**
benchmark engine, not just the kvoff variant. For the thesis workload
this is a strict no-op: every model is pure full-attention
(`FullAttentionSpec` for all layers), so `unify_hybrid_kv_cache_specs`
early-returns on uniform specs
(`vllm/v1/core/kv_cache_utils.py:get_kv_cache_groups`) and both paths
land in `UnitaryKVCacheCoordinator`. Forcing the flag off unconditionally
removes a confound in the `OffloadingConnector` A/B — previously only the
kvoff arm had HMA disabled, so any HMA-vs-no-HMA difference would have
been attributed to kv_offload.

```python
gen_vllm_config = {
    ...,
    "disable_hybrid_kv_cache_manager": True,  # no-op on uniform-spec models
}
ver_vllm_config = {
    ...,
    "disable_hybrid_kv_cache_manager": True,
}
```

A hybrid-attention verifier (Gemma 3, gpt-oss, Llama 4, Mamba models)
would need to revisit this — HMA-off upcasts sliding-window specs to
full attention and loses the per-layer KV memory savings.

### 12c. Per-run stats extraction (prefix cache + KV transfer traffic)

**Files:** `models/vllm_wrapper.py`, `benchmarks/run_benchmarks.py`

#### What we capture

Three categories of per-run stats, all sourced directly from vLLM's
engine internals:

1. **GPU prefix cache** — `PrefixCacheStats(queries, hits, requests)`
   from `scheduler.kv_cache_manager`. Only **prefill admission**
   contributes (`get_computed_blocks` gated by `num_computed_tokens == 0`);
   decode steps never touch the counter.
2. **CPU prefix cache** — same shape, from
   `scheduler.connector_prefix_cache_stats` (populated only when
   `OffloadingConnector` is active). CPU hit rate is conditional on a GPU
   miss: `(1 − gpu_hit) × cpu_hit` gives the unconditional fraction of
   tokens kv_offload recovered.
3. **KV transfer volume** — per-direction (`cpu_to_gpu`, `gpu_to_cpu`)
   totals of `bytes`, `time_s`, and `count` from
   `OffloadingConnectorStats.record_transfer`. Bytes are hardware-exact
   (`dst_sub_block_count * total_block_size_in_bytes`); times are
   CUDA-event-measured (`start_event.elapsed_time(end_event)`). These
   quantify the PCIe traffic kv_offload produces — critical for
   determining how much H2D bandwidth remains for weight prefetch
   (Phase 3).

#### Per-step reset vs run-total: accumulator pattern

**Upstream gotcha:** vLLM's `EngineCore.step` loop calls
`scheduler.make_stats()` every engine step. `make_stats` drains and
resets all three counter families atomically. A naive "read at shutdown"
captures only the idle tail window and produces all-zero output.

**Fix:** install accumulator wrappers at worker init. In
`models/vllm_wrapper.py`:

```python
class _CacheStatsAcc:
    """Running sum of PrefixCacheStats across engine steps."""
    __slots__ = ("queries", "hits", "requests")
    def __init__(self):
        self.queries = self.hits = self.requests = 0
    def add(self, s):
        if s is None: return
        self.queries += getattr(s, "queries", 0) or 0
        self.hits    += getattr(s, "hits", 0) or 0
        self.requests += getattr(s, "requests", 0) or 0


class _TransferStatsAcc:
    """Running sum of KV transfer bytes/time per direction."""
    def __init__(self):
        self.by_type = {}  # "cpu_to_gpu" -> {bytes, time_s, count}
    def add(self, kv_connector_stats):
        if not kv_connector_stats: return
        for xfer_type, ops in kv_connector_stats.items():
            if not isinstance(ops, list): continue
            if xfer_type not in self.by_type:
                self.by_type[xfer_type] = {"bytes": 0, "time_s": 0.0, "count": 0}
            acc = self.by_type[xfer_type]
            for op in ops:
                acc["bytes"]  += op.get("op_size", 0)
                acc["time_s"] += op.get("op_time", 0)
                acc["count"]  += 1


def _install_prefix_cache_accumulator(scheduler):
    """Wrap scheduler.make_stats to sum per-step stats for end-of-run reporting."""
    if getattr(scheduler, "_prefix_cache_acc_installed", False):
        return
    scheduler._acc_gpu_prefix = _CacheStatsAcc()
    scheduler._acc_cpu_prefix = _CacheStatsAcc()
    scheduler._acc_transfers  = _TransferStatsAcc()
    original = scheduler.make_stats
    def wrapped(*args, **kwargs):
        stats = original(*args, **kwargs)
        if stats is not None:
            scheduler._acc_gpu_prefix.add(getattr(stats, "prefix_cache_stats", None))
            scheduler._acc_cpu_prefix.add(getattr(stats, "connector_prefix_cache_stats", None))
            scheduler._acc_transfers.add(getattr(stats, "kv_connector_stats", None))
        return stats
    scheduler.make_stats = wrapped
    scheduler._prefix_cache_acc_installed = True
```

The wrapper is installed once per worker, right after
`TTSLLM(**model_kwargs)` returns, before the `status: initialized` ack
is sent. Every subsequent engine step feeds its per-step counters into
the accumulators before they're reset for the next step.

A fourth accumulator, `_BatchStatsAcc`, tracks the per-step batch size
distribution (min/mean/max `num_running_reqs`, max waiting queue depth,
peak `kv_cache_usage`, and a histogram by bucket). Same wrapping site,
same cost profile — a single stats read per step already drives the
wrapper; adding one more sub-accumulator is noise.

`_handle_get_run_stats` reads the four accumulators directly:

```python
def _handle_get_run_stats(ctx, request):
    scheduler = ctx.model.llm_engine.engine_core.engine_core.scheduler
    gpu  = getattr(scheduler, "_acc_gpu_prefix", None)
    cpu  = getattr(scheduler, "_acc_cpu_prefix", None)
    xfer = getattr(scheduler, "_acc_transfers", None)
    batch = getattr(scheduler, "_acc_batch", None)
    return {"result": {
        "gpu": gpu.to_dict() if gpu else None,
        "cpu": cpu.to_dict() if (cpu and cpu.requests > 0) else None,
        "transfers": xfer.to_dict() if xfer else None,
        "batch": batch.to_dict() if batch else None,
    }}
```

Zero-admission steps and zero-transfer steps contribute zero summands —
correct by construction. Overhead is ~100 ns of integer additions per
engine step (6 adds for prefix caches + a small dict scan for transfers
+ a bin-search for batch), negligible against the millisecond-scale step
cost.

`BaseVLLMModelWrapper.get_run_stats()` public method dispatches the
action across the multiprocessing pipe and returns the nested dict.

**Benchmark sidecar** — in `run_benchmarks.py`'s `finally` block, snapshot
both engines before `fast_tts.shutdown()` and dump
`{jsonl_stem}.runstats.json` next to the results jsonl. If the sidecar
already exists (from a prior partial run), it is **not overwritten** — the
earlier file covers more problems and is more representative. Expected
schema for a kvoff run:

```json
{
  "generator": {
    "gpu": {"queries": 468646, "hits": 458880, "requests": 768},
    "cpu": {"queries": 9766, "hits": 16, "requests": 768},
    "transfers": {
      "cpu_to_gpu": {"bytes": 123456789, "time_s": 1.23, "count": 456},
      "gpu_to_cpu": {"bytes": 987654321, "time_s": 4.56, "count": 789}
    },
    "batch": {
      "steps_total": 12043, "steps_nonzero": 11980,
      "mean_running": 187.4, "max_running": 256,
      "max_waiting": 42, "max_kv_usage": 0.82,
      "histogram": {"0": 63, "1-7": 210, "8-31": 418,
                    "32-63": 902, "64-127": 1834, "128-255": 8573, "256+": 43}
    }
  },
  "verifier": { "gpu": {...}, "cpu": {...}, "transfers": {...}, "batch": {...} },
  "num_problems": 30
}
```

For the fasttts (no offload) variant, both `cpu` and `transfers` are
`null`. Snapshot failures are logged as warnings and never abort a
completed run.

**PCIe contention analysis** from the sidecar: compare
`transfers.cpu_to_gpu.bytes / wall_clock_time` against the RTX 4090's
PCIe Gen4 ×16 peak (~25 GB/s) to determine what fraction of H2D
bandwidth kv_offload consumes. That fraction is the budget the thesis's
weight prefetch (Phase 3) would have to share.

### 12d. POC harness (`bench_kv_offload.py`)

**File:** `David/Benchmarks/phase0/bench_kv_offload.py`

Extended from a single-dataset sweep to a dataset × method × N matrix:

- `DATASETS = ["aime", "math500"]`, triple-nested loop, 16 cells total
- `collect_results()` merges each cell's `.runstats.json` sidecar into
  the metrics dict (and falls back to legacy `.cachestats.json` when
  present), exposing `gen_gpu_hit`, `gen_cpu_hit`, `ver_gpu_hit`,
  `ver_cpu_hit` fields (hit rate = `hits / queries`, or `None` when not
  applicable) plus the generator batch-stats fields
- `print_table` adds four hit-rate columns per row, one block per dataset
- `plot_results` produces a 6-panel PDF: two rows (one per dataset) × three
  columns (latency / goodput / paired speedup vs N)
- Log filenames include the dataset prefix: `{dataset}_{method}_n{N}.log`
- `summary.json` is now keyed by `dataset → method → N` (was `method → N`)

Sanity expectations:

- `fasttts` variant: `cpu_hit_rate` should be `None` (no CPU cache exists)
- `fasttts_kvoff` variant at N ≥ 16: `cpu_hit_rate > 0` (GPU cache spills
  under pressure, CPU tier catches evictions)
- GPU hit rates should be very close between the two variants for each
  cell — kv_offload doesn't change GPU prefix caching semantics, it only
  catches what GPU evicts
- Expected: CPU hit rate **correlates** with the latency speedup across
  cells; this is the attribution signal

---

## 13. Generator System Prompt Standardization

**File:** `config.py` (`DEFAULT_SYSTEM_PROMPT`)

Replaced the inherited search-and-learn "long schema" prompt
(`"Solve ... efficiently and clearly: ## Step 1: [Concise description] ..."`)
with the OpenAI MATH / PRM800K / OpenR standard:

```
Please reason step by step, and put your final answer within \boxed{}.
```

**Rationale:**
- Community-standard prompt used by OpenR, PRM800K labeling, Math-Shepherd,
  Skywork/Qwen PRM training, and `lm-evaluation-harness`. The long schema
  was search-and-learn-specific.
- PRM scoring is prompt-agnostic: `models/reward_utils.py:prepare_input`
  builds raw `[BOS] + problem + "\n" + response`, no chat template — the
  PRM never sees the system prompt. Cross-prompt PRM compatibility preserved.

**n=4 AIME validation** (bw=4, iter=10, 30 problems):

| | Long (old) | Short (new) |
|---|---|---|
| Mean tokens/beam | 735 | 734 |
| Accuracy | 3/30 | 6/30 |
| `\boxed{}` extraction | 28/30 | 30/30 |
| Uses `## Step N:` | 14/30 | 2/30 |
| Uses numbered `1. 2. …` | 5/30 | 15/30 |

Format shifts from `## Step N:` to numbered lists but `\n\n` separators and
terminal `\boxed{}` remain; PRM per-step scores stay monotonic. Dropping
brevity cues did **not** lengthen output — mean tokens unchanged.

**Temperature update:** the prior note deferred an OpenR 0.7 A/B. When
BoN joined the benchmark grid (Section 14), the reference framework's
BoN default of 0.7 (Liu et al. `run.sh:99`) became load-bearing for
paper-aligned pass@1 reporting. All yamls under `benchmarks/configs/`
were normalized to `temperature: 0.7` in one pass, replacing the
inherited 0.8. The 7B/math500 fasttts config was already at 0.7 (hand
tuning artifact), so the change is a no-op there; every other file
moves from 0.8 to 0.7.

**Reproduction:** `benchmarks/configs/7B-instruct/aime/fasttts_openr_prompt/n4.yaml`.

**Baseline re-run required.** Existing results under
`benchmarks/benchmark_results/7B-instruct/*/fasttts*` were generated with the
old prompt; archive before re-running (e.g.
`mv .../fasttts .../fasttts_longprompt_archive`) — `run_benchmarks.py` skips
already-processed IDs and would otherwise mix prompts into the same JSONL.
The temperature shift from 0.8 → 0.7 also invalidates pre-standardization
numbers; re-run is therefore unconditional across the full `(search,
optimization)` grid.

---

## 14. Best-of-N as a First-Class Benchmark Method

Best-of-N was already implemented as a search strategy in
`search/best_of_n.py` and registered in `fasttts.py:_SEARCH_STRATEGIES`,
but the experiment sweep (`run_all_experiments.py`) hardcoded
`METHODS = ["fasttts", "baseline"]` — both beam_search variants — so BoN
had never been run through the pipeline. This section adds BoN as a peer
strategy so the thesis's end-to-end story is not restricted to
beam-search-family methods.

### Which FastTTS optimizations apply to BoN

Three optimizations live under the existing `fasttts/` configs. Their
surface for BoN:

1. **SBE** (`enable_spec_diff`) — two mechanisms: (a) strip
   `SamplingParams.stop` before submit so the engine ignores `\n\n`,
   and (b) Phase 1 / Phase 2 queue-pressure logic across a multi-iteration
   loop. BoN passes no `stop` strings (`best_of_n.py:44-50`) and does not
   use the iterative `SearchState` / `_filter_active` loop —
   **no-op for BoN.**

2. **Prefix-aware scheduling** (`prefix_aware_scheduling`) — has exactly
   one effect: `_duplicate_beams` in `search/common.py:346-370` reorders
   duplicate beams so they sit adjacent to their parent. BoN never calls
   `_duplicate_beams` — **no-op for BoN.**

3. **Asymmetric GPU memory split** — yaml-only. The fasttts configs give
   the generator 0.74 and the verifier 0.16 (vs. baseline's 0.68/0.22 on
   7B). **This IS meaningful for BoN:** the single big `generate(n=N)`
   call holds N parallel KV caches in the generator, while the verifier
   runs exactly once at the end over N short scoring inputs. BoN benefits
   at least as much as beam_search from the generator-heavy allocation.

`best_of_n.py` now carries a comment near the `SamplingParams`
construction documenting the no-op relationship for SBE and
prefix-aware scheduling.

### Config tree restructure (flat → two-axis)

Old layout conflated search strategy with optimization:

```
configs/{gen}/{dataset}/{baseline,fasttts,baseline_kvoff,fasttts_kvoff}/n{N}.yaml
```

`baseline` and `fasttts` were implicitly beam_search. Adding BoN
without restructuring would leave the naming inconsistent. New layout:

```
configs/{gen}/{dataset}/{search_strategy}/{optimization}/n{N}.yaml
```

- All existing 56 yamls moved under `configs/{gen}/{dataset}/beam_search/`
  (including the transient `baseline_kvoff`, `fasttts_kvoff` variants).
  `output_dir:` lines updated to mirror the new path.
- 20 new `configs/{gen}/{dataset}/best_of_n/fasttts/n{1,4,16,64,256}.yaml`
  generated from the corresponding `beam_search/fasttts` yamls via sed:
  `approach: beam_search → best_of_n`, `output_dir` substitution, drop
  `beam_width` and `num_iterations` (BoN ignores them), flip
  `enable_spec_diff: true → false` and `prefix_aware_scheduling: true → false`
  (no-ops; keeping them false avoids installing `SBETracker` and the
  `make_stats` scheduler hook so the A/B against beam_search is clean).
  Asymmetric memory split preserved verbatim.

**`(best_of_n, baseline)` is intentionally omitted from the grid.** For
BoN the only difference between a baseline and a fasttts yaml is the
generator/verifier memory split (SBE and prefix-aware scheduling are
no-ops). Running both duplicates effort without producing a new data
point; a single BoN row is sufficient. Revisit only if 7B / N=256 OOMs
and the baseline (less generator headroom) behaves differently.

### Experiment runner refactor (`run_all_experiments.py`)

`METHODS` replaced with an explicit `COMBO_ORDER_KEYS` list of
`(search_strategy, optimization)` tuples. The sweep runs two combos:
`(beam_search, fasttts)` and `(best_of_n, fasttts)`. Planned-run count
is `len(GENERATORS) × len(DATASETS) × 2 × len(N_VALUES)` — with the
current `GENERATORS = ["7B-instruct"]` and two datasets, that is
**20 runs**.

`(beam_search, baseline)` is intentionally omitted from the sweep. It
is the un-optimized reference for full fasttts — comparing against it
would just duplicate the thesis's own internal SBE / prefix-aware
ablations. The primary thesis comparison is full-fasttts beam search vs.
BoN (both with the asymmetric memory split). The `beam_search/baseline`
yamls remain on disk so the ablation can be re-run on demand, but are
not part of the headline sweep.

Display labels: `COMBO_DISPLAY_MAP` maps the two combos to `"Beam Search"`
and `"BoN"`. Both runs apply the fasttts optimization stack, so the axis
that actually differs between the two plotted lines is the **search
strategy**, not the optimization — labels name the strategy accordingly.

Helper changes:

- `_combo_key(strategy, opt) → "{strategy}/{optimization}"` — JSON-friendly
  key used throughout `collect_results`, `main_results.json`, and
  `accuracy.json`.
- `_planned_runs()` yields 5-tuples `(gen, dataset, strategy, optimization, n)`.
- `run_experiments()` builds config path from the 5-tuple; `label` and
  `output_dir` mirror the new shape.
- `collect_results()` and `evaluate_accuracy()` walk
  `{data_dir}/{gen}/{dataset}/{strategy}/{optimization}/`.
- `_evaluate_combo()` replaces `_evaluate_method()`. The result-file glob
  widens `_iter10` → `_iter*` so any `num_iterations` value matches. Both
  strategies currently use `num_iterations=40`: BoN's single-call
  semantics make it a no-op (the loop exits after iteration 1), and
  beam_search was raised from 10 to 40 after a wall-clock measurement
  showed iter=40 is actually ~8% faster on 7B AIME — the earlier
  last-iteration free-run in iter=10 triggers expensive PRM-unguided tail
  generation on beams that would otherwise have been pruned. Keeping the
  glob pattern iteration-agnostic means future tuning of this knob won't
  require runner changes. The `_specdiff` suffix is only applied when
  SBE is actually active (`strategy == "beam_search" and optimization == "fasttts"`);
  `(best_of_n, fasttts)` keeps `enable_spec_diff: false`, so its filenames
  carry no suffix.

Plotting generalizes over `COMBO_ORDER`:

- `plot_goodput` rebuilds `color_palette` from
  `{label: sns.color_palette()[i] for i, label in enumerate(COMBO_ORDER)}`
  and loops over `COMBO_ORDER` for legend patches.
- `plot_latency` computes `k = len(COMBO_ORDER)` and lays out `2*k` bars
  per N-group: `k` generator bars, then `k` verifier bars (twin-colored
  via `_lighten` + hatched). `offsets = np.linspace(-(2*k-1)/2, (2*k-1)/2, 2*k)`
  replaces the hardcoded 4-bar layout.
- `plot_accuracy` replaces `_ACCURACY_METHOD_STYLES` with
  `_ACCURACY_COMBO_STYLES` keyed on `(strategy, optimization)`. Active
  styles: `(beam_search, fasttts)` solid+filled, `(best_of_n, fasttts)`
  dash-dot+filled. The `(beam_search, baseline)` style (dashed+hollow) is
  retained in the dict so an ad-hoc re-run of the baseline ablation still
  plots correctly without further code changes.

### Temperature alignment

In the same pass, all yamls were normalized to `temperature: 0.7` to match
the reference framework's BoN default (see Section 13 for the rationale).

### Files touched

- `search/best_of_n.py` — SBE + prefix-aware no-op rationale comment.
- `run_all_experiments.py` — two-axis sweep (full rewrite of runner,
  collector, evaluator, plotting helpers).
- `accuracy_evaluation/evaluation/evaluate.py` — `ProblemMetrics` enriched
  with `reference_answer`, `selected_answer`, `selected_idx`;
  `_select_prm_vote` returns `(correct, idx_in_valid, selected_key)`.
- `benchmarks/configs/` — 56 yamls moved under `beam_search/`, 20 new
  yamls created under `best_of_n/fasttts/`, all 76 normalized to T=0.7.
