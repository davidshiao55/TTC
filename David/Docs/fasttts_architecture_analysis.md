# FastTTS Architecture Analysis

## Overview

FastTTS is a two-model test-time search framework for math reasoning:

- **Generator** (`Qwen/Qwen2.5-Math-1.5B-Instruct`) — produces candidate reasoning steps
- **Verifier / PRM** (`Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B`) — scores each step with a Process Reward Model

The search loop generates one step at a time, has the PRM score it, keeps the top-k beams, and repeats. The stop string `\n\n` marks the boundary between reasoning steps.

---

## Core Search Loop (Beam Search)

**File:** `search/beam_search.py` → `_beam_search()`

```
for each iteration k:
    1. Duplicate survivors back to n beams
    2. Build prompts: [system_prompt + question + step_1 + ... + step_k]
    3. Generator extends each beam by one step (stop at \n\n)
    4. Verifier scores all beams
    5. Prune to top (n // beam_width) beams
    6. Repeat
```

**Key config defaults** (`config.py`):
| Parameter | Generator | Verifier |
|---|---|---|
| `gpu_memory_utilization` | 0.45 | 0.45 |
| `enable_prefix_caching` | True | True |
| `max_model_len` | 4096 | 4096 |
| `beam_width` | — | 2 |
| `n` (total beams) | — | 8 |

Beam prompts are built via `build_conversation()` (`search/utils.py`):

```python
conversation = [
    {"role": "system",    "content": system_prompt},
    {"role": "user",      "content": question},
    {"role": "assistant", "content": steps_so_far},  # omitted at step 0
]
```

---

## Three FastTTS Optimizations

### Optimization 1 — Speculative Beam Extension (SBE)

**Problem:** In standard beam search, generating step k+1 cannot begin until the verifier has scored step k and the pruning decision is made. The generator is idle during verifier scoring.

**Solution:** Overlap generator and verifier work. While the verifier scores step k, the generator keeps running past the `\n\n` stop boundary, producing a speculative head start into step k+1. When the verifier finishes and pruning is done, surviving beams already have partial step k+1 tokens in context — the generator resumes mid-step rather than from scratch.

**Paper claim location:** §6.3, §6.5.1 (ablation, blue bars in Fig. 16). The speedup comes from amortising the generator's idle time during verifier scoring.

---

#### Layer 1 — `SpecStopChecker`: deferred stop

**File:** `models/spec_stopchecker.py`

Normal vLLM `StopChecker.maybe_stop_sequence()` on a stop string match: truncates output to before the stop string, sets `seq.status = FINISHED_STOPPED`. The sequence then exits the scheduler's running queue.

`SpecStopChecker` overrides this with a two-phase dispatch, keyed on `len(self.scheduler.waiting)`:

- **Phase 1** (`waiting` non-empty): beams are still being submitted as prefill requests. Normal stop applies — sequences exit the batch as usual.
- **Phase 2** (`waiting` empty): all beams are in-flight. Stop is deferred:

| Trigger | Normal `StopChecker` | `SpecStopChecker` (Phase 2) |
|---|---|---|
| EOS token | truncate + `FINISHED_STOPPED` | **same** — EOS is always a hard stop |
| Stop token | truncate + `FINISHED_STOPPED` | truncate, `seq.stop_reason = token_id`, status stays `RUNNING` |
| Stop string (`\n\n`) | truncate + `FINISHED_STOPPED` | `seq.stop_reason = stop_str`, **no truncation**, status stays `RUNNING` |
| Max length | `FINISHED_LENGTH_CAPPED` | `FINISHED_LENGTH_CAPPED` |

In `SingleStepOutputProcessor._process_sequence_group_outputs()` (original vLLM 0.9.2):
```python
self.stop_checker.maybe_stop_sequence(seq, new_char_count, sampling_params, ...)
if seq.is_finished():
    scheduler.free_seq(seq)
```
A deferred-stop sequence has `status = RUNNING` → `is_finished() = False` → NOT freed → stays in the running queue → keeps generating past `\n\n`.

The **no truncation** on stop string is intentional: the tokens after `\n\n` are the speculative head start and must be preserved in the sequence's output.

`SpecStopChecker` holds a live reference to `self.scheduler[0]` passed at construction time, which is how it checks the waiting queue state on every token step.

**Injection** (`generator_engine.py:99-129`, `enable_spec_beam_extension()`):
```python
self.output_processor = SingleStepOutputProcessor(
    ...
    stop_checker=SpecStopChecker(self.scheduler_config.max_model_len,
                                  get_tokenizer_for_seq,
                                  self.scheduler[0]),)
```
This replaces the engine's output processor in-place. The `scheduler_config.policy` is also set to `"priority"` here so the priority-based ordering in `_process_model_outputs_spec` takes effect.

The helper predicate used throughout:
```python
def is_finished_stopped_with_stop(seq: Sequence) -> bool:
    return not seq.is_finished() and seq.stop_reason is not None
```
Identifies sequences that have hit `\n\n` (or a stop token) but are still physically running.

---

#### Layer 2 — `_process_model_outputs_spec`: tracking, flushing, and the memory valve

**File:** `models/generator_engine.py`

`_process_model_outputs` dispatches here only when `spec_beam_extension_enabled=True` AND the current batch has `sampling_params.stop` set (the generator call with `\n\n` stopping). Verifier calls (no stop params) go through the normal path.

Per sequence group, after `process_outputs()` runs `SpecStopChecker`:

1. **Priority assignment** (line 237-238): any newly deferred-stop sequence gets `seq_group.priority = SPEC_BEAM_CANDIDATE_PRIORITY`. This high priority ensures it is scheduled ahead of regular beams by the priority scheduler in subsequent steps — speculative candidates get GPU time to keep building their head start.

2. **`all_finished` accumulator** (line 239):
   ```python
   all_finished &= is_finished_stopped_with_stop(seq) or seq.is_finished()
   ```
   `True` iff every sequence in the batch is either deferred-stop or truly finished. This is the termination signal for the SBE generation phase.

3. **Memory pressure valve** (lines 255-263): if `len(spec_beam_extension_now) + len(running_now) > 256`, forcibly finish deferred-stop sequences (FIFO) until the batch is within limit. Prevents unbounded batch growth when many beams converge on `\n\n` simultaneously.

4. **`all_finished` flush** (lines 265-275): when every sequence is deferred-stop, force-set `status = FINISHED_STOPPED` and `free_seq()` for all. This releases the speculative sequences from the scheduler and returns their outputs (including the post-`\n\n` tokens) to the caller.

State transitions:
```
[normal running]
    → generate token → hit \n\n
    → SpecStopChecker: stop_reason = "\n\n", status = RUNNING (deferred)
    → _process_model_outputs_spec: priority = SPEC_BEAM_CANDIDATE_PRIORITY
    → stays in running queue, keeps generating

[all sequences deferred-stop]
    → all_finished = True
    → flush: status = FINISHED_STOPPED, free_seq() for all
    → RequestOutput with full post-\n\n text returned to caller
```

---

#### Layer 3 — `split_string_by_separator` and `future_texts`

**File:** `search/beam_search.py` (lines 374-379), `search/utils.py` (lines 21-48)

After `generate_beam()` returns, the raw output `gen_result.next_texts[0]` contains the completed step plus everything the model generated after `\n\n` (the speculative tokens). For example:

```
"To solve this, multiply both sides by x.\n\nNext, factor the result."
                                           ^^^^
                                           SpecStopChecker kept going past here
```

`split_string_by_separator(text, "\n\n")` splits this into:
- `current_text` = `"To solve this, multiply both sides by x.\n\n"` — the confirmed step
- `future_texts` = `[("Next, factor the result.", False)]` — stored on the beam

Each `future_texts` entry is a `(text, is_finished_this_step)` tuple:
- `is_finished_this_step = True`: this chunk itself ends with `\n\n` — a complete additional step that can be consumed directly without a generator call
- `is_finished_this_step = False`: a partial step (no trailing `\n\n`) — gives a head start but the generator still needs to complete this step

If the model ran long enough to produce multiple complete steps before being flushed, `future_texts` can contain several `(step, True)` entries followed by one `(partial, False)`.

---

#### Layer 4 — Verifier call: scoring confirmed vs. speculative text

**File:** `search/beam_search.py` (lines 397-404)

The verifier is always called with `current_text` (confirmed steps), not speculative tokens. However, there is one exception: if `future_texts[0][1] = True` (the first queued future step is a complete step), the verifier scores `current_text + future_texts[0][0]`. This scores a step that hasn't been "officially" confirmed yet, allowing the pruning decision to incorporate the already-generated next step.

Beams that already have a score for this iteration (from a previous SBE hit) skip the verifier call entirely — their stored `all_scores` are reused.

This matches the paper's description: "Verifier evaluates `B`" (all beams), not just speculative candidates `sp`.

---

#### Layer 5 — `DuplicateThenTruncate`: creating diversity in duplicates

**File:** `search/beam_search.py` (lines 256-284)

When survivors (n // beam_width beams) are expanded back to n, each survivor produces `repeats` copies. The **original** beam keeps its `future_texts` intact. Each **duplicate** has `truncate_sentence_by_tokens` applied to `future_texts[-1][0]` (the last, partial speculative chunk):

```python
truncate_sentence_by_tokens(text, tokenizer, mean_ratio=0.85, std_ratio=0.1)
# draws ratio ~ N(0.85, 0.1), clipped to [0, 1]
# returns the first (ratio × total_tokens) tokens of the text
```

The duplicate's head start is truncated to ~85% of the original's length (with random variation). This forces divergence: if original and duplicate had identical prompts (including identical speculative tokens), they would generate identical next steps. The truncation gives them different starting points, so their generations diverge — maintaining beam diversity.

Only `future_texts[-1]` (the partial chunk at the end) is truncated. Complete steps in earlier `future_texts` entries (where `is_finished_this_step=True`) are preserved — confirmed steps need no diversity manipulation.

---

#### Layer 6 — Head start consumption

**File:** `search/beam_search.py` (lines 291-315)

At the top of each iteration, before building generator prompts:

```python
for beam in active_beams:
    if beam.future_texts:
        next_text, is_finished_this_step = beam.future_texts[0]
        if is_finished_this_step:
            beam.skipped_this_step = True   # full step available, skip generator
        else:
            beam.current_text += next_text  # pre-pend partial head start
            beam.future_texts.pop(0)
```

Beams with `skipped_this_step=True` are excluded from the generator call entirely — their next step came from the previous SBE run for free. Beams with a partial head start call the generator but with a longer `current_text` — the model continues from mid-step rather than from the beginning of a new step.

Beams that are skipped still go through the verifier path (line 354-363): they pop the next `future_texts` entry, append it to `current_text`, and participate in scoring normally.

---

#### Full SBE data flow (one iteration cycle)

```
Iteration k, survivors = n // beam_width beams:

1. DuplicateThenTruncate:
   n // bw survivors → n beams
   Original: future_texts intact
   Duplicate: future_texts[-1] truncated to ~N(0.85, 0.1) of original length

2. Head start consumption:
   Partial head start pre-pended to beam.current_text
   → generator prompt starts mid-step k+1

3. Generator call (non-skipped beams):
   SpecStopChecker (Phase 2) defers stop at \n\n, keeps generating
   Returns: full step k+1 + \n\n + partial step k+2

4. split_string_by_separator:
   current_text ← step k+1 (appended to beam.current_text)
   future_texts ← [(partial_step_k+2, False)] (stored for next iteration)

5. Verifier: scores all n beams on current_text
   Beams with a complete future step: scored on current_text + future_texts[0][0]

6. Prune to top n // beam_width survivors by verifier score
   → goto 1, next iteration starts with speculative head start already loaded
```

The core savings: step k+1 tokens are generated while the verifier scores step k (overlapped work), and those tokens are already in context when step k+2's generator call begins — a free head start worth ~R=0.85 of a full step's tokens.

---

### Optimization 2 — Dynamic Prefix-Aware Scheduling

**Problem:** When n beams are submitted as a batch, they share varying amounts of prefix. With random scheduling order, a beam's KV blocks may be evicted before its duplicate is processed, forcing full prefix recomputation.

**Paper's solution:** Order request scheduling so that beams sharing the longest prefix are served consecutively, keeping shared KV blocks hot in cache throughout the group. The paper evaluates this in terms of **KV cache memory efficiency** (Fig. 18) — smaller peak KV footprint across the batch — with goodput gain as a downstream effect (Fig. 16, most significant in memory-constrained configs like 1.5B+7B).

**Paper claim location:** §6.5.1 (ablation, green bars in Fig. 16) and §6.5.3 (Fig. 18, compared against Random and Worst-Case scheduling baselines).

#### What is actually implemented in the open-source code

There are two distinct parts in the code:

**Part 1 — Adjacent beam placement** (`beam_search.py`, duplication logic):

When `prefix_aware_scheduling=True`, after pruning survivors are expanded back to n by placing each beam and all its copies adjacent in the list:
```
# prefix_aware_scheduling=False (original): [A, B, C, D, A', B', C', D']
# prefix_aware_scheduling=True:             [A, A', B, B', C, C', D, D']
```
A and its copy A' are now consecutive → KV blocks for A's full history are guaranteed hot when A' is processed. This is an O(n) rearrangement with zero tokenization overhead.

**Part 2 — Priority assignment** (`search/utils.py` → `assign_prefix_priorities`):

A general O(n²×L) algorithm that finds the largest group of sequences sharing the longest common prefix, assigns them a priority tier, then repeats for remaining sequences. This handles partial prefix sharing between non-identical sibling beams — the "dynamic" part of the name.

**This part is commented out in the upstream repo and never runs.** Both call sites in `beam_search.py` have the priority assignment code commented out. This is upstream dead code from the original FastTTS authors, not disabled by us.

#### Benchmark configs

The experiment runner (`run_all_experiments.py`) only tests `["baseline", "spec_prefix"]` — both SBE and prefix-aware scheduling are always bundled together in the published results. A standalone `prefix/` config folder exists (used for the paper's ablation) but is not part of the main experiment script.

The `prefix/` configs use deliberately low `gpu_memory_utilization: 0.20` to amplify KV eviction pressure, making the scheduling effect measurable.

#### What `prefix_aware_scheduling=True` actually enables

Only adjacent beam placement. The priority-based scheduling (which the paper's Fig. 18 evaluates) is not active. When documenting experimental setup, note that only adjacent placement is used, not the full priority-based variant.

---

### Optimization 3 — Asymmetric Multi-Model Memory Allocation

**Problem:** Generator and verifier have different memory demands depending on model sizes and workload phases. Giving each a fixed, equal share of GPU memory is suboptimal — the bottleneck model is starved while the other has idle headroom.

**Solution:** Dynamically balance GPU memory between generator and verifier rather than splitting it statically. 

#### Subsection: Offloading (sleep/wake)

The optimization space can be extended with an offloading strategy for cases where GPU memory 𝑀 is extremely constrained.
**Implementation:** `vllm/device_allocator/cumem.py` — `CuMemAllocator` (singleton)

Uses CUDA virtual memory API (`cuMemCreate`, `cuMemMap`, `cuMemUnmap`, `cuMemRelease`) to manage all GPU allocations by tag ("weights", "kv_cache"):

- `sleep(level=1)` in `vllm/v1/worker/gpu_worker.py`:
  - Weights → `cudaMemcpy` GPU→CPU pinned RAM, then `cuMemUnmap + cuMemRelease` (physical GPU pages freed)
  - KV cache → physical pages released with NO CPU backup (discarded — stale across model alternation anyway)
  - Virtual address space preserved; on `wake_up()`, new physical pages remapped to same addresses and weights copied back H2D
- `sleep(level=2)`: discards everything, saves nothing

**Initialization sequence:**
```
Generator inits (0.45 GPU) → sleeps → signals ready
Verifier inits (0.45 GPU) → sleeps → signals ready
Search loop: wake generator → generate → sleep generator → wake verifier → score → sleep verifier → …
```

Combined `gpu_memory_utilization` can exceed 100% (e.g. 60% + 60% = 120%) because only one is awake at a time.

**Note — paper vs code discrepancy:** The paper describes "KV cache offloaded to CPU"; the actual code offloads *weights* and discards KV cache. Effect on available memory is identical.

**V1 CPU KV offload** (`vllm/v1/kv_offload/`) is a separate, independent vLLM feature — not used by FastTTS by default.

---

## Prefix Cache Behavior Analysis

**`enable_prefix_caching: True`** is set for both models. This uses a radix tree to deduplicate identical prefix token blocks.

### Within a single generator call (strong hits)

After pruning to `n // beam_width` survivors and duplicating back to `n`, pairs of identical beams are submitted simultaneously:
- Each duplicate pair has **100% identical prompts** → guaranteed prefix cache hit for the duplicate
- Sibling beams (different survivors) share `[system_prompt + question]` → partial hit

### Cross-iteration without sleep (KV stays GPU-resident)

When sleep mode is OFF, KV cache blocks from iteration `k` are not freed immediately. With `enable_prefix_caching`, finished-request blocks are returned to the radix tree as *evictable* (cached) blocks and remain on GPU until evicted by memory pressure.

Iteration `k+1` sends prompts that are exactly `[previous_k_steps + new_step]`. vLLM finds the prefix in the radix tree, marks those blocks in-use, and **skips prefill entirely for the matched tokens**. Only the new step tokens are prefilled.

This compounds with depth: by iteration 20, only 1 step's worth of tokens is prefilled per call, regardless of total context length.

### Cross-iteration with V1 KV offload enabled

When GPU memory is tight, the radix tree's evictable blocks can be moved to CPU RAM instead of discarded. On a prefix hit for a CPU-resident block, vLLM triggers an H2D transfer before decode.

Cost comparison for a 4096-token prefix (Qwen-1.5B, BF16):
- Recompute: full prefill (slow, quadratic-ish)
- CPU-resident hit: ~512 MB × PCIe H2D @ ~32 GB/s ≈ **~16 ms transfer** (much faster than recompute)
- GPU-resident hit: **~0 ms** (just attend to cached KV)

KV offload trades PCIe bandwidth for avoiding recompute — worthwhile for long contexts.

### Cross-iteration with sleep mode

Sleep discards GPU KV cache (physical pages freed). Cross-iteration KV reuse is **impossible** — cold cache on every wake_up, even with `enable_prefix_caching`. Within-call duplicate hits still work (they're within the same wake period).

### Summary table

| Scenario | Cross-iter prefix hit? | Cost |
|---|---|---|
| No sleep, GPU-resident block | Yes | 0 — KV already on GPU |
| No sleep, CPU-resident block (KV offload) | Yes | PCIe H2D transfer |
| Sleep mode (any) | No | Full prefill recompute |
| Duplicate beams, same wake period | Always | 0 |
| Sibling beams, same wake period | Partial | New-step tokens only |

---

## Answer Selection and Accuracy Metrics

### How FastTTS selects the final answer

The evaluation pipeline (`accuracy_evaluation/evaluation/evaluate.py`) uses **PRM-weighted majority voting**:

```python
scores = [np.prod(score) for score in sample['solutions']['scores'][0]]
max_indices = np.argsort(scores)[-top_n:]
completions = [sample['solutions']['completions'][0][i] for i in max_indices]
sample['pred'] = [extract_answer(completion, data_name) for completion in completions]
sample['pred'] = [max(sample['pred'], key=lambda x: sample['pred'].count(x))]
```

1. Rank all completed beams by **product of per-step PRM scores**
2. Select the top `top_n` beams
3. Extract the `\boxed{}` answer from each
4. **Majority vote** among those extracted answers

The `top_n` parameter controls the balance between verifier trust and voting:
- `top_n=1`: Pure verifier selection (highest PRM-product beam wins)
- `top_n=N`: Majority vote among top-N PRM-ranked beams
- `top_n=all`: Full majority voting across all beams

### How accuracy is reported (and the problems with it)

**Problem 1 — Test-set hyperparameter tuning.** The original FastTTS codebase (`run_all_experiments.py`) **sweeps** `top_n` over all N values (8, 16, 32, ..., 512) and **reports the maximum accuracy** achieved across the sweep. This is hyperparameter tuning on the test set — the reported number is optimistic.

**Problem 2 — Completion count exceeds n.** FastTTS beam search runs for a fixed `num_iterations` and does **not** abort when n completions are reached. The `_check_n_completion` function (and its AE equivalent at `beam_search.py:249`) only **records a latency timestamp** — it does not stop the search or truncate results. The search continues, accumulating completions beyond n. In practice, n=8 produces ~11 completed beams.

Neither the search loop nor the evaluation pipeline truncates to n. All completions (>n) are passed to `evaluate.py`.

**Problem 3 — Inconsistent latency/accuracy sets.** FastTTS reports:
- **Latency**: time to reach n completions (`n_gen_latency` snapshot at the iteration where `completed >= n`)
- **Accuracy**: computed over **all** completions (which is > n)

This makes the system look better than it is — fast latency (measured at n) paired with high accuracy (benefiting from extra completions beyond n). A fair report would use the same set of completions for both metrics.

**Comparison: Liu et al. (compute-optimal-tts) aborts at n.** In contrast, Liu et al.'s beam search (`tree.py:289`) terminates immediately when n completions are reached:
```python
if len(end_nodes) == beam_size:
    break
```
They also dynamically reduce expansion width as solutions accumulate (`k = beam_size - len(end_nodes)`). This guarantees exactly n completed solutions, making their pass@N clean.

### Comparison with other TTC frameworks

| Framework | Answer selection | Termination | Reporting |
|---|---|---|---|
| **FastTTS** | PRM-product rank → top_n → majority vote | Fixed iterations (no abort at n) | Sweep top_n, report max (optimistic) |
| **compute-optimal-tts** (Liu et al.) | 7 methods independently | Abort at n completions | Report each method separately (transparent) |
| **search-and-learn** (Beeching et al.) | 3 methods at multiple N | Fixed iterations | Report all at each N (transparent) |

### Metric definitions

- **Top-1 accuracy**: A single answer is selected from N beams (via voting or verifier ranking). Check if it matches ground truth. This is the deployed-system metric.
- **Pass@N accuracy**: Does *at least one* of the N beams contain the correct answer? Measures search coverage regardless of selection quality. Always >= Top-1.
- **Gap (Pass@N − Top-1)**: How much accuracy the verifier/voting leaves on the table. Correct answers exist but aren't selected.

### PRM-Vote vs PRM-Max for answer selection (Liu et al. findings)

Liu et al. ("Can 1B Surpass 405B") explicitly benchmark voting methods for search-based TTS results (Table 2, MATH-500):

| Method | Skywork-PRM-7B | Qwen2.5-Math-PRM-7B |
|---|---|---|
| Majority Vote | 86.8 | 87.6 |
| PRM-Min-Max | 83.0 | 87.4 |
| PRM-Min-Vote | 86.6 | 87.6 |
| PRM-Last-Max | 84.4 | 87.6 |
| **PRM-Last-Vote** | **87.0** | 87.6 |
| PRM-Avg-Max | 85.8 | **87.8** |
| PRM-Avg-Vote | 86.8 | 87.6 |

Key findings:
- **Vote variants consistently match or beat Max variants** — using PRM to weight a majority vote is better than trusting PRM to pick the single best beam
- **Skywork-PRM is sensitive to voting method**: PRM-Last-Vote (87.0) beats PRM-Last-Max (84.4) by 2.6 points. This is our PRM — so PRM-Max (our `top_n=1`) would underperform
- **Qwen2.5-Math-PRM is insensitive**: all methods give ~87.6 because it's well-calibrated (trained with LLM-as-judge data cleaning)
- The gap between PRM-Vote and PRM-Max reflects PRM calibration quality. Poorly calibrated PRMs benefit more from voting as a correction mechanism