#!/usr/bin/env python
# Shared infrastructure for all iterative search strategies.
# Extracted from beam_search.py to eliminate code duplication across
# dvts, dynamic_branching, vg_search, and beam_search.

import copy
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
from tqdm import tqdm
from vllm import SamplingParams
import torch.cuda.nvtx as nvtx

from config import SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.beam import Beam, StepChunk, _next_beam_id, reset_beam_id_counter, step_hash
from search.results import SearchResults
from search.utils import build_conversation, aggregate_scores, truncate_sentence_by_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parse raw generator output into StepChunks
# ---------------------------------------------------------------------------

def parse_generation_into_chunks(
    gen: Beam,
    stop: str,
    prompt_token_len: int,
    max_model_len: int = 4096,
) -> List[StepChunk]:
    """Split a raw generator output into ordered StepChunks.

    At most one chunk — always the last — carries terminal=True, and
    only when the generation genuinely terminated the beam. Length-cap
    recovery is encoded directly in the terminal_gen waterfall below.

    Decision table (see David/Docs plan file for full context):

      1. token_length >= max_model_len        → terminal (context exhausted)
      2. stop_reason == stop and text != ""   → not terminal (clean boundary
                                                or SBE force-finish)
      3. finish_reason == "length" and
         text contains at least one `stop`    → not terminal (length-cap
                                                recovery: budget shared
                                                across multiple steps)
      4. otherwise                            → terminal (EOS, single
                                                mega-step length cap,
                                                empty text, other)
    """
    text = gen.gen_text[0]
    stop_reason = gen.stop_reasons[0] if gen.stop_reasons else None
    finish_reason = gen.finish_reasons[0] if gen.finish_reasons else None
    token_length = prompt_token_len + gen.step_tokens

    if token_length >= max_model_len:
        terminal_gen = True
    elif stop_reason == stop and text != "":
        terminal_gen = False
    elif finish_reason == "length" and text.count(stop) >= 1:
        terminal_gen = False
    else:
        terminal_gen = True

    parts = text.split(stop)
    chunks: List[StepChunk] = []
    for p in parts[:-1]:
        chunks.append(StepChunk(text=p + stop, is_complete_step=True, terminal=False))
    if parts[-1] != "":
        chunks.append(StepChunk(text=parts[-1], is_complete_step=False, terminal=False))

    if not chunks:
        return [StepChunk(text="", is_complete_step=False, terminal=terminal_gen)]

    if terminal_gen:
        chunks[-1].terminal = True
    return chunks


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def score_beam(
    verifier: VerifierVLLMModelWrapper,
    prompts: List[str],
    completions: List[List[str]],
    *,
    skip_reading_prefix_cache: bool = False,
):
    """Score a beam of completions.

    With the default ``skip_reading_prefix_cache=False``, shared prefixes
    hit the PRM KV cache and those step boundaries come back as ``None``
    — callers must run step-hash propagation. Pass ``True`` for single-shot
    callers that don't propagate.
    """
    verifier_time = time.time()
    scores = verifier.score(
        prompts, completions,
        skip_reading_prefix_cache=skip_reading_prefix_cache,
    )
    verifier_time = time.time() - verifier_time
    return scores, verifier_time


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate_beam(
    templated_convs: List[str],
    generator: GeneratorVLLMModelWrapper,
    sampling_params: SamplingParams,
    beam_width: int = 1,
    tokenizer=None,
):
    """Generate one step of tokens for a batch of beams."""

    num_convs = len(templated_convs)
    gen_results = [
        {
            "index": i,
            "initial_prompt": text,
            "text": "",
            "stop_reason": None,
            "finish_reason": None,
            "completion_tokens": 0,
            "prompt_token_len": 0,
        }
        for i, text in enumerate(templated_convs)
        for _ in range(beam_width)
    ]

    gen_prompts = [r["initial_prompt"] for r in gen_results]

    start_time = time.time()
    nvtx.range_push("generate")
    llm_outputs = generator.generate(gen_prompts, sampling_params=sampling_params)
    nvtx.range_pop()
    end_time = time.time()

    for gen_result, output in zip(gen_results, llm_outputs):
        gen_text = output.outputs[0].text
        out_tokens = len(output.outputs[0].token_ids)
        gen_result["text"] = gen_text
        gen_result["stop_reason"] = output.outputs[0].stop_reason
        gen_result["finish_reason"] = output.outputs[0].finish_reason
        gen_result["completion_tokens"] = out_tokens
        gen_result["prompt_token_len"] = len(output.prompt_token_ids)

    # Convert to beam format
    outputs = []
    counter = 0
    for i in range(num_convs):
        texts = []
        stop_reasons = []
        finish_reasons = []
        completion_tokens = []
        prompt_token_lens = []
        for _ in range(beam_width):
            gen_result = gen_results[counter]
            texts.append(gen_result["text"])
            stop_reasons.append(gen_result["stop_reason"])
            finish_reasons.append(gen_result["finish_reason"])
            completion_tokens.append(gen_result["completion_tokens"])
            prompt_token_lens.append(gen_result["prompt_token_len"])
            counter += 1

        beam_result = Beam(
            prompt=templated_convs[i],
            current_text="",
            gen_text=texts,
            stop_reasons=stop_reasons,
            finish_reasons=finish_reasons,
            prompt_token_lens=prompt_token_lens,
            scores=[],
            pruned=False,
            gen_history=[],
            step_tokens=completion_tokens[0],
        )
        outputs.append(beam_result)

    return outputs, end_time - start_time


# ---------------------------------------------------------------------------
# Dataclasses for decomposed search state
# ---------------------------------------------------------------------------

@dataclass
class SearchState:
    """Mutable state shared across phases within one iterative search."""
    # Beams
    all_beams: List[Beam]
    active_beams: List[Beam]
    completed_beams: List[Beam]

    # Iteration control
    search_config: SearchConfig
    generator_max_model_len: int = 4096
    iteration: int = 0
    is_last: bool = False

    # Config-derived (computed once at init)
    system_prompt: str = ""
    base_sampling_params: SamplingParams = None
    final_sampling_params: SamplingParams = None

    # Metrics accumulators
    total_gen_latency: float = 0.0
    total_ver_latency: float = 0.0
    total_tokens: int = 0

    # Per-iteration transients (reset each iteration)
    gen_results: List[Beam] = field(default_factory=list)
    agg_scores: List[float] = field(default_factory=list)

    # Per-iteration counters (for logging)
    skipped_beam_count: int = 0
    extended_beam_count: int = 0

    # Persistent map of step_hash -> score, survives pruning
    step_hash_bank: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScoringBatch:
    """Inputs for the verifier scoring call."""
    prompts: List[str]
    completions: List[List[str]]


# ---------------------------------------------------------------------------
# Phase functions
# ---------------------------------------------------------------------------

def _init_state(
    batch_of_prompts: List[str],
    search_config: SearchConfig,
    tokenizer,
) -> SearchState:
    """Initialize search state: create beams, sampling params, compute prompt length."""
    reset_beam_id_counter()

    base_sampling_params = SamplingParams(
        temperature=search_config.temperature,
        max_tokens=search_config.max_tokens,
        top_p=search_config.top_p,
        stop=[search_config.stop],
        include_stop_str_in_output=True,
        n=1,
    )
    final_sampling_params = SamplingParams(
        temperature=search_config.temperature,
        max_tokens=search_config.max_tokens,
        top_p=search_config.top_p,
        n=1,
    )

    beams: List[Beam] = []
    for prompt in batch_of_prompts:
        for idx in range(search_config.n):
            beams.append(Beam(
                prompt=prompt,
                current_text="",
                pruned=False,
                completed=False,
                stop_reasons=None,
                scores=[],
                gen_history=[],
                step_tokens=0,
                total_tokens_generated=0,
                time_to_complete=0.0,
                beam_id=_next_beam_id(),
                parent_id=None,
                born_at_iteration=0,
            ))

    return SearchState(
        all_beams=beams,
        active_beams=[],
        completed_beams=[],
        search_config=search_config,
        system_prompt=search_config.system_prompt,
        base_sampling_params=base_sampling_params,
        final_sampling_params=final_sampling_params,
    )


def _filter_active(state: SearchState):
    """Filter out pruned beams."""
    if state.iteration == 0:
        state.active_beams = [b for b in state.all_beams if not b.pruned]
    else:
        state.active_beams = [b for b in state.active_beams if not b.pruned]

    # Under SBE, sort active beams low→high by aggregate score so low-score
    # beams are submitted to the engine first. They drain during Phase 1
    # (no speculation) while the waiting queue is non-empty; high-score
    # beams are admitted last and hit their stops after the queue empties,
    # making them the ones that speculate in Phase 2.
    # NOTE: this sort exists in FastTTS-AE as a commented-out line in each
    # search strategy (e.g. search/dynamic_branching.py:249); re-enabled here
    if state.search_config.spec_beam_extension:
        state.active_beams.sort(
            key=lambda b: aggregate_scores(b.scores, state.search_config.agg_strategy)
        )


def _check_n_completion(state: SearchState) -> bool:
    """Return True when the search has produced enough beams to stop.

    All n_* metrics are set in `_finalize` once — the loop breaks
    immediately on `True`, so the snapshot we used to take here is
    always equal to the end-of-loop total.
    """
    has_enough = (
        len(state.completed_beams) >= state.search_config.n
        or len(state.active_beams) == 0
    )
    if has_enough:
        logger.info(
            f"Reached target n: {len(state.completed_beams)} completed beams "
            f"after {state.total_gen_latency + state.total_ver_latency:.2f}s"
        )
    return has_enough


def _duplicate_beams(state: SearchState, tokenizer):
    """Expand active beams to search_config.n via even-split duplication."""
    config = state.search_config
    active = state.active_beams
    i = state.iteration

    if len(active) == config.n:
        return

    repeats = config.n // len(active)

    if getattr(config, 'prefix_aware_scheduling', False):
        # Prefix-aware: place duplicates adjacent, reversed iteration
        final_beams = []
        remainder = config.n % len(active)

        for x, beam in reversed(list(enumerate(active))):
            repeats_for_this_beam = repeats + (1 if x < remainder else 0)
            duplicates = []
            for _ in range(repeats_for_this_beam - 1):
                duplicate = copy.deepcopy(beam)
                duplicate.beam_id = _next_beam_id()
                duplicate.parent_id = beam.beam_id
                duplicate.born_at_iteration = i
                if beam.pending_steps:
                    if config.spec_truncation_ratio <= 0.0:
                        # True vanilla equivalence: duplicate regenerates from
                        # current_text, no speculative seed.
                        duplicate.pending_steps = []
                    else:
                        first_text = truncate_sentence_by_tokens(
                            beam.pending_steps[0].text, tokenizer,
                            mean_ratio=config.spec_truncation_ratio,
                        )
                        duplicate.pending_steps = [
                            StepChunk(text=first_text, is_complete_step=False, terminal=False)
                        ]
                    duplicate.scores = beam.scores[:i]
                    duplicate.step_hashes = beam.step_hashes[:i]
                duplicates.append(duplicate)
            final_beams.extend([beam] + duplicates)
        state.active_beams = final_beams
    else:
        # Standard: place duplicates at the end (matches AE beam_search).
        extended_active_beams = [copy.deepcopy(b) for b in (active * repeats)]
        for b in extended_active_beams:
            b.beam_id = _next_beam_id()
            b.born_at_iteration = i
            if b.pending_steps:
                if config.spec_truncation_ratio <= 0.0:
                    b.pending_steps = []
                else:
                    first_text = truncate_sentence_by_tokens(
                        b.pending_steps[0].text, tokenizer,
                        mean_ratio=config.spec_truncation_ratio,
                    )
                    b.pending_steps = [
                        StepChunk(text=first_text, is_complete_step=False, terminal=False)
                    ]
                b.scores = b.scores[:i]
                b.step_hashes = b.step_hashes[:i]
        state.active_beams = (active + extended_active_beams)[:config.n]

        for idx, dup in enumerate(extended_active_beams):
            original_idx = idx % len(active)
            dup.parent_id = active[original_idx].beam_id

    assert len(state.active_beams) == config.n, (
        f"Expected {config.n} active beams, got {len(state.active_beams)}"
    )


def _prepare_step_source(state: SearchState, tokenizer):
    """Decide per-beam whether this iteration skips generation (consuming
    a queued StepChunk) or needs a fresh generator call."""
    state.extended_beam_count = 0
    for beam in state.active_beams:
        if not beam.pending_steps:
            beam.skipped_this_step = False
            continue

        state.extended_beam_count += 1
        head = beam.pending_steps[0]

        if state.is_last and any(c.terminal for c in beam.pending_steps):
            beam.skipped_this_step = True
        elif head.is_complete_step or head.terminal:
            beam.skipped_this_step = True
        else:
            num_tokens = len(tokenizer.encode(head.text))
            beam.step_tokens = num_tokens
            beam.total_tokens_generated += num_tokens
            beam.current_text += head.text
            beam.pending_steps.pop(0)
            beam.skipped_this_step = False


def _generate(state: SearchState, generator: GeneratorVLLMModelWrapper, tokenizer,
              sampling_params_override: Optional[SamplingParams] = None):
    """Build conversations for non-skipped beams and call the generator."""
    config = state.search_config
    i = state.iteration

    if sampling_params_override is not None:
        sampling_params = sampling_params_override
    else:
        sampling_params = state.final_sampling_params if state.is_last else state.base_sampling_params

    convs = [
        build_conversation(b.prompt, b.current_text, config.system_prompt)
        for b in state.active_beams if not b.skipped_this_step
    ]
    add_generation_prompt = (i == 0)
    continue_final_message = (i > 0)

    if convs:
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )

        gen_results, gen_time = generate_beam(
            templated_convs, generator, sampling_params, tokenizer=tokenizer
        )
        state.gen_results = gen_results
        state.total_gen_latency += gen_time
    else:
        state.gen_results = []


def _process_results(state: SearchState, tokenizer) -> ScoringBatch:
    """Process generation results, update beam state, build scoring batch."""
    config = state.search_config
    i = state.iteration

    prompts, completions = [], []
    state.skipped_beam_count = 0
    counter = 0

    for beam in state.active_beams:
        if beam.skipped_this_step:
            current_step = beam.pending_steps.pop(0)
            beam.current_text += current_step.text
            num_tokens = len(tokenizer.encode(current_step.text))
            beam.step_tokens = num_tokens
            beam.total_tokens_generated += num_tokens
            beam.gen_history.append("")
            state.skipped_beam_count += 1
        else:
            gen_result = state.gen_results[counter]
            counter += 1

            chunks = parse_generation_into_chunks(
                gen_result, config.stop,
                gen_result.prompt_token_lens[0],
                max_model_len=state.generator_max_model_len,
            )
            current_step = chunks[0]
            beam.pending_steps = chunks[1:]
            beam.gen_text = gen_result.gen_text
            beam.stop_reasons = gen_result.stop_reasons
            beam.finish_reasons = gen_result.finish_reasons
            num_tokens = len(tokenizer.encode(current_step.text))
            beam.step_tokens = num_tokens
            beam.total_tokens_generated += num_tokens
            beam.current_text += current_step.text
            beam.gen_history.append(gen_result.gen_text[0])

        # Completion decision
        if current_step.terminal:
            if not beam.completed:
                beam.completed = True
                beam.time_to_complete = (
                    state.total_gen_latency + state.total_ver_latency
                )
            state.completed_beams.append(beam)

        # Scoring batch — skipped beams never re-enter the PRM batch
        if not beam.skipped_this_step:
            beam.step_hashes.append(step_hash(beam.current_text))
            scoring_text = beam.current_text
            for chunk in beam.pending_steps:
                if not chunk.is_complete_step and not chunk.terminal:
                    break
                scoring_text += chunk.text
                beam.step_hashes.append(step_hash(scoring_text))
            prompts.append(beam.prompt)
            completions.append([scoring_text])

    # is_last cleanup
    if state.is_last:
        for beam in state.active_beams:
            for chunk in beam.pending_steps:
                beam.current_text += chunk.text
            beam.pending_steps = []
            if not beam.completed:
                beam.completed = True
                beam.time_to_complete = (
                    state.total_gen_latency + state.total_ver_latency
                )
                state.completed_beams.append(beam)

    return ScoringBatch(prompts=prompts, completions=completions)


def _validate_scores(scores: list, state: SearchState) -> None:
    """Check for unfilled None scores and raise with beam context."""
    scored_beams = [b for b in state.active_beams if not b.skipped_this_step]
    skipped_beams = [b for b in state.active_beams if b.skipped_this_step]

    victims = []
    for idx, score in enumerate(scores):
        none_positions = [j for j, s in enumerate(score[0]) if s is None]
        if none_positions:
            victims.append((idx, none_positions))

    if not victims:
        return

    lines = [
        f"PRM score propagation failed at iter {state.iteration}: "
        f"{len(victims)} beam(s) have None scores",
    ]
    for idx, none_pos in victims:
        beam = scored_beams[idx]
        lines.append(
            f"  victim: scored[{idx}] beam {beam.beam_id} "
            f"(parent={beam.parent_id} born@{beam.born_at_iteration}): "
            f"none_positions={none_pos} "
            f"steps=[{','.join(beam.step_hashes)}] prev_scores={len(beam.scores)}"
        )
    lines.append("")
    scores_fmt = lambda s: ','.join(f'{v:.3f}' if v is not None else 'None' for v in s)
    for si, sb in enumerate(scored_beams):
        lines.append(
            f"  scored[{si}] beam {sb.beam_id} "
            f"(parent={sb.parent_id} born@{sb.born_at_iteration}): "
            f"steps=[{','.join(sb.step_hashes)}] "
            f"scores=[{scores_fmt(sb.scores)}]"
        )
    for si, sb in enumerate(skipped_beams):
        lines.append(
            f"  skipped[{si}] beam {sb.beam_id} "
            f"(parent={sb.parent_id} born@{sb.born_at_iteration}): "
            f"steps=[{','.join(sb.step_hashes)}] "
            f"scores=[{scores_fmt(sb.scores)}]"
        )
    raise RuntimeError("\n".join(lines))


def _propagate_by_step_hash(scores: list, scored_beams: List[Beam], state: SearchState) -> None:
    """Fill None scores from the persistent step_hash bank."""
    for idx, score in enumerate(scores):
        beam = scored_beams[idx]
        for j, s in enumerate(score[0]):
            if s is None and j < len(beam.step_hashes) and beam.step_hashes[j] in state.step_hash_bank:
                score[0][j] = state.step_hash_bank[beam.step_hashes[j]]


def _score_and_assign(
    state: SearchState,
    verifier: VerifierVLLMModelWrapper,
    tokenizer,
    batch: ScoringBatch,
):
    """Call verifier, propagate scores, then assign to beams."""
    config = state.search_config
    i = state.iteration

    if batch.prompts:
        scores, verifier_time = score_beam(
            verifier, batch.prompts, batch.completions,
        )
        state.total_ver_latency += verifier_time

        scored_beams = [b for b in state.active_beams if not b.skipped_this_step]

        # Layer 0: pad truncated scores
        for idx, score in enumerate(scores):
            expected_len = len(scored_beams[idx].step_hashes)
            if len(score[0]) < expected_len:
                pad_len = expected_len - len(score[0])
                score[0] = [None] * pad_len + score[0]

        # Layer 1: lock prev_scores
        for idx, score in enumerate(scores):
            beam = scored_beams[idx]
            for j in range(min(len(score[0]), len(beam.scores))):
                if beam.scores[j] is not None:
                    score[0][j] = beam.scores[j]

        # Update persistent bank
        for idx, score in enumerate(scores):
            for j, s in enumerate(score[0]):
                if s is not None and j < len(scored_beams[idx].step_hashes):
                    state.step_hash_bank[scored_beams[idx].step_hashes[j]] = s

        # Layer 2: fill from bank
        if any(None in s[0] for s in scores):
            _propagate_by_step_hash(scores, scored_beams, state)

        # Layer 3: validate
        _validate_scores(scores, state)
    else:
        scores = []

    # Assign scores and compute aggregated scores for pruning
    if scores:
        agg_index = i + 1 if not state.is_last else max(len(s[0]) for s in scores)
    else:
        agg_index = i + 1
    counter = 0
    state.agg_scores = []

    for beam in state.active_beams:
        if not beam.skipped_this_step:
            score = scores[counter]
            beam.scores = score[0]
            state.agg_scores.append(
                aggregate_scores(score[0][:agg_index], config.agg_strategy)
            )
            counter += 1
        else:
            state.agg_scores.append(
                aggregate_scores(beam.scores[:agg_index], config.agg_strategy)
            )

    assert counter == len(scores), f"counter: {counter}, len(scores): {len(scores)}"


def _filter_completed_and_prune(state: SearchState):
    """Remove completed beams from active set, prune lowest-scoring (top-k)."""
    config = state.search_config

    state.agg_scores = [
        state.agg_scores[idx]
        for idx, b in enumerate(state.active_beams)
        if not b.completed
    ]
    state.active_beams = [b for b in state.active_beams if not b.completed]

    if len(state.active_beams) == 0:
        return

    top_indices = np.argsort(np.array(state.agg_scores).flatten())[
        -(config.n // config.beam_width):
    ]
    for idx, beam in enumerate(state.active_beams):
        if idx not in top_indices:
            beam.pruned = True


def _log_iteration(state: SearchState, tokenizer):
    """Log iteration summary (INFO) and per-beam detail (DEBUG)."""
    config = state.search_config
    i = state.iteration

    unpruned = [b for b in state.active_beams if not b.pruned]

    logger.info(
        f"iter {i}: active={len(unpruned)} completed={len(state.completed_beams)} "
        f"skipped={state.skipped_beam_count} "
        f"latency={state.total_gen_latency + state.total_ver_latency:.2f}s"
    )

    if logger.isEnabledFor(logging.DEBUG):
        def _fmt_beam(beam):
            ct_tokens = len(tokenizer.encode(beam.current_text)) if beam.current_text else 0
            conv = build_conversation(beam.prompt, beam.current_text, config.system_prompt)
            templated = tokenizer.apply_chat_template(
                conv, add_generation_prompt=False,
                continue_final_message=True, tokenize=True,
            )
            prompt_tokens = len(templated)
            n_pending = len(beam.pending_steps)
            future_tokens = (
                sum(len(tokenizer.encode(c.text)) for c in beam.pending_steps)
                if beam.pending_steps else 0
            )
            if not beam.pending_steps:
                tail_label = ""
            else:
                tail = beam.pending_steps[-1]
                if tail.terminal and tail.is_complete_step:
                    tail_label = "|T"
                elif tail.terminal:
                    tail_label = "+partial|T"
                elif not tail.is_complete_step:
                    tail_label = "+cont"
                else:
                    tail_label = ""
            scores_str = ','.join(f'{s:.3f}' for s in beam.scores) if beam.scores else '—'
            flags = (
                f"{'PRUNED ' if beam.pruned else ''}"
                f"{'SKIPPED ' if beam.skipped_this_step else ''}"
            ).strip()
            return (
                f"  beam {beam.beam_id} "
                f"(parent={beam.parent_id}, born@{beam.born_at_iteration}) "
                f"ct={ct_tokens} prompt={prompt_tokens} "
                f"pending={n_pending}{tail_label}({future_tokens}tok) "
                f"scores=[{scores_str}] stop={beam.stop_label} "
                f"steps=[{','.join(beam.step_hashes)}] "
                f"{flags}"
            )

        if state.active_beams:
            logger.debug(f"  [active ({len(state.active_beams)})]")
            for beam in state.active_beams:
                logger.debug(_fmt_beam(beam))
        if state.completed_beams:
            logger.debug(f"  [completed ({len(state.completed_beams)})]")
            for beam in state.completed_beams:
                logger.debug(_fmt_beam(beam))


def _finalize(state: SearchState):
    """Post-loop: compute final metrics, sort completed beams."""
    config = state.search_config

    for beam in state.completed_beams:
        assert not beam.pending_steps, (
            f"beam {beam.beam_id} has {len(beam.pending_steps)} leftover "
            f"pending_steps at finalize — this should not happen"
        )

    state.total_tokens += sum(b.total_tokens_generated for b in state.completed_beams)

    state.completed_beams = sorted(
        state.completed_beams,
        key=lambda b: aggregate_scores(b.scores, config.agg_strategy),
        reverse=True,
    )[:config.n]

    # n_completion_tokens references the same top-n beam set that
    # `evaluate.py` scores for accuracy — can differ from total_tokens
    # when burst completion pushed M > n before truncation.
    n_completion_tokens = sum(b.total_tokens_generated for b in state.completed_beams)

    return (
        state.completed_beams,
        state.total_gen_latency,
        state.total_ver_latency,
        state.total_tokens,
        n_completion_tokens,
    )


# ---------------------------------------------------------------------------
# Result packaging (shared by all strategies)
# ---------------------------------------------------------------------------

def package_results(
    problems: List[str],
    completed_beams: List[Beam],
    total_generator_latency_s: float,
    total_verifier_latency_s: float,
    total_num_tokens: int,
    n_completion_tokens: int,
    search_config: SearchConfig,
) -> SearchResults:
    """Package completed beams into the canonical SearchResults container."""
    grouped_results = defaultdict(list)
    for beam in completed_beams:
        grouped_results[beam.prompt].append(beam)

    results = SearchResults(
        total_num_tokens=total_num_tokens,
        n_completion_tokens=n_completion_tokens,
        total_generator_latency_s=total_generator_latency_s,
        total_verifier_latency_s=total_verifier_latency_s,
    )

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.scores, search_config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)] if agg_scores else ""
        results.pred.append(pred)
        results.completions.append(completions)
        results.scores.append([b.scores for b in beams])
        results.completion_time.append([b.time_to_complete for b in beams])
        results.effective_num_tokens.append([b.total_tokens_generated for b in beams])

    return results
