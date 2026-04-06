#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from vllm import SamplingParams
import torch.cuda.nvtx as nvtx

from config import SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.beam import Beam, StepChunk, _next_beam_id, reset_beam_id_counter, step_hash
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
        # Length-cap recoverable: output contains at least one complete
        # step, so the per-call budget was shared across multiple steps —
        # no single step violated the per-step limit. The trailing
        # partial (if any) becomes a (False, False) continuation prefix
        # that _prepare_step_source will consume next iteration.
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
        # Empty generator output (e.g., immediate EOS with zero tokens).
        # Emit a synthetic chunk so the beam can still be consumed this
        # iteration normally.
        return [StepChunk(text="", is_complete_step=False, terminal=terminal_gen)]

    if terminal_gen:
        chunks[-1].terminal = True
    return chunks


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def score_beam(verifier: VerifierVLLMModelWrapper, prompts: List[str], completions: List[List[str]]):
    """Score a beam of completions. Returns raw scores (may contain Nones from prefix caching)."""
    verifier_time = time.time()
    scores = verifier.score(prompts, completions)
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
    tokenizer = None,
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
        # Preserve BOTH stop_reason (the matched stop string, if any)
        # and finish_reason ("stop" / "length" / "abort"). The parser
        # needs the distinction for length-cap recovery — see
        # parse_generation_into_chunks.
        gen_result["text"] = gen_text
        gen_result["stop_reason"] = output.outputs[0].stop_reason
        gen_result["finish_reason"] = output.outputs[0].finish_reason
        gen_result["completion_tokens"] = out_tokens
        gen_result["prompt_token_len"] = len(output.prompt_token_ids)

    # Convert to beam format
    outputs = []
    counter = 0
    for i in range(num_convs):
        next_texts = []
        stop_reasons = []
        finish_reasons = []
        completion_tokens = []
        prompt_token_lens = []
        for _ in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result["text"])
            stop_reasons.append(gen_result["stop_reason"])
            finish_reasons.append(gen_result["finish_reason"])
            completion_tokens.append(gen_result["completion_tokens"])
            prompt_token_lens.append(gen_result["prompt_token_len"])
            counter += 1

        beam_result = Beam(
            prompt=templated_convs[i],
            current_text="",
            gen_text=next_texts,
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
# Dataclasses for decomposed beam search
# ---------------------------------------------------------------------------

@dataclass
class BeamSearchState:
    """Mutable state shared across phases within one beam search."""
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
    n_gen_latency: float = 0.0
    n_ver_latency: float = 0.0
    n_completion_tokens: int = 0
    extended_tokens_list: List[List[int]] = field(default_factory=list)

    # Per-iteration transients (reset each iteration)
    gen_results: List[Beam] = field(default_factory=list)
    extended_tokens: List[int] = field(default_factory=list)
    agg_scores: List[float] = field(default_factory=list)

    # Per-iteration counters (for logging)
    skipped_beam_count: int = 0
    extended_beam_count: int = 0


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
) -> BeamSearchState:
    """Initialize beam search state: create beams, sampling params, compute prompt length."""
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

    return BeamSearchState(
        all_beams=beams,
        active_beams=[],
        completed_beams=[],
        search_config=search_config,
        system_prompt=search_config.system_prompt,
        base_sampling_params=base_sampling_params,
        final_sampling_params=final_sampling_params,
    )


def _filter_active(state: BeamSearchState):
    """Filter out pruned beams."""
    if state.iteration == 0:
        state.active_beams = [b for b in state.all_beams if not b.pruned]
    else:
        state.active_beams = [b for b in state.active_beams if not b.pruned]


def _check_n_completion(state: BeamSearchState):
    """Record metrics when we first reach n completions."""
    has_enough = (
        len(state.completed_beams) >= state.search_config.n
        or len(state.active_beams) == 0
    )
    if has_enough and state.n_gen_latency == 0:
        state.n_gen_latency = state.total_gen_latency
        state.n_ver_latency = state.total_ver_latency
        state.n_completion_tokens = state.total_tokens
        logger.info(
            f"Reached target n: {len(state.completed_beams)} completed beams "
            f"after {state.n_gen_latency + state.n_ver_latency:.2f}s, "
            f"{state.n_completion_tokens} total tokens"
        )


def _duplicate_beams(state: BeamSearchState, tokenizer):
    """Expand active beams to search_config.n via duplication."""
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
                    # Paper Algorithm 1, line 19: truncate first speculative
                    # chunk into a continuation prefix → immediate divergence.
                    first_text = truncate_sentence_by_tokens(
                        beam.pending_steps[0].text, tokenizer
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
        # Standard: place duplicates at the end
        extended_active_beams = [copy.deepcopy(b) for b in (active * repeats)]
        for b in extended_active_beams:
            b.beam_id = _next_beam_id()
            b.born_at_iteration = i
            if b.pending_steps:
                first_text = truncate_sentence_by_tokens(
                    b.pending_steps[0].text, tokenizer
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


def _prepare_step_source(state: BeamSearchState, tokenizer):
    """Decide per-beam whether this iteration skips generation (consuming
    a queued StepChunk) or needs a fresh generator call.

    After this function:
    - beam.skipped_this_step == True  → the head of pending_steps is a
      committable chunk (complete step or terminal); no generator call
      needed this iteration.
    - beam.skipped_this_step == False → pending_steps is empty or had a
      non-terminal partial prefix consumed as context; the generator
      will be called for this beam in _generate.
    """
    state.extended_beam_count = 0
    for beam in state.active_beams:
        if not beam.pending_steps:
            beam.skipped_this_step = False
            continue

        state.extended_beam_count += 1
        head = beam.pending_steps[0]

        if state.is_last and any(c.terminal for c in beam.pending_steps):
            # Option A: terminal chunk already queued on the final
            # iteration. Skip generation — the skipped branch of
            # _process_results will pop chunks one at a time, but since
            # this is the last iteration the is_last cleanup in
            # _process_results will flush everything into current_text.
            beam.skipped_this_step = True
        elif head.is_complete_step or head.terminal:
            # A committable chunk sits at the head — no generation needed.
            beam.skipped_this_step = True
        else:
            # Head is a non-terminal partial (duplicate-truncation prefix
            # or force-finish/length-cap-recovery tail that has reached
            # the front of the queue). Consume it as prompt context and
            # let the generator extend it.
            num_tokens = len(tokenizer.encode(head.text))
            beam.step_tokens = num_tokens
            beam.total_tokens_generated += num_tokens
            beam.current_text += head.text
            beam.pending_steps.pop(0)
            state.extended_tokens.append(num_tokens)
            beam.skipped_this_step = False


def _generate(state: BeamSearchState, generator: GeneratorVLLMModelWrapper, tokenizer):
    """Build conversations for non-skipped beams and call the generator."""
    config = state.search_config
    i = state.iteration

    sampling_params = state.final_sampling_params if state.is_last else state.base_sampling_params

    convs = [
        build_conversation(b.prompt, b.current_text, config.system_prompt)
        for b in state.active_beams if not b.skipped_this_step
    ]
    add_generation_prompt = (i == 0)
    continue_final_message = (i > 0)

    if convs:
        if hasattr(config, 'custom_chat_template') and config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
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


def _process_results(state: BeamSearchState, tokenizer) -> ScoringBatch:
    """Process generation results, update beam state, build scoring batch.

    Returns a ScoringBatch with the inputs needed for the verifier call.

    Key semantic change vs. the original code:

    - Completion decision is `current_step.terminal`, not
      `beam_is_completed(gen_result, …)`. The parser assigns
      terminal=True only on the *last* chunk, and only when the
      generation genuinely terminated the beam.
    - Skipped beams (consuming a previously-scored chunk from
      pending_steps) NEVER re-enter the PRM batch.
    - The scoring-loop gate `not is_complete_step and not terminal`
      filters out non-terminal continuation prefixes (from
      _duplicate_beams, force-finish, or length-cap recovery).
    """
    config = state.search_config
    i = state.iteration

    prompts, completions = [], []
    state.skipped_beam_count = 0
    counter = 0

    for beam in state.active_beams:
        # ── Commit one step ────────────────────────────────────────
        if beam.skipped_this_step:
            # Chunk was already generated and scored at a prior
            # iteration's lookahead pass. Just consume it.
            current_step = beam.pending_steps.pop(0)
            beam.current_text += current_step.text
            num_tokens = len(tokenizer.encode(current_step.text))
            beam.step_tokens = num_tokens
            beam.total_tokens_generated += num_tokens
            beam.gen_history.append("")
            state.skipped_beam_count += 1
            state.extended_tokens.append(num_tokens)
        else:
            # Fresh generation result. Parse into StepChunks.
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

        # ── Completion decision (the semantic fix) ─────────────────
        if current_step.terminal:
            if not beam.completed:
                beam.completed = True
                beam.time_to_complete = (
                    state.total_gen_latency + state.total_ver_latency
                )
            state.completed_beams.append(beam)

        # ── Scoring batch ──────────────────────────────────────────
        # Gate 1: skipped beams never re-enter the PRM batch — their
        # scores were already computed at the speculation iteration
        # that generated the chunk.
        if not beam.skipped_this_step:
            # Gate 2: build scoring_text by walking pending_steps.
            # Break at any (is_complete_step=False, terminal=False)
            # chunk — a non-terminal continuation prefix that is not a
            # scorable step. Two producers for that shape:
            #   (a) SBE force-finish tails.
            #   (b) Length-cap-recoverable tails.
            beam.step_hashes.append(step_hash(beam.current_text))
            scoring_text = beam.current_text
            for chunk in beam.pending_steps:
                if not chunk.is_complete_step and not chunk.terminal:
                    break
                scoring_text += chunk.text
                beam.step_hashes.append(step_hash(scoring_text))
            prompts.append(beam.prompt)
            completions.append([scoring_text])

    # ── is_last cleanup (single pass) ──────────────────────────────
    # After scoring, flush any remaining pending_steps text into
    # current_text and force-complete every beam. For skipped beams,
    # scores/hashes were already computed at the speculation iteration.
    # For non-skipped beams, scoring_text above already covered them.
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


def _validate_scores(scores: list, state: BeamSearchState) -> None:
    """Check for unfilled None scores and raise with beam context.

    Remaining Nones indicate a pruned beam's KV cache was reused — the
    pruned beam's scores are permanently lost. Reports beam step hashes
    so you can see which beams share prefixes.
    """
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


def _propagate_by_step_hash(scores: list, scored_beams: List[Beam], state: BeamSearchState) -> None:
    """Fill None scores using step_hash matching across all beams.

    Each step_hash is a hash of the cumulative text through that step,
    so matching by single hash is equivalent to matching the full prefix.

    Bank is built from historical scores (beam.scores) AND fresh PRM
    output (scores) so beams in the same batch can donate to each other.
    """
    # Build bank: hash → score, from all available sources
    bank = {}
    # Historical scores from all active beams
    for beam in state.active_beams:
        for j, h in enumerate(beam.step_hashes):
            if j < len(beam.scores) and beam.scores[j] is not None:
                bank[h] = beam.scores[j]
    # Fresh scores from this iteration's PRM batch
    for idx, score in enumerate(scores):
        for j, s in enumerate(score[0]):
            if s is not None and j < len(scored_beams[idx].step_hashes):
                bank[scored_beams[idx].step_hashes[j]] = s

    # Fill Nones
    for idx, score in enumerate(scores):
        beam = scored_beams[idx]
        for j, s in enumerate(score[0]):
            if s is None and j < len(beam.step_hashes) and beam.step_hashes[j] in bank:
                score[0][j] = bank[beam.step_hashes[j]]


def _score_and_assign(
    state: BeamSearchState,
    verifier: VerifierVLLMModelWrapper,
    tokenizer,
    batch: ScoringBatch,
):
    """Call verifier, propagate scores, then assign to beams.

    Score propagation (Layers 1-3) runs here using step_hashes, not in
    the PRM subprocess. This avoids tokenization mismatches from
    prepare_input's truncation logic.
    """
    config = state.search_config
    i = state.iteration

    # --- Call verifier (returns raw scores, may contain Nones) ---
    if batch.prompts:
        scores, verifier_time = score_beam(
            verifier, batch.prompts, batch.completions,
        )
        state.total_ver_latency += verifier_time

        # --- Score propagation ---
        scored_beams = [b for b in state.active_beams if not b.skipped_this_step]

        # Layer 0: pad truncated scores to match step_hashes length.
        # When the verifier's prepare_input truncates early steps, score[0]
        # has fewer entries than beam.step_hashes. Pad the front with None
        # so that subsequent layers operate on positionally-aligned data.
        for idx, score in enumerate(scores):
            expected_len = len(scored_beams[idx].step_hashes)
            if len(score[0]) < expected_len:
                pad_len = expected_len - len(score[0])
                score[0] = [None] * pad_len + score[0]

        # Layer 1: lock prev_scores (fill from beam's own history)
        for idx, score in enumerate(scores):
            beam = scored_beams[idx]
            for j in range(min(len(score[0]), len(beam.scores))):
                if beam.scores[j] is not None:
                    score[0][j] = beam.scores[j]

        # Layer 2: step_hash propagation (fill from any beam with matching steps)
        if any(None in s[0] for s in scores):
            _propagate_by_step_hash(scores, scored_beams, state)

        # Layer 3: validate no missing
        _validate_scores(scores, state)
    else:
        scores = []

    # --- Assign scores to beams and compute aggregated scores for pruning ---
    # agg_index: how many steps to use for the aggregate (one per
    # iteration, or all available on the last iteration).
    if scores:
        agg_index = i + 1 if not state.is_last else max(len(s[0]) for s in scores)
    else:
        agg_index = i + 1
    counter = 0  # indexes into `scores` (only scored beams, not skipped)
    state.agg_scores = []

    for beam in state.active_beams:
        if not beam.skipped_this_step:
            # Beam was scored this iteration — consume from scores list.
            # (Old criterion was `state.is_last or len(beam.scores) < agg_index`,
            # which assumed all beams are scored on is_last. Under delayed
            # completion, terminal-queued beams are skipped even on is_last.)
            score = scores[counter]
            beam.scores = score[0]  # store per-step PRM scores
            state.agg_scores.append(
                aggregate_scores(score[0][:agg_index], config.agg_strategy)
            )
            counter += 1
        else:
            # Beam was skipped — reuse existing scores for aggregate
            state.agg_scores.append(
                aggregate_scores(beam.scores[:agg_index], config.agg_strategy)
            )

    assert counter == len(scores), f"counter: {counter}, len(scores): {len(scores)}"


def _filter_completed_and_prune(state: BeamSearchState) -> bool:
    """Remove completed beams, prune lowest-scoring. Returns True if should break."""
    config = state.search_config

    # Filter out completed beams
    state.agg_scores = [
        state.agg_scores[idx]
        for idx, b in enumerate(state.active_beams)
        if not b.completed
    ]
    state.active_beams = [b for b in state.active_beams if not b.completed]

    # Early stopping
    if len(state.active_beams) == 0:
        logger.info(
            f"Early exit: 0 active, {len(state.completed_beams)} completed"
        )
        return True

    # Prune: keep top (n / beam_width) beams
    top_indices = np.argsort(np.array(state.agg_scores).flatten())[
        -(config.n // config.beam_width):
    ]
    for idx, beam in enumerate(state.active_beams):
        if idx not in top_indices:
            beam.pruned = True

    return False


def _log_iteration(state: BeamSearchState, tokenizer):
    """Log iteration summary (INFO) and per-beam detail (DEBUG)."""
    config = state.search_config
    i = state.iteration

    unpruned = [b for b in state.active_beams if not b.pruned]

    logger.info(
        f"iter {i}: active={len(unpruned)} completed={len(state.completed_beams)} "
        f"skipped={state.skipped_beam_count} "
        f"latency={state.total_gen_latency + state.total_ver_latency:.2f}s"
    )

    # Per-beam state table — only at DEBUG level (expensive tokenization)
    if logger.isEnabledFor(logging.DEBUG):
        for beam in state.active_beams:
            ct_tokens = len(tokenizer.encode(beam.current_text)) if beam.current_text else 0
            conv = build_conversation(beam.prompt, beam.current_text, config.system_prompt)
            templated = tokenizer.apply_chat_template(
                conv, add_generation_prompt=False,
                continue_final_message=True, tokenize=True,
            )
            prompt_tokens = len(templated)
            # Pending steps: tail chunk characterises the pipeline state.
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
                    tail_label = "|T"           # terminal complete step
                elif tail.terminal:
                    tail_label = "+partial|T"   # terminal partial tail
                elif not tail.is_complete_step:
                    tail_label = "+cont"        # continuation prefix
                else:
                    tail_label = ""             # all regular complete steps
            # Format scores list compactly
            scores_str = ','.join(f'{s:.3f}' for s in beam.scores) if beam.scores else '—'
            flags = (
                f"{'PRUNED ' if beam.pruned else ''}"
                f"{'DONE ' if beam.completed else ''}"
                f"{'SKIPPED ' if beam.skipped_this_step else ''}"
            ).strip()
            logger.debug(
                f"  beam {beam.beam_id} "
                f"(parent={beam.parent_id}, born@{beam.born_at_iteration}) "
                f"ct={ct_tokens} prompt={prompt_tokens} "
                f"pending={n_pending}{tail_label}({future_tokens}tok) "
                f"scores=[{scores_str}] stop={beam.stop_label} "
                f"steps=[{','.join(beam.step_hashes)}] "
                f"{flags}"
            )


def _finalize(state: BeamSearchState):
    """Post-loop: compute final metrics, sort completed beams.

    Under the new delayed-completion semantics, a beam in
    completed_beams always has pending_steps == [] by the time it was
    added (terminal chunks are consumed at their scheduled iteration,
    and is_last flushes any remaining). The old dump loop that
    moved leftover future_texts into current_text is no longer needed.
    """
    config = state.search_config

    for beam in state.completed_beams:
        assert not beam.pending_steps, (
            f"beam {beam.beam_id} has {len(beam.pending_steps)} leftover "
            f"pending_steps at finalize — this should not happen"
        )

    state.total_tokens += sum(b.total_tokens_generated for b in state.completed_beams)
    if state.n_gen_latency == 0:
        state.n_completion_tokens = state.total_tokens
        state.n_gen_latency = state.total_gen_latency
        state.n_ver_latency = state.total_ver_latency

    if config.sort_completed:
        state.completed_beams = sorted(
            state.completed_beams,
            key=lambda b: aggregate_scores(b.scores, config.agg_strategy),
            reverse=True,
        )

    return (
        state.completed_beams,
        state.total_gen_latency,
        state.total_ver_latency,
        state.n_gen_latency,
        state.n_ver_latency,
        state.total_tokens,
        state.n_completion_tokens,
        state.extended_tokens_list,
    )


# ---------------------------------------------------------------------------
# Main beam search loop
# ---------------------------------------------------------------------------

def _beam_search(
    batch_of_prompts: List[str],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> tuple:
    """Beam search implementation decomposed into named phases."""

    tokenizer = generator.get_tokenizer()
    state = _init_state(batch_of_prompts, search_config, tokenizer)
    state.generator_max_model_len = generator.config.generator_vllm_config.get("max_model_len", 4096)

    logger.info("Starting beam search iterations")
    for i in tqdm(range(search_config.num_iterations), desc="Beam search iterations"):
        state.iteration = i
        state.is_last = (i == search_config.num_iterations - 1)
        state.extended_tokens = []

        _filter_active(state)
        _duplicate_beams(state, tokenizer)
        _prepare_step_source(state, tokenizer)
        _generate(state, generator, tokenizer)
        scoring_batch = _process_results(state, tokenizer)

        state.extended_tokens_list.append(state.extended_tokens)

        _score_and_assign(state, verifier, tokenizer, scoring_batch)

        should_stop = _filter_completed_and_prune(state)
        _check_n_completion(state)

        if should_stop:
            break

        _log_iteration(state, tokenizer)

    return _finalize(state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def beam_search(
    examples: Dict[str, Any],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> Dict[str, Any]:
    """Run beam search for a batch of examples."""
    problems = examples["problem"]
    assert len(problems) == 1, "batch_of_prompts should be a list of length 1 for now"

    nvtx.range_push("Total")
    (
        completed_beams, total_generator_latency_s, total_verifier_latency_s,
        n_generator_latency_s, n_verifier_latency_s,
        total_num_tokens, n_completion_tokens, extended_tokens_list,
    ) = _beam_search(problems, search_config, generator, verifier)
    nvtx.range_pop()

    # Group results by prompt
    grouped_results = defaultdict(list)
    for beam in completed_beams:
        grouped_results[beam.prompt].append(beam)

    results = {
        "completions": [],
        "pred": [],
        "completion_tokens": [],
        "scores": [],
        "effective_num_tokens": [],
        "total_num_tokens": total_num_tokens,
        "n_completion_tokens": n_completion_tokens,
        "total_generator_latency_s": total_generator_latency_s,
        "total_verifier_latency_s": total_verifier_latency_s,
        "n_generator_latency_s": n_generator_latency_s,
        "n_verifier_latency_s": n_verifier_latency_s,
        "completion_time": [],
        "vllm_metrics": {},
        "vllm_metrics_summary": {},
        "extended_tokens_list": extended_tokens_list,
    }

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.scores, search_config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["pred"].append(pred)
        results["completions"].append(completions)
        results["scores"].append([b.scores for b in beams])
        results["completion_tokens"].append([b.step_tokens for b in beams])
        results["completion_time"].append([b.time_to_complete for b in beams])
        results["effective_num_tokens"].append([b.total_tokens_generated for b in beams])

    return results
