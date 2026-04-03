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
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm
from vllm import SamplingParams
import torch.cuda.nvtx as nvtx

from config import SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.beam import Beam, _next_beam_id, reset_beam_id_counter
from search.utils import build_conversation, aggregate_scores, split_string_by_separator, truncate_sentence_by_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Completion check
# ---------------------------------------------------------------------------

def beam_is_completed(beam: Beam, token_length: int, max_model_len: int = 4096) -> bool:
    """A beam continues only if it stopped at the step separator with text remaining."""
    if token_length >= max_model_len:
        return True
    # The only "continue" case: stopped at step separator with non-empty output
    if beam.stop_reasons[0] == "\n\n" and beam.next_texts[0] != "":
        return False
    return True  # EOS, length, empty text, or any non-separator stop


# ---------------------------------------------------------------------------
# Scoring helper (unchanged)
# ---------------------------------------------------------------------------

def score_beam(verifier: VerifierVLLMModelWrapper, prompts: List[str], completions: List[List[str]], tokenizer=None, prev_scores=None, skipped_beam_context=None):
    """Score a beam of completions."""
    prefix_priorities = None
    verifier_time = time.time()
    scores = verifier.score(prompts, completions, priority=prefix_priorities,
                            prev_scores=prev_scores, skipped_beam_context=skipped_beam_context)
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
            "completion_tokens": 0,
        }
        for i, text in enumerate(templated_convs)
        for _ in range(beam_width)
    ]

    gen_prompts = [r["initial_prompt"] for r in gen_results]

    # Log prompt lengths when approaching limit (DEBUG only)
    if tokenizer is not None and logger.isEnabledFor(logging.DEBUG):
        for j, prompt in enumerate(gen_prompts):
            tok_len = len(tokenizer.encode(prompt))
            if tok_len >= 4000:
                logger.debug(
                    f"generate_beam: prompt {j} has {tok_len} tokens "
                    f"(max_model_len=4096, headroom={4096 - tok_len})"
                )

    start_time = time.time()
    nvtx.range_push("generate")
    llm_outputs = generator.generate(gen_prompts, sampling_params=sampling_params)
    nvtx.range_pop()
    end_time = time.time()

    for gen_result, output in zip(gen_results, llm_outputs):
        gen_text = output.outputs[0].text
        out_tokens = len(output.outputs[0].token_ids)
        finish_reason = output.outputs[0].finish_reason
        stop_reason = output.outputs[0].stop_reason

        if tokenizer is not None and finish_reason == "length" and logger.isEnabledFor(logging.DEBUG):
            prompt_tok = len(tokenizer.encode(gen_result["initial_prompt"]))
            logger.debug(
                f"generate_beam: beam {gen_result['index']} hit length cap. "
                f"prompt={prompt_tok}, output={out_tokens}, total={prompt_tok + out_tokens}, "
                f"finish_reason={finish_reason}, stop_reason={stop_reason}"
            )

        gen_result["text"] = gen_text
        gen_result["stop_reason"] = stop_reason if stop_reason is not None else "EOS"
        gen_result["completion_tokens"] = out_tokens

    # Convert to beam format
    outputs = []
    counter = 0
    for i in range(num_convs):
        next_texts = []
        stop_reasons = []
        completion_tokens = []
        for _ in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result["text"])
            stop_reasons.append(gen_result["stop_reason"])
            completion_tokens.append(gen_result["completion_tokens"])
            counter += 1

        beam_result = Beam(
            prompt=templated_convs[i],
            index=i,
            current_text="",
            next_texts=next_texts,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
            completion_tokens=completion_tokens,
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
    iteration: int = 0
    is_last: bool = False

    # Config-derived (computed once at init)
    system_prompt: str = ""
    prompt_token_length: int = 0
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
    verified_beam_count: int = 0


@dataclass
class ScoringBatch:
    """Inputs for the verifier scoring call."""
    prompts: List[str]
    completions: List[List[str]]
    prev_scores: List[List[float]]
    skipped_beam_context: List[Tuple]


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
                index=idx,
                current_text="",
                next_texts=None,
                pruned=False,
                completed=False,
                stop_reasons=None,
                history=[],
                best_scores=[],
                all_scores=[],
                previous_text=None,
                completion_tokens=0,
                total_completion_tokens=0,
                completion_time=0.0,
                beam_id=_next_beam_id(),
                parent_id=None,
                born_at_iteration=0,
            ))

    # Compute prompt token length (from empty response — existing behavior)
    conv = build_conversation(batch_of_prompts[0], "", search_config.system_prompt)
    conv_str = tokenizer.apply_chat_template(
        conv, add_generation_prompt=True,
        continue_final_message=False, tokenize=True,
    )
    prompt_token_length = len(conv_str)

    return BeamSearchState(
        all_beams=beams,
        active_beams=[],
        completed_beams=[],
        search_config=search_config,
        system_prompt=search_config.system_prompt,
        prompt_token_length=prompt_token_length,
        base_sampling_params=base_sampling_params,
        final_sampling_params=final_sampling_params,
    )


def _filter_active(state: BeamSearchState):
    """Filter out pruned beams to get current active set."""
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
                if beam.future_texts:
                    # Paper Algorithm 1, line 19: truncate first speculative
                    # step, clear rest -> immediate divergence
                    first_text = truncate_sentence_by_tokens(beam.future_texts[0][0], tokenizer)
                    duplicate.future_texts = [(first_text, False)]
                    duplicate.all_scores = beam.all_scores[:i]
                duplicates.append(duplicate)
            final_beams.extend([beam] + duplicates)
        state.active_beams = final_beams
    else:
        # Standard: place duplicates at the end
        extended_active_beams = [copy.deepcopy(b) for b in (active * repeats)]
        for b in extended_active_beams:
            b.beam_id = _next_beam_id()
            b.parent_id = b.beam_id  # parent is the original (before deepcopy overwrote beam_id)
            b.born_at_iteration = i
            if b.future_texts:
                first_text = truncate_sentence_by_tokens(b.future_texts[0][0], tokenizer)
                b.future_texts = [(first_text, False)]
                b.all_scores = b.all_scores[:i]
        state.active_beams = (active + extended_active_beams)[:config.n]

        # Fix parent_id: deepcopy copied the old beam_id, but we need the
        # original beam's beam_id as parent. Since active * repeats duplicates
        # each beam, the parent_id should reference the corresponding original.
        for idx, dup in enumerate(extended_active_beams):
            original_idx = idx % len(active)
            dup.parent_id = active[original_idx].beam_id

    assert len(state.active_beams) == config.n, (
        f"Expected {config.n} active beams, got {len(state.active_beams)}"
    )


def _consume_future_texts(state: BeamSearchState, tokenizer):
    """Pop SBE future_texts into current_text. Sets skipped_this_step."""
    state.extended_beam_count = 0
    for beam in state.active_beams:
        if len(beam.future_texts) > 0:
            state.extended_beam_count += 1
            next_text, is_finished_this_step = beam.future_texts[0]
            if state.is_last:
                # Last iteration: dump ALL remaining future texts
                while beam.future_texts:
                    next_text, _ = beam.future_texts.pop(0)
                    num_tokens = len(tokenizer.encode(next_text))
                    beam.completion_tokens = num_tokens
                    beam.total_completion_tokens += beam.completion_tokens
                    beam.current_text += next_text
                    state.extended_tokens.append(num_tokens)
                beam.skipped_this_step = beam.completed
            elif is_finished_this_step:
                beam.skipped_this_step = True
            else:
                num_tokens = len(tokenizer.encode(next_text))
                beam.completion_tokens = num_tokens
                beam.total_completion_tokens += beam.completion_tokens
                beam.current_text += next_text
                beam.future_texts.pop(0)
                state.extended_tokens.append(num_tokens)


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
    """
    config = state.search_config
    i = state.iteration

    prompts, completions, beam_prev_scores = [], [], []
    skipped_beam_context = []
    state.skipped_beam_count = 0
    state.verified_beam_count = 0
    counter = 0

    for beam in state.active_beams:
        skipped_this_step = beam.skipped_this_step

        if skipped_this_step and not state.is_last:
            # --- Skipped beam: consume next future_text ---
            next_text, _ = beam.future_texts.pop(0)
            beam.current_text += next_text
            num_tokens = len(tokenizer.encode(next_text))
            beam.completion_tokens = num_tokens
            beam.total_completion_tokens += beam.completion_tokens
            beam.history.append("")
            beam.skipped_this_step = False
            is_completed = beam.completed
            state.skipped_beam_count += 1
            state.extended_tokens.append(num_tokens)
        else:
            # --- Regular beam: use generation result ---
            gen_result = state.gen_results[counter]
            counter += 1
            is_completed = beam_is_completed(
                gen_result, state.prompt_token_length + gen_result.completion_tokens[0]
            )
            if state.is_last:
                current_text = gen_result.next_texts[0]
                future_texts = []
            else:
                current_text, future_texts, _ = split_string_by_separator(
                    gen_result.next_texts[0], config.stop
                )
            beam.future_texts = future_texts
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            num_tokens = len(tokenizer.encode(current_text))
            beam.completion_tokens = num_tokens
            beam.total_completion_tokens += beam.completion_tokens
            beam.current_text += current_text
            beam.history.append(gen_result.next_texts[0])

        # --- Common completion check ---
        if is_completed:
            if not beam.completed:
                beam.completed = True
                beam.completion_time = state.total_gen_latency + state.total_ver_latency
            if not beam.future_texts:
                state.completed_beams.append(beam)

        # --- Build scoring batch ---
        if len(beam.all_scores) >= i + 1 and not state.is_last:
            # Beam already has enough scores — skip verification
            state.verified_beam_count += 1
            skipped_beam_context.append((beam.prompt, beam.current_text, beam.all_scores))
        elif beam.future_texts and beam.future_texts[0][1]:
            # Lookahead: include next step text for scoring
            prompts.append(beam.prompt)
            completions.append([beam.current_text + beam.future_texts[0][0]])
            beam_prev_scores.append(beam.all_scores)
        else:
            prompts.append(beam.prompt)
            completions.append([beam.current_text])
            beam_prev_scores.append(beam.all_scores)

    return ScoringBatch(
        prompts=prompts,
        completions=completions,
        prev_scores=beam_prev_scores,
        skipped_beam_context=skipped_beam_context,
    )


def _score_and_assign(
    state: BeamSearchState,
    verifier: VerifierVLLMModelWrapper,
    tokenizer,
    batch: ScoringBatch,
):
    """Call verifier, then assign scores and compute aggregated scores."""
    config = state.search_config
    i = state.iteration

    # --- Call verifier ---
    if batch.prompts:
        scores, verifier_time = score_beam(
            verifier, batch.prompts, batch.completions, tokenizer,
            prev_scores=batch.prev_scores,
            skipped_beam_context=batch.skipped_beam_context if batch.skipped_beam_context else None,
        )
        state.total_ver_latency += verifier_time
    else:
        scores = []

    # --- Assign scores to beams ---
    agg_index = i + 1 if not state.is_last else max([len(s) for s in scores])
    counter = 0
    state.agg_scores = []

    for beam in state.active_beams:
        if state.is_last or len(beam.all_scores) < agg_index:
            score = scores[counter]
            state.agg_scores.append(
                aggregate_scores(score[0][:agg_index], config.agg_strategy)
            )
            beam.all_scores = score[0]
            counter += 1
        else:
            state.agg_scores.append(
                aggregate_scores(beam.all_scores[:agg_index], config.agg_strategy)
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
    num_steps = [b.current_text.count("\n\n") for b in unpruned]
    score_lengths = [len(b.all_scores) for b in unpruned]
    stop_reasons = [b.stop_reasons[0] if b.stop_reasons else "?" for b in unpruned]

    logger.info(
        f"{'—' * 80}\n"
        f"iter {i}: completed={len(state.completed_beams)} "
        f"skipped={state.skipped_beam_count} extended={state.extended_beam_count} "
        f"verified={state.verified_beam_count} "
        f"latency={state.total_gen_latency + state.total_ver_latency:.2f}s "
        f"score_lens={score_lengths} steps={num_steps} stops={stop_reasons}"
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
            n_future = len(beam.future_texts) if beam.future_texts else 0
            future_tokens = (
                sum(len(tokenizer.encode(ft[0])) for ft in beam.future_texts)
                if beam.future_texts else 0
            )
            n_scores = len(beam.all_scores)
            last_score = f"{beam.all_scores[-1]:.4f}" if beam.all_scores else "—"
            flags = (
                f"{'PRUNED ' if beam.pruned else ''}"
                f"{'DONE ' if beam.completed else ''}"
            ).strip()
            logger.debug(
                f"  beam {beam.beam_id} "
                f"(parent={beam.parent_id}, born@{beam.born_at_iteration}) "
                f"ct_tok={ct_tokens} prompt_tok={prompt_tokens} "
                f"future={n_future}({future_tokens}tok) "
                f"scores={n_scores} last={last_score} "
                f"stop={beam.stop_reasons[0] if beam.stop_reasons else '?'} "
                f"{flags}"
            )

    # Warn on step count mismatch
    for x, beam in enumerate(unpruned):
        if num_steps[x] != i + 1:
            logger.warning(
                f"Beam {beam.beam_id} has {num_steps[x]} steps, expected {i + 1}. "
                f"stop_reasons={beam.stop_reasons}"
            )


def _finalize(state: BeamSearchState):
    """Post-loop: compute final metrics, sort completed beams."""
    config = state.search_config

    state.total_tokens += sum(b.completion_tokens for b in state.completed_beams)
    if state.n_gen_latency == 0:
        state.n_completion_tokens = state.total_tokens
        state.n_gen_latency = state.total_gen_latency
        state.n_ver_latency = state.total_ver_latency

    if config.sort_completed:
        state.completed_beams = sorted(
            state.completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
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

    logger.info("Starting beam search iterations")
    for i in tqdm(range(search_config.num_iterations), desc="Beam search iterations"):
        state.iteration = i
        state.is_last = (i == search_config.num_iterations - 1)
        state.extended_tokens = []

        _filter_active(state)
        _check_n_completion(state)
        _duplicate_beams(state, tokenizer)
        _consume_future_texts(state, tokenizer)
        _generate(state, generator, tokenizer)
        scoring_batch = _process_results(state, tokenizer)

        state.extended_tokens_list.append(state.extended_tokens)

        _score_and_assign(state, verifier, tokenizer, scoring_batch)

        if _filter_completed_and_prune(state):
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
            aggregate_scores(b.all_scores, search_config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["pred"].append(pred)
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["completion_tokens"].append([b.completion_tokens for b in beams])
        results["completion_time"].append([b.completion_time for b in beams])
        results["effective_num_tokens"].append([b.total_completion_tokens for b in beams])

    return results
