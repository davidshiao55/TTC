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
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
import torch.cuda.nvtx as nvtx

from config import SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.beam import Beam, StepChunk, _next_beam_id
from search.common import (
    SearchState,
    _init_state,
    _filter_active,
    _check_n_completion,
    _prepare_step_source,
    _generate,
    _process_results,
    _score_and_assign,
    _filter_completed_and_prune,
    _log_iteration,
    _finalize,
    package_results,
)
from search.utils import aggregate_scores, truncate_sentence_by_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dynamic branching-specific: score-proportional duplication
# ---------------------------------------------------------------------------

def _dynamic_branching_duplicate_beams(state: SearchState, tokenizer):
    """Duplicate beams proportional to their scores."""
    config = state.search_config
    active = state.active_beams
    i = state.iteration

    if len(active) == config.n:
        return

    # Compute scores for proportional allocation
    if i == 0 or not state.agg_scores:
        beam_scores = np.ones(len(active))
    else:
        beam_scores = np.array([
            aggregate_scores(b.scores, config.agg_strategy)
            for b in active
        ])
        beam_scores = beam_scores - np.min(beam_scores) + 1e-6

    # Proportional allocation with floor + remainder
    total_score = np.sum(beam_scores)
    target_beams = config.n
    duplication_ratios = beam_scores / total_score * target_beams
    duplication_counts = np.floor(duplication_ratios).astype(int)

    remaining_beams = target_beams - np.sum(duplication_counts)
    if remaining_beams > 0:
        fractional_parts = duplication_ratios - duplication_counts
        top_indices = np.argsort(fractional_parts)[::-1][:int(remaining_beams)]
        for idx in top_indices:
            duplication_counts[idx] += 1

    # Build duplicated beam list
    final_beams = []
    for beam_idx, beam in enumerate(active):
        num_duplicates = duplication_counts[beam_idx]
        if num_duplicates > 0:
            final_beams.append(beam)
            for _ in range(num_duplicates - 1):
                duplicate = copy.deepcopy(beam)
                duplicate.beam_id = _next_beam_id()
                duplicate.parent_id = beam.beam_id
                duplicate.born_at_iteration = i
                if beam.pending_steps:
                    first_text = truncate_sentence_by_tokens(
                        beam.pending_steps[0].text, tokenizer
                    )
                    duplicate.pending_steps = [
                        StepChunk(text=first_text, is_complete_step=False, terminal=False)
                    ]
                    duplicate.scores = beam.scores[:i]
                    duplicate.step_hashes = beam.step_hashes[:i]
                final_beams.append(duplicate)

    state.active_beams = final_beams

    assert len(state.active_beams) == config.n, (
        f"Expected {config.n} active beams, got {len(state.active_beams)}"
    )


# ---------------------------------------------------------------------------
# Main dynamic branching loop
# ---------------------------------------------------------------------------

def _dynamic_branching_search(
    batch_of_prompts: List[str],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> tuple:
    """Dynamic branching search with score-based beam duplication."""

    tokenizer = generator.get_tokenizer()
    state = _init_state(batch_of_prompts, search_config, tokenizer)
    state.generator_max_model_len = generator.config.generator_vllm_config.get("max_model_len", 4096)

    logger.info("Starting dynamic branching search iterations")
    for i in tqdm(range(search_config.num_iterations), desc="Dynamic branching search iterations"):
        state.iteration = i
        state.is_last = (i == search_config.num_iterations - 1)

        _filter_active(state)
        _dynamic_branching_duplicate_beams(state, tokenizer)
        _prepare_step_source(state, tokenizer)
        _generate(state, generator, tokenizer)
        scoring_batch = _process_results(state, tokenizer)

        _score_and_assign(state, verifier, tokenizer, scoring_batch)

        _filter_completed_and_prune(state)
        reached_n = _check_n_completion(state)
        _log_iteration(state, tokenizer)

        if reached_n:
            break

    return _finalize(state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dynamic_branching_search(
    examples: Dict[str, Any],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> Dict[str, Any]:
    """Dynamic branching search for a batch of examples."""
    problems = examples["problem"]
    assert len(problems) == 1, "batch_of_prompts should be a list of length 1 for now"

    nvtx.range_push("Total")
    (
        completed_beams, total_generator_latency_s, total_verifier_latency_s,
        total_num_tokens, n_completion_tokens,
    ) = _dynamic_branching_search(problems, search_config, generator, verifier)
    nvtx.range_pop()

    return package_results(
        problems, completed_beams,
        total_generator_latency_s, total_verifier_latency_s,
        total_num_tokens, n_completion_tokens,
        search_config,
    )
