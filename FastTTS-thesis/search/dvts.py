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
    _log_iteration,
    _finalize,
    package_results,
)
from search.utils import aggregate_scores, truncate_sentence_by_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DVTS-specific: subtree-aware duplication
# ---------------------------------------------------------------------------

def _dvts_duplicate_beams(state: SearchState, tokenizer, subtree_beams: List[List[Beam]]):
    """Expand active beams to n via duplication, rebuilding subtree structure.

    Returns the updated subtree_beams list-of-lists.
    """
    config = state.search_config
    active = state.active_beams
    i = state.iteration

    if len(active) == config.n:
        return subtree_beams

    repeats = config.n // len(active)
    remainder = config.n % len(active)
    new_subtree_beams = []

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
        new_subtree_beams.append([beam] + duplicates)

    state.active_beams = [b for subtree in new_subtree_beams for b in subtree]

    assert len(state.active_beams) == config.n, (
        f"Expected {config.n} active beams, got {len(state.active_beams)}"
    )

    return new_subtree_beams


# ---------------------------------------------------------------------------
# DVTS-specific: per-subtree pruning
# ---------------------------------------------------------------------------

def _dvts_filter_completed_and_prune(state: SearchState, subtree_beams: List[List[Beam]]):
    """Remove completed beams, keep best per subtree.

    Returns updated subtree_beams.
    """
    config = state.search_config

    # Remove completed beams from each subtree
    subtree_beams = [
        [b for b in subtree if not b.completed]
        for subtree in subtree_beams
    ]
    state.active_beams = [b for b in state.active_beams if not b.completed]

    if len(state.active_beams) == 0:
        return subtree_beams

    # For each non-empty subtree, keep only the best beam
    for subtree in subtree_beams:
        if subtree:
            subtree_scores = [
                aggregate_scores(beam.scores, config.agg_strategy)
                for beam in subtree
            ]
            best_idx = int(np.argmax(subtree_scores))
            for idx, beam in enumerate(subtree):
                if idx != best_idx:
                    beam.pruned = True

    return subtree_beams


# ---------------------------------------------------------------------------
# Main DVTS loop
# ---------------------------------------------------------------------------

def _dvts_search(
    batch_of_prompts: List[str],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> tuple:
    """DVTS implementation with subtree-based beam selection."""

    tokenizer = generator.get_tokenizer()
    state = _init_state(batch_of_prompts, search_config, tokenizer)
    state.generator_max_model_len = generator.config.generator_vllm_config.get("max_model_len", 4096)

    beam_width = search_config.beam_width
    # Initialize subtrees: partition n beams into groups of beam_width
    subtree_beams = [
        state.all_beams[i * beam_width:(i + 1) * beam_width]
        for i in range(search_config.n // beam_width)
    ]

    logger.info("Starting DVTS search iterations")
    for i in tqdm(range(search_config.num_iterations), desc="DVTS search iterations"):
        state.iteration = i
        state.is_last = (i == search_config.num_iterations - 1)

        _filter_active(state)
        subtree_beams = _dvts_duplicate_beams(state, tokenizer, subtree_beams)
        _prepare_step_source(state, tokenizer)
        _generate(state, generator, tokenizer)
        scoring_batch = _process_results(state, tokenizer)

        _score_and_assign(state, verifier, tokenizer, scoring_batch)

        subtree_beams = _dvts_filter_completed_and_prune(state, subtree_beams)
        reached_n = _check_n_completion(state)
        _log_iteration(state, tokenizer)

        if reached_n:
            break

    return _finalize(state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dvts_search(
    examples: Dict[str, Any],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> Dict[str, Any]:
    """DVTS (Diverse Tree Search) for a batch of examples."""
    problems = examples["problem"]
    assert len(problems) == 1, "batch_of_prompts should be a list of length 1 for now"

    nvtx.range_push("Total")
    (
        completed_beams, total_generator_latency_s, total_verifier_latency_s,
        total_num_tokens, n_completion_tokens,
    ) = _dvts_search(problems, search_config, generator, verifier)
    nvtx.range_pop()

    return package_results(
        problems, completed_beams,
        total_generator_latency_s, total_verifier_latency_s,
        total_num_tokens, n_completion_tokens,
        search_config,
    )
