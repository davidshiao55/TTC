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

import logging
from typing import List, Dict, Any

from tqdm import tqdm
import torch.cuda.nvtx as nvtx

from config import SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.common import (
    SearchState,
    _init_state,
    _filter_active,
    _check_n_completion,
    _duplicate_beams,
    _prepare_step_source,
    _generate,
    _process_results,
    _score_and_assign,
    _filter_completed_and_prune,
    _log_iteration,
    _finalize,
    package_results,
)

logger = logging.getLogger(__name__)


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

        _filter_completed_and_prune(state)
        reached_n = _check_n_completion(state)
        _log_iteration(state, tokenizer)

        if reached_n:
            break

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

    return package_results(
        problems, completed_beams,
        total_generator_latency_s, total_verifier_latency_s,
        n_generator_latency_s, n_verifier_latency_s,
        total_num_tokens, n_completion_tokens, extended_tokens_list,
        search_config,
    )
