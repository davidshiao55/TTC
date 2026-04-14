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
from vllm import SamplingParams
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
# Main VG search loop
# ---------------------------------------------------------------------------

def _vg_search(
    batch_of_prompts: List[str],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> tuple:
    """VG search with two-stage max_tokens approach."""

    tokenizer = generator.get_tokenizer()
    state = _init_state(batch_of_prompts, search_config, tokenizer)
    state.generator_max_model_len = generator.config.generator_vllm_config.get("max_model_len", 4096)

    # VG-specific: 3-stage sampling params
    stage1_max_tokens = min(64, search_config.max_tokens // 4)

    stage1_sampling_params = SamplingParams(
        temperature=search_config.temperature,
        max_tokens=stage1_max_tokens,
        top_p=search_config.top_p,
        stop=[search_config.stop],
        include_stop_str_in_output=True,
        n=1,
    )
    stage2_sampling_params = SamplingParams(
        temperature=search_config.temperature,
        max_tokens=search_config.max_tokens,
        top_p=search_config.top_p,
        stop=[search_config.stop],
        include_stop_str_in_output=True,
        n=1,
    )
    # final_sampling_params is already created by _init_state (no stop string)

    logger.info("Starting VG search iterations")
    for i in tqdm(range(search_config.num_iterations), desc="VG search iterations"):
        state.iteration = i
        state.is_last = (i == search_config.num_iterations - 1)

        # Select sampling params based on stage
        if i < 3:
            current_sampling_params = stage1_sampling_params
        elif state.is_last:
            current_sampling_params = state.final_sampling_params
        else:
            current_sampling_params = stage2_sampling_params

        _filter_active(state)
        _duplicate_beams(state, tokenizer)
        _prepare_step_source(state, tokenizer)
        _generate(state, generator, tokenizer, sampling_params_override=current_sampling_params)
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

def vg_search(
    examples: Dict[str, Any],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> Dict[str, Any]:
    """VG search for a batch of examples."""
    problems = examples["problem"]
    assert len(problems) == 1, "batch_of_prompts should be a list of length 1 for now"

    nvtx.range_push("Total")
    (
        completed_beams, total_generator_latency_s, total_verifier_latency_s,
        total_num_tokens, n_completion_tokens,
    ) = _vg_search(problems, search_config, generator, verifier)
    nvtx.range_pop()

    return package_results(
        problems, completed_beams,
        total_generator_latency_s, total_verifier_latency_s,
        total_num_tokens, n_completion_tokens,
        search_config,
    )
