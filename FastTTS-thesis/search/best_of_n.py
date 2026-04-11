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
import time
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
from vllm import SamplingParams
import torch.cuda.nvtx as nvtx

from config import SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.beam import Beam
from search.common import score_beam
from search.utils import build_conversation, aggregate_scores

logger = logging.getLogger(__name__)


def _best_of_n_search(
    batch_of_prompts: List[str],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> tuple:
    """Best of N search: generate n completions in a single call, score at end."""

    tokenizer = generator.get_tokenizer()
    max_model_len = generator.config.generator_vllm_config.get("max_model_len", 4096)

    sampling_params = SamplingParams(
        temperature=search_config.temperature,
        max_tokens=max_model_len,
        top_p=search_config.top_p,
        include_stop_str_in_output=True,
        n=search_config.n,
    )

    completed_beams: List[Beam] = []
    total_generator_latency_s = 0.0
    total_verifier_latency_s = 0.0
    total_num_tokens = 0

    for prompt in batch_of_prompts:
        conv = build_conversation(prompt, "", search_config.system_prompt)

        if hasattr(search_config, 'custom_chat_template') and search_config.custom_chat_template is not None:
            tokenizer.chat_template = search_config.custom_chat_template
        templated_conv = tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=True,
            continue_final_message=False,
            tokenize=False,
        )

        start_time = time.time()
        nvtx.range_push("generate")
        llm_outputs = generator.generate([templated_conv], sampling_params=sampling_params)
        nvtx.range_pop()
        end_time = time.time()

        total_generator_latency_s += (end_time - start_time)

        for output in llm_outputs[0].outputs:
            completion_text = output.text
            completion_tokens = len(output.token_ids)

            beam = Beam(
                prompt=prompt,
                current_text=completion_text,
                gen_text=[completion_text],
                stop_reasons=[output.stop_reason],
                finish_reasons=[output.finish_reason],
                scores=[],
                gen_history=[completion_text],
                completed=True,
                step_tokens=completion_tokens,
                total_tokens_generated=completion_tokens,
                time_to_complete=total_generator_latency_s,
            )

            completed_beams.append(beam)
            total_num_tokens += completion_tokens

    # Score all completions with verifier
    if completed_beams:
        prompts = [beam.prompt for beam in completed_beams]
        completions = [[beam.current_text] for beam in completed_beams]

        scores, verifier_time = score_beam(verifier, prompts, completions)
        total_verifier_latency_s += verifier_time

        for beam, score in zip(completed_beams, scores):
            beam.scores = score[0]

        logger.info(f"Scored {len(completed_beams)} completions in {verifier_time:.2f}s")

    # Always sort by aggregate score (descending)
    completed_beams = sorted(
        completed_beams,
        key=lambda b: aggregate_scores(b.scores, search_config.agg_strategy),
        reverse=True,
    )

    return (
        completed_beams,
        total_generator_latency_s,
        total_verifier_latency_s,
        total_generator_latency_s,  # n_generator_latency = total (single-shot)
        total_verifier_latency_s,   # n_verifier_latency = total
        total_num_tokens,
        total_num_tokens,           # n_completion_tokens = total
        [],                         # extended_tokens_list (not applicable)
    )


def best_of_n_search(
    examples: Dict[str, Any],
    search_config: SearchConfig,
    generator: GeneratorVLLMModelWrapper,
    verifier: VerifierVLLMModelWrapper,
) -> Dict[str, Any]:
    """Best of N search for a batch of examples."""
    problems = examples["problem"]
    assert len(problems) == 1, "batch_of_prompts should be a list of length 1 for now"

    nvtx.range_push("Total")
    (
        completed_beams, total_generator_latency_s, total_verifier_latency_s,
        n_generator_latency_s, n_verifier_latency_s,
        total_num_tokens, n_completion_tokens, extended_tokens_list,
    ) = _best_of_n_search(problems, search_config, generator, verifier)
    nvtx.range_pop()

    # Package results
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
        if beams and beams[0].scores:
            agg_scores = [
                aggregate_scores(b.scores, search_config.agg_strategy) for b in beams
            ]
            best_idx = int(np.argmax(agg_scores))
            pred = completions[best_idx]
        else:
            pred = completions[0] if completions else ""

        results["pred"].append(pred)
        results["completions"].append(completions)
        results["scores"].append([b.scores for b in beams])
        results["completion_tokens"].append([b.step_tokens for b in beams])
        results["completion_time"].append([b.time_to_complete for b in beams])
        results["effective_num_tokens"].append([b.total_tokens_generated for b in beams])

    return results
