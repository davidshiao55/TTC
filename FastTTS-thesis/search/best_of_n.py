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

import asyncio
import copy
import logging
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional
import random

import numpy as np
from tqdm import tqdm
from vllm import SamplingParams
import torch.cuda.nvtx as nvtx

from config import SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.beam import Beam
from search.utils import build_conversation, aggregate_scores, split_string_by_separator, truncate_sentence_by_tokens

logger = logging.getLogger(__name__)


def beam_is_completed(beam: Beam, token_length: int) -> bool:
    if token_length >= 4096:
        logger.warning(f"Token length {token_length} exceeds max length 4096")
    return beam.stop_reasons[0] == "EOS" or beam.stop_reasons[0] == "length" or beam.next_texts[0] == "" or token_length >= 4096 or (beam.stop_reasons[0] != "\n\n")


def generate_beam(
    templated_convs: List[str],
    lookahead_steps: int,
    generator: GeneratorVLLMModelWrapper,
    sampling_params: SamplingParams,
    beam_width: int = 1,
    tokenizer = None,
):
    """Optimized async version of generate_beam."""
        
    # Pre-allocate results array for better memory efficiency
    num_convs = len(templated_convs)
    gen_results = [
        {
            "index": i,
            "initial_prompt": text,
            "first_step_text": "",
            "lookahead_text": "",
            "stop_reason": None,
            "first_step_stop_reason": None,
            "completion_tokens": 0,
        }
        for i, text in enumerate(templated_convs)
        for _ in range(beam_width)
    ]

    # Pre-compute sampling parameters for lookahead
    gen_sampling_params = copy.deepcopy(sampling_params)
    lookahead_sampling_params = copy.deepcopy(sampling_params)
    lookahead_sampling_params.temperature = 0.0  # Greedy for lookahead steps
    
    for i in range(lookahead_steps + 1):
        # Use appropriate sampling parameters
        current_sampling_params = lookahead_sampling_params if i == 1 else gen_sampling_params
            
        # Get all generations that did not finish with EOS
        current_gen = [
            gen_result
            for gen_result in gen_results
            if gen_result["stop_reason"] != "EOS"
        ]
        
        if not current_gen:
            break
            
        gen_prompts = [
            gen_result["initial_prompt"] + gen_result["lookahead_text"]
            for gen_result in current_gen
        ]
        
        # Prefix-aware scheduling: assign priorities if enabled
        prefix_priorities = None
        # if getattr(generator.config, 'prefix_aware_scheduling', False) and len(gen_prompts) >= 256:
        #     tokenized_prompts = [tokenizer.tokenize(p) for p in gen_prompts]
        #     from search.utils import assign_prefix_priorities
        #     prefix_priorities = assign_prefix_priorities(tokenized_prompts)
            # prefix_priorities = [-p for p in prefix_priorities]
        
        # NVTX profiling for generation
        start_time = time.time()
        nvtx.range_push("generate")
        llm_outputs = generator.generate(gen_prompts, sampling_params=current_sampling_params, priority=prefix_priorities)
        nvtx.range_pop()
        end_time = time.time()

        # Process results more efficiently
        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output.outputs[0].text
            if i == 0:
                gen_result["first_step_text"] = gen_text
                gen_result["first_step_stop_reason"] = output.outputs[0].stop_reason
                if gen_result["first_step_stop_reason"] is None:
                    gen_result["first_step_stop_reason"] = "EOS"
                gen_result["completion_tokens"] = len(output.outputs[0].token_ids) 

            gen_result["lookahead_text"] = gen_result["lookahead_text"] + gen_text
            gen_result["stop_reason"] = output.outputs[0].stop_reason
            if gen_result["stop_reason"] is None:
                gen_result["stop_reason"] = "EOS"
        
    # Convert to beam format more efficiently
    outputs = []
    counter = 0
    for i in range(num_convs):
        next_texts = []
        stop_reasons = []
        lookahead_texts = []
        completion_tokens = []
        for _ in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result["first_step_text"])
            lookahead_texts.append(gen_result["lookahead_text"])
            stop_reasons.append(gen_result["first_step_stop_reason"])
            completion_tokens.append(gen_result["completion_tokens"])
            counter += 1

        beam_result = Beam(
            prompt=templated_convs[i],
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
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


def _best_of_n_search(
    batch_of_prompts: List[str], 
    search_config: SearchConfig, 
    generator: GeneratorVLLMModelWrapper, 
    verifier: VerifierVLLMModelWrapper,
) -> List[Beam]:
    """Best of N search implementation without verification and selection."""
    
    # Pre-compute sampling parameters
    sampling_params = SamplingParams(
        temperature=search_config.temperature,
        # max_tokens=search_config.max_tokens,
        max_tokens=2048,
        top_p=search_config.top_p,
        # stop=[search_config.stop],
        include_stop_str_in_output=True,
        n=search_config.n,  # Generate n completions directly
    )

    # Initialize beams for each prompt
    beams: List[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(search_config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
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
                )
            )

    completed_beams: List[Beam] = []
    total_generator_latency_s = 0
    total_verifier_latency_s = 0
    n_generator_latency_s = 0
    n_verifier_latency_s = 0
    total_num_tokens = 0
    n_completion_tokens = 0
    extended_tokens_list = []
    
    # Pre-compute tokenizer to avoid repeated calls
    tokenizer = generator.get_tokenizer()
    
    # Get the prompt token number length
    conv = build_conversation(batch_of_prompts[0], "", search_config.system_prompt)
    conv_str = tokenizer.apply_chat_template(conv, 
                                             add_generation_prompt=True, 
                                             continue_final_message=False, tokenize=True)
    prompt_token_length = len(conv_str)

    logger.info("Starting best_of_n search")
    
    # For best_of_n, we generate all completions in a single step
    for prompt_idx, prompt in enumerate(batch_of_prompts):
        # Build conversation for this prompt
        conv = build_conversation(prompt, "", search_config.system_prompt)
        
        # Apply chat template
        if hasattr(search_config, 'custom_chat_template') and search_config.custom_chat_template is not None:
            tokenizer.chat_template = search_config.custom_chat_template
        templated_conv = tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=True,
            continue_final_message=False,
            tokenize=False,
        )
        
        # Generate n completions for this prompt
        start_time = time.time()
        nvtx.range_push("generate")
        llm_outputs = generator.generate([templated_conv], sampling_params=sampling_params)
        nvtx.range_pop()
        end_time = time.time()
        
        total_generator_latency_s += (end_time - start_time)
        
        # Process the generated completions
        for i, output in enumerate(llm_outputs[0].outputs):
            completion_text = output.text
            completion_tokens = len(output.token_ids)
            
            # Create a beam for this completion
            beam = Beam(
                prompt=prompt,
                index=i,
                current_text=completion_text,
                next_texts=[completion_text],
                lookahead_texts=[completion_text],
                pruned=False,
                completed=True,
                stop_reasons=[output.stop_reason or "EOS"],
                history=[completion_text],
                best_scores=[0.0],  # Will be updated after verification
                all_scores=[],      # Will be updated after verification
                previous_text=None,
                completion_tokens=completion_tokens,
                total_completion_tokens=completion_tokens,
                completion_time=total_generator_latency_s,
            )
            
            completed_beams.append(beam)
            total_num_tokens += completion_tokens
    
    # Call verifier once at the end to score all completions
    if completed_beams:
        prompts = [beam.prompt for beam in completed_beams]
        completions = [[beam.current_text] for beam in completed_beams]
        
        # Score with verifier
        verifier_time = time.time()
        scores = verifier.score(prompts, completions)
        verifier_time = time.time() - verifier_time
        total_verifier_latency_s += verifier_time
        
        # Update beams with their scores
        for beam, score in zip(completed_beams, scores):
            beam.all_scores = score[0]
            beam.best_scores = [aggregate_scores(score[0], search_config.agg_strategy)]
        
        logger.info(f"Scored {len(completed_beams)} completions with verifier in {verifier_time:.2f}s")
    
    n_completion_tokens = total_num_tokens
    n_generator_latency_s = total_generator_latency_s
    
    # Sort completed beams if requested
    if search_config.sort_completed:
        # Since there's no scoring, we can sort by completion time or keep original order
        completed_beams = sorted(completed_beams, key=lambda b: b.completion_time)

    # Return values
    return completed_beams, total_generator_latency_s, total_verifier_latency_s, n_generator_latency_s, n_verifier_latency_s, total_num_tokens, n_completion_tokens, extended_tokens_list


def best_of_n_search(
    examples: Dict[str, Any], 
    search_config: SearchConfig, 
    generator: GeneratorVLLMModelWrapper, 
    verifier: VerifierVLLMModelWrapper, 
) -> Dict[str, Any]:
    """Best of N search for a batch of examples."""
    problems = examples["problem"]
    assert len(problems) == 1, "batch_of_prompts should be a list of length 1 for now"
    
    # NVTX profiling for entire best_of_n search
    nvtx.range_push("Total")
    completed_beams, total_generator_latency_s, total_verifier_latency_s, n_generator_latency_s, n_verifier_latency_s, total_num_tokens, n_completion_tokens, extended_tokens_list = _best_of_n_search(problems, search_config, generator, verifier)
    nvtx.range_pop()

    # Group results by prompt
    grouped_results = defaultdict(list)
    for results in completed_beams:
        grouped_results[results.prompt].append(results)

    # Pre-allocate results dictionary
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
        
        # For best_of_n, we can choose the best completion based on scores
        # or implement other heuristics
        if beams and beams[0].all_scores:
            # Use scores to select the best completion
            agg_scores = [
                aggregate_scores(b.all_scores, search_config.agg_strategy) for b in beams
            ]
            best_idx = np.argmax(agg_scores)
            pred = completions[best_idx]
            logger.info(f"Selected completion {best_idx} with score {agg_scores[best_idx]:.4f}")
        else:
            # Fallback: take the first completion if no scores available
            pred = completions[0]
            logger.info("No scores available, using first completion as fallback")
        
        results["pred"].append(pred)
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["completion_tokens"].append([b.completion_tokens for b in beams])
        results["completion_time"].append([b.completion_time for b in beams])
        results["effective_num_tokens"].append([b.total_completion_tokens for b in beams])

    return results
