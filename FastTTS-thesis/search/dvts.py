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


def score_beam(verifier: VerifierVLLMModelWrapper, prompts: List[str], completions: List[List[str]], tokenizer=None, prev_scores=None, skipped_beam_context=None):
    """Score a beam of completions."""
    # Prefix-aware scheduling: assign priorities if enabled
    prefix_priorities = None
    # if getattr(verifier.config, 'prefix_aware_scheduling', False):
    #     tokenized_prompts = [tokenizer.tokenize(p) for p in prompts]
    #     from search.utils import assign_prefix_priorities
    #     prefix_priorities = assign_prefix_priorities(tokenized_prompts)
    #     prefix_priorities = [-p for p in prefix_priorities]

    # Score with verifier
    verifier_time = time.time()
    scores = verifier.score(prompts, completions, priority=prefix_priorities,
                            prev_scores=prev_scores, skipped_beam_context=skipped_beam_context)
    verifier_time = time.time() - verifier_time
    return scores, verifier_time


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


def _dvts_search(
    batch_of_prompts: List[str], 
    search_config: SearchConfig, 
    generator: GeneratorVLLMModelWrapper, 
    verifier: VerifierVLLMModelWrapper,
) -> List[Beam]:
    """DVTS (Diverse Tree Search) implementation with subtree-based beam selection."""
    
    # Pre-compute sampling parameters to avoid repeated creation
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

    # Initialize beams more efficiently
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
    total_generator_latency_s = 0  # Total generator latency for the entire search
    total_verifier_latency_s = 0   # Total verifier latency for the entire search
    n_generator_latency_s = 0      # Generator latency for collecting n completions
    n_verifier_latency_s = 0       # Verifier latency for collecting n completions
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

    beam_width = search_config.beam_width
    subtree_beams = [beams[i*beam_width:(i+1)*beam_width] for i in range(search_config.n // beam_width)]
    logger.info("Starting DVTS search iterations")
    for i in tqdm(range(search_config.num_iterations), desc="DVTS search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for subtree in subtree_beams for b in subtree if not b.pruned]
        logger.info(f"active_beams: {len(active_beams)}")
        # active_beams = sorted(active_beams, key=lambda b: aggregate_scores(b.all_scores, search_config.agg_strategy))
        if (len(completed_beams) >= search_config.n or len(active_beams) == 0) and n_generator_latency_s == 0:
            n_generator_latency_s = total_generator_latency_s
            n_verifier_latency_s = total_verifier_latency_s
            n_completion_tokens = total_num_tokens
            logger.info(f"Reached target n: {len(completed_beams)} completed beams after {n_generator_latency_s + n_verifier_latency_s:.2f}s, {n_completion_tokens} total tokens")
        
        # Optimize beam duplication logic
        if len(active_beams) != search_config.n:
            repeats = (search_config.n // len(active_beams))
            
            subtree_beams = []
            remainder = search_config.n % len(active_beams)

            for x, beam in reversed(list(enumerate(active_beams))):
                repeats_for_this_beam = repeats + (1 if x < remainder else 0)
                duplicates = []
                for _ in range(repeats_for_this_beam-1):
                    duplicate = copy.deepcopy(beam)
                    if beam.future_texts:
                        first_text = truncate_sentence_by_tokens(beam.future_texts[0][0], tokenizer)
                        duplicate.future_texts = [(first_text, False)]
                        duplicate.all_scores = beam.all_scores[:i]
                    duplicates.append(duplicate)
                subtree_beams.append([beam] + duplicates)
            active_beams = [b for subtree in subtree_beams for b in subtree]
            
            # if not prefix-aware scheduling, we need to randomize the order of the beams
            if not getattr(generator.config, 'prefix_aware_scheduling', False):
                random.shuffle(active_beams)
            
            if len(active_beams) != search_config.n:
                raise ValueError(
                    f"Expected {search_config.n} active beams, but got {len(active_beams)}"
                )
        
        # For spec beam extension, we need to add the future texts to the current text
        extended_beams = 0
        extended_tokens = []
        for beam in active_beams:
            if len(beam.future_texts) > 0:
                extended_beams += 1
                next_text, is_finished_this_step = beam.future_texts[0]
                if i == search_config.num_iterations - 1:
                    while beam.future_texts:
                        next_text, _ = beam.future_texts.pop(0)
                        num_tokens = len(tokenizer.encode(next_text))
                        beam.completion_tokens = num_tokens
                        beam.total_completion_tokens += beam.completion_tokens
                        beam.current_text += next_text
                        extended_tokens.append(num_tokens)
                    beam.skipped_this_step = beam.completed
                elif is_finished_this_step:
                    beam.skipped_this_step = True
                else: 
                    num_tokens = len(tokenizer.encode(next_text))
                    beam.completion_tokens = num_tokens
                    beam.total_completion_tokens += beam.completion_tokens
                    beam.current_text += next_text
                    beam.future_texts.pop(0)
                    extended_tokens.append(num_tokens)

        # Use appropriate sampling parameters
        current_sampling_params = final_sampling_params if i == search_config.num_iterations - 1 else base_sampling_params

        # Build conversations more efficiently
        convs = [
            build_conversation(b.prompt, b.current_text, search_config.system_prompt)
            for b in active_beams if not b.skipped_this_step
        ]
        add_generation_prompt = i == 0
        continue_final_message = i > 0

        if convs:
            # Apply chat template once
            if hasattr(search_config, 'custom_chat_template') and search_config.custom_chat_template is not None:
                tokenizer.chat_template = search_config.custom_chat_template
            templated_convs = tokenizer.apply_chat_template(
                convs,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=continue_final_message,
                tokenize=False,
            )
            
            lookahead = 0 if i == search_config.num_iterations - 1 else search_config.lookahead
            
            gen_results, gen_time = generate_beam(
                    templated_convs, lookahead, generator, current_sampling_params, tokenizer=tokenizer
                )
            total_generator_latency_s += gen_time

        # Process generation results more efficiently
        prompts, completions, beam_prev_scores = [], [], []
        skipped_beam_context = []
        skipped_beams = 0
        verified_beams = 0
        counter = 0
        for beam in active_beams:
            # Handle skipped beams (using future texts)
            skipped_this_step = beam.skipped_this_step
            if skipped_this_step and i < search_config.num_iterations - 1:
                next_text, _ = beam.future_texts.pop(0)
                beam.current_text += next_text
                num_tokens = len(tokenizer.encode(next_text))
                beam.completion_tokens = num_tokens
                beam.total_completion_tokens += beam.completion_tokens
                beam.history.append("")
                beam.skipped_this_step = False
                is_completed = beam.completed
                skipped_beams += 1
                extended_tokens.append(num_tokens)
            else:
                # Handle regular beams (using generation results)
                gen_result = gen_results[counter]
                counter += 1
                is_completed = beam_is_completed(gen_result, prompt_token_length + gen_result.completion_tokens[0])
                if i == search_config.num_iterations - 1:
                    current_text = gen_result.next_texts[0]
                    future_texts = []
                else:
                    current_text, future_texts, _ = split_string_by_separator(
                        gen_result.next_texts[0], search_config.stop
                    )
                # current_text = gen_result.next_texts[0]
                # future_texts = []
                beam.future_texts = future_texts
                beam.next_texts = gen_result.next_texts
                beam.stop_reasons = gen_result.stop_reasons
                beam.lookahead_texts = gen_result.lookahead_texts
                num_tokens = len(tokenizer.encode(current_text))
                beam.completion_tokens = num_tokens
                beam.total_completion_tokens += beam.completion_tokens
                beam.current_text += current_text
                beam.history.append(gen_result.next_texts[0])

            # Common completion check
            if is_completed:
                if not beam.completed:
                    beam.completed = True
                    beam.completion_time = total_generator_latency_s + total_verifier_latency_s
                if not beam.future_texts:
                    completed_beams.append(beam)

            if len(beam.all_scores) >= i + 1 and i < search_config.num_iterations - 1:
                verified_beams += 1
                skipped_beam_context.append((beam.prompt, beam.current_text, beam.all_scores))
            elif beam.future_texts and beam.future_texts[0][1]:
                prompts.append(beam.prompt)
                completions.append([beam.current_text + beam.future_texts[0][0]])
                beam_prev_scores.append(beam.all_scores)
            else:
                prompts.append(beam.prompt)
                completions.append([beam.current_text])
                beam_prev_scores.append(beam.all_scores)

        extended_tokens_list.append(extended_tokens)

        if prompts:
            scores, verifier_time = score_beam(verifier, prompts, completions, tokenizer,
                                               prev_scores=beam_prev_scores,
                                               skipped_beam_context=skipped_beam_context if skipped_beam_context else None)
            total_verifier_latency_s += verifier_time
        else:
            scores = []

        # Aggregate scores more efficiently
        agg_index = i + 1 if i < search_config.num_iterations - 1 else max([len(s) for s in scores])
        counter = 0
        for beam in active_beams:
            if i == search_config.num_iterations - 1 or len(beam.all_scores) < agg_index:
                score = scores[counter]
                beam.all_scores = score[0]
                beam.best_scores = aggregate_scores(score[0][:agg_index], search_config.agg_strategy)
                counter += 1
                # assert len(score[0]) >= i + 1 or beam.completed, f"length of score: {i+1}, {len(score[0])} {beam.completed} {beam.current_text}"
            else:
                beam.best_scores = aggregate_scores(beam.all_scores[:agg_index], search_config.agg_strategy)
        assert counter == len(scores), f"counter: {counter}, len(scores): {len(scores)}"
        # agg_scores = [
        #     [aggregate_scores(s[:agg_index], search_config.agg_strategy) for s in score]
        #     for score in scores
        # ]

        # for beam, score in zip(active_beams, scores, strict=True):
        #     beam.all_scores = score[0]

        # Filter completed beams
        subtree_beams = [[b for b in subtree if not b.completed] for subtree in subtree_beams]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping 
        if len(active_beams) == 0:
            logger.info(f"Early exit: {len(active_beams)} active, {len(completed_beams)} completed")
            break
    
        # Group beams into subtrees and select the best from each    
        num_completed_subtrees = 0
        for subtree in subtree_beams:
            if subtree:
                best_in_subtree_idx = np.argmax([beam.best_scores for beam in subtree])
                for idx, beam in enumerate(subtree):
                    if idx != best_in_subtree_idx:
                        beam.pruned = True
            else:
                num_completed_subtrees += 1 
        # unprune num_completed_subtrees beams
        # top_indices = np.argsort(np.array([beam.best_scores for beam in pruned_beams]).flatten())[
        #     -num_completed_subtrees :]
        # for idx, beam in enumerate(pruned_beams):
        #     if idx in top_indices:
        #         beam.pruned = False
        #         subtree_beams.append([beam])
            
        
        num_steps = [beam.current_text.count("\n\n") for beam in active_beams if not beam.pruned]
        agg_scores_length = [len(beam.all_scores) for beam in active_beams if not beam.pruned]
        stop_reasons = [beam.stop_reasons[0] for beam in active_beams if not beam.pruned]
        logger.info(f"-" * 100)
        logger.info(f"Iteration {i} completed beams: {len(completed_beams)}, skipped beams: {skipped_beams}, extended beams: {extended_beams}, verifier beams: {verified_beams}, total latency: {total_generator_latency_s + total_verifier_latency_s:.2f}s, length of agg_scores: {agg_scores_length}, num_steps: {num_steps}, stop reasons: {stop_reasons}")
        for x, beam in enumerate([b for b in active_beams if not b.pruned]):
            if num_steps[x] != i + 1:
                logger.warning(f"Beam {x} has {num_steps[x]} steps, expected {i + 1}")
                logger.warning(f"Beam {x} current text: {beam.current_text}")
                logger.warning(f"Beam {x} history: {beam.history}, stop reasons: {beam.stop_reasons}")
    total_num_tokens += sum([b.completion_tokens for b in completed_beams])
    n_completion_tokens = total_num_tokens if n_generator_latency_s == 0 else n_completion_tokens
    n_generator_latency_s = total_generator_latency_s if n_generator_latency_s == 0 else n_generator_latency_s
    n_verifier_latency_s = total_verifier_latency_s if n_verifier_latency_s == 0 else n_verifier_latency_s
    # Sort completed beams if requested
    if search_config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, search_config.agg_strategy),
            reverse=True,
        )

    # Return values unrelated to VLLM metrics
    return completed_beams, total_generator_latency_s, total_verifier_latency_s, n_generator_latency_s, n_verifier_latency_s, total_num_tokens, n_completion_tokens, extended_tokens_list


def dvts_search(
    examples: Dict[str, Any], 
    search_config: SearchConfig, 
    generator: GeneratorVLLMModelWrapper, 
    verifier: VerifierVLLMModelWrapper, 
) -> Dict[str, Any]:
    """DVTS (Diverse Tree Search) for a batch of examples."""
    problems = examples["problem"]
    assert len(problems) == 1, "batch_of_prompts should be a list of length 1 for now"
    
    # NVTX profiling for entire DVTS search
    nvtx.range_push("Total")
    completed_beams, total_generator_latency_s, total_verifier_latency_s, n_generator_latency_s, n_verifier_latency_s, total_num_tokens, n_completion_tokens, extended_tokens_list = _dvts_search(problems, search_config, generator, verifier)
    nvtx.range_pop()

    # Group results by prompt more efficiently
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
        "total_num_tokens": total_num_tokens,  # total number of tokens generated
        "n_completion_tokens": n_completion_tokens,  # number of tokens generated for n completions
        "total_generator_latency_s": total_generator_latency_s,  # total generator latency for the entire search
        "total_verifier_latency_s": total_verifier_latency_s,    # total verifier latency for the entire search
        "n_generator_latency_s": n_generator_latency_s,          # generator latency for collecting n completions
        "n_verifier_latency_s": n_verifier_latency_s,            # verifier latency for collecting n completions
        "completion_time": [],
        "vllm_metrics": {}, # No VLLM metrics here
        "vllm_metrics_summary": {}, # No VLLM metrics summary here
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
