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
import multiprocessing as mp
if mp.get_start_method() != 'spawn':
    mp.set_start_method('spawn', force=True)
import os
# Force V0 since reward models are not supported in V1
# os.environ["VLLM_USE_V1"] = "3"

"""
Example usage of FastTTS for test time search.

This script demonstrates how to use FastTTS to perform beam search
with a generator and verifier model running in separate processes.
"""

import asyncio
import logging
from datasets import load_dataset
from fasttts import create_fasttts
from fasttts import FastTTSConfig, SearchConfig
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def infer_example():
    """Example using async interface."""
    logger.info("Running async example...")
    
    # Create FastTTS instance
    fasttts = create_fasttts(
        generator_vllm_config={
            "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "gpu_memory_utilization": 0.2,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            # "max_num_seqs": 2
        },
        verifier_vllm_config={
            "model": "peiyi9979/math-shepherd-mistral-7b-prm",
            "gpu_memory_utilization": 0.6,
            "enable_prefix_caching": False,
            # "max_num_seqs": 2
        },
        offload_enabled=True,
        spec_beam_extension=False,  # Enable speculative beam extension
    )
    
    # Test problems
    problems = [
        # "What is 2 + 2?",
        "Solve for x: 3x + 5 = 20",
    ]
    
    search_config = SearchConfig(
        approach="beam_search",
        beam_width=2,
        n=4,
        num_iterations=2,  
        temperature=0.8,
        max_tokens=512, 
    )
    
    try:
        # Initialize models
        fasttts.initialize()
        
        # Perform search with default configuration
        results = fasttts.search(problems, search_config=search_config)
        
        logger.info("\n--- Results ---")
        logger.info(f"Total num tokens: {results['total_num_tokens']}")
        logger.info(f"Effective num tokens: {results['effective_num_tokens']}")
        logger.info(f"N completion tokens: {results['n_completion_tokens']}")
        logger.info(f"Total generator latency: {results['total_generator_latency_s']:.2f}s")
        logger.info(f"Total verifier latency: {results['total_verifier_latency_s']:.2f}s")
        logger.info(f"N generator latency: {results['n_generator_latency_s']:.2f}s")
        logger.info(f"N verifier latency: {results['n_verifier_latency_s']:.2f}s")
        logger.info(f"Completions: {len(results['completions'][0])}")
        logger.info(f"Predictions: {results['completions'][0][0]}")
        logger.info(f"Completion time: {results['completion_time']:.2f}s")
        logger.info(f"Extended beams: {results['extended_beams']}")
            
    finally:
        # Cleanup
        fasttts.shutdown()



def aime_example():
    """Example using AIME dataset with a specific sample index."""
    
    indices = [0]
    # Load AIME dataset
    try:
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        logger.info(f"Loaded AIME dataset with {len(dataset)} samples")
        
        # Get the specific sample
        samples = [dataset[index] for index in indices]
        problems = [sample["problem"] for sample in samples]
        reference_answers = [sample["answer"] for sample in samples]
        
    except Exception as e:
        logger.error(f"Failed to load AIME dataset: {e}")
        return
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    # Create FastTTS instance
    fasttts = create_fasttts(
        generator_vllm_config={
            # "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "model": "Qwen/Qwen2.5-Math-7B-Instruct",
            # "model": "meta-llama/Llama-3.2-3B-Instruct",
            "gpu_memory_utilization": 0.74,
            "disable_log_stats": False,
            # "max_model_len": 4096,
            # "enforce_eager": True,
        },
        verifier_vllm_config={
            # "model": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
            # "model": "peiyi9979/math-shepherd-mistral-7b-prm",
            "model": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
            "gpu_memory_utilization": 0.16,
            "disable_log_stats": False,
            # "max_model_len": 4096
        },
        offload_enabled=False,
        spec_beam_extension=False,
        prefix_aware_scheduling=False,  # Enable prefix-aware scheduling
    )
    
    search_config = SearchConfig(
        approach="best_of_n",
        beam_width=4,
        n=16,
        num_iterations=10,
        temperature=0.8,
        # max_tokens=5,
    )
    
    try:
        # Initialize models
        fasttts.initialize()
        
        results = fasttts.search(problems, search_config=search_config)
        
        # Print results
        logger.info("\n--- Results ---")
        logger.info(f"Total num tokens: {results['total_num_tokens']}")
        # logger.info(f"Effective num tokens: {sum(results['effective_num_tokens'][0])/len(results['effective_num_tokens'][0])}")
        logger.info(f"Effective num tokens: {sum(results['effective_num_tokens'][0])}")
        logger.info(f"Effective num tokens per step: {sum(results['effective_num_tokens'][0])/len(results['effective_num_tokens'][0])}")
        number_of_tokens = [len(tokenizer.encode(results['completions'][0][i])) for i in range(len(results['completions'][0]))]
        logger.info(f"Number of tokens in 1 completion: {sum(number_of_tokens)/len(number_of_tokens)}")
        logger.info(f"N completion tokens: {results['n_completion_tokens']}")
        logger.info(f"Total generator latency: {results['total_generator_latency_s']:.2f}s")
        logger.info(f"Total verifier latency: {results['total_verifier_latency_s']:.2f}s")
        logger.info(f"N generator latency: {results['n_generator_latency_s']:.2f}s")
        logger.info(f"N verifier latency: {results['n_verifier_latency_s']:.2f}s")
        logger.info(f"Goodput: {sum(results['effective_num_tokens'][0])/(results['n_generator_latency_s'] + results['n_verifier_latency_s']):.2f}")
        logger.info(f"Per-token generator goodput: {sum(results['effective_num_tokens'][0])/len(results['effective_num_tokens'][0])/(results['n_generator_latency_s'] + results['n_verifier_latency_s']):.2f}")
        logger.info(f"Completions: {len(results['completions'][0])}")
        logger.info(f"Completion time: {sum(results['completion_time'][0])/len(results['completion_time'][0]):.2f}s")
        num_steps = sum([results['completions'][0][i].count('\n\n') for i in range(len(results['completions'][0]))]) / len(results['completions'][0])
        logger.info(f"Number of steps in 1 completion: {num_steps}")
        logger.info(f"Extended tokens: {results['extended_tokens_list']}")
        # for i in range(len(results['completions'][0])):
        #     logger.info("-" * 100)
        #     logger.info(f"Completion {i}: {results['completions'][0][i]}")
        
        #  save results to json
        # import json
        # with open(f"aime_results_{index}_spec.json", "w") as f:
        #     json.dump(results, f)
        
    finally:
        # Cleanup
        fasttts.shutdown()


def examine_extended_tokens_example():
    """Example using AIME dataset with a specific sample index."""
    
    indices = [0]
    # Load AIME dataset
    try:
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        logger.info(f"Loaded AIME dataset with {len(dataset)} samples")
        
        # Get the specific sample
        samples = [dataset[index] for index in indices]
        problems = [sample["problem"] for sample in samples]
        reference_answers = [sample["answer"] for sample in samples]
        
    except Exception as e:
        logger.error(f"Failed to load AIME dataset: {e}")
        return
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    # Create FastTTS instance
    fasttts = create_fasttts(
        generator_vllm_config={
            "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
            # "model": "Qwen/Qwen2.5-Math-7B-Instruct",
            # "model": "meta-llama/Llama-3.2-3B-Instruct",
            "gpu_memory_utilization": 0.2,
            "disable_log_stats": False,
            # "max_model_len": 4096,
            # "enforce_eager": True,
        },
        verifier_vllm_config={
            # "model": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
            "model": "peiyi9979/math-shepherd-mistral-7b-prm",
            # "model": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
            "gpu_memory_utilization": 0.65,
            "disable_log_stats": False,
            # "max_model_len": 4096
        },
        offload_enabled=False,
        spec_beam_extension=True,
        prefix_aware_scheduling=False,  # Enable prefix-aware scheduling
    )
    # Initialize models
    fasttts.initialize()
    
    # n_values = [8, 16, 32, 64, 128, 256, 512]
    temperatures = [0, 0.5, 1.0, 1.5, 2.0]
    result_per_n = {}
    
    for temperature in temperatures:
        search_config = SearchConfig(
            approach="beam_search",
            beam_width=4,
            n=64,
            num_iterations=5,
            temperature=temperature,
            )
        
        results = fasttts.search(problems, search_config=search_config)
        result_per_n[temperature] = results['extended_tokens_list']
            
        #  save results to json
        import json
        with open(f"figures/spec_ablation_temperature.json", "w") as f:
            json.dump(result_per_n, f)
    fasttts.shutdown()

if __name__ == "__main__":
    logger.info("FastTTS Example Script")
    logger.info("=" * 50)
    
    # Note: These examples require actual model paths and may take time to run
    # Uncomment the examples you want to run
    
    # infer_example()
    aime_example()  # Test first AIME problem
    # examine_extended_tokens_example()
    
    logger.info("\nTo run examples, uncomment the desired function calls above.")
    logger.info("Note: Make sure you have the required models available.") 