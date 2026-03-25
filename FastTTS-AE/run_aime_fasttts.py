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

import argparse
import json
import logging
import os
from datetime import datetime
from datasets import load_dataset
from fasttts import create_fasttts
from fasttts import FastTTSConfig, SearchConfig
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run FastTTS on the first AIME dataset with configurable parameters"
    )
    
    # Search configuration parameters
    parser.add_argument(
        "--num_iterations", 
        type=int, 
        default=2,
        help="Number of search iterations (default: 10)"
    )
    parser.add_argument(
        "--n", 
        type=int, 
        default=128,
        help="Number of completions to generate (default: 8)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=2,
        help="Temperature for generation (default: 0.8)"
    )
    parser.add_argument(
        "--beam_width", 
        type=int, 
        default=4,
        help="Beam width for beam search (default: 4)"
    )
    
    # Model configuration parameters
    parser.add_argument(
        "--generator_model", 
        type=str, 
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Generator model name (default: Qwen/Qwen2.5-Math-1.5B-Instruct)"
    )
    parser.add_argument(
        "--verifier_model", 
        type=str, 
        default="peiyi9979/math-shepherd-mistral-7b-prm",
        help="Verifier model name (default: peiyi9979/math-shepherd-mistral-7b-prm)"
    )
    parser.add_argument(
        "--generator_gpu_memory", 
        type=float, 
        default=0.3,
        help="GPU memory utilization for generator (default: 0.2)"
    )
    parser.add_argument(
        "--verifier_gpu_memory", 
        type=float, 
        default=0.62,
        help="GPU memory utilization for verifier (default: 0.65)"
    )
    
    # FastTTS configuration parameters
    parser.add_argument(
        "--offload_enabled", 
        action="store_true",
        help="Enable offloading (default: False)"
    )
    parser.add_argument(
        "--spec_beam_extension", 
        action="store_true",
        default=False,
        help="Enable speculative beam extension (default: False)"
    )
    parser.add_argument(
        "--prefix_aware_scheduling", 
        action="store_true",
        help="Enable prefix-aware scheduling (default: False)"
    )
    
    return parser.parse_args()


def run_aime_fasttts(args):
    """Run FastTTS on the first AIME dataset with the given parameters."""
    
    logger.info("Starting FastTTS AIME experiment")
    logger.info(f"Parameters: {vars(args)}")
    
    # Load AIME dataset
    try:
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        logger.info(f"Loaded AIME dataset with {len(dataset)} samples")
        
        # Get the first sample
        sample = dataset[0]
        problem = sample["problem"]
        reference_answer = sample["answer"]
        
        logger.info(f"Problem: {problem}")
        logger.info(f"Reference answer: {reference_answer}")
        
    except Exception as e:
        logger.error(f"Failed to load AIME dataset: {e}")
        return None
    
    # Load tokenizer for analysis
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    
    # Create FastTTS instance
    fasttts = create_fasttts(
        generator_vllm_config={
            "model": args.generator_model,
            "gpu_memory_utilization": args.generator_gpu_memory,
            "disable_log_stats": False,
        },
        verifier_vllm_config={
            "model": args.verifier_model,
            "gpu_memory_utilization": args.verifier_gpu_memory,
            "disable_log_stats": True,
        },
        offload_enabled=args.offload_enabled,
        spec_beam_extension=args.spec_beam_extension,
        prefix_aware_scheduling=args.prefix_aware_scheduling,
    )
    
    # Create search configuration
    search_config = SearchConfig(
        approach="beam_search",
        beam_width=args.beam_width,
        n=args.n,
        num_iterations=args.num_iterations,
        temperature=args.temperature,
    )
    
    try:
        # Initialize models
        logger.info("Initializing FastTTS models...")
        fasttts.initialize()
        
        # Perform search
        logger.info("Starting search...")
        results = fasttts.search([problem], search_config=search_config)
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("RESULTS")
        logger.info("="*50)
        logger.info(f"Total num tokens: {results['total_num_tokens']}")
        logger.info(f"Effective num tokens: {sum(results['effective_num_tokens'][0])}")
        logger.info(f"Effective num tokens per step: {sum(results['effective_num_tokens'][0])/len(results['effective_num_tokens'][0])}")
        
        number_of_tokens = [len(tokenizer.encode(results['completions'][0][i])) for i in range(len(results['completions'][0]))]
        logger.info(f"Number of tokens in 1 completion: {sum(number_of_tokens)/len(number_of_tokens)}")
        logger.info(f"N completion tokens: {results['n_completion_tokens']}")
        
        logger.info(f"Total generator latency: {results['total_generator_latency_s']:.2f}s")
        logger.info(f"Total verifier latency: {results['total_verifier_latency_s']:.2f}s")
        logger.info(f"N generator latency: {results['n_generator_latency_s']:.2f}s")
        logger.info(f"N verifier latency: {results['n_verifier_latency_s']:.2f}s")
        
        goodput = sum(results['effective_num_tokens'][0])/(results['n_generator_latency_s'] + results['n_verifier_latency_s'])
        logger.info(f"Goodput: {goodput:.2f}")
        
        per_token_goodput = sum(results['effective_num_tokens'][0])/len(results['effective_num_tokens'][0])/(results['n_generator_latency_s'] + results['n_verifier_latency_s'])
        logger.info(f"Per-token generator goodput: {per_token_goodput:.2f}")
        
        logger.info(f"Completions: {len(results['completions'][0])}")
        logger.info(f"Completion time: {sum(results['completion_time'][0])/len(results['completion_time'][0]):.2f}s")
        
        num_steps = sum([results['completions'][0][i].count('\n\n') for i in range(len(results['completions'][0]))]) / len(results['completions'][0])
        logger.info(f"Number of steps in 1 completion: {num_steps}")
        
        if 'extended_tokens_list' in results:
            logger.info(f"Extended tokens: {results['extended_tokens_list']}")
        
        return results
        
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        fasttts.shutdown()


def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    logger.info("FastTTS AIME Experiment")
    logger.info("=" * 50)
    
    results = run_aime_fasttts(args)
    
    if results:
        logger.info("Experiment completed successfully!")
    else:
        logger.error("Experiment failed!")
        exit(1)


if __name__ == "__main__":
    main() 