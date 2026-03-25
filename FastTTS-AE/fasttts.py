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
import logging
from typing import Dict, List, Any, Optional
import time

from config import FastTTSConfig, SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.beam_search import beam_search
from search.dvts import dvts_search
from search.dynamic_branching import dynamic_branching_search
from search.vg_search import vg_search
from search.best_of_n import best_of_n_search

logger = logging.getLogger(__name__)


class FastTTS:
    """Fast Test Time Search interface with async generator and verifier models."""
    
    def __init__(self, config: FastTTSConfig):
        """Initialize FastTTS with the given configuration."""
        self.config = config
        self.generator = None
        self.verifier = None
        self._initialized = False
        
    def initialize(self):
        """Initialize the generator and verifier models asynchronously."""
        if self._initialized:
            return
            
        logger.info("Initializing FastTTS models...")
        
        # Initialize generator
        self.generator = GeneratorVLLMModelWrapper(
            config=self.config,
            enable_sleep_mode=self.config.offload_enabled,
        )
        
        # Initialize verifier
        self.verifier = VerifierVLLMModelWrapper(
            config=self.config,
            enable_sleep_mode=self.config.offload_enabled,
        )
        
        self._initialized = True
        logger.info("FastTTS models initialized successfully")
        
    def _process_batch(
        self, 
        problems: List[str], 
        search_config: SearchConfig,
        **search_kwargs
    ) -> Dict[str, Any]:
        """Process a single batch of problems."""
        # Override with any additional kwargs
        if search_kwargs:
            search_config = search_config.copy(**search_kwargs)
            
        # Prepare examples format
        examples = {"problem": problems}
        
        if search_config.approach == "beam_search":
            return beam_search(examples, search_config, self.generator, self.verifier)
        elif search_config.approach == "dvts":
            return dvts_search(examples, search_config, self.generator, self.verifier)
        elif search_config.approach == "best_of_n":
            return best_of_n_search(examples, search_config, self.generator, self.verifier)
        elif search_config.approach == "dynamic_branching":
            return dynamic_branching_search(examples, search_config, self.generator, self.verifier)
        elif search_config.approach == "vg_search":
            return vg_search(examples, search_config, self.generator, self.verifier)
        else:
            raise ValueError(f"Unknown approach: {search_config.approach}")
        
    def search(
        self, 
        problems: List[str], 
        search_config: Optional[SearchConfig] = None,
        **search_kwargs
    ) -> Dict[str, Any]:
        """Perform test time search asynchronously with batch processing."""
        if not self._initialized:
            self.initialize()
            
        # Get search configuration
        search_config = self.config.get_search_config(search_config)
        
        # Override with any additional kwargs
        if search_kwargs:
            search_config = search_config.copy(**search_kwargs)
            
        batch_size = search_config.batch_size
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        # If batch_size is greater than the number of problems, process all at once
        if batch_size >= len(problems):
            logger.info(f"Processing {len(problems)} problems at once")
            return self._process_batch(problems, search_config)
        
        # Process problems in batches
        logger.info(f"Processing {len(problems)} problems in batches of {batch_size}")
        
        all_results = {
            "completions": [],
            "pred": [],
            "completion_tokens": [],
            "scores": [],
            "total_num_tokens": [], 
            "n_completion_tokens": [],
            "total_generator_latency_s": [],
            "total_verifier_latency_s": [],
            "n_generator_latency_s": [],
            "n_verifier_latency_s": [],
            "completion_time": [],
        }
        
        # Process batches sequentially
        for i in range(0, len(problems), batch_size):
            batch_problems = problems[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(problems) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_problems)} problems")
            
            batch_results = self._process_batch(batch_problems, search_config)
            
            # Merge results
            for key in all_results:
                if key in batch_results:
                    all_results[key].append(batch_results[key])
                    
            # Log progress
            logger.info(f"Completed batch {batch_num}/{total_batches}")
        
        logger.info(f"Completed processing all {len(problems)} problems")
        return all_results
        
    def search_single(
        self, 
        problem: str, 
        search_config: Optional[SearchConfig] = None,
        **search_kwargs
    ) -> Dict[str, Any]:
        """Search for a single problem asynchronously."""
        results = self.search([problem], search_config, **search_kwargs)
        
        # Extract single result
        single_result = {}
        for key, value in results.items():
            if isinstance(value, list) and len(value) > 0:
                single_result[key] = value[0]
            else:
                single_result[key] = value
                
        return single_result
        
    def create_search_config(self, **kwargs) -> SearchConfig:
        """Create a search configuration with optional overrides."""
        return self.config.create_search_config(**kwargs)
        
    def shutdown(self):
        """Shutdown the models gracefully."""
        if self.generator:
            self.generator.shutdown()
        if self.verifier:
            self.verifier.shutdown()
        self._initialized = False
        logger.info("FastTTS shutdown complete")
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        asyncio.run(self.shutdown())


def create_fasttts_config(
    generator_vllm_config: Optional[Dict] = None,
    verifier_vllm_config: Optional[Dict] = None,
    **kwargs
) -> FastTTSConfig:
    # Separate FastTTSConfig parameters from SearchConfig parameters
    from dataclasses import fields
    fasttts_config_param_names = {field.name for field in fields(FastTTSConfig)}
    
    # Create default config to get default VLLM configs
    default_config = FastTTSConfig()
    
    # Merge VLLM configs with defaults
    merged_generator_config = default_config.generator_vllm_config.copy()
    if generator_vllm_config:
        merged_generator_config.update(generator_vllm_config)
        
    merged_verifier_config = default_config.verifier_vllm_config.copy()
    if verifier_vllm_config:
        merged_verifier_config.update(verifier_vllm_config)
    
    # Extract FastTTSConfig parameters from kwargs
    fasttts_config_params = {
        'generator_vllm_config': merged_generator_config,
        'verifier_vllm_config': merged_verifier_config,
    }
    
    # Dynamically extract parameters that exist in kwargs
    for param_name in fasttts_config_param_names:
        if param_name in kwargs:
            fasttts_config_params[param_name] = kwargs.pop(param_name)
    
    config = FastTTSConfig(**fasttts_config_params)
    return config

# Convenience function for quick usage
def create_fasttts(
    generator_vllm_config: Optional[Dict] = None,
    verifier_vllm_config: Optional[Dict] = None,
    approach: str = "beam_search",
    **kwargs
) -> FastTTS:
    """Create a FastTTS instance with default configuration."""
    config = create_fasttts_config(generator_vllm_config, verifier_vllm_config, **kwargs)
    
    # Set the default search approach and any remaining search config parameters
    config.search_config.approach = approach
    for key, value in kwargs.items():
        if hasattr(config.search_config, key):
            setattr(config.search_config, key, value)
    
    return FastTTS(config) 