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
from typing import Dict, List, Any, Optional, Union

from fasttts.config import FastTTSConfig, SearchConfig
from fasttts.models.vllm_wrapper import AsyncGeneratorVLLMModelWrapper, AsyncVerifierVLLMModelWrapper
from fasttts.search.beam_search import beam_search_async, beam_search

logger = logging.getLogger(__name__)


class FastTTS:
    """Fast Test Time Search interface with async generator and verifier models."""
    
    def __init__(self, config: FastTTSConfig):
        """Initialize FastTTS with the given configuration."""
        self.config = config
        self.generator = None
        self.verifier = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the generator and verifier models asynchronously."""
        if self._initialized:
            return
            
        logger.info("Initializing FastTTS models...")
        
        # Initialize generator
        self.generator = AsyncGeneratorVLLMModelWrapper(
            model_path=self.config.generator_model_path,
            config=self.config,
            enable_prefix_caching=True,
            enable_sleep_mode=self.config.offload_enabled,
        )
        
        # Initialize verifier
        self.verifier = AsyncVerifierVLLMModelWrapper(
            model_path=self.config.verifier_model_path,
            config=self.config,
            enable_prefix_caching=True,
            enable_sleep_mode=self.config.offload_enabled,
        )
        
        # Wait a bit for models to initialize
        await asyncio.sleep(2)
        
        self._initialized = True
        logger.info("FastTTS models initialized successfully")
        
    async def _process_batch_async(
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
            return await beam_search_async(examples, search_config, self.generator, self.verifier)
        elif search_config.approach == "best_of_n":
            # TODO: Implement best_of_n
            raise NotImplementedError("best_of_n not implemented yet")
        else:
            raise ValueError(f"Unknown approach: {search_config.approach}")
        
    async def search_async(
        self, 
        problems: List[str], 
        search_config: Optional[SearchConfig] = None,
        **search_kwargs
    ) -> Dict[str, Any]:
        """Perform test time search asynchronously with batch processing."""
        if not self._initialized:
            await self.initialize()
            
        # Get search configuration
        search_config = self.config.get_search_config(search_config)
        
        # Override with any additional kwargs
        if search_kwargs:
            search_config = search_config.copy(**search_kwargs)
            
        batch_size = search_config.batch_size
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        # If batch_size is 1 or greater than the number of problems, process all at once
        if batch_size == 1 or batch_size >= len(problems):
            logger.info(f"Processing {len(problems)} problems in a single batch")
            return await self._process_batch_async(problems, search_config)
        
        # Process problems in batches
        logger.info(f"Processing {len(problems)} problems in batches of {batch_size}")
        
        all_results = {
            "completions": [],
            "pred": [],
            "completion_tokens": [],
            "scores": [],
            "total_num_tokens": []
        }
        
        # Process batches sequentially
        for i in range(0, len(problems), batch_size):
            batch_problems = problems[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(problems) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_problems)} problems")
            
            batch_results = await self._process_batch_async(batch_problems, search_config)
            
            # Merge results
            for key in all_results:
                if key in batch_results:
                    all_results[key].extend(batch_results[key])
                    
            # Log progress
            logger.info(f"Completed batch {batch_num}/{total_batches}")
        
        logger.info(f"Completed processing all {len(problems)} problems")
        return all_results
            
    def search(
        self, 
        problems: List[str], 
        search_config: Optional[SearchConfig] = None,
        **search_kwargs
    ) -> Dict[str, Any]:
        """Perform test time search synchronously with batch processing."""
        return asyncio.run(self.search_async(problems, search_config, **search_kwargs))
        
    async def search_single_async(
        self, 
        problem: str, 
        search_config: Optional[SearchConfig] = None,
        **search_kwargs
    ) -> Dict[str, Any]:
        """Search for a single problem asynchronously."""
        results = await self.search_async([problem], search_config, **search_kwargs)
        
        # Extract single result
        single_result = {}
        for key, value in results.items():
            if isinstance(value, list) and len(value) > 0:
                single_result[key] = value[0]
            else:
                single_result[key] = value
                
        return single_result
        
    def search_single(
        self, 
        problem: str, 
        search_config: Optional[SearchConfig] = None,
        **search_kwargs
    ) -> Dict[str, Any]:
        """Search for a single problem synchronously."""
        return asyncio.run(self.search_single_async(problem, search_config, **search_kwargs))
        
    def create_search_config(self, **kwargs) -> SearchConfig:
        """Create a search configuration with optional overrides."""
        return self.config.create_search_config(**kwargs)
        
    async def shutdown(self):
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


# Convenience function for quick usage
def create_fasttts(
    generator_model_path: str = "meta-llama/Llama-3.2-1B-Instruct",
    verifier_model_path: str = "peiyi9979/math-shepherd-mistral-7b-prm",
    approach: str = "beam_search",
    **kwargs
) -> FastTTS:
    """Create a FastTTS instance with default configuration."""
    config = FastTTSConfig(
        generator_model_path=generator_model_path,
        verifier_model_path=verifier_model_path,
        **kwargs
    )
    
    # Set the default search approach
    config.default_search_config.approach = approach
    
    return FastTTS(config) 