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
from typing import Dict, List, Any, Optional

from config import FastTTSConfig, SearchConfig
from models.vllm_wrapper import GeneratorVLLMModelWrapper, VerifierVLLMModelWrapper
from search.beam_search import beam_search
from search.dvts import dvts_search
from search.dynamic_branching import dynamic_branching_search
from search.vg_search import vg_search
from search.best_of_n import best_of_n_search
from search.results import SearchResults

logger = logging.getLogger(__name__)


_SEARCH_STRATEGIES = {
    "beam_search": beam_search,
    "dvts": dvts_search,
    "best_of_n": best_of_n_search,
    "dynamic_branching": dynamic_branching_search,
    "vg_search": vg_search,
}


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
    ) -> SearchResults:
        """Process a single batch of problems."""
        strategy = _SEARCH_STRATEGIES.get(search_config.approach)
        if strategy is None:
            raise ValueError(f"Unknown approach: {search_config.approach}")
        return strategy({"problem": problems}, search_config, self.generator, self.verifier)

    def search(
        self,
        problems: List[str],
        search_config: Optional[SearchConfig] = None,
        **search_kwargs,
    ) -> SearchResults:
        """Perform test-time search with batch processing."""
        if not self._initialized:
            self.initialize()

        if search_config is None:
            search_config = self.config.search_config
        if search_kwargs:
            search_config = search_config.copy(**search_kwargs)
        search_config = search_config.copy(
            spec_beam_extension=self.config.spec_beam_extension,
        )

        batch_size = search_config.batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if batch_size >= len(problems):
            logger.info(f"Processing {len(problems)} problems at once")
            return self._process_batch(problems, search_config)

        logger.info(f"Processing {len(problems)} problems in batches of {batch_size}")
        total_batches = (len(problems) + batch_size - 1) // batch_size
        merged = SearchResults()
        for batch_num, start in enumerate(range(0, len(problems), batch_size), start=1):
            batch_problems = problems[start:start + batch_size]
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_problems)} problems")
            merged.append_batch(self._process_batch(batch_problems, search_config))
            logger.info(f"Completed batch {batch_num}/{total_batches}")

        logger.info(f"Completed processing all {len(problems)} problems")
        return merged

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
        self.shutdown()


def _merge_vllm_config(defaults: Dict, overrides: Optional[Dict]) -> Dict:
    """Shallow-merge a caller-supplied vLLM config dict over the defaults."""
    merged = defaults.copy()
    if overrides:
        merged.update(overrides)
    return merged


def create_fasttts_config(
    generator_vllm_config: Optional[Dict] = None,
    verifier_vllm_config: Optional[Dict] = None,
    **kwargs,
) -> FastTTSConfig:
    """Build a FastTTSConfig. kwargs matching FastTTSConfig fields are consumed;
    the rest are left in the caller's dict for downstream use (search params)."""
    from dataclasses import fields
    fasttts_field_names = {f.name for f in fields(FastTTSConfig)}
    defaults = FastTTSConfig()

    fasttts_params = {
        "generator_vllm_config": _merge_vllm_config(defaults.generator_vllm_config, generator_vllm_config),
        "verifier_vllm_config": _merge_vllm_config(defaults.verifier_vllm_config, verifier_vllm_config),
    }
    for name in list(kwargs):
        if name in fasttts_field_names:
            fasttts_params[name] = kwargs.pop(name)

    return FastTTSConfig(**fasttts_params)


def create_fasttts(
    generator_vllm_config: Optional[Dict] = None,
    verifier_vllm_config: Optional[Dict] = None,
    approach: str = "beam_search",
    **kwargs,
) -> FastTTS:
    """Create a FastTTS instance with default configuration."""
    config = create_fasttts_config(generator_vllm_config, verifier_vllm_config, **kwargs)
    # Remaining kwargs target SearchConfig fields; let SearchConfig.copy validate.
    config.search_config = config.search_config.copy(approach=approach, **{
        k: v for k, v in kwargs.items() if hasattr(config.search_config, k)
    })
    return FastTTS(config)
