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

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict

@dataclass
class SearchConfig:
    """Configuration for search strategies that can vary per request."""
    
    # Search approach
    approach: Literal["beam_search", "best_of_n", "dvts", "dynamic_branching", "vg_search"] = "beam_search"
    
    # Generation parameters for search
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 2048
    stop: Optional[str] = "\n\n"
    
    # Beam search specific parameters
    beam_width: int = 4
    num_iterations: int = 40
    lookahead: int = 0
    n: int = 8  # Number of beams to maintain
    
    # Best of N specific parameters (for future use)
    num_samples: int = 4  # Number of samples for best_of_n
    
    # System prompt and chat template
    system_prompt: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
    # system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    custom_chat_template: Optional[str] = None
    
    # Aggregation strategy for scores
    agg_strategy: str = "last"  # Options: "last", "min", "prod", "mean"
    
    # Filtering options
    filter_duplicates: bool = False
    sort_completed: bool = False
    
    # Batch processing
    batch_size: int = 1  # Number of problems to process at once (1 = sequential processing)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.approach == "beam_search":
            if self.batch_size != 1:
                raise ValueError("batch_size should be 1 for beam_search")
            if self.n % self.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")
        elif self.approach == "best_of_n":
            if self.num_samples <= 0:
                raise ValueError("num_samples should be positive for best_of_n")
                
    def copy(self, **kwargs):
        """Create a copy of the search config with optional overrides."""
        import copy
        new_config = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
        new_config.__post_init__()  # Re-validate
        return new_config


@dataclass
class FastTTSConfig:
    """Configuration for FastTTS test time search.
    
    quantization options: None (default), 'awq', 'gptq', 'squeezellm', etc. See vLLM docs for supported quantization types.
    """
    
    # Model paths
    generator_vllm_config: Optional[Dict] = field(default_factory=lambda: {
        "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.45,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "seed": 42,
        "disable_log_stats": False,
    })
    verifier_vllm_config: Optional[Dict] = field(default_factory=lambda: {
        "model": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.45,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "seed": 42,
        "disable_log_stats": False,
    })
    
    # Global generation parameters (can be overridden by SearchConfig)
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 2048
    
    # System prompt and chat template (used as defaults for SearchConfig)
    system_prompt: str = "Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."
    custom_chat_template: Optional[str] = None
    
    # Default search configuration
    search_config: SearchConfig = field(default_factory=SearchConfig)
    
    # Optimization
    spec_beam_extension: bool = False  # Enable speculative beam extension for better performance
    offload_enabled: bool = False  # Enable offloading for memory management
    prefix_aware_scheduling: bool = False  # Enable prefix-aware scheduling for request batching
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure default search config uses global parameters
        if self.search_config.temperature != self.temperature:
            self.search_config.temperature = self.temperature
        if self.search_config.top_p != self.top_p:
            self.search_config.top_p = self.top_p
        if self.search_config.max_tokens != self.max_tokens:
            self.search_config.max_tokens = self.max_tokens
        if self.search_config.system_prompt != self.system_prompt:
            self.search_config.system_prompt = self.system_prompt
        if self.search_config.custom_chat_template != self.custom_chat_template:
            self.search_config.custom_chat_template = self.custom_chat_template
            
    def create_search_config(self, **kwargs) -> SearchConfig:
        """Create a search configuration with optional overrides."""
        return self.search_config.copy(**kwargs)

    def get_search_config(self, search_config: Optional[SearchConfig] = None) -> SearchConfig:
        """Get search configuration, using default if none provided."""
        if search_config is None:
            return self.search_config
        return search_config