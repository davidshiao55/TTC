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

"""Configuration dataclasses for FastTTS.

``FastTTSConfig`` holds engine-level settings (vLLM model paths, offloading
flags). ``SearchConfig`` holds per-request search settings (approach,
generation params, beam width, system prompt). The two are kept separate so
that the generator/verifier engines stay stable across successive searches
with different strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

DEFAULT_GENERATOR_VLLM_CONFIG: Dict = {
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.45,
    "tensor_parallel_size": 1,
    "enable_prefix_caching": True,
    "seed": 42,
    "disable_log_stats": False,
}

DEFAULT_VERIFIER_VLLM_CONFIG: Dict = {
    "model": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.45,
    "tensor_parallel_size": 1,
    "enable_prefix_caching": True,
    "seed": 42,
    "disable_log_stats": False,
}


@dataclass
class SearchConfig:
    """Per-request search configuration."""

    # Strategy selection
    approach: Literal["beam_search", "best_of_n", "dvts", "dynamic_branching", "vg_search"] = "beam_search"

    # Generation parameters
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 2048
    stop: Optional[str] = "\n\n"

    # Beam search / DVTS specific
    beam_width: int = 4
    num_iterations: int = 40
    lookahead: int = 0
    n: int = 8

    # System prompt
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Score aggregation / post-processing
    agg_strategy: str = "last"  # "last" | "min" | "prod" | "mean"
    filter_duplicates: bool = False

    # Batch processing
    batch_size: int = 1  # 1 = sequential per-problem processing

    # Mirror of FastTTSConfig.spec_beam_extension so search-layer code
    # (e.g. score-ordered active_beams) can gate on SBE without needing
    # access to the engine-level config. Set by FastTTS.search() at dispatch.
    spec_beam_extension: bool = False

    # Speculative-step truncation ratio (paper §4.1.1). Controls how much
    # of a parent beam's speculative first step a duplicated child inherits.
    #   0.0  → true vanilla beam search equivalence: duplicates start with
    #           empty pending_steps and regenerate from current_text.
    #   0.85 → paper-recommended SBE value; duplicates inherit ~85% of the
    #           parent's speculative first step as a starting seed.
    spec_truncation_ratio: float = 0.0

    def __post_init__(self):
        if self.approach == "beam_search":
            if self.batch_size != 1:
                raise ValueError("batch_size should be 1 for beam_search")
            if self.n % self.beam_width != 0:
                raise ValueError("n should be a multiple of beam_width")
        elif self.approach == "best_of_n":
            if self.n <= 0:
                raise ValueError("n should be positive for best_of_n")

    def copy(self, **kwargs) -> "SearchConfig":
        """Return a validated deep-copy with field overrides applied."""
        import copy
        new_config = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
        new_config.__post_init__()
        return new_config


@dataclass
class FastTTSConfig:
    """Engine-level configuration for FastTTS.

    Generation parameters (temperature, top_p, max_tokens, system_prompt)
    live on :class:`SearchConfig`.
    """

    generator_vllm_config: Optional[Dict] = field(
        default_factory=lambda: DEFAULT_GENERATOR_VLLM_CONFIG.copy()
    )
    verifier_vllm_config: Optional[Dict] = field(
        default_factory=lambda: DEFAULT_VERIFIER_VLLM_CONFIG.copy()
    )

    search_config: SearchConfig = field(default_factory=SearchConfig)

    spec_beam_extension: bool = False
    offload_enabled: bool = False
    prefix_aware_scheduling: bool = False

    def create_search_config(self, **kwargs) -> SearchConfig:
        """Create a search configuration with optional overrides."""
        return self.search_config.copy(**kwargs)
