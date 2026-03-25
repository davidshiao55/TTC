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

from dataclasses import dataclass
from typing import List, Optional, Tuple
from search.utils import aggregate_scores


@dataclass
class Beam:
    """Represents a beam in beam search."""
    
    prompt: str
    index: int
    current_text: str
    next_texts: Optional[List[str]] = None
    lookahead_texts: Optional[List[str]] = None
    stop_reasons: Optional[List[Optional[str]]] = None
    best_scores: List[float] = None  # PRM scores
    all_scores: List[List[float]] = None  # All PRM scores
    previous_text: Optional[str] = None
    pruned: bool = False
    history: List[str] = None
    completed: bool = False
    completion_tokens: int = 0
    total_completion_tokens: int = 0
    completion_time: float = 0.0
    
    # Parameters for spec beam extension
    future_texts: List[Tuple[str, bool]] = None # texts that is generated for the next search steps
    skipped_this_step: bool = False
    
    def __post_init__(self):
        """Initialize default values."""
        if self.best_scores is None:
            self.best_scores = [0.0]
        if self.all_scores is None:
            self.all_scores = []
        if self.history is None:
            self.history = []
        if self.future_texts is None:
            self.future_texts = []
            
    def add_generation(self, next_text: str, stop_reason: Optional[str] = None):
        """Add a new generation to the beam."""
        self.current_text += next_text
        self.history.append(next_text)
        self.completion_tokens += 1
        
        if stop_reason in ["EOS", "length"] or not next_text:
            self.completed = True
            
    def clone(self):
        """Create a deep copy of the beam."""
        import copy
        return copy.deepcopy(self)
        
    def get_score(self, agg_strategy: str = "last") -> float:
        """Get the aggregated score for this beam."""
        if not self.all_scores:
            return 0.0
            
        return aggregate_scores(self.all_scores, agg_strategy) 