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


# ---------------------------------------------------------------------------
# Beam identity counter — reset at the start of each search invocation
# ---------------------------------------------------------------------------
_beam_id_counter = 0


def _next_beam_id() -> int:
    global _beam_id_counter
    _beam_id_counter += 1
    return _beam_id_counter


def reset_beam_id_counter():
    """Reset for each new search — called at start of _beam_search."""
    global _beam_id_counter
    _beam_id_counter = 0


@dataclass
class Beam:
    """Represents a beam in beam search."""

    prompt: str
    index: int
    current_text: str
    next_texts: Optional[List[str]] = None
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

    # Parameters for spec beam extension
    future_texts: List[Tuple[str, bool]] = None # texts that is generated for the next search steps
    skipped_this_step: bool = False

    # Identity tracking (survives duplication)
    beam_id: int = -1
    parent_id: Optional[int] = None
    born_at_iteration: int = -1

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
