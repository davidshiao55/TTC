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
from typing import List, Optional


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


def step_hash(text: str) -> str:
    """Short hash of step text for identity comparison in logs."""
    return format(hash(text) & 0xFFFFFFFF, '08x')


@dataclass
class StepChunk:
    """One segment of a generator output after splitting on the step separator.

    The two flags are orthogonal:

    - `is_complete_step`: text ends at a step boundary ("\n\n"). Scorable
      by the PRM as a whole step. Hash-identifiable.
    - `terminal`: consuming this chunk completes the beam. Set only on
      the last chunk of a generator output, and only when the parser's
      waterfall decided the generation genuinely terminated (EOS, real
      length-cap on a single mega-step, context exhaustion, etc).

    Four shapes arise:

    - (True, False):  regular speculative step, not the beam's last.
    - (True, True):   final step that ends exactly at "\n\n" followed by
                      EOS / length / context-exhaustion.
    - (False, True):  terminal partial tail (e.g. EOS mid-step). Scored
                      by the PRM via `reward_utils.prepare_input`'s
                      final-split handling, matching baseline.
    - (False, False): non-terminal continuation prefix. Produced by
                      (a) `_duplicate_beams` truncation (always head),
                      (b) SBE force-finish tails, or (c) length-cap
                      recovery (finish_reason="length" with at least
                      one complete step in the output). Consumed as
                      a prefix by `_prepare_step_source` next
                      iteration; never submitted to the PRM.
    """

    text: str
    is_complete_step: bool
    terminal: bool = False


@dataclass
class Beam:
    """Represents a beam in beam search.

    Fields:
        prompt:         The original question text.
        current_text:   Accumulated solution text (grows each iteration).
        gen_text:       Raw generation output from the last generate call
                        (single-element list [text], before split).
        stop_reasons:   Stop reason from vLLM for the last generation.
                        String from `SamplingParams.stop` that matched, or
                        None if generation stopped for another reason.
        finish_reasons: Finish reason from vLLM for the last generation.
                        One of "stop" / "length" / "abort". Needed to
                        distinguish a real length-cap from an EOS, which
                        the parser's waterfall uses for length-cap
                        recovery (see StepChunk docstring).
        scores:         Per-step PRM scores [s0, s1, ..., sN].
        pruned:         Whether this beam was pruned (filtered next iteration).
        gen_history:    Raw generation text per iteration (for other search strategies).
        completed:      Whether this beam has finished generating.
        step_tokens:    Token count of the last step consumed (overwritten each iteration).
        total_tokens_generated: Cumulative tokens generated across all iterations.
        time_to_complete: Wall-clock latency when beam was marked completed.
        pending_steps:  Queue of StepChunks that have been generated
                        speculatively (or manufactured as divergence
                        prefixes by `_duplicate_beams`) but not yet
                        committed to `current_text`. One chunk is
                        consumed per iteration.
        skipped_this_step: Whether this beam skipped generation this iteration.
        step_hashes:    Hash of cumulative text at each \n\n boundary, aligned 1:1
                        with scores. Each hash encodes the full prefix through
                        that step, so single-hash comparison suffices for prefix
                        matching. Used for score propagation and debug logging.
        beam_id:        Unique ID (auto-incrementing, survives iterations).
        parent_id:      Parent beam_id (set on duplication).
        born_at_iteration: Iteration when this beam was created/duplicated.
    """

    prompt: str
    current_text: str
    gen_text: Optional[List[str]] = None
    stop_reasons: Optional[List[Optional[str]]] = None
    finish_reasons: Optional[List[Optional[str]]] = None
    prompt_token_lens: Optional[List[int]] = None
    scores: List[float] = None
    pruned: bool = False
    gen_history: List[str] = None
    completed: bool = False
    step_tokens: int = 0
    total_tokens_generated: int = 0
    time_to_complete: float = 0.0

    # Parameters for spec beam extension
    pending_steps: List[StepChunk] = None
    skipped_this_step: bool = False

    # Step identity — aligned 1:1 with scores. Used for score propagation
    # (prefix caching None-fill) and debug logging.
    step_hashes: List[str] = None

    # Identity tracking (survives duplication)
    beam_id: int = -1
    parent_id: Optional[int] = None
    born_at_iteration: int = -1

    def __post_init__(self):
        """Initialize default values."""
        if self.scores is None:
            self.scores = []
        if self.gen_history is None:
            self.gen_history = []
        if self.pending_steps is None:
            self.pending_steps = []
        if self.step_hashes is None:
            self.step_hashes = []

    @property
    def stop_label(self) -> str:
        """Readable stop reason for logging."""
        raw = self.stop_reasons[0] if self.stop_reasons else None
        if raw == "\n\n":
            return "step"
        if raw == "EOS":
            return "eos"
        if raw == "length":
            return "length"
        if raw is None:
            fr = self.finish_reasons[0] if self.finish_reasons else None
            if fr == "length":
                return "length"
            if fr == "stop":
                return "eos"
            return "?"
        return repr(raw)
