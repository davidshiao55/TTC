"""GeneratorLLMEngineV1 — V1 LLMEngine subclass for the FastTTS generator.

Implements Speculative Beam Extension (SBE) as described in FastTTS §4.1:
- Phase 1 (Continuous Beam Batching): finish beams at stop strings when
  the waiting queue has work.
- Phase 2 (Speculative Execution): continue generating past stop strings
  when the waiting queue is empty.
- Force-finish all speculative beams when every standard beam completes.
- Preemption cleanup for speculative beams evicted by the scheduler.
- Overflow protection (256-beam cap).

Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 for direct scheduler access.
"""

from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass, field

from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import FinishReason
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine

from .numbers import SPEC_BEAM_CANDIDATE_PRIORITY

logger = logging.getLogger(__name__)

MAX_SPEC_BEAMS = 256


@dataclass
class SBETracker:
    """Per-batch state for Speculative Beam Extension."""

    stop_strings: list[str] = field(default_factory=list)
    include_stop_str: bool = False
    total_requests: int = 0

    # req_id → char offset where the *first* stop string was detected
    stopped_at: dict[str, int] = field(default_factory=dict)

    # req_ids currently in Phase 2 (speculative continuation)
    speculative: set[str] = field(default_factory=set)

    # req_ids that are truly finished (naturally or Phase 1)
    finished: set[str] = field(default_factory=set)


class GeneratorLLMEngineV1(V1LLMEngine):
    """V1 LLMEngine subclass for the FastTTS generator with SBE support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sbe_enabled: bool = False
        self._sbe_tracker: SBETracker | None = None
        self._scheduler = None  # direct scheduler reference

    # ------------------------------------------------------------------
    # SBE lifecycle
    # ------------------------------------------------------------------

    def enable_spec_beam_extension(self) -> None:
        """Activate SBE.  Must be called after engine creation.

        Requires:
        - VLLM_ENABLE_V1_MULTIPROCESSING=0 (InprocClient)
        - scheduling_policy="priority" in EngineArgs
        """
        from vllm.v1.core.sched.request_queue import SchedulingPolicy
        from vllm.v1.engine.core_client import InprocClient

        if not isinstance(self.engine_core, InprocClient):
            raise RuntimeError(
                "SBE requires VLLM_ENABLE_V1_MULTIPROCESSING=0 "
                "(InprocClient), but got "
                f"{type(self.engine_core).__name__}"
            )

        self._scheduler = self.engine_core.engine_core.scheduler
        if self._scheduler.policy != SchedulingPolicy.PRIORITY:
            raise RuntimeError(
                "SBE requires scheduling_policy='priority', but got "
                f"'{self._scheduler.policy.value}'"
            )

        self._sbe_enabled = True
        logger.info("Speculative Beam Extension enabled (V1)")

    # ------------------------------------------------------------------
    # add_request override — strip stop strings when SBE is active
    # ------------------------------------------------------------------

    def add_request(self, request_id, prompt, params, **kwargs):
        if (
            self._sbe_enabled
            and isinstance(params, SamplingParams)
            and params.stop
        ):
            # Lazily create a fresh tracker for this batch.
            if self._sbe_tracker is None:
                stops = params.stop
                if isinstance(stops, str):
                    stops = [stops]
                self._sbe_tracker = SBETracker(
                    stop_strings=list(stops),
                    include_stop_str=params.include_stop_str_in_output,
                )

            self._sbe_tracker.total_requests += 1

            # Strip stop strings so neither the EngineCore nor the
            # detokenizer will trigger early termination.
            params = copy(params)
            params.stop = None

        return super().add_request(request_id, prompt, params, **kwargs)

    # ------------------------------------------------------------------
    # step override — SBE main loop
    # ------------------------------------------------------------------

    def step(self):
        if self._sbe_tracker is not None:
            return self._step_sbe()
        return super().step()

    def _step_sbe(self) -> list[RequestOutput]:
        tracker = self._sbe_tracker
        assert tracker is not None

        # 1) Run one engine step (token generation + detokenization).
        parent_outputs = super().step()

        # 2) Collect naturally-finished requests (EOS / max_tokens).
        finished_outputs: list[RequestOutput] = []
        for output in parent_outputs:
            if output.finished:
                tracker.finished.add(output.request_id)
                finished_outputs.append(output)

        # 3) For every still-active request, check for stop strings.
        active_req_ids = list(self.output_processor.request_states.keys())
        for req_id in active_req_ids:
            if req_id in tracker.finished or req_id in tracker.stopped_at:
                continue

            text = self._get_cumulative_text(req_id)
            if text is None:
                continue

            stop_pos = self._find_stop_string(text, tracker.stop_strings)
            if stop_pos < 0:
                continue

            # Stop string detected — record it.
            tracker.stopped_at[req_id] = stop_pos

            if self._scheduler.waiting:
                # Phase 1: finish beam, free resources for queued beams.
                output = self._finish_beam(req_id, tracker, truncate=True)
                if output is not None:
                    tracker.finished.add(req_id)
                    finished_outputs.append(output)
            else:
                # Phase 2: mark as speculative, keep generating.
                tracker.speculative.add(req_id)
                request = self._scheduler.requests.get(req_id)
                if request is not None:
                    request.priority = SPEC_BEAM_CANDIDATE_PRIORITY

        # 4) Check already-speculative beams that now see a waiting queue
        #    (Phase 2 → Phase 1 transition for new stop hits is handled
        #    above on the *next* step; here we handle speculative beams
        #    that were preempted by the scheduler).
        self._cleanup_preempted_speculative(tracker, finished_outputs)

        # 5) Overflow protection.
        self._enforce_overflow_limit(tracker, finished_outputs)

        # 6) Force-finish if every non-speculative beam is done.
        if self._check_all_done(tracker):
            self._force_finish_all_speculative(tracker, finished_outputs)

        # 7) If tracker is exhausted, reset for next batch.
        if not self.output_processor.has_unfinished_requests():
            self._sbe_tracker = None

        return finished_outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_cumulative_text(self, req_id: str) -> str | None:
        req_state = self.output_processor.request_states.get(req_id)
        if req_state is None or req_state.detokenizer is None:
            return None
        return req_state.detokenizer.output_text

    @staticmethod
    def _find_stop_string(text: str, stop_strings: list[str]) -> int:
        """Return char offset of the first stop-string match, or -1."""
        earliest = -1
        for s in stop_strings:
            pos = text.find(s)
            if pos >= 0 and (earliest < 0 or pos < earliest):
                earliest = pos
        return earliest

    def _finish_beam(
        self,
        req_id: str,
        tracker: SBETracker,
        *,
        truncate: bool,
    ) -> RequestOutput | None:
        """Create a finished RequestOutput for *req_id*, then abort it.

        If *truncate* is True (Phase 1), the text is cut at the first
        stop-string boundary.  If False (force-finish), full text is kept.
        """
        req_state = self.output_processor.request_states.get(req_id)
        if req_state is None:
            return None

        stop_pos = tracker.stopped_at.get(req_id)

        if truncate and stop_pos is not None and req_state.detokenizer is not None:
            end = stop_pos
            # Find which stop string matched so we can include it if needed
            for s in tracker.stop_strings:
                if req_state.detokenizer.output_text[stop_pos:].startswith(s):
                    if tracker.include_stop_str:
                        end = stop_pos + len(s)
                    break
            req_state.detokenizer.output_text = (
                req_state.detokenizer.output_text[:end]
            )

        # Determine the stop_reason string (first matching stop string)
        stop_reason: str | None = None
        if stop_pos is not None and req_state.detokenizer is not None:
            full_text = req_state.detokenizer.output_text
            for s in tracker.stop_strings:
                # After possible truncation, the stop string may be at the
                # very end (include_stop_str) or just past the end.
                if stop_pos < len(full_text) + len(s):
                    stop_reason = s
                    break
        if stop_reason is None and tracker.stop_strings:
            stop_reason = tracker.stop_strings[0]

        # Create the finished RequestOutput via the existing API.
        output = req_state.make_request_output(
            new_token_ids=[],
            pooling_output=None,
            finish_reason=FinishReason.STOP,
            stop_reason=stop_reason,
        )

        # Clean up: remove from output_processor, abort in engine_core.
        self.output_processor._finish_request(req_state)
        self.engine_core.abort_requests([req_id])

        # Also remove from speculative set if present.
        tracker.speculative.discard(req_id)

        return output

    def _cleanup_preempted_speculative(
        self,
        tracker: SBETracker,
        finished_outputs: list[RequestOutput],
    ) -> None:
        """Abort speculative beams that were preempted by the scheduler."""
        from vllm.v1.request import RequestStatus

        to_remove: list[str] = []
        for req_id in tracker.speculative:
            request = self._scheduler.requests.get(req_id)
            if request is None:
                # Already cleaned up externally.
                to_remove.append(req_id)
                continue
            if request.status == RequestStatus.PREEMPTED:
                # Preempted by scheduler — abort with full text (no truncation).
                output = self._finish_beam(req_id, tracker, truncate=False)
                if output is not None:
                    tracker.finished.add(req_id)
                    finished_outputs.append(output)
                to_remove.append(req_id)

        for req_id in to_remove:
            tracker.speculative.discard(req_id)

    def _enforce_overflow_limit(
        self,
        tracker: SBETracker,
        finished_outputs: list[RequestOutput],
    ) -> None:
        """Cap total active (running + speculative) at MAX_SPEC_BEAMS."""
        active_count = len(self.output_processor.request_states)
        speculative_list = sorted(tracker.speculative)  # deterministic order

        idx = 0
        while active_count > MAX_SPEC_BEAMS and idx < len(speculative_list):
            req_id = speculative_list[idx]
            idx += 1
            output = self._finish_beam(req_id, tracker, truncate=False)
            if output is not None:
                tracker.finished.add(req_id)
                finished_outputs.append(output)
                active_count -= 1

    def _check_all_done(self, tracker: SBETracker) -> bool:
        """True when every non-speculative beam is finished."""
        # finished + speculative should account for all requests
        return (
            len(tracker.finished) + len(tracker.speculative)
            >= tracker.total_requests
        )

    def _force_finish_all_speculative(
        self,
        tracker: SBETracker,
        finished_outputs: list[RequestOutput],
    ) -> None:
        """Terminate all speculative beams — all standard beams are done."""
        spec_ids = list(tracker.speculative)
        for req_id in spec_ids:
            output = self._finish_beam(req_id, tracker, truncate=False)
            if output is not None:
                tracker.finished.add(req_id)
                finished_outputs.append(output)
