"""Logits processor used by the Phase 2 forced-context parity probe.

The processor records logprobs from the unmodified model logits, then forces
the next token to follow a reference continuation. This lets the parity probe
compare GPU-only and hybrid KV over the same token context after the split.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor


@dataclass
class _ReqState:
    prompt_tail: int
    output_ids: list[int]
    forced_tokens: list[int]


class CaptureForceLogitsProcessor(LogitsProcessor):
    """Capture raw logprobs and force a reference token continuation."""

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra = sampling_params.extra_args or {}
        if "forced_by_prompt_tail" not in extra:
            raise ValueError(
                "CaptureForceLogitsProcessor requires forced_by_prompt_tail"
            )

    def __init__(
        self,
        vllm_config: Any,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        self.req_info: dict[int, _ReqState] = {}
        self.mode = os.environ.get("COTS_FORCE_LOGITS_MODE", "unknown")
        self.out_path = os.environ.get(
            "COTS_FORCE_LOGITS_OUT", "/tmp/phase2_forced_logits.jsonl"
        )
        self.topk = int(os.environ.get("COTS_FORCE_LOGITS_TOPK", "20"))
        self._fh = open(self.out_path, "a", buffering=1)

    def is_argmax_invariant(self) -> bool:
        return False

    def _new_state(
        self,
        params: SamplingParams,
        prompt_ids: list[int] | None,
        output_ids: list[int],
    ) -> _ReqState | None:
        if prompt_ids is None:
            return None
        extra = params.extra_args or {}
        forced_by_tail = extra.get("forced_by_prompt_tail")
        if not isinstance(forced_by_tail, dict):
            return None
        prompt_tail = int(prompt_ids[-1])
        forced = forced_by_tail.get(str(prompt_tail))
        if forced is None:
            forced = forced_by_tail.get(prompt_tail)
        if forced is None:
            return None
        return _ReqState(
            prompt_tail=prompt_tail,
            output_ids=output_ids,
            forced_tokens=[int(token) for token in forced],
        )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.req_info, batch_update, self._new_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_info:
            return logits
        with torch.no_grad():
            for req_idx, state in list(self.req_info.items()):
                step = len(state.output_ids)
                if step >= len(state.forced_tokens):
                    continue
                row = logits[req_idx]
                forced = int(state.forced_tokens[step])
                row_f32 = row.float()
                logsumexp = torch.logsumexp(row_f32, dim=-1)
                forced_logprob = float(row_f32[forced].item() - logsumexp.item())
                k = min(self.topk, row.numel())
                top_vals, top_ids = torch.topk(row_f32, k=k)
                top_logprobs = (top_vals - logsumexp).cpu().tolist()
                rec = {
                    "mode": self.mode,
                    "time": time.time(),
                    "req_idx": int(req_idx),
                    "prompt_tail": state.prompt_tail,
                    "step": int(step),
                    "forced_token": forced,
                    "forced_logprob": forced_logprob,
                    "top_ids": [int(token) for token in top_ids.cpu().tolist()],
                    "top_logprobs": [float(value) for value in top_logprobs],
                }
                self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                row.fill_(float("-inf"))
                row[forced] = 0.0
        return logits
