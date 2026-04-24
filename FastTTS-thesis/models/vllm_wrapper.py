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

"""Process-isolated wrappers around the generator and verifier vLLM engines.

Each model lives in its own ``multiprocessing`` process so that the two
engines can each hold their own CUDA context without interfering (vLLM's
sleep-mode allocator in particular conflicts between colocated engines).
Requests flow over a ``Pipe`` as ``{"action": name, ...}`` dicts; the worker
dispatches via :data:`_WORKER_HANDLERS`. ``ProcessTokenizerWrapper`` mirrors
the HuggingFace tokenizer API over the same pipe.
"""

import os


def _ensure_v1_env() -> None:
    """Set env vars required by FastTTS's V1 integration (idempotent)."""
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Prevent fragmentation OOM when two engines share one GPU (see migration
    # doc §3). Incompatible with CuMemAllocator (sleep mode); when offloading
    # is enabled, sleep puts one engine out of the way so fragmentation
    # headroom is no longer an issue.
    cur = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" not in cur:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            f"{cur},expandable_segments:True" if cur else "expandable_segments:True"
        )


_ensure_v1_env()

import logging
import multiprocessing as mp
import traceback
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import torch

from config import FastTTSConfig
from .tts_llm import TTSLLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker request handlers
# ---------------------------------------------------------------------------

def _with_sleep_wake(fn: Callable) -> Callable:
    """Wrap a handler so the model is woken before the call and slept after.

    No-op when ``ctx.enable_sleep_mode`` is False.
    """
    def wrapped(ctx: "WorkerContext", request: dict) -> Any:
        if ctx.enable_sleep_mode:
            ctx.model.wake_up()
        try:
            return fn(ctx, request)
        finally:
            if ctx.enable_sleep_mode:
                ctx.model.sleep()
    return wrapped


@_with_sleep_wake
def _handle_generate(ctx, request):
    return {"result": ctx.model.generate_text(request["prompts"], **request.get("kwargs", {}))}


@_with_sleep_wake
def _handle_score(ctx, request):
    return {"result": ctx.model.score_outputs(
        request["questions"], request["outputs"],
        request.get("system_prompt", ""), **request.get("kwargs", {}),
    )}


def _handle_tokenizer_info(ctx, request):
    return {"tokenizer_info": {
        "name": getattr(ctx.tokenizer, "name_or_path", "unknown"),
        "vocab_size": getattr(ctx.tokenizer, "vocab_size", 0),
    }}


def _handle_apply_chat_template(ctx, request):
    return {"result": ctx.tokenizer.apply_chat_template(
        request["conversations"],
        add_generation_prompt=request.get("add_generation_prompt", True),
        continue_final_message=request.get("continue_final_message", False),
        tokenize=request.get("tokenize", False),
    )}


def _handle_tokenize(ctx, request):
    return {"tokens": ctx.tokenizer.tokenize(request["text"])}


def _handle_encode(ctx, request):
    return {"token_ids": ctx.tokenizer.encode(request["text"])}


def _handle_decode(ctx, request):
    return {"text": ctx.tokenizer.decode(request["token_ids"])}


class _CacheStatsAcc:
    """Running sum of PrefixCacheStats across engine steps."""
    __slots__ = ("queries", "hits", "requests")

    def __init__(self):
        self.queries = 0
        self.hits = 0
        self.requests = 0

    def add(self, s) -> None:
        if s is None:
            return
        self.queries += getattr(s, "queries", 0) or 0
        self.hits += getattr(s, "hits", 0) or 0
        self.requests += getattr(s, "requests", 0) or 0

    def to_dict(self) -> dict:
        return {"queries": self.queries, "hits": self.hits, "requests": self.requests}


class _TransferStatsAcc:
    """Running sum of KV transfer bytes/time across engine steps."""

    def __init__(self):
        self.by_type: dict[str, dict] = {}  # e.g. "cpu_to_gpu" -> {bytes, time_s, count}

    def add(self, kv_connector_stats) -> None:
        """Accumulate from SchedulerStats.kv_connector_stats (a dict or None)."""
        if not kv_connector_stats:
            return
        for transfer_type, ops in kv_connector_stats.items():
            if not isinstance(ops, list):
                continue
            if transfer_type not in self.by_type:
                self.by_type[transfer_type] = {"bytes": 0, "time_s": 0.0, "count": 0}
            acc = self.by_type[transfer_type]
            for op in ops:
                acc["bytes"] += op.get("op_size", 0) if isinstance(op, dict) else 0
                acc["time_s"] += op.get("op_time", 0) if isinstance(op, dict) else 0
                acc["count"] += 1

    def to_dict(self) -> dict | None:
        return self.by_type if self.by_type else None


class _BatchStatsAcc:
    """Per-step histogram of scheduler batch sizes.

    Populated from SchedulerStats.num_running_reqs / num_waiting_reqs /
    kv_cache_usage on every make_stats() call. Useful for answering
    "how many beams were actually in-flight concurrently" without adding
    per-step logging noise.
    """

    # Right-open bin edges: a step's running_reqs lands in the first bin
    # whose upper edge is > running_reqs. The implicit final bin catches
    # anything above the last edge (>=256 under our configs).
    BINS = (1, 8, 32, 64, 128, 256)
    BIN_LABELS = ("0", "1-7", "8-31", "32-63", "64-127", "128-255", "256+")

    def __init__(self):
        self.steps_total = 0
        self.steps_nonzero = 0  # steps with num_running_reqs > 0
        self.steps_queued = 0  # steps with num_waiting_reqs > 0 — throttling signal
        self.running_sum = 0
        self.running_max = 0
        self.waiting_sum = 0  # sum over steps_queued, for mean-when-queued
        self.waiting_max = 0
        self.kv_usage_max = 0.0
        self.histogram = [0] * (len(self.BINS) + 1)

    def add(self, stats) -> None:
        if stats is None:
            return
        n_run = getattr(stats, "num_running_reqs", 0) or 0
        n_wait = getattr(stats, "num_waiting_reqs", 0) or 0
        kv = getattr(stats, "kv_cache_usage", 0.0) or 0.0
        self.steps_total += 1
        if n_run > 0:
            self.steps_nonzero += 1
            self.running_sum += n_run
        if n_wait > 0:
            self.steps_queued += 1
            self.waiting_sum += n_wait
        if n_run > self.running_max:
            self.running_max = n_run
        if n_wait > self.waiting_max:
            self.waiting_max = n_wait
        if kv > self.kv_usage_max:
            self.kv_usage_max = kv
        # Find the first bin edge strictly greater than n_run; if none, land
        # in the final overflow bucket.
        idx = len(self.BINS)
        for i, edge in enumerate(self.BINS):
            if n_run < edge:
                idx = i
                break
        self.histogram[idx] += 1

    def to_dict(self) -> dict:
        mean_running = (
            self.running_sum / self.steps_nonzero if self.steps_nonzero else 0.0
        )
        mean_waiting_when_queued = (
            self.waiting_sum / self.steps_queued if self.steps_queued else 0.0
        )
        frac_steps_queued = (
            self.steps_queued / self.steps_total if self.steps_total else 0.0
        )
        return {
            "steps_total": self.steps_total,
            "steps_nonzero": self.steps_nonzero,
            "steps_queued": self.steps_queued,
            "frac_steps_queued": frac_steps_queued,
            "mean_running": mean_running,  # mean over non-idle steps
            "max_running": self.running_max,
            "mean_waiting_when_queued": mean_waiting_when_queued,
            "max_waiting": self.waiting_max,
            "max_kv_usage": self.kv_usage_max,
            "histogram": dict(zip(self.BIN_LABELS, self.histogram)),
        }


def _install_prefix_cache_accumulator(scheduler) -> None:
    """Wrap scheduler.make_stats so per-step stats are summed for end-of-run reporting.

    Accumulates four categories:
    - GPU prefix cache (queries/hits)
    - CPU prefix cache (queries/hits, from OffloadingConnector)
    - KV transfer volume (bytes/time per direction, from OffloadingConnectorStats)
    - Per-step batch sizes (num_running_reqs histogram, KV usage peak)

    All are drained every engine step by make_stats; the wrapper intercepts
    the returned SchedulerStats and adds to the accumulators.
    """
    if getattr(scheduler, "_prefix_cache_acc_installed", False):
        return
    scheduler._acc_gpu_prefix = _CacheStatsAcc()
    scheduler._acc_cpu_prefix = _CacheStatsAcc()
    scheduler._acc_transfers = _TransferStatsAcc()
    scheduler._acc_batch = _BatchStatsAcc()
    original = scheduler.make_stats

    def wrapped(*args, **kwargs):
        stats = original(*args, **kwargs)
        if stats is not None:
            scheduler._acc_gpu_prefix.add(getattr(stats, "prefix_cache_stats", None))
            scheduler._acc_cpu_prefix.add(getattr(stats, "connector_prefix_cache_stats", None))
            scheduler._acc_transfers.add(getattr(stats, "kv_connector_stats", None))
            scheduler._acc_batch.add(stats)
        return stats

    scheduler.make_stats = wrapped
    scheduler._prefix_cache_acc_installed = True


def _handle_get_run_stats(ctx, request):
    """Report run-total prefix cache counters, KV transfer volume, batch stats."""
    scheduler = ctx.model.llm_engine.engine_core.engine_core.scheduler
    gpu = getattr(scheduler, "_acc_gpu_prefix", None)
    cpu = getattr(scheduler, "_acc_cpu_prefix", None)
    xfer = getattr(scheduler, "_acc_transfers", None)
    batch = getattr(scheduler, "_acc_batch", None)
    return {
        "result": {
            "gpu": gpu.to_dict() if gpu else None,
            "cpu": (cpu.to_dict() if (cpu and cpu.requests > 0) else None),
            "transfers": xfer.to_dict() if xfer else None,
            "batch": batch.to_dict() if batch else None,
        }
    }


_WORKER_HANDLERS: Dict[str, Callable] = {
    "generate": _handle_generate,
    "score": _handle_score,
    "get_tokenizer_info": _handle_tokenizer_info,
    "apply_chat_template": _handle_apply_chat_template,
    "tokenize": _handle_tokenize,
    "encode": _handle_encode,
    "decode": _handle_decode,
    "get_run_stats": _handle_get_run_stats,
}


class WorkerContext:
    """Per-worker state shared across every request handler."""

    __slots__ = ("model", "tokenizer", "enable_sleep_mode")

    def __init__(self, model: TTSLLM, tokenizer, enable_sleep_mode: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.enable_sleep_mode = enable_sleep_mode


# ---------------------------------------------------------------------------
# Base wrapper
# ---------------------------------------------------------------------------

class BaseVLLMModelWrapper(ABC):
    """Base class for VLLM model wrappers."""

    def __init__(self, config: FastTTSConfig, enable_sleep_mode: bool = False):
        self.config = config
        self.enable_sleep_mode = enable_sleep_mode
        self.model = None
        self.process = None
        self._initialize_model()

    def _initialize_model(self):
        """Spawn the worker process and wait for its initialization ack."""
        vllm_cfg = (
            self.config.generator_vllm_config if self.model_type == "generator"
            else self.config.verifier_vllm_config
        )
        logger.info(f"Initializing {self.model_type} model: {vllm_cfg['model']}")

        self.parent_conn, child_conn = mp.Pipe()
        self.process = mp.Process(
            target=_model_process_worker,
            args=(child_conn, self._get_model_kwargs(), self.model_type),
        )
        if mp.get_start_method() != "spawn":
            logger.warning("Setting start method to 'spawn'")
            mp.set_start_method("spawn", force=True)
        self.process.start()

        result = self.parent_conn.recv()
        if result.get("error") or result.get("status") != "initialized":
            raise RuntimeError(f"Failed to initialize {self.model_type} model: {result.get('error')}")
        logger.info(f"{self.model_type.capitalize()} model initialized successfully in separate process")

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type (generator or verifier)."""

    @abstractmethod
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model initialization arguments."""

    def get_tokenizer(self):
        if self.process:
            return ProcessTokenizerWrapper(self.parent_conn)
        return self.model.get_tokenizer()

    def get_tokenizer_info(self):
        if self.process:
            self.parent_conn.send({"action": "get_tokenizer_info"})
            return self.parent_conn.recv().get("tokenizer_info")
        return None

    def get_run_stats(self):
        """Snapshot per-engine run totals. Should be called once near
        end-of-run before shutdown. Returns a dict with keys 'gpu', 'cpu',
        'transfers', 'batch' — each either None or a stats sub-dict.
        """
        if self.process:
            try:
                self.parent_conn.send({"action": "get_run_stats"})
                return self.parent_conn.recv().get("result")
            except (BrokenPipeError, EOFError):
                return None
        return None

    def shutdown(self):
        if self.process:
            try:
                self.parent_conn.send({"action": "shutdown"})
                self.process.join(timeout=10)
            except (BrokenPipeError, EOFError):
                pass
            finally:
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
                self.parent_conn.close()
                self.process = None
        elif self.model:
            del self.model
        logger.info(f"{self.model_type.capitalize()} model shutdown complete")


# ---------------------------------------------------------------------------
# Worker entrypoint
# ---------------------------------------------------------------------------

def _model_process_worker(conn, model_kwargs, model_type):
    """Worker target: initialize model, then dispatch requests via handler registry."""
    try:
        _ensure_v1_env()

        model = TTSLLM(**model_kwargs)
        # Wrap make_stats to accumulate prefix-cache counters across steps.
        # Required because vLLM resets counters on every read (make_stats is
        # called each engine step), so the raw end-of-run read is ~empty.
        try:
            _install_prefix_cache_accumulator(
                model.llm_engine.engine_core.engine_core.scheduler
            )
        except Exception as install_exc:  # non-fatal
            logger.warning(f"prefix cache accumulator not installed: {install_exc}")
        ctx = WorkerContext(
            model=model,
            tokenizer=model.get_tokenizer(),
            enable_sleep_mode=model_kwargs.get("enable_sleep_mode", False),
        )
        if ctx.enable_sleep_mode:
            model.sleep()
        conn.send({"status": "initialized"})

        while True:
            try:
                request = conn.recv()
            except EOFError:
                break
            action = request.get("action")
            if action == "shutdown":
                break
            handler = _WORKER_HANDLERS.get(action)
            if handler is None:
                conn.send({"error": f"unknown action: {action!r}"})
                continue
            conn.send(handler(ctx, request))
    except Exception as e:
        conn.send({"error": f"{e}\n{traceback.format_exc()}"})
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        conn.close()


# ---------------------------------------------------------------------------
# Generator / verifier wrappers
# ---------------------------------------------------------------------------

class GeneratorVLLMModelWrapper(BaseVLLMModelWrapper):
    """VLLM model wrapper for generator models."""

    @property
    def model_type(self) -> str:
        return "generator"

    def _get_model_kwargs(self) -> Dict[str, Any]:
        return {
            **self.config.generator_vllm_config,
            "enable_sleep_mode": self.enable_sleep_mode,
            "spec_beam_extension": self.config.spec_beam_extension,
            "prefix_aware_scheduling": self.config.prefix_aware_scheduling,
        }

    def generate(self, prompts: List[str], **kwargs):
        if self.process:
            self.parent_conn.send({
                "action": "generate",
                "prompts": prompts,
                "kwargs": kwargs,
            })
            result = self.parent_conn.recv()
            if result.get("error"):
                raise RuntimeError(f"Failed to generate: {result['error']}")
            return result["result"]
        results = self.model.generate_text(prompts, **kwargs)
        logger.info(f"Generator model generated {len(results)} results")
        return results


class VerifierVLLMModelWrapper(BaseVLLMModelWrapper):
    """VLLM model wrapper for verifier models."""

    @property
    def model_type(self) -> str:
        return "verifier"

    def _get_model_kwargs(self) -> Dict[str, Any]:
        return {
            **self.config.verifier_vllm_config,
            "enable_sleep_mode": self.enable_sleep_mode,
            "prefix_aware_scheduling": self.config.prefix_aware_scheduling,
            "runner": "pooling",
        }

    def score(self, questions: List[str], outputs: List[List[str]], **kwargs):
        if self.process:
            self.parent_conn.send({
                "action": "score",
                "questions": questions,
                "outputs": outputs,
                "system_prompt": self.config.search_config.system_prompt,
                "kwargs": kwargs,
            })
            result = self.parent_conn.recv()
            if result.get("error"):
                raise RuntimeError(f"Failed to score: {result['error']}")
            return result["result"]
        return self.model.score_outputs(
            questions, outputs, self.config.search_config.system_prompt, **kwargs,
        )


# ---------------------------------------------------------------------------
# ProcessTokenizerWrapper — RPC-backed HF tokenizer proxy
# ---------------------------------------------------------------------------

class ProcessTokenizerWrapper:
    """Wrapper for tokenizer functionality when using process-based models."""

    def __init__(self, parent_conn):
        self.parent_conn = parent_conn

    def _rpc(self, action: str, response_key: str, default=None, **payload):
        self.parent_conn.send({"action": action, **payload})
        return self.parent_conn.recv().get(response_key, default)

    def apply_chat_template(self, conversations, add_generation_prompt=True,
                            continue_final_message=False, tokenize=False):
        return self._rpc(
            "apply_chat_template", "result",
            conversations=conversations,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=tokenize,
        )

    def tokenize(self, text):
        return self._rpc("tokenize", "tokens", default=[], text=text)

    def encode(self, text):
        return self._rpc("encode", "token_ids", default=[], text=text)

    def decode(self, token_ids):
        return self._rpc("decode", "text", default="", token_ids=token_ids)
