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


_WORKER_HANDLERS: Dict[str, Callable] = {
    "generate": _handle_generate,
    "score": _handle_score,
    "get_tokenizer_info": _handle_tokenizer_info,
    "apply_chat_template": _handle_apply_chat_template,
    "tokenize": _handle_tokenize,
    "encode": _handle_encode,
    "decode": _handle_decode,
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
