# SPDX-License-Identifier: Apache-2.0
"""Stage 5 — Python runner + graph capture is a HARD ERROR.

The Python runner's substrate is `ThreadPoolExecutor.submit` +
`future.result()`. Neither is graph-capturable: under
`torch.cuda.graph(...)` capture the host-side `submit`/`result` calls
would execute at capture time but be ABSENT from the captured stream,
so replay would skip the CPU GEMM entirely and produce silently-wrong
output (zeros from y_pinned).

Stage 5 enforces this at engine launch — `CotsOffloader.post_init`
raises `RuntimeError` when `cpu_runner='python'` and
`enforce_eager=False`. The check is conditional: native runner is now
allowed under graph capture (the §1.14 collapse path).
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
import torch.nn as nn

from vllm.config import (
    CompilationConfig,
    CotsOffloadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.offloader import CotsOffloader, set_offloader

pytestmark = pytest.mark.needs_cuda


HIDDEN = 256
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
MAX_NUM_TOKENS = 64


class _QkvLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=HIDDEN,
            head_size=HEAD_DIM,
            total_num_heads=NUM_HEADS,
            total_num_kv_heads=NUM_KV_HEADS,
            bias=False,
            disable_tp=True,
            params_dtype=torch.bfloat16,
            prefix="qkv_proj",
        )


def _make_vllm_config(*, enforce_eager: bool) -> VllmConfig:
    mc = ModelConfig.__new__(ModelConfig)
    object.__setattr__(mc, "enforce_eager", enforce_eager)
    sc = SchedulerConfig.__new__(SchedulerConfig)
    object.__setattr__(sc, "max_num_batched_tokens", MAX_NUM_TOKENS)
    cc = CompilationConfig.__new__(CompilationConfig)
    object.__setattr__(cc, "cudagraph_capture_sizes", [MAX_NUM_TOKENS])
    object.__setattr__(cc, "custom_ops", ["none"])
    object.__setattr__(cc, "enabled_custom_ops", Counter())
    object.__setattr__(cc, "disabled_custom_ops", Counter())
    pc = ParallelConfig.__new__(ParallelConfig)
    object.__setattr__(pc, "tensor_parallel_size", 1)
    vc = VllmConfig.__new__(VllmConfig)
    object.__setattr__(vc, "model_config", mc)
    object.__setattr__(vc, "scheduler_config", sc)
    object.__setattr__(vc, "compilation_config", cc)
    object.__setattr__(vc, "parallel_config", pc)
    return vc


def _drive_post_init(
    *,
    cpu_runner: str,
    enforce_eager: bool,
    expect_raise: type[BaseException] | None,
    raise_match: str | None = None,
) -> None:
    """Build an offloader + run post_init in the same
    `set_current_vllm_config` context (post_init reads the global
    config, which is only set inside the manager). Optionally assert
    that post_init raises a specific exception type/message."""
    vc = _make_vllm_config(enforce_eager=enforce_eager)
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=0.10,
                cpu_runner=cpu_runner,
                kv_biased=True,
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))
        torch.manual_seed(0)
        q_size = NUM_HEADS * HEAD_DIM
        kv_size = NUM_KV_HEADS * HEAD_DIM
        q = torch.randn(q_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, q, "q")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, k, "k")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, v, "v")
        try:
            if expect_raise is not None:
                with pytest.raises(expect_raise, match=raise_match):
                    offloader.post_init()
            else:
                offloader.post_init()  # should not raise
        finally:
            if offloader._runner is not None:
                offloader._runner.close()


def test_python_runner_with_graph_capture_raises() -> None:
    """`cpu_runner='python'` + `enforce_eager=False` → post_init must
    raise RuntimeError. Capturing the python runner would silently drop
    CPU GEMM work from the graph — producing wrong outputs at replay.
    """
    _drive_post_init(
        cpu_runner="python",
        enforce_eager=False,
        expect_raise=RuntimeError,
        raise_match="cpu_runner='python' requires",
    )


def test_python_runner_with_eager_passes() -> None:
    """`cpu_runner='python'` + `enforce_eager=True` is the documented
    Phase 1a/1b configuration; must continue to work."""
    _drive_post_init(
        cpu_runner="python", enforce_eager=True, expect_raise=None
    )


def test_native_runner_with_graph_capture_passes() -> None:
    """Stage 5 production path: `cpu_runner='native'` +
    `enforce_eager=False` is now allowed (the §1.14 orch-collapse
    path). post_init completes without raising; the actual graph
    capture round-trip is verified by
    `test_native_runner_graph_capture_e2e`."""
    _drive_post_init(
        cpu_runner="native", enforce_eager=False, expect_raise=None
    )


def test_native_runner_with_eager_passes() -> None:
    """`cpu_runner='native'` + `enforce_eager=True` is also allowed —
    Stage 3's parity tests + Stage 4's thread-policy tests run in
    this configuration."""
    _drive_post_init(
        cpu_runner="native", enforce_eager=True, expect_raise=None
    )
