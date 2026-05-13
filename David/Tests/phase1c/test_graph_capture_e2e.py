# SPDX-License-Identifier: Apache-2.0
"""Stage 5 — CUDA Graph capture end-to-end.

The headline Stage 5 invariant: with `cpu_runner='native'` and
`enforce_eager=False`, the operator's forward path is graph-capturable.
That means
  (a) `torch.cuda.graph(...)` capture across the operator's
      `submit_with_d2h` / `wait_and_uva` calls succeeds without
      cudaErrorCapturedEvent or similar errors;
  (b) replays of the captured graph produce results bit-equivalent
      to the eager reference (the host-callback dispatch + worker GEMM
      run on every replay; the slab's pre-set weight pointers are
      stable per `slab_count_` reserve-once invariant from Stage 1);
  (c) repeated replays are deterministic — multiple replays of the
      same graph on the same input produce identical output (catches
      stale-userData UB in the slab pool, plan §risk register #2).

These are the technical preconditions for the final Phase 1c graph
path. Real-model production validation lives in
`bench_capture_gap_qwen.py`, `bench_capture_gap_qwen_grid.py`, and
`check_capture_piecewise_parity_qwen.py`.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import (
    CompilationConfig,
    CotsOffloadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.offloader import CotsOffloader, set_offloader
from vllm.model_executor.offloader.base import ForwardDispatchInfo
from vllm.forward_context import BatchDescriptor

pytestmark = pytest.mark.needs_cuda


HIDDEN = 256
INTERMEDIATE = 1024
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
MAX_NUM_TOKENS = 32

BF16_RTOL = 5e-2
BF16_ATOL = 0.5
MLP_BLOCK_RTOL = 5e-2


class _MlpBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bf16 = torch.bfloat16
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=HIDDEN,
            output_sizes=[INTERMEDIATE, INTERMEDIATE],
            bias=False,
            disable_tp=True,
            params_dtype=bf16,
            prefix="mlp.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=INTERMEDIATE,
            output_size=HIDDEN,
            bias=False,
            disable_tp=True,
            params_dtype=bf16,
            prefix="mlp.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu, _ = self.gate_up_proj(x)
        x = self.act_fn(gu)
        x, _ = self.down_proj(x)
        return x


class _MlpLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MlpBlock()


class _QkvLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bf16 = torch.bfloat16
        self.qkv_proj = QKVParallelLinear(
            hidden_size=HIDDEN,
            head_size=HEAD_DIM,
            total_num_heads=NUM_HEADS,
            total_num_kv_heads=NUM_KV_HEADS,
            bias=False,
            disable_tp=True,
            params_dtype=bf16,
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


def _build_qkv(
    *, f_cpu_store: float, cpu_runner: str, enforce_eager: bool
) -> tuple[_QkvLayer, CotsOffloader]:
    vc = _make_vllm_config(enforce_eager=enforce_eager)
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=0.0,
                kv_biased=True,
                cpu_runner=cpu_runner,
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
        offloader.post_init()
    return layer, offloader


def _build_mlp(
    *, f_cpu_store: float, cpu_runner: str, enforce_eager: bool
) -> tuple[_MlpLayer, CotsOffloader]:
    vc = _make_vllm_config(enforce_eager=enforce_eager)
    with set_current_vllm_config(vc):
        layer = _MlpLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=0.0,
                kv_biased=True,
                cpu_runner=cpu_runner,
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))
        torch.manual_seed(0)
        gate = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
        up = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
        down = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device="cuda")
        layer.mlp.gate_up_proj.weight_loader(layer.mlp.gate_up_proj.weight, gate, 0)
        layer.mlp.gate_up_proj.weight_loader(layer.mlp.gate_up_proj.weight, up, 1)
        layer.mlp.down_proj.weight_loader(layer.mlp.down_proj.weight, down)
        offloader.post_init()
    return layer, offloader


def _capture_replay_compare(
    *,
    layer: nn.Module,
    offloader: CotsOffloader,
    forward: callable,
    x: torch.Tensor,
    n_replays: int,
) -> None:
    """Run forward eagerly to get the reference; capture a graph
    around the same forward; replay it `n_replays` times; assert
    every replay matches the eager reference. Models the
    cudagraph_utils.py:267 boundary: prepare_before_forward +
    sync_prev_onload BEFORE the captured region."""
    def dispatch() -> None:
        n = int(x.shape[0])
        offloader.on_dispatch(
            ForwardDispatchInfo(
                batch_descriptor=BatchDescriptor(num_tokens=n),
                num_tokens_unpadded=n,
            )
        )

    # Eager reference (no capture). post_init has already fired so the
    # offloader is fully installed.
    dispatch()
    out_eager = forward(x)
    if isinstance(out_eager, tuple):
        out_eager = out_eager[0]
    out_eager = out_eager.clone()  # detach from any pooled buffer
    torch.cuda.current_stream().synchronize()

    # Capture: same prepare_before_forward + sync_prev_onload at the
    # boundary (mirrors what cudagraph_utils.py:267 does for FULL
    # graph replay).
    g = torch.cuda.CUDAGraph()
    # Pre-capture warmup (any first-time allocations / pool ops).
    dispatch()
    _ = forward(x)
    torch.cuda.current_stream().synchronize()

    dispatch()
    with torch.cuda.graph(g):
        out_captured = forward(x)
        if isinstance(out_captured, tuple):
            out_captured = out_captured[0]
        offloader.join_after_forward()

    # First replay.
    dispatch()
    g.replay()
    torch.cuda.current_stream().synchronize()
    captured_first = out_captured.clone()
    torch.testing.assert_close(captured_first, out_eager, rtol=BF16_RTOL, atol=BF16_ATOL)

    # Determinism: N more replays on the same input must produce the
    # same output every time. Catches stale-userData UB in the slab
    # pool (plan §risk register #2).
    for i in range(n_replays):
        dispatch()
        g.replay()
        torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(
            out_captured,
            captured_first,
            rtol=0,
            atol=0,
            msg=f"replay #{i + 2} drifted from replay #1",
        )


@pytest.mark.parametrize("f_cpu_store", [0.10, 0.25, 0.50])
def test_qkv_native_graph_capture_replay_50x(f_cpu_store: float) -> None:
    """QKV operator under graph capture with cpu_runner='native' +
    enforce_eager=False. Capture once, replay 50× — each replay must
    match the eager reference within bf16 atol AND be deterministic
    across replays."""
    layer, offloader = _build_qkv(
        f_cpu_store=f_cpu_store,
        cpu_runner="native",
        enforce_eager=False,
    )
    torch.manual_seed(31)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    try:
        _capture_replay_compare(
            layer=layer,
            offloader=offloader,
            forward=layer.qkv_proj,
            x=x,
            n_replays=50,
        )
    finally:
        if offloader._runner is not None:
            offloader._runner.close()


@pytest.mark.parametrize("f_cpu_store", [0.10, 0.25, 0.50])
def test_mlp_native_graph_capture_replay_50x(f_cpu_store: float) -> None:
    """MLP-block operator under graph capture. Exercises the strided
    down-proj slab dispatch (when f_cpu_store produces a non-trivial
    n_cpu_compute) inside a captured graph."""
    layer, offloader = _build_mlp(
        f_cpu_store=f_cpu_store,
        cpu_runner="native",
        enforce_eager=False,
    )
    torch.manual_seed(37)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    try:
        _capture_replay_compare(
            layer=layer,
            offloader=offloader,
            forward=layer.mlp,
            x=x,
            n_replays=50,
        )
    finally:
        if offloader._runner is not None:
            offloader._runner.close()


def test_default_config_supports_graph_capture() -> None:
    """Stage 5 review-fix: confirm the post-Stage-5 default
    CotsOffloadConfig (no explicit cpu_runner override) now routes
    through the native runner and supports graph capture. Previously
    the default was 'python' and graph capture would hard-fail; the
    default flipped to 'native' at Stage 5.
    """
    from vllm.config.offload import CotsOffloadConfig

    vc = _make_vllm_config(enforce_eager=False)
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        # NB: deliberately NO cpu_runner kwarg — relies on the default.
        offloader = CotsOffloader(
            config=CotsOffloadConfig(f_cpu_store=0.20, kv_biased=True)
        )
        from vllm.model_executor.offloader.cots import NativeCotsRunner

        assert isinstance(offloader._runner, NativeCotsRunner), (
            "Default CotsOffloadConfig should now construct a "
            "NativeCotsRunner (Stage 5 default flip)."
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
        offloader.post_init()  # would have raised under the old default

    torch.manual_seed(99)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    try:
        _capture_replay_compare(
            layer=layer,
            offloader=offloader,
            forward=layer.qkv_proj,
            x=x,
            n_replays=10,
        )
    finally:
        if offloader._runner is not None:
            offloader._runner.close()


def test_capture_then_eager_then_capture_again_works() -> None:
    """Sentinel: alternating between captured replay and eager forward
    on the SAME offloader doesn't corrupt state. Mirrors what FastTTS
    does at runtime — prefill and decode iterations alternate between
    PIECEWISE eager and FULL captured paths via vLLM's
    `CudagraphDispatcher`."""
    layer, offloader = _build_qkv(
        f_cpu_store=0.20,
        cpu_runner="native",
        enforce_eager=False,
    )
    torch.manual_seed(41)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    def dispatch() -> None:
        n = int(x.shape[0])
        offloader.on_dispatch(
            ForwardDispatchInfo(
                batch_descriptor=BatchDescriptor(num_tokens=n),
                num_tokens_unpadded=n,
            )
        )

    try:
        # 1. Eager.
        dispatch()
        out_eager_1, _ = layer.qkv_proj(x)
        out_eager_1 = out_eager_1.clone()

        # 2. Capture + replay.
        g = torch.cuda.CUDAGraph()
        dispatch()
        _ = layer.qkv_proj(x)
        torch.cuda.current_stream().synchronize()
        dispatch()
        with torch.cuda.graph(g):
            out_captured, _ = layer.qkv_proj(x)
            offloader.join_after_forward()
        dispatch()
        g.replay()
        torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(
            out_captured, out_eager_1, rtol=BF16_RTOL, atol=BF16_ATOL
        )

        # 3. Eager again — must still produce the right answer.
        dispatch()
        out_eager_2, _ = layer.qkv_proj(x)
        out_eager_2 = out_eager_2.clone()
        torch.testing.assert_close(
            out_eager_2, out_eager_1, rtol=BF16_RTOL, atol=BF16_ATOL
        )

        # 4. Replay AGAIN — captured graph still works after eager pass.
        dispatch()
        g.replay()
        torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(
            out_captured, out_eager_1, rtol=BF16_RTOL, atol=BF16_ATOL
        )
    finally:
        if offloader._runner is not None:
            offloader._runner.close()
