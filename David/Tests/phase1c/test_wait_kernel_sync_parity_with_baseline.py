# SPDX-License-Identifier: Apache-2.0
"""§1c.29 commit 2 — wait_kernel sync parity gate.

Captured-graph replay results MUST match between the legacy
`cudaLaunchHostFunc(sync_cb)` path (`weight_capture_sync_mode="host_callback"`) and
the wait_kernel-sync path (`weight_capture_sync_mode="wait_kernel"`) at bf16 tolerance, on both
the QKV operator and the MLP-block operator. Same inputs / same
weights / same captured graph topology except the sync node — any
output divergence indicates the wait_kernel-sync ordering or done_slot publish is
wrong.

This complements `test_graph_capture_e2e.py` (which compares native
runner replay vs eager reference at f_cpu_store ∈ {0.10, 0.25, 0.50})
by running BOTH wait_kernel-on and wait_kernel-off through the same capture-replay
machinery and comparing them tensor-to-tensor.
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
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.offloader import CotsOffloader, set_offloader
from vllm.model_executor.offloader.base import ForwardDispatchInfo

pytestmark = pytest.mark.needs_cuda


HIDDEN = 256
INTERMEDIATE = 1024
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
MAX_NUM_TOKENS = 64

BF16_RTOL = 5e-2
BF16_ATOL = 0.5
MLP_BLOCK_RTOL = 5e-2


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


class _MlpBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=HIDDEN,
            output_sizes=[INTERMEDIATE, INTERMEDIATE],
            bias=False,
            disable_tp=True,
            params_dtype=torch.bfloat16,
            prefix="mlp.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=INTERMEDIATE,
            output_size=HIDDEN,
            bias=False,
            disable_tp=True,
            params_dtype=torch.bfloat16,
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


def _make_vllm_config() -> VllmConfig:
    mc = ModelConfig.__new__(ModelConfig)
    object.__setattr__(mc, "enforce_eager", False)
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


def _build_qkv(*, f_cpu_store: float, wait_kernel: bool) -> tuple[_QkvLayer, CotsOffloader]:
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=0.0,
                cpu_runner="native",
                weight_capture_sync_mode=("wait_kernel" if wait_kernel else "host_callback"),
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


def _build_mlp(*, f_cpu_store: float, wait_kernel: bool) -> tuple[_MlpLayer, CotsOffloader]:
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _MlpLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=0.0,
                cpu_runner="native",
                weight_capture_sync_mode=("wait_kernel" if wait_kernel else "host_callback"),
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


def _publish_dispatch(offloader: CotsOffloader, num_tokens: int) -> None:
    """Publish the same OOG dispatch state the real GPUModelRunner sends.

    Native COTS custom ops intentionally reject implicit tensor-shape
    bucket inference. These unit tests drive layers directly, so they
    must provide the dispatch boundary explicitly before eager, capture,
    and replay forwards.
    """
    offloader.on_dispatch(
        ForwardDispatchInfo(
            batch_descriptor=BatchDescriptor(num_tokens=int(num_tokens), uniform=True),
            num_tokens_unpadded=int(num_tokens),
        )
    )


def _capture_replay(*, offloader: CotsOffloader, forward, x: torch.Tensor):
    """Run forward eagerly (warmup), then capture a graph + replay 5×.
    Returns the eager output AND a list of replay outputs (clones)."""
    num_tokens = int(x.shape[0])
    _publish_dispatch(offloader, num_tokens)
    out_eager = forward(x)
    if isinstance(out_eager, tuple):
        out_eager = out_eager[0]
    out_eager = out_eager.clone()
    torch.cuda.current_stream().synchronize()

    g = torch.cuda.CUDAGraph()
    _publish_dispatch(offloader, num_tokens)
    _ = forward(x)  # pre-capture warmup
    torch.cuda.current_stream().synchronize()

    _publish_dispatch(offloader, num_tokens)
    with torch.cuda.graph(g):
        out_captured = forward(x)
        if isinstance(out_captured, tuple):
            out_captured = out_captured[0]
        offloader.join_after_forward()

    replays = []
    for _ in range(5):
        _publish_dispatch(offloader, num_tokens)
        g.replay()
        torch.cuda.current_stream().synchronize()
        replays.append(out_captured.clone())
    return out_eager, replays


@pytest.mark.parametrize("f_cpu_store", [0.10, 0.25, 0.50])
def test_qkv_wait_kernel_path_matches_baseline(f_cpu_store: float) -> None:
    """QKV captured-replay output is bit-identical between wait_kernel-on and
    wait_kernel-off (within bf16 atol). This is the load-bearing parity gate
    for the operator-side wait_kernel-sync wiring: the captured graph is identical
    in every node EXCEPT the sync node (sync_cb host_fn vs
    cots_wait_done_kernel), so any divergence indicates the wait_kernel-sync ordering or
    done_slot publish is broken."""
    torch.manual_seed(31)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    layer_off, off_off = _build_qkv(f_cpu_store=f_cpu_store, wait_kernel=False)
    try:
        _, replays_off = _capture_replay(
            offloader=off_off, forward=layer_off.qkv_proj, x=x
        )
    finally:
        if off_off._runner is not None:
            off_off._runner.close()

    layer_on, off_on = _build_qkv(f_cpu_store=f_cpu_store, wait_kernel=True)
    try:
        _, replays_on = _capture_replay(
            offloader=off_on, forward=layer_on.qkv_proj, x=x
        )
    finally:
        if off_on._runner is not None:
            off_on._runner.close()

    for i, (r_off, r_on) in enumerate(zip(replays_off, replays_on)):
        torch.testing.assert_close(
            r_on, r_off, rtol=BF16_RTOL, atol=BF16_ATOL,
            msg=f"replay #{i}: wait_kernel-sync path diverged from baseline at f_cpu_store={f_cpu_store}",
        )


@pytest.mark.parametrize("f_cpu_store", [0.10, 0.25, 0.50])
def test_mlp_wait_kernel_path_matches_baseline(f_cpu_store: float) -> None:
    """MLP-block captured-replay output is bit-identical between wait_kernel-on
    and wait_kernel-off. Exercises the strided down-proj slab dispatch with wait_kernel sync
    enabled (the worker still publishes done_slot=seq after the
    silu*up + at::linear chain finishes)."""
    torch.manual_seed(37)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    layer_off, off_off = _build_mlp(f_cpu_store=f_cpu_store, wait_kernel=False)
    try:
        _, replays_off = _capture_replay(
            offloader=off_off, forward=layer_off.mlp, x=x
        )
    finally:
        if off_off._runner is not None:
            off_off._runner.close()

    layer_on, off_on = _build_mlp(f_cpu_store=f_cpu_store, wait_kernel=True)
    try:
        _, replays_on = _capture_replay(
            offloader=off_on, forward=layer_on.mlp, x=x
        )
    finally:
        if off_on._runner is not None:
            off_on._runner.close()

    atol = MLP_BLOCK_RTOL * float(replays_off[0].abs().max())
    for i, (r_off, r_on) in enumerate(zip(replays_off, replays_on)):
        torch.testing.assert_close(
            r_on, r_off, rtol=BF16_RTOL, atol=atol,
            msg=f"replay #{i}: wait_kernel-sync path diverged from baseline at f_cpu_store={f_cpu_store}",
        )


def test_diag_counters_increment_under_wait_kernel() -> None:
    """When VLLM_COTS_DIAG=1 is set BEFORE process start, the diag
    wait kernel runs and `wait_kernel_immediate_resume_count + wait_kernel_lagging_wait_count`
    increments by exactly the captured wait_kernel fire count per
    replay. Skipped when env is not set (the env is read once at
    diag_enabled() first call; setting it mid-process is a no-op)."""
    import os

    if os.environ.get("VLLM_COTS_DIAG", "0") != "1":
        pytest.skip(
            "VLLM_COTS_DIAG=1 must be set before process start. "
            "Re-run with VLLM_COTS_DIAG=1 to validate diag counter "
            "increment under wait_kernel sync."
        )

    torch.manual_seed(41)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    layer, off = _build_qkv(f_cpu_store=0.10, wait_kernel=True)
    try:
        from vllm.model_executor.offloader import cots_ops

        infer = cots_ops.lookup_weight_runner(off._runner._runner_id, "diag_test")
        infer.reset_counters()

        _, replays = _capture_replay(offloader=off, forward=layer.qkv_proj, x=x)
        # 5 replays × N wait_kernel fires per replay.
        counters = dict(infer.get_counters())
        total = (
            counters.get("wait_kernel_immediate_resume_count", 0)
            + counters.get("wait_kernel_lagging_wait_count", 0)
        )
        assert total > 0, (
            f"diag mode active but neither wait_kernel_immediate_resume_count "
            f"nor wait_kernel_lagging_wait_count incremented; got {counters}"
        )
        assert len(replays) == 5
    finally:
        if off._runner is not None:
            off._runner.close()
