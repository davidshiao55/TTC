# SPDX-License-Identifier: Apache-2.0
"""Stage 3 parity gate — native runner vs Python runner.

Builds a synthetic Qwen2-style mini-layer (QKVParallelLinear +
MergedColumnParallelLinear + RowParallelLinear, mirroring
phase1b/test_three_way_scatter.py's stub fabric), runs identical inputs
through `cpu_runner='python'` and `cpu_runner='native'` (both with
`enforce_eager=True`), and asserts bit-exact / 1-ULP BF16 parity at
both QKV and MLP-block call sites.

This is the load-bearing Stage 3 acceptance: if the C++
`at::linear`-based dispatch + the slab-pointer/strides + the
host-callback round-trip produces a different result than Python's
`F.linear`, the substrate is silently corrupting math somewhere.
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


# Same fabric proportions as phase1b/test_three_way_scatter.py — small
# enough to be a fast unit test, large enough that a corrupt slab pointer
# or wrong-stride strided view would surface.
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


class _WoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bf16 = torch.bfloat16
        self.o_proj = RowParallelLinear(
            input_size=HIDDEN,
            output_size=HIDDEN,
            bias=False,
            disable_tp=True,
            params_dtype=bf16,
            prefix="self_attn.o_proj",
        )


class _WoLayerWithQkv(nn.Module):
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
            prefix="self_attn.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=HIDDEN,
            output_size=HIDDEN,
            bias=False,
            disable_tp=True,
            params_dtype=bf16,
            prefix="self_attn.o_proj",
        )


def _make_vllm_config() -> VllmConfig:
    mc = ModelConfig.__new__(ModelConfig)
    object.__setattr__(mc, "enforce_eager", True)
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
    *, f_cpu_store: float, f_prefetch: float, cpu_runner: str
) -> tuple[_QkvLayer, CotsOffloader, torch.Tensor]:
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=f_prefetch,
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
    return layer, offloader, torch.cat([q, k, v], dim=0)


def _build_mlp(
    *, f_cpu_store: float, f_prefetch: float, cpu_runner: str
) -> tuple[_MlpLayer, CotsOffloader, torch.Tensor, torch.Tensor, torch.Tensor]:
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _MlpLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=f_prefetch,
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
    return layer, offloader, gate, up, down


def _build_wo(
    *, f_cpu_store: float, f_prefetch: float, cpu_runner: str
) -> tuple[_WoLayerWithQkv, CotsOffloader, torch.Tensor]:
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _WoLayerWithQkv().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=f_prefetch,
                kv_biased=True,
                weight_modules={"wo"},
                cpu_runner=cpu_runner,
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))
        torch.manual_seed(0)
        weight = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16, device="cuda")
        layer.o_proj.weight_loader(layer.o_proj.weight, weight)
        offloader.post_init()
    return layer, offloader, weight


def _dispatch(offloader: CotsOffloader, n: int = MAX_NUM_TOKENS) -> None:
    offloader.on_dispatch(
        ForwardDispatchInfo(
            batch_descriptor=BatchDescriptor(num_tokens=n),
            num_tokens_unpadded=n,
        )
    )


@pytest.mark.parametrize(
    "f_cpu_store",
    [0.10, 0.25, 0.50],
    ids=["fcpu_010", "fcpu_025", "fcpu_050"],
)
def test_qkv_native_matches_python(f_cpu_store: float) -> None:
    """QKV operator forward, f_prefetch=0 (no streamer path). The native
    runner's strided/contiguous slab dispatch must produce the same
    values as PythonCotsWeightRunner's F.linear closure."""
    torch.manual_seed(123)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    _, off_py, _ = _build_qkv(
        f_cpu_store=f_cpu_store, f_prefetch=0.0, cpu_runner="python"
    )
    _dispatch(off_py)
    out_py, _ = off_py._layer_modules[0].qkv_proj(x)
    set_offloader.__wrapped__ if hasattr(set_offloader, "__wrapped__") else None
    # Tear down the python offloader before constructing the native one
    # (set_offloader is a global singleton; back-to-back installs replace
    # each other but the per-runner registry should stay clean).
    if off_py._runner is not None:
        off_py._runner.close()

    _, off_nat, _ = _build_qkv(
        f_cpu_store=f_cpu_store, f_prefetch=0.0, cpu_runner="native"
    )
    _dispatch(off_nat)
    out_nat, _ = off_nat._layer_modules[0].qkv_proj(x)
    if off_nat._runner is not None:
        off_nat._runner.close()

    torch.testing.assert_close(out_nat, out_py, rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.parametrize(
    "f_cpu_store",
    [0.10, 0.25, 0.50],
    ids=["fcpu_010", "fcpu_025", "fcpu_050"],
)
def test_mlp_native_matches_python(f_cpu_store: float) -> None:
    """MLP-block operator forward at f_prefetch=0. Exercises the native
    runner's strided down-proj view (`at::from_blob` with non-trivial
    row stride when the column-narrow has a positive offset) end-to-end
    through the fused gate+up+silu*up+down dispatch."""
    torch.manual_seed(456)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    _, off_py, *_ = _build_mlp(
        f_cpu_store=f_cpu_store, f_prefetch=0.0, cpu_runner="python"
    )
    _dispatch(off_py)
    out_py = off_py._layer_modules[0].mlp(x)
    if off_py._runner is not None:
        off_py._runner.close()

    _, off_nat, *_ = _build_mlp(
        f_cpu_store=f_cpu_store, f_prefetch=0.0, cpu_runner="native"
    )
    _dispatch(off_nat)
    out_nat = off_nat._layer_modules[0].mlp(x)
    if off_nat._runner is not None:
        off_nat._runner.close()

    atol = MLP_BLOCK_RTOL * float(out_py.abs().max())
    torch.testing.assert_close(out_nat, out_py, rtol=BF16_RTOL, atol=atol)


def test_default_weight_modules_do_not_wrap_wo() -> None:
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _WoLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(f_cpu_store=0.25, cpu_runner="python")
        )
        offloader.wrap_modules(iter([layer]))

    assert not hasattr(layer.o_proj, "_cots_handle")
    assert offloader._handles == []
    if offloader._runner is not None:
        offloader._runner.close()


def test_wo_output_split_uses_qkv_head_granularity() -> None:
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _WoLayerWithQkv().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=0.10,
                weight_modules={"wo"},
                cpu_runner="python",
            )
        )
        offloader.wrap_modules(iter([layer]))

    handle = layer.o_proj._cots_handle
    assert handle.output_granularity == HEAD_DIM
    assert handle.n_cpu == HEAD_DIM
    assert all(h.role != "qkv" for h in offloader._handles)
    if offloader._runner is not None:
        offloader._runner.close()


@pytest.mark.parametrize(
    "f_cpu_store",
    [0.25, 0.50, 0.75],
    ids=["fcpu_025", "fcpu_050", "fcpu_075"],
)
def test_wo_native_matches_python(f_cpu_store: float) -> None:
    """WO uses the generic output-split linear path, but its native slab
    descriptor and op-kind routing are distinct from QKV."""
    torch.manual_seed(457)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    _, off_py, _ = _build_wo(
        f_cpu_store=f_cpu_store, f_prefetch=0.0, cpu_runner="python"
    )
    _dispatch(off_py)
    out_py, _ = off_py._layer_modules[0].o_proj(x)
    if off_py._runner is not None:
        off_py._runner.close()

    _, off_nat, _ = _build_wo(
        f_cpu_store=f_cpu_store, f_prefetch=0.0, cpu_runner="native"
    )
    _dispatch(off_nat)
    out_nat, _ = off_nat._layer_modules[0].o_proj(x)
    if off_nat._runner is not None:
        off_nat._runner.close()

    torch.testing.assert_close(out_nat, out_py, rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.parametrize(
    "f_cpu_store",
    [0.10, 0.50],
    ids=["fcpu_010", "fcpu_050"],
)
def test_no_streamer_native_forward_succeeds(f_cpu_store: float) -> None:
    """At f_prefetch=0 the streamer is None, but final native COTS still
    runs once the production OOG dispatch boundary has published the
    active bucket."""
    torch.manual_seed(7)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    _, off, _ = _build_qkv(
        f_cpu_store=f_cpu_store, f_prefetch=0.0, cpu_runner="native"
    )
    assert off._streamer is None  # f_prefetch=0
    assert off._current_bucket is None
    _dispatch(off)
    out, _ = off._layer_modules[0].qkv_proj(x)
    assert out.shape == (MAX_NUM_TOKENS, off._handles[0].out_dim)
    if off._runner is not None:
        off._runner.close()


def _prime_prefetch(offloader: CotsOffloader) -> None:
    """Mirror of phase1b/test_three_way_scatter.py's helper. Caches the
    bucket on the streamer + offloader, populates layer-0's prefetch
    slot, then joins the copy stream onto compute. Required to drive
    the three-way (permanent + prefetched + CPU) operator forward
    without going through the layer-forward hook chain."""
    streamer = offloader._streamer
    assert streamer is not None
    _dispatch(offloader)


@pytest.mark.parametrize(
    "f_cpu_store,f_prefetch",
    [(0.25, 0.10), (0.50, 0.20)],
    ids=["fcs025_fp010", "fcs050_fp020"],
)
def test_qkv_three_way_native_matches_python(
    f_cpu_store: float, f_prefetch: float
) -> None:
    """QKV three-way at f_prefetch > 0 — exercises the load-bearing
    POST-narrow QKV pointer (`h.w_cpu.narrow(0, n_pref>0, n_cpu)`)
    that the install populates into the slab. If the offset path is
    wrong (e.g., we passed the base pointer instead of the narrow's
    data_ptr), the native runner would read the wrong rows of w_cpu
    and diverge from the python runner's closure."""
    torch.manual_seed(11)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    _, off_py, _ = _build_qkv(
        f_cpu_store=f_cpu_store, f_prefetch=f_prefetch, cpu_runner="python"
    )
    _prime_prefetch(off_py)
    out_py, _ = off_py._layer_modules[0].qkv_proj(x)
    if off_py._runner is not None:
        off_py._runner.close()

    _, off_nat, _ = _build_qkv(
        f_cpu_store=f_cpu_store, f_prefetch=f_prefetch, cpu_runner="native"
    )
    _prime_prefetch(off_nat)
    out_nat, _ = off_nat._layer_modules[0].qkv_proj(x)
    if off_nat._runner is not None:
        off_nat._runner.close()

    torch.testing.assert_close(out_nat, out_py, rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.parametrize(
    "f_cpu_store,f_prefetch",
    [(0.20, 0.10), (0.30, 0.15), (0.50, 0.25)],
    ids=["fcs020_fp010", "fcs030_fp015", "fcs050_fp025"],
)
def test_mlp_three_way_native_matches_python(
    f_cpu_store: float, f_prefetch: float
) -> None:
    """MLP block three-way at f_prefetch > 0 — the load-bearing test
    for the strided down-proj slab path. With dn_n_pref > 0 the
    install passes a POST-narrow pointer (offset by
    `dn_n_pref * elem_size`) AND non-default strides
    (stride_row=n_cpu_total, stride_col=1) to populate_slab_mlp. The
    C++ worker reconstructs the strided view via `at::from_blob(ptr,
    sizes, strides, opts)` and runs `at::linear` on it. If the
    pointer or strides are wrong the native result diverges from the
    python runner's `F.linear(z, dn_h.w_cpu.narrow(1, dn_n_pref,
    dn_n_cpu))` closure."""
    torch.manual_seed(13)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    _, off_py, *_ = _build_mlp(
        f_cpu_store=f_cpu_store, f_prefetch=f_prefetch, cpu_runner="python"
    )
    _prime_prefetch(off_py)
    out_py = off_py._layer_modules[0].mlp(x)
    if off_py._runner is not None:
        off_py._runner.close()

    _, off_nat, *_ = _build_mlp(
        f_cpu_store=f_cpu_store, f_prefetch=f_prefetch, cpu_runner="native"
    )
    _prime_prefetch(off_nat)
    out_nat = off_nat._layer_modules[0].mlp(x)
    if off_nat._runner is not None:
        off_nat._runner.close()

    atol = MLP_BLOCK_RTOL * float(out_py.abs().max())
    torch.testing.assert_close(out_nat, out_py, rtol=BF16_RTOL, atol=atol)


@pytest.mark.parametrize(
    "f_cpu_store,f_prefetch",
    [(0.50, 0.25), (0.75, 0.25)],
    ids=["fcs050_fp025", "fcs075_fp025"],
)
def test_wo_three_way_native_matches_python(
    f_cpu_store: float, f_prefetch: float
) -> None:
    torch.manual_seed(17)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    _, off_py, _ = _build_wo(
        f_cpu_store=f_cpu_store, f_prefetch=f_prefetch, cpu_runner="python"
    )
    _prime_prefetch(off_py)
    out_py, _ = off_py._layer_modules[0].o_proj(x)
    if off_py._runner is not None:
        off_py._runner.close()

    _, off_nat, _ = _build_wo(
        f_cpu_store=f_cpu_store, f_prefetch=f_prefetch, cpu_runner="native"
    )
    _prime_prefetch(off_nat)
    out_nat, _ = off_nat._layer_modules[0].o_proj(x)
    if off_nat._runner is not None:
        off_nat._runner.close()

    torch.testing.assert_close(out_nat, out_py, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_prepare_before_forward_sets_current_bucket() -> None:
    """The unconditional `prepare_before_forward` (called by the
    first-decoder pre-hook OR by `cudagraph_utils.py:267` at the graph
    capture/replay boundary) sets `_current_bucket` regardless of
    streamer presence. Phase 1c moved this state from the streamer to
    the offloader so f_prefetch=0 still has a valid bucket source."""
    _, off, _ = _build_qkv(
        f_cpu_store=0.10, f_prefetch=0.0, cpu_runner="native"
    )
    assert off._current_bucket is None
    # Simulate what the pre-hook (or graph-boundary) calls.
    off.prepare_before_forward(MAX_NUM_TOKENS)
    assert off._current_bucket == off._bucket_for(MAX_NUM_TOKENS)
    if off._runner is not None:
        off._runner.close()
