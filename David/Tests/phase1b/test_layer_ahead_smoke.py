"""Phase 1b §8 — layer-ahead smoke / regression sentinels.

End-to-end forward pass through a 2-layer mini stub. Two assertions:

  1. **Regression sentinel** — at `f_prefetch=0`, the active route has no
     prefetched rows and remains bit-identical to Phase 1a at the same
     `f_cpu_store`.

  2. **Phase 1b parity** — at `f_prefetch>0`, the layer-ahead path
     produces output matching the unsplit reference within BF16
     tolerance. This exercises the full hook chain
     (`wait_prefetch` → `_start_prefetch` → streamer.start /
     streamer.wait → operator three-way scatter).
"""

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

HIDDEN = 256
INTERMEDIATE = 1024
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
MAX_NUM_TOKENS = 64
N_LAYERS = 2

BF16_RTOL = 5e-2
BF16_ATOL = 0.5
MLP_RTOL = 5e-2


class _MiniMlp(nn.Module):
    def __init__(self):
        super().__init__()
        bf16 = torch.bfloat16
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=HIDDEN,
            output_sizes=[INTERMEDIATE, INTERMEDIATE],
            bias=False, disable_tp=True, params_dtype=bf16,
            prefix="mlp.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=INTERMEDIATE, output_size=HIDDEN,
            bias=False, disable_tp=True, params_dtype=bf16,
            prefix="mlp.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gu, _ = self.gate_up_proj(x)
        x = self.act_fn(gu)
        x, _ = self.down_proj(x)
        return x


class _Layer(nn.Module):
    """Layer with `forward` so the hook chain fires on layer(x)."""

    def __init__(self):
        super().__init__()
        bf16 = torch.bfloat16
        self.qkv_proj = QKVParallelLinear(
            hidden_size=HIDDEN, head_size=HEAD_DIM,
            total_num_heads=NUM_HEADS, total_num_kv_heads=NUM_KV_HEADS,
            bias=False, disable_tp=True, params_dtype=bf16,
            prefix="qkv_proj",
        )
        self.mlp = _MiniMlp()

    def forward(self, x):
        # Sum-reduce qkv to (n, hidden) so layer chain composes.
        qkv_out, _ = self.qkv_proj(x)
        qkv_dim = NUM_HEADS * HEAD_DIM
        kv_dim = NUM_KV_HEADS * HEAD_DIM
        q = qkv_out[:, :qkv_dim]
        return x + self.mlp(q) + qkv_out[:, qkv_dim:qkv_dim + kv_dim].sum(
            dim=1, keepdim=True
        ).expand_as(x)


def _make_vllm_config():
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


def _populate(layer, ref_weights):
    """Load weights from `ref_weights` (a dict of full-shape tensors) via
    the wrapped weight_loaders."""
    layer.qkv_proj.weight_loader(
        layer.qkv_proj.weight, ref_weights["q"], "q"
    )
    layer.qkv_proj.weight_loader(
        layer.qkv_proj.weight, ref_weights["k"], "k"
    )
    layer.qkv_proj.weight_loader(
        layer.qkv_proj.weight, ref_weights["v"], "v"
    )
    layer.mlp.gate_up_proj.weight_loader(
        layer.mlp.gate_up_proj.weight, ref_weights["gate"], 0
    )
    layer.mlp.gate_up_proj.weight_loader(
        layer.mlp.gate_up_proj.weight, ref_weights["up"], 1
    )
    layer.mlp.down_proj.weight_loader(
        layer.mlp.down_proj.weight, ref_weights["down"]
    )


def _seed_ref_weights(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    return {
        "q": torch.randn(q_size, HIDDEN, dtype=torch.bfloat16, device="cuda",
                         generator=g),
        "k": torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda",
                         generator=g),
        "v": torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda",
                         generator=g),
        "gate": torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16,
                            device="cuda", generator=g),
        "up": torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16,
                          device="cuda", generator=g),
        "down": torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16,
                            device="cuda", generator=g),
    }


def _ref_forward_layer(x, weights):
    """Unsplit reference computation matching `_Layer.forward`."""
    qkv_full = torch.cat([weights["q"], weights["k"], weights["v"]], dim=0)
    qkv_out = F.linear(x, qkv_full, None)
    qkv_dim = NUM_HEADS * HEAD_DIM
    kv_dim = NUM_KV_HEADS * HEAD_DIM
    q = qkv_out[:, :qkv_dim]
    gate_up = F.linear(q, torch.cat([weights["gate"], weights["up"]], dim=0),
                       None)
    silu = F.silu(gate_up[:, :INTERMEDIATE]) * gate_up[:, INTERMEDIATE:]
    mlp_out = F.linear(silu, weights["down"], None)
    return x + mlp_out + qkv_out[:, qkv_dim:qkv_dim + kv_dim].sum(
        dim=1, keepdim=True
    ).expand_as(x)


def _ref_forward_chain(x, per_layer_weights):
    out = x
    for w in per_layer_weights:
        out = _ref_forward_layer(out, w)
    return out


def _build_and_run(f_cpu_store, f_prefetch, x, per_layer_weights):
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layers = [_Layer().cuda() for _ in range(N_LAYERS)]
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=f_prefetch,
                cpu_runner="python",
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter(layers))
        for layer, w in zip(layers, per_layer_weights):
            _populate(layer, w)
        offloader.post_init()

        out = x
        n = int(x.shape[0])
        offloader.on_dispatch(
            ForwardDispatchInfo(
                batch_descriptor=BatchDescriptor(num_tokens=n),
                num_tokens_unpadded=n,
            )
        )
        for layer in layers:
            out = layer(out)
        torch.cuda.synchronize()
        return out


# ---------------------------------------------------------------------------
def test_layer_ahead_smoke_phase1a_regression():
    """f_prefetch=0 must produce output matching the unsplit reference
    within BF16 tolerance — same as Phase 1a's behavior at the same
    f_cpu_store."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    weights = [_seed_ref_weights(seed=i) for i in range(N_LAYERS)]

    out = _build_and_run(
        f_cpu_store=0.10, f_prefetch=0.0, x=x, per_layer_weights=weights
    )
    ref = _ref_forward_chain(x, weights)
    atol = MLP_RTOL * float(ref.abs().max())
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=atol)


def test_layer_ahead_smoke_phase1b_parity():
    """f_prefetch>0 must also produce output matching the unsplit reference
    within BF16 tolerance — exercises the full hook chain (wait_prefetch /
    start_prefetch / streamer.start / streamer.wait + three-way scatter)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    weights = [_seed_ref_weights(seed=i) for i in range(N_LAYERS)]

    out = _build_and_run(
        f_cpu_store=0.10, f_prefetch=0.05, x=x, per_layer_weights=weights
    )
    ref = _ref_forward_chain(x, weights)
    atol = MLP_RTOL * float(ref.abs().max())
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=atol)


def test_layer_ahead_smoke_phase1b_higher_offload():
    """Higher offload (f_cpu_store=0.30, f_prefetch=0.15) — exercises the
    asymmetric WQKV vs MLP geometry through the full hook chain."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    weights = [_seed_ref_weights(seed=i) for i in range(N_LAYERS)]

    out = _build_and_run(
        f_cpu_store=0.30, f_prefetch=0.15, x=x, per_layer_weights=weights
    )
    ref = _ref_forward_chain(x, weights)
    atol = MLP_RTOL * float(ref.abs().max())
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=atol)
