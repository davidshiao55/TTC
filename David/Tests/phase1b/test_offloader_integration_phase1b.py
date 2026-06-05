"""Phase 1b §8 — end-to-end CotsOffloader lifecycle with prefetch enabled.

Exercises the full Phase 1b path on a 2-layer mini stub:
  1. wrap_modules walks layers, builds handles, applies per-bucket prefetch
     split, allocates the buffer pool + streamer, and installs layer hooks.
  2. post_init verifies enforce_eager + sets eager_fallback_entry.
  3. A forward pass through the layer chain triggers wait_prefetch /
     start_prefetch hooks, which delegate to the streamer's H2D + sync.

This file verifies the Phase 1b additions: option-A streamer allocation, hook
installation, slot indices, active zero-prefetch routes, and the multi-layer
prefetch chain.
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
from vllm.model_executor.offloader.cots import CotsPrefetchBufferPool
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


class _MiniDecoderLayer(nn.Module):
    """Minimal layer with a `forward` so the layer-level prefetch hook fires
    when called as `layer(x)`. Forward does qkv → MLP → residual add."""

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
        # qkv produces (n, q+k+v); for the integration test we just sum it
        # to (n, hidden) so x shape is preserved across layers.
        qkv_out, _ = self.qkv_proj(x)
        # Reduce qkv_out back to hidden by summing the q/k/v slices.
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
    object.__setattr__(cc, "cudagraph_capture_sizes", [8, 32, MAX_NUM_TOKENS])
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


def _populate_weights(layer):
    torch.manual_seed(0)
    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    q = torch.randn(q_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
    layer.qkv_proj.weight_loader(layer.qkv_proj.weight, q, "q")
    layer.qkv_proj.weight_loader(layer.qkv_proj.weight, k, "k")
    layer.qkv_proj.weight_loader(layer.qkv_proj.weight, v, "v")
    gate = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
    down = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device="cuda")
    layer.mlp.gate_up_proj.weight_loader(
        layer.mlp.gate_up_proj.weight, gate, 0
    )
    layer.mlp.gate_up_proj.weight_loader(
        layer.mlp.gate_up_proj.weight, up, 1
    )
    layer.mlp.down_proj.weight_loader(layer.mlp.down_proj.weight, down)


def _build_phase1b(f_cpu_store, f_prefetch):
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layers = [_MiniDecoderLayer().cuda() for _ in range(N_LAYERS)]
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=f_prefetch,
                cpu_runner="python",
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter(layers))
        for layer in layers:
            _populate_weights(layer)
        offloader.post_init()
    return layers, offloader


def _dispatch(offloader: CotsOffloader, n: int = MAX_NUM_TOKENS) -> None:
    offloader.on_dispatch(
        ForwardDispatchInfo(
            batch_descriptor=BatchDescriptor(num_tokens=n),
            num_tokens_unpadded=n,
        )
    )


# ---------------------------------------------------------------------------
def test_phase1b_lifecycle_allocates_streamer_and_pool():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    layers, offloader = _build_phase1b(f_cpu_store=0.20, f_prefetch=0.05)

    assert offloader._streamer is not None
    assert offloader._prefetch_buffer_pool is not None

    # K=2 slot rotation: layer 0 → slot 0, layer 1 → slot 1.
    assert CotsPrefetchBufferPool.K == 2
    for handles in offloader._layer_handles:
        for h in handles:
            assert h.slot_idx == h.layer_idx % CotsPrefetchBufferPool.K
            if h.max_n_prefetch > 0:
                assert len(h.w_prefetch_slots) == 2

    # Per-layer events allocated, init flags False.
    n = len(offloader._layer_modules)
    assert len(offloader._streamer._copy_done_events) == n
    assert all(not v for v in offloader._streamer._event_valid_for_eager)
    assert all(not v for v in offloader._streamer._prefetch_in_capture)


def test_phase1b_lifecycle_at_zero_prefetch_reserves_option_a_streamer():
    """Option-A buffer accounting reserves prefetch infrastructure from
    f_cpu_store even when this uniform table copies zero prefetch rows."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    _, offloader = _build_phase1b(f_cpu_store=0.20, f_prefetch=0.0)
    assert offloader._streamer is not None
    assert offloader._prefetch_buffer_pool is not None
    for handles in offloader._layer_handles:
        for h in handles:
            assert h.n_prefetch_by_bucket
            assert all(n == 0 for n in h.n_prefetch_by_bucket.values())
            assert h.max_n_prefetch == h.n_cpu
            assert len(h.w_prefetch_slots) == CotsPrefetchBufferPool.K


def test_phase1b_layer_forward_hooks_wired():
    """Layer.forward is wrapped — calling it once should not raise (the
    hook chain emits wait_prefetch/start_prefetch which route through the
    registered offloader to the streamer)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    layers, offloader = _build_phase1b(f_cpu_store=0.20, f_prefetch=0.05)

    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    # Call layers in sequence — first layer's pre-hook caches the bucket
    # and repairs layer 0, then each forward waits for the current layer
    # and starts prefetch for the next layer before compute.
    out = x
    _dispatch(offloader)
    for layer in layers:
        out = layer(out)
    torch.cuda.synchronize()
    assert out.shape == (MAX_NUM_TOKENS, HIDDEN)
    assert torch.isfinite(out).all()


def test_phase1b_dispatch_table_populated_in_wrap_modules():
    """Dispatch table is built in wrap_modules (Phase 1b — must exist
    before the buffer pool is sized). Per-bucket f_prefetch is uniform."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    _, offloader = _build_phase1b(f_cpu_store=0.20, f_prefetch=0.05)

    assert offloader._dispatch_table  # non-empty
    for bucket, (f_cpu, f_pref) in offloader._dispatch_table.items():
        assert abs(f_cpu - 0.15) < 1e-9, f"f_cpu mismatch at {bucket}"
        assert abs(f_pref - 0.05) < 1e-9, f"f_prefetch mismatch at {bucket}"


def test_phase1b_dispatch_boundary_prepares_layer0():
    """The final runtime uses OOG dispatch to cache the active bucket and
    prepare layer 0 before the layer chain runs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    layers, offloader = _build_phase1b(f_cpu_store=0.20, f_prefetch=0.05)
    assert offloader._streamer is not None
    _dispatch(offloader)
    assert offloader._current_bucket == offloader._dispatch_bucket_for(MAX_NUM_TOKENS)
    for h in offloader._layer_handles[0]:
        if h.n_prefetch_by_bucket.get(offloader._current_bucket, 0) == 0:
            continue
        assert h.prefetch_owner_in_slot[h.slot_idx] is h


def test_phase1b_post_init_fallback_entry_matches_largest_bucket():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    _, offloader = _build_phase1b(f_cpu_store=0.20, f_prefetch=0.05)
    assert (
        offloader._eager_fallback_entry
        == offloader._dispatch_table[offloader._dispatch_buckets[-1]]
    )
