"""Phase 1b §5 — Operator forward-pass three-way split correctness.

Exercises `CotsQKVOp.apply` and `CotsSwiGLUMLPOp.__call__` at `f_prefetch > 0`,
verifying numerical parity with the unsplit reference within BF16 tolerance.

The three GPU/CPU compute paths must be set up correctly:
  * GPU permanent slice on `layer.weight`
  * GPU prefetched slice on the buffer-pool slot view (populated by the
    streamer's H2D ahead of the layer's forward)
  * CPU compute slice on `w_cpu` rows past the prefetched prefix

Phase 1a regression: `f_prefetch == 0` → test_mlp_block_fusion / dispatcher_split
already cover this; here we additionally cover `f_prefetch > 0`.
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
HEAD_DIM = HIDDEN // NUM_HEADS  # 32
MAX_NUM_TOKENS = 64

BF16_RTOL = 5e-2
BF16_ATOL = 0.5
MLP_BLOCK_RTOL = 5e-2


class MiniMlpBlock(nn.Module):
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


class _MlpLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MiniMlpBlock()


class _QkvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        bf16 = torch.bfloat16
        self.qkv_proj = QKVParallelLinear(
            hidden_size=HIDDEN, head_size=HEAD_DIM,
            total_num_heads=NUM_HEADS, total_num_kv_heads=NUM_KV_HEADS,
            bias=False, disable_tp=True, params_dtype=bf16,
            prefix="qkv_proj",
        )


def _make_vllm_config(custom_ops: str = "none"):
    """`custom_ops="all"` matches the engine default and exercises the
    CUDA custom-op paths (SiluAndMul, etc.) — required to catch crashes
    that only manifest with native kernels on zero-size inputs."""
    mc = ModelConfig.__new__(ModelConfig)
    object.__setattr__(mc, "enforce_eager", True)
    sc = SchedulerConfig.__new__(SchedulerConfig)
    object.__setattr__(sc, "max_num_batched_tokens", MAX_NUM_TOKENS)
    cc = CompilationConfig.__new__(CompilationConfig)
    object.__setattr__(cc, "cudagraph_capture_sizes", [MAX_NUM_TOKENS])
    object.__setattr__(cc, "custom_ops", [custom_ops])
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


def _build_mlp(f_cpu_store, f_prefetch):
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _MlpLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=f_prefetch,
                kv_biased=True,
                cpu_runner="python",
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


def _build_qkv(f_cpu_store, f_prefetch):
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=f_prefetch,
                kv_biased=True,
                cpu_runner="python",
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


def _prime_prefetch(offloader):
    """Simulate the layer-forward hook chain: cache the bucket and start
    prefetch for layer 0 so its slot is populated before forward()."""
    streamer = offloader._streamer
    assert streamer is not None
    offloader.on_dispatch(
        ForwardDispatchInfo(
            batch_descriptor=BatchDescriptor(num_tokens=MAX_NUM_TOKENS),
            num_tokens_unpadded=MAX_NUM_TOKENS,
        )
    )


# ---------------------------------------------------------------------------
# MLP block — three-way split parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "f_cpu_store,f_prefetch", [(0.20, 0.10), (0.30, 0.15), (0.50, 0.25)]
)
def test_mlp_three_way_matches_unsplit(f_cpu_store, f_prefetch):
    """At `f_prefetch > 0`, the three-way MLP block output equals the
    unsplit `MLP1 → SwiGLU → MLP2` reference within BF16 tolerance."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    layer, offloader, gate, up, down = _build_mlp(f_cpu_store, f_prefetch)

    # Layer is offloaded → has prefetch handles; prime the slot manually.
    assert offloader._streamer is not None
    assert offloader._prefetch_buffer_pool is not None
    _prime_prefetch(offloader)

    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    # Bypass the layer-forward hook (we already primed the slot manually
    # above and we want to test the operator directly, not the hook chain).
    # CotsSwiGLUMLPOp is installed as `layer.mlp.forward`; calling layer.mlp(x)
    # triggers it.
    out = layer.mlp(x)

    gate_up = F.linear(x, torch.cat([gate, up], dim=0), None)
    silu_full = F.silu(gate_up[:, :INTERMEDIATE]) * gate_up[:, INTERMEDIATE:]
    ref = F.linear(silu_full, down, None)

    atol = MLP_BLOCK_RTOL * float(ref.abs().max())
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=atol)


def test_mlp_phase1a_regression_at_zero_prefetch():
    """Sentinel: f_prefetch=0 → behaves identically to Phase 1a (no
    streamer / pool / hooks allocated)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    _, offloader, _, _, _ = _build_mlp(f_cpu_store=0.25, f_prefetch=0.0)
    assert offloader._streamer is None
    assert offloader._prefetch_buffer_pool is None


# ---------------------------------------------------------------------------
# QKV — three-way split parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "f_cpu_store,f_prefetch",
    [
        # f_cpu_store at K/V boundary — n_q_tail=0, prefetch consumes K/V.
        (0.25, 0.10),
        # f_cpu_store > K/V — n_q_tail > 0, prefetch consumes Q-tail then K/V.
        (0.50, 0.10),
    ],
)
def test_qkv_three_way_matches_unsplit(f_cpu_store, f_prefetch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    layer, offloader, qkv_full = _build_qkv(f_cpu_store, f_prefetch)
    if offloader._streamer is not None:
        _prime_prefetch(offloader)

    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    # QKVParallelLinear.forward returns (output, bias). We compare the output.
    out, _ = layer.qkv_proj(x)
    ref = F.linear(x, qkv_full, None)

    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=BF16_ATOL)


# ---------------------------------------------------------------------------
# Pure-prefetch fast path (n_cpu_compute == 0)
# ---------------------------------------------------------------------------
def test_pure_prefetch_zero_cpu_compute_no_residual_qkv():
    """At `f_cpu_store == f_prefetch`, QKV's `n_cpu_compute` must be exactly
    0 — no residual rows from head-aligned snapping. Repro for Codex
    Finding 1: `round(f_prefetch * out_dim)` < snapped `n_cpu` left a few
    rows on the CPU compute path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=0.20, f_prefetch=0.20, kv_biased=True,
                cpu_runner="python",
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))
        torch.manual_seed(0)
        q_size, kv_size = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
        q = torch.randn(q_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, q, "q")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, k, "k")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, v, "v")
        offloader.post_init()

    qkv_h = offloader._handles[0]
    bucket = offloader._capture_buckets[0]
    # The fix: n_cpu_compute must be exactly zero — every CPU-stored byte
    # is prefetched, no residual.
    assert qkv_h.n_cpu_compute_by_bucket[bucket] == 0
    assert qkv_h.n_prefetch_by_bucket[bucket] == qkv_h.n_cpu


def test_pure_prefetch_qkv_forward_parity():
    """End-to-end: QKV at f_cpu_store == f_prefetch must match the unsplit
    reference. Exercises the zero-CPU fast path — runner.submit/wait/UVA
    are all skipped, only GPU permanent + GPU prefetched paths run."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=0.20, f_prefetch=0.20, kv_biased=True,
                cpu_runner="python",
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))
        torch.manual_seed(0)
        q_size, kv_size = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
        q = torch.randn(q_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, q, "q")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, k, "k")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, v, "v")
        offloader.post_init()

    # Confirm fast path will trigger.
    qkv_h = offloader._handles[0]
    assert qkv_h.n_cpu_compute_by_bucket[offloader._capture_buckets[0]] == 0

    # Manually populate slot 0 (in real runs the streamer's hooks do this).
    if offloader._streamer is not None:
        _prime_prefetch(offloader)

    qkv_full = torch.cat([q, k, v], dim=0)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out, _ = layer.qkv_proj(x)
    ref = F.linear(x, qkv_full, None)
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_pure_prefetch_mlp_forward_parity():
    """End-to-end: MLP block at f_cpu_store == f_prefetch must match the
    unsplit reference. Both gate/up halves and down's input cols are 100%
    prefetched; CPU runner / D2H / UVA are skipped."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    layer, offloader, gate, up, down = _build_mlp(
        f_cpu_store=0.20, f_prefetch=0.20
    )
    # Confirm both handles have zero CPU compute.
    bucket = offloader._capture_buckets[0]
    for h in offloader._handles:
        assert h.n_cpu_compute_by_bucket[bucket] == 0, (
            f"{h.kind} {h.qualified_name}: expected n_cpu_compute=0, "
            f"got {h.n_cpu_compute_by_bucket[bucket]}"
        )

    if offloader._streamer is not None:
        _prime_prefetch(offloader)

    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = layer.mlp(x)

    gate_up = F.linear(x, torch.cat([gate, up], dim=0), None)
    silu_full = F.silu(gate_up[:, :INTERMEDIATE]) * gate_up[:, INTERMEDIATE:]
    ref = F.linear(silu_full, down, None)
    atol = MLP_BLOCK_RTOL * float(ref.abs().max())
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=atol)


# ---------------------------------------------------------------------------
# Full offload (f_cpu_store=1.0) with custom_ops=["all"]
#
# Repro for the silent engine crash where empty permanent slices
# (gate_up.weight shape (0, in_dim), down.weight shape (out_dim, 0)) made
# F.linear emit (B, 0) which CUDA SiluAndMul could not handle. Tests with
# `custom_ops=["none"]` masked the crash because PyTorch native SiLU
# tolerates zero-size inputs.
# ---------------------------------------------------------------------------
def _build_mlp_full_offload(f_prefetch):
    """f_cpu_store=1.0 with custom_ops="all" so SiluAndMul uses its CUDA op."""
    vc = _make_vllm_config(custom_ops="all")
    with set_current_vllm_config(vc):
        layer = _MlpLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=1.0, f_prefetch=f_prefetch, kv_biased=True,
                cpu_runner="python",
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


@pytest.mark.parametrize("f_prefetch", [0.0, 1.0])
def test_full_offload_mlp_with_cuda_custom_ops(f_prefetch):
    """Empty-permanent-slice fix: f_cpu_store=1.0 must run without crashing
    when CUDA custom ops (SiluAndMul) are enabled. Verifies output parity."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    layer, offloader, gate, up, down = _build_mlp_full_offload(f_prefetch)
    if offloader._streamer is not None:
        _prime_prefetch(offloader)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = layer.mlp(x)

    gate_up = F.linear(x, torch.cat([gate, up], dim=0), None)
    silu_full = F.silu(gate_up[:, :INTERMEDIATE]) * gate_up[:, INTERMEDIATE:]
    ref = F.linear(silu_full, down, None)
    atol = MLP_BLOCK_RTOL * float(ref.abs().max())
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=atol)


@pytest.mark.parametrize("f_prefetch", [0.0, 1.0])
def test_full_offload_qkv_with_cuda_custom_ops(f_prefetch):
    """Empty-permanent-slice fix for QKV at f_cpu_store=1.0 with CUDA
    custom ops enabled."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    vc = _make_vllm_config(custom_ops="all")
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=1.0, f_prefetch=f_prefetch, kv_biased=True,
                cpu_runner="python",
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))
        torch.manual_seed(0)
        q_size, kv_size = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
        q = torch.randn(q_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, q, "q")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, k, "k")
        layer.qkv_proj.weight_loader(layer.qkv_proj.weight, v, "v")
        offloader.post_init()

    if offloader._streamer is not None:
        _prime_prefetch(offloader)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out, _ = layer.qkv_proj(x)
    ref = F.linear(x, torch.cat([q, k, v], dim=0), None)
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=BF16_ATOL)



# ---------------------------------------------------------------------------
# Planner-path regressions
# ---------------------------------------------------------------------------
def test_factory_prefetch_installs_machinery_even_with_config_zero():
    """A `dispatch_table_factory` may emit non-zero f_prefetch while the
    config knob is 0.0 — common when the Planner overrides the manual
    fallback. The offloader must still install the streamer + buffer
    pool; otherwise prefetched bytes are silently dropped from the
    forward result. Repro for Codex Finding 1."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _MlpLayer().cuda()
        # Config: f_cpu_store=0.50, f_prefetch=0.0 (manual fallback says
        # "no prefetch"). Factory: emit f_prefetch=0.20 per bucket.
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=0.50, f_prefetch=0.0, kv_biased=True,
                cpu_runner="python",
            ),
            dispatch_table_factory=lambda buckets: {b: (0.30, 0.20) for b in buckets},
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

    # Streamer + pool MUST be installed even though config f_prefetch=0.
    assert offloader._streamer is not None, "streamer should install when factory emits prefetch"
    assert offloader._prefetch_buffer_pool is not None
    bucket = offloader._capture_buckets[0]
    assert any(h.n_prefetch_by_bucket[bucket] > 0 for h in offloader._handles)

    # Forward output must match unsplit reference — the prefetched bytes
    # are accounted for, not silently dropped.
    _prime_prefetch(offloader)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = layer.mlp(x)
    gate_up = F.linear(x, torch.cat([gate, up], dim=0), None)
    silu_full = F.silu(gate_up[:, :INTERMEDIATE]) * gate_up[:, INTERMEDIATE:]
    ref = F.linear(silu_full, down, None)
    atol = MLP_BLOCK_RTOL * float(ref.abs().max())
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=atol)


def test_streamer_clears_stale_state_on_noop_start():
    """Per-bucket Planner strategies can have a layer that prefetches in
    bucket A but not in bucket B. The streamer's start() early-return for
    bucket B must clear `_event_valid_for_eager[idx]` so a later
    wait_for_layer doesn't sync on a stale event. Repro for Codex
    Finding 2."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from vllm.model_executor.offloader.cots import (
        WeightPrefetchStreamer,
    )

    streamer = WeightPrefetchStreamer(n_layers=4)
    # Simulate a prior bucket where layer 0 prefetched: eager-valid flag set.
    streamer._event_valid_for_eager[0] = True
    streamer._prefetch_in_capture[0] = True

    # Now drive start() for bucket 1 with handles that have no prefetch
    # for bucket 1. Use minimal stub handles since start() only inspects
    # `n_prefetch_by_bucket`.
    class _StubHandle:
        n_prefetch_by_bucket = {1: 0}

    streamer.current_bucket = 1
    streamer.start(0, [_StubHandle()])

    # Stale flags must be cleared.
    assert streamer._event_valid_for_eager[0] is False, (
        "stale eager-valid flag must be cleared on no-op start"
    )
    assert streamer._prefetch_in_capture[0] is False, (
        "stale capture flag must be cleared on no-op start"
    )
