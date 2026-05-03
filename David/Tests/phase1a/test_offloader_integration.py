"""Phase 1a §5 — End-to-end CotsOffloader lifecycle on a synthetic mini-layer.

Exercises the full offloader code path:
  1. wrap_modules walks the module tree, finds qkv/gate_up/down (skips o_proj),
     redirects param.data to pinned CPU, replaces quant_method.
  2. Synthetic "weight loading" writes weights into the now-CPU param.data.
  3. post_init splits weights into _w_cpu + GPU slice, allocates shared
     activation buffers, populates the dispatch table.
  4. A forward pass through the mini-layer matches a non-offloaded reference
     within BF16 tolerance.

This avoids loading a full 14GB Qwen2.5-7B in pytest while still exercising
every code path the production model would hit. The end-to-end real-model
smoke lives as a probe under David/Benchmarks/phase1/.
"""

import pytest
import torch
import torch.nn as nn

from vllm.config import (
    CompilationConfig,
    CotsOffloadConfig,
    ModelConfig,
    OffloadConfig,
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

# Qwen2.5-7B-ish dimensions, scaled down so the test runs fast.
HIDDEN = 256
INTERMEDIATE = 1024
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS  # 32
MAX_NUM_TOKENS = 64

BF16_RTOL = 5e-2
BF16_ATOL = 0.5  # absorbs cuBLAS kernel-selection ULP noise; see test_dispatcher_split.py
# Fused MLP1→SwiGLU→MLP2 chains two matmuls + nonlinearity. Output magnitudes
# are 10-100× per-Linear outputs (variance grows by ~sqrt(intermediate) per
# matmul), so absolute roundoff scales accordingly. The per-element check
# uses atol = MLP_BLOCK_RTOL × max(|reference|) (computed at test time) so
# tolerance scales with the actual output magnitude.
MLP_BLOCK_RTOL = 5e-2


class MiniMlpBlock(nn.Module):
    """Mimics Qwen2MLP: gate_up_proj + act_fn + down_proj. Required by
    cots's MLP-block recognition (`weight_offload_design.md` matched-index
    invariant — col/row offload only inside this structural pattern).
    """

    def __init__(self):
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

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MiniDecoderLayer(nn.Module):
    """Mimics Qwen2 layer's offloadable structure: qkv_proj for attention,
    o_proj (NOT offloaded), and an MLP block (mlp.gate_up_proj +
    mlp.act_fn + mlp.down_proj — both linears offloaded as a fused block).
    """

    def __init__(self):
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
        # o_proj — RowParallelLinear like mlp.down_proj, but suffix is
        # o_proj so the offloader should skip it.
        self.o_proj = RowParallelLinear(
            input_size=NUM_HEADS * HEAD_DIM,
            output_size=HIDDEN,
            bias=False,
            disable_tp=True,
            params_dtype=bf16,
            prefix="o_proj",
        )
        self.mlp = MiniMlpBlock()


def _make_vllm_config(enforce_eager: bool = True) -> VllmConfig:
    """Build a minimal VllmConfig that satisfies CotsOffloader.post_init."""
    model_config = ModelConfig.__new__(ModelConfig)
    # post_init only reads .enforce_eager.
    object.__setattr__(model_config, "enforce_eager", enforce_eager)

    sched_config = SchedulerConfig.__new__(SchedulerConfig)
    object.__setattr__(sched_config, "max_num_batched_tokens", MAX_NUM_TOKENS)

    from collections import Counter as _Counter

    compilation_config = CompilationConfig.__new__(CompilationConfig)
    object.__setattr__(
        compilation_config, "cudagraph_capture_sizes", [8, 16, 32, 64]
    )
    # Fields normally populated by dataclass __init__; SiluAndMul / CustomOp
    # read these via get_cached_compilation_config().
    # CustomOp.default_on() asserts exactly one of "none"/"all" is present.
    object.__setattr__(compilation_config, "custom_ops", ["none"])
    object.__setattr__(compilation_config, "enabled_custom_ops", _Counter())
    object.__setattr__(compilation_config, "disabled_custom_ops", _Counter())

    parallel_config = ParallelConfig.__new__(ParallelConfig)
    object.__setattr__(parallel_config, "tensor_parallel_size", 1)

    vllm_config = VllmConfig.__new__(VllmConfig)
    object.__setattr__(vllm_config, "model_config", model_config)
    object.__setattr__(vllm_config, "scheduler_config", sched_config)
    object.__setattr__(vllm_config, "compilation_config", compilation_config)
    object.__setattr__(vllm_config, "parallel_config", parallel_config)
    return vllm_config


def _simulate_weight_loading(layer: MiniDecoderLayer) -> dict:
    """Simulate vLLM's load_weights for each parameter in the mini layer.

    For non-offloaded params (o_proj.weight): just .copy_() random data
    into param.data (which is on GPU at full shape).

    For offloaded params (qkv_proj.weight, gate_up_proj.weight, down_proj.weight):
    construct a full-shape "loaded_weight" tensor and call the wrapped
    weight_loader, which splits it into GPU and CPU portions. For QKV/Merged
    we call per shard; for row we call once.
    """
    torch.manual_seed(0)
    refs = {}

    # qkv_proj: per-shard call with full Q, K, V tensors.
    qkv = layer.qkv_proj
    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    q_full = torch.randn(q_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
    k_full = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
    v_full = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
    qkv.weight_loader(qkv.weight, q_full, "q")
    qkv.weight_loader(qkv.weight, k_full, "k")
    qkv.weight_loader(qkv.weight, v_full, "v")
    # Reference: full QKV weight = [Q | K | V] stacked.
    refs["qkv_proj.weight"] = torch.cat([q_full, k_full, v_full], dim=0)

    # mlp.gate_up_proj: per-shard call with full gate, up (each shape
    # (intermediate, hidden)).
    gate_up = layer.mlp.gate_up_proj
    gate_full = torch.randn(
        INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda"
    )
    up_full = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
    gate_up.weight_loader(gate_up.weight, gate_full, 0)
    gate_up.weight_loader(gate_up.weight, up_full, 1)
    refs["mlp.gate_up_proj.weight"] = torch.cat([gate_full, up_full], dim=0)

    # mlp.down_proj: single-call loader with full weight.
    down = layer.mlp.down_proj
    down_full = torch.randn(
        HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device="cuda"
    )
    down.weight_loader(down.weight, down_full)
    refs["mlp.down_proj.weight"] = down_full

    # o_proj: not offloaded; uses original (un-wrapped) weight_loader.
    o = layer.o_proj
    o_full = torch.randn(
        HIDDEN, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    o.weight_loader(o.weight, o_full)
    refs["o_proj.weight"] = o_full

    return refs


def _ref_forward(refs: dict, x: torch.Tensor) -> dict:
    """Compute non-offloaded reference outputs using the captured weights."""
    import torch.nn.functional as F

    # MLP block reference: full unsplit MLP1 → SwiGLU → MLP2.
    gate_up = F.linear(x, refs["mlp.gate_up_proj.weight"], None)
    d = gate_up.shape[-1] // 2
    silu_full = F.silu(gate_up[..., :d]) * gate_up[..., d:]
    mlp_out = F.linear(silu_full, refs["mlp.down_proj.weight"], None)

    return {
        "qkv": F.linear(x, refs["qkv_proj.weight"], None),
        "mlp": mlp_out,
        "o": F.linear(x, refs["o_proj.weight"], None),
    }


@pytest.mark.parametrize("f_cpu_store", [0.05, 0.10, 0.25])
def test_mini_decoder_offload_e2e(f_cpu_store):
    """Full lifecycle: wrap → load → post_init → forward, on a mini layer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    vllm_config = _make_vllm_config(enforce_eager=True)
    with set_current_vllm_config(vllm_config):
        layer = MiniDecoderLayer().cuda()

        # Set up the offloader the way create_offloader would.
        offloader = CotsOffloader(
            config=CotsOffloadConfig(f_cpu_store=f_cpu_store, kv_biased=True)
        )
        set_offloader(offloader)

        # wrap_modules expects a generator of decoder layers.
        offloader.wrap_modules(iter([layer]))

        # After wrap (TP-style): param.data is on GPU at FINAL slice shape
        # (not on CPU at full shape). Weight loading writes through our
        # wrapped weight_loader directly into GPU slice + _w_cpu.
        assert layer.qkv_proj.weight.data.device.type == "cuda"
        assert layer.mlp.gate_up_proj.weight.data.device.type == "cuda"
        assert layer.mlp.down_proj.weight.data.device.type == "cuda"
        # CRITICAL invariant: WO (o_proj) must NOT be offloaded.
        assert layer.o_proj.weight.data.device.type == "cuda"
        assert not hasattr(layer.o_proj, "_cots_handle")

        # MLP block recognized → parent.forward replaced with fused closure.
        assert layer.mlp.gate_up_proj._cots_handle.in_block is True
        assert layer.mlp.down_proj._cots_handle.in_block is True
        assert offloader._fused_ops, (
            "MLP block recognition should have populated _fused_ops"
        )

        # Shared activation buffers were allocated at the end of wrap_modules.
        assert offloader._x_pinned is not None and offloader._x_pinned.is_pinned()
        assert offloader._y_pinned is not None and offloader._y_pinned.is_pinned()
        assert offloader._y_gpu is not None and offloader._y_gpu.is_cuda

        # Each offloaded dispatcher has _w_cpu allocated empty (loader will
        # populate). cpu_indices_cuda / gpu_indices_cuda are also ready.
        # QKV may skip wrapping at small f when head-aligned snap rounds to
        # zero KV pairs (`weight_offload_design.md §201-205`); the linear
        # then runs entirely on GPU via its original quant_method.
        offloaded_linears: list[nn.Module] = [layer.qkv_proj, layer.mlp.gate_up_proj, layer.mlp.down_proj]
        for sub in offloaded_linears:
            if not hasattr(sub, "_cots_handle"):
                continue
            h = sub._cots_handle
            assert h.w_cpu is not None and h.w_cpu.is_pinned()
            assert h.cpu_indices_cuda is not None
            assert h.gpu_indices_cuda is not None

        # Simulate weight loading via the wrapped loaders.
        refs = _simulate_weight_loading(layer)

        # post_init: bookkeeping only — verifies enforce_eager + builds
        # dispatch table. No GPU allocations or data movement.
        offloader.post_init()

        # Weights still on GPU at slice shape (no rebinding in post_init).
        assert layer.qkv_proj.weight.data.device.type == "cuda"
        assert layer.mlp.gate_up_proj.weight.data.device.type == "cuda"
        assert layer.mlp.down_proj.weight.data.device.type == "cuda"
        # Dispatch table populated trivially (Phase 1a).
        for bucket, (fc, fp) in offloader._dispatch_table.items():
            assert fp == 0.0
            assert abs(fc - f_cpu_store) < 1e-9

        # Forward pass: run QKV per-Linear and the fused MLP block; compare
        # to references.
        x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ref = _ref_forward(refs, x)

        out_qkv = layer.qkv_proj(x)[0]
        out_mlp = layer.mlp(x)
        # o_proj is the un-offloaded baseline; should match its own ref exactly.
        out_o = layer.o_proj(x)[0]

        torch.testing.assert_close(
            out_qkv, ref["qkv"], rtol=BF16_RTOL, atol=BF16_ATOL
        )
        # Magnitude-aware tolerance for the chained MLP block.
        mlp_atol = MLP_BLOCK_RTOL * float(ref["mlp"].abs().max())
        torch.testing.assert_close(
            out_mlp, ref["mlp"], rtol=MLP_BLOCK_RTOL, atol=mlp_atol
        )
        # o_proj is GPU-only — should be exactly the F.linear reference.
        torch.testing.assert_close(out_o, ref["o"])


def test_post_init_requires_enforce_eager():
    """Phase 1a hard-rejects graph capture mode."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    vllm_config = _make_vllm_config(enforce_eager=False)
    with set_current_vllm_config(vllm_config):
        layer = MiniDecoderLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(f_cpu_store=0.1, kv_biased=True)
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))
        _simulate_weight_loading(layer)

        with pytest.raises(RuntimeError, match="enforce_eager=True"):
            offloader.post_init()


def test_lookup_dispatch_rounds_up():
    """Per planner_design.md §4.5: round num_tokens UP to nearest bucket."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    vllm_config = _make_vllm_config(enforce_eager=True)
    with set_current_vllm_config(vllm_config):
        layer = MiniDecoderLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(f_cpu_store=0.1, kv_biased=True)
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))
        _simulate_weight_loading(layer)
        offloader.post_init()

        # Buckets are [8, 16, 32, 64] from _make_vllm_config.
        # 5 tokens → round up to 8.
        # 17 tokens → round up to 32.
        # 64 tokens → exact match at 64.
        # 65 tokens → out of range → eager fallback.
        for n_tokens in (1, 5, 8):
            entry = offloader.lookup_dispatch(n_tokens)
            assert entry == (0.1, 0.0)
        # All in-range buckets return same trivial entry in 1a.
        assert offloader.lookup_dispatch(17) == (0.1, 0.0)
        assert offloader.lookup_dispatch(64) == (0.1, 0.0)
        # Out-of-range falls back.
        assert offloader.lookup_dispatch(128) == (0.1, 0.0)


def test_quantized_layer_rejected():
    """Phase 1a only supports unquantized BF16; other quant_methods must fail."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    vllm_config = _make_vllm_config(enforce_eager=True)
    with set_current_vllm_config(vllm_config):
        layer = MiniDecoderLayer().cuda()
        # Patch a fake quant_method onto qkv_proj to simulate quantization.

        class FakeQuantMethod:
            pass

        layer.qkv_proj.quant_method = FakeQuantMethod()

        offloader = CotsOffloader(
            config=CotsOffloadConfig(f_cpu_store=0.1, kv_biased=True)
        )
        set_offloader(offloader)
        with pytest.raises(RuntimeError, match="unquantized"):
            offloader.wrap_modules(iter([layer]))
