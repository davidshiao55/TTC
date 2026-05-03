"""Phase 1a §6 — Fused MLP block forward correctness + transfer count.

Verifies the matched-index invariant from `weight_offload_design.md`:

1. `CotsSwiGLUMLPOp` (installed as `Qwen2MLP.forward`) produces output equal
   to the unsplit reference within BF16 tolerance.
2. **Exactly one UVA copy fires per MLP-block forward** — down from three
   in the old per-Linear path. This is the load-bearing perf invariant
   that motivates the block-level operator.

Mirrors `phase0/bench_split_correctness.py §D` (the MLP1→MLP2 col→row
pipeline reference) on a synthetic mini block.
"""

from collections import Counter
from unittest.mock import patch

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
    RowParallelLinear,
)
from vllm.model_executor.offloader import CotsOffloader, set_offloader

HIDDEN = 256
INTERMEDIATE = 1024
MAX_NUM_TOKENS = 64
BF16_RTOL = 5e-2


class MiniMlpBlock(nn.Module):
    """Qwen2MLP shape for cots's MLP-block recognition."""

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
        gu, _ = self.gate_up_proj(x)
        x = self.act_fn(gu)
        x, _ = self.down_proj(x)
        return x


class _Layer(nn.Module):
    """Wraps the MLP so cots's wrap_modules generator yields one layer
    that contains the fusable parent (matches Qwen2DecoderLayer shape)."""

    def __init__(self):
        super().__init__()
        self.mlp = MiniMlpBlock()


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


def _build_block(f_cpu_store: float):
    """Build the layer + offloader, populate weights, post_init."""
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _Layer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(f_cpu_store=f_cpu_store, kv_biased=True)
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))

        # Capture full reference weights, populate via wrapped loaders.
        torch.manual_seed(0)
        gate_full = torch.randn(
            INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda"
        )
        up_full = torch.randn(
            INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda"
        )
        down_full = torch.randn(
            HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device="cuda"
        )
        layer.mlp.gate_up_proj.weight_loader(
            layer.mlp.gate_up_proj.weight, gate_full, 0
        )
        layer.mlp.gate_up_proj.weight_loader(
            layer.mlp.gate_up_proj.weight, up_full, 1
        )
        layer.mlp.down_proj.weight_loader(layer.mlp.down_proj.weight, down_full)

        offloader.post_init()
    return layer, offloader, gate_full, up_full, down_full


@pytest.mark.parametrize("f_cpu_store", [0.10, 0.25, 0.50])
def test_fused_mlp_matches_unsplit_reference(f_cpu_store):
    """The fused block output equals the unsplit `MLP1 → SwiGLU → MLP2`
    reference within BF16 tolerance scaled to output magnitude."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    layer, _, gate, up, down = _build_block(f_cpu_store)

    # MLP-block recognition must have fired:
    assert layer.mlp.gate_up_proj._cots_handle.in_block is True
    assert layer.mlp.down_proj._cots_handle.in_block is True

    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    out = layer.mlp(x)

    gate_up = F.linear(x, torch.cat([gate, up], dim=0), None)
    silu_full = F.silu(gate_up[:, :INTERMEDIATE]) * gate_up[:, INTERMEDIATE:]
    ref = F.linear(silu_full, down, None)

    # Magnitude-aware atol — chained matmul output has output magnitude
    # ~sqrt(intermediate)·O(per-Linear); rtol covers large-value tail,
    # atol = rtol × peak covers near-zero positions.
    atol = BF16_RTOL * float(ref.abs().max())
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=atol)


def test_fused_mlp_emits_exactly_one_uva_copy():
    """Counts UVA-copy invocations during one MLP-block forward.

    Per-Linear path (pre-fusion) would emit 3 transfers per block:
      MLP1 CPU output → GPU (UVA), GPU intermediate → CPU (D2H), CPU MLP2
      output → GPU (UVA).
    Fused path emits just **1**: only MLP2's CPU partial returns to GPU.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    layer, _, _, _, _ = _build_block(f_cpu_store=0.25)
    x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, device="cuda")

    import vllm.model_executor.offloader.cots as cots_mod

    counter = {"calls": 0}
    real = cots_mod.uva_copy_into_gpu

    def counting_uva(src, dst):
        counter["calls"] += 1
        return real(src, dst)

    with patch.object(cots_mod, "uva_copy_into_gpu", counting_uva):
        # CotsSwiGLUMLPOp.__call__ resolves uva_copy_into_gpu from module
        # scope on each call, so patch.object reaches it.
        _ = layer.mlp(x)
        torch.cuda.synchronize()

    assert counter["calls"] == 1, (
        f"Expected exactly 1 UVA copy per MLP block (matched-index invariant); "
        f"got {counter['calls']}"
    )


def test_orphan_col_row_raises():
    """Phase 1a contract: a wrapped MergedCol/Row outside an MLP block
    structure raises at wrap_modules — no silent wrong output."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    class OrphanLayer(nn.Module):
        def __init__(self):
            super().__init__()
            bf16 = torch.bfloat16
            # Standalone MergedCol with no parent MLP structure (no act_fn,
            # no down_proj sibling at same level).
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=HIDDEN,
                output_sizes=[INTERMEDIATE, INTERMEDIATE],
                bias=False,
                disable_tp=True,
                params_dtype=bf16,
                prefix="orphan.gate_up_proj",
            )

    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = OrphanLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(f_cpu_store=0.25, kv_biased=True)
        )
        set_offloader(offloader)
        with pytest.raises(RuntimeError, match="MLP block"):
            offloader.wrap_modules(iter([layer]))
