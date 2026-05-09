# SPDX-License-Identifier: Apache-2.0
"""Stage 5 — multi-engine coexistence.

FastTTS launches a generator + verifier as two `LLM` engines in the
same process. Each engine has its own `CotsOffloader` instance, which
constructs its own `NativeCotsRunner` registered in the module-private
`cots_ops._COTS_RUNNERS` weak map under a unique `runner_id`. The
custom-op impls (`vllm.cots_submit_gemm`, `vllm.cots_sync_then_uva`)
look up the right runner by id, so two offloaders coexist with
independent slab pools and no C++ singleton.

This test confirms:
  * Two NativeCotsRunner instances get distinct runner_ids.
  * Both register in the registry; both look up correctly.
  * Interleaved submits across two CotsCpuInfer instances don't
    cross-talk — each runner's slab pool stays its own.
  * Closing one runner unregisters it without disturbing the other.
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
from vllm.model_executor.offloader import CotsOffloader, cots_ops, set_offloader

pytestmark = pytest.mark.needs_cuda


HIDDEN = 256
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
MAX_NUM_TOKENS = 32

BF16_RTOL = 5e-2
BF16_ATOL = 0.5


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


def _build_qkv_offloader(
    *, f_cpu_store: float, seed: int
) -> tuple[_QkvLayer, CotsOffloader]:
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                cpu_runner="native",
                kv_biased=True,
            )
        )
        # NB: set_offloader is global; the test cycles which offloader
        # is "current" so the custom-op impls (which call
        # `_lookup_runner(runner_id)` via the registry, not through
        # set_offloader) still find both. The test below alternates
        # forwards explicitly, so we set whichever is most recent.
        set_offloader(offloader)
        offloader.wrap_modules(iter([layer]))
        torch.manual_seed(seed)
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


def test_two_runners_have_distinct_runner_ids() -> None:
    """Trivial registry sanity: each NativeCotsRunner gets a unique id,
    and each id resolves to a distinct `CotsCpuInfer` in the registry
    (§1c.19: registry now holds infers, not runners)."""
    from vllm.model_executor.offloader import cots

    r1 = cots.NativeCotsRunner(dry_run=False)
    r2 = cots.NativeCotsRunner(dry_run=False)
    try:
        assert r1._runner_id != r2._runner_id
        infer1 = cots_ops._COTS_INFER.get(r1._runner_id)
        infer2 = cots_ops._COTS_INFER.get(r2._runner_id)
        assert infer1 is not None
        assert infer2 is not None
        assert infer1 is not infer2
    finally:
        r1.close()
        r2.close()


def test_close_one_runner_does_not_affect_other() -> None:
    """Tearing down one runner unregisters its id; the other's id
    stays valid in the registry."""
    from vllm.model_executor.offloader import cots

    r1 = cots.NativeCotsRunner(dry_run=False)
    r2 = cots.NativeCotsRunner(dry_run=False)
    try:
        rid1, rid2 = r1._runner_id, r2._runner_id
        assert cots_ops._COTS_INFER.get(rid1) is not None
        infer2 = cots_ops._COTS_INFER.get(rid2)
        assert infer2 is not None

        r1.close()
        assert cots_ops._COTS_INFER.get(rid1) is None
        assert cots_ops._COTS_INFER.get(rid2) is infer2
    finally:
        r2.close()


def test_two_offloaders_produce_independent_outputs() -> None:
    """Build two offloaders with DIFFERENT random weights (different
    seeds). Run identical inputs through both; outputs must differ
    (proving they're not sharing weights / slabs / runners)."""
    layer_a, off_a = _build_qkv_offloader(f_cpu_store=0.20, seed=10)
    layer_b, off_b = _build_qkv_offloader(f_cpu_store=0.20, seed=20)
    torch.manual_seed(99)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    try:
        # Two distinct runners.
        assert off_a._runner is not None and off_b._runner is not None
        assert off_a._runner._runner_id != off_b._runner._runner_id

        set_offloader(off_a)
        out_a, _ = layer_a.qkv_proj(x)
        set_offloader(off_b)
        out_b, _ = layer_b.qkv_proj(x)

        # Different weights → different outputs.
        diff = (out_a.float() - out_b.float()).abs().max().item()
        assert diff > 0.5, (
            f"Two offloaders with different weights produced near-"
            f"identical output (max abs diff {diff:.4f}). Likely shared "
            f"slab pool or registry collision."
        )

        # Re-run the first to confirm it's still self-consistent (its
        # registry entry didn't get clobbered by the second's
        # construction).
        set_offloader(off_a)
        out_a_again, _ = layer_a.qkv_proj(x)
        torch.testing.assert_close(out_a_again, out_a, rtol=0, atol=0)
    finally:
        if off_a._runner is not None:
            off_a._runner.close()
        if off_b._runner is not None:
            off_b._runner.close()


def test_interleaved_forwards_do_not_cross_talk() -> None:
    """A more aggressive variant: alternate forwards between two
    offloaders inside the same loop. If their slab pools shared
    state, outputs would drift. Each forward should match its first-
    iteration result bit-exactly because nothing in the offloader
    state varies between iterations."""
    layer_a, off_a = _build_qkv_offloader(f_cpu_store=0.10, seed=33)
    layer_b, off_b = _build_qkv_offloader(f_cpu_store=0.40, seed=44)
    torch.manual_seed(123)
    x = torch.randn(MAX_NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    try:
        set_offloader(off_a)
        out_a_first, _ = layer_a.qkv_proj(x)
        out_a_first = out_a_first.clone()
        set_offloader(off_b)
        out_b_first, _ = layer_b.qkv_proj(x)
        out_b_first = out_b_first.clone()

        for _ in range(10):
            set_offloader(off_a)
            out_a_n, _ = layer_a.qkv_proj(x)
            torch.testing.assert_close(out_a_n, out_a_first, rtol=0, atol=0)
            set_offloader(off_b)
            out_b_n, _ = layer_b.qkv_proj(x)
            torch.testing.assert_close(out_b_n, out_b_first, rtol=0, atol=0)
    finally:
        if off_a._runner is not None:
            off_a._runner.close()
        if off_b._runner is not None:
            off_b._runner.close()
