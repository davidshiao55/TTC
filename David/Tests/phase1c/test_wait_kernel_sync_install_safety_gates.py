# SPDX-License-Identifier: Apache-2.0
"""§1c.29 commit 2 — wait-kernel sync safety gates.

`CotsOffloader.post_init` enforces the four hard-fail preconditions
documented in `David/Docs/phase1c_findings.md` §1c.29 when
`cots_capture_sync_mode="wait_kernel"`:

  1. cpu_runner != 'native' → RuntimeError
  2. enforce_eager=True     → RuntimeError
  3. CUDA not available     → RuntimeError (skipped on this runner)
  4. _cots_C not built      → RuntimeError (covered indirectly: tests
                              run only when the extension imports)

The gates are checked BEFORE the slab pool is allocated so a
misconfiguration cannot leave a half-installed runner behind.
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
    m3: bool,
    expect_raise: type[BaseException] | None,
    raise_match: str | None = None,
) -> None:
    """Build the offloader and run post_init under a vllm config
    context. Optionally assert that post_init raises a specific
    exception type/message.
    """
    vc = _make_vllm_config(enforce_eager=enforce_eager)
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=0.10,
                cpu_runner=cpu_runner,
                cots_capture_sync_mode=("wait_kernel" if m3 else "host_callback"),
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
                offloader.post_init()
        finally:
            if offloader._runner is not None:
                offloader._runner.close()


def test_m3_with_python_runner_raises() -> None:
    """Gate 1: wait-kernel sync + cpu_runner='python' is rejected. Python runner
    has no host-mapped done_slot, no worker thread to publish, no
    slab pool — none of the wait-kernel-sync mechanism applies. The gate fires
    even with enforce_eager=True (which would otherwise satisfy the
    earlier python+graph check)."""
    _drive_post_init(
        cpu_runner="python",
        enforce_eager=True,
        m3=True,
        expect_raise=RuntimeError,
        raise_match=r"cots_capture_sync_mode='wait_kernel' requires cpu_runner='native'",
    )


def test_m3_with_eager_mode_raises() -> None:
    """Gate 2: wait-kernel sync + enforce_eager=True is rejected. The wait kernel
    replaces a captured sync_cb host_fn node; under enforce_eager
    there is no captured node to replace, so wait-kernel sync adds round-trip
    cost without removing any captured cost — net negative."""
    _drive_post_init(
        cpu_runner="native",
        enforce_eager=True,
        m3=True,
        expect_raise=RuntimeError,
        raise_match=r"cots_capture_sync_mode='wait_kernel' requires enforce_eager=False",
    )


def test_m3_native_graph_passes() -> None:
    """Production config: wait-kernel sync + cpu_runner='native' + enforce_eager=False.
    No gates fire; the offloader installs the slab pool AND the wait-kernel sync
    host-mapped slots successfully."""
    _drive_post_init(
        cpu_runner="native",
        enforce_eager=False,
        m3=True,
        expect_raise=None,
    )


def test_m3_disabled_default_path_unchanged() -> None:
    """Default config: cots_capture_sync_mode="host_callback". None of the wait-kernel sync
    gates fire; the legacy sync_cb host_fn path is wired (verified
    indirectly — post_init succeeds under both eager and graph-capture
    mode regardless of cpu_runner, as the existing test suite
    already covers)."""
    _drive_post_init(
        cpu_runner="native",
        enforce_eager=False,
        m3=False,
        expect_raise=None,
    )


def test_m3_install_marks_every_slab() -> None:
    """After post_init with wait-kernel sync enabled, every slab in the pool has
    `wait_kernel_sync_installed=True`. Confirms the install_m3 walk reaches all
    task_ids (no off-by-one) and that the per-slab gate inside
    `sync_or_wait_on_stream` will dispatch to the wait kernel for
    every dispatch."""
    pytest.importorskip("vllm._cots_C")

    vc = _make_vllm_config(enforce_eager=False)
    with set_current_vllm_config(vc):
        layer = _QkvLayer().cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=0.10,
                cpu_runner="native",
                cots_capture_sync_mode="wait_kernel",
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
        offloader.post_init()
        try:
            from vllm.model_executor.offloader import cots_ops

            infer = cots_ops._lookup_infer(
                offloader._runner._runner_id, "test_m3_install_marks_every_slab"
            )
            n_slabs = offloader._runner._n_slabs
            assert n_slabs > 0, "test fixture should produce at least one slab"
            for tid in range(n_slabs):
                assert infer.wait_kernel_sync_installed_for_task(tid), (
                    f"task_id={tid} not flagged wait_kernel_sync_installed after post_init"
                )
        finally:
            if offloader._runner is not None:
                offloader._runner.close()
