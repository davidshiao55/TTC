# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for Phase 1c tests.

Mirrors phase1a/phase1b conftests: brings up a single-rank tensor
parallel group at session scope so vLLM's `QKVParallelLinear` /
`MergedColumnParallelLinear` / `RowParallelLinear` constructors (used
by parity tests) find the TP group at init time. Idempotent — skips
re-init if a prior session-scoped fixture already initialized.
"""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def _init_vllm_distributed():
    """Initialize a single-rank TP group; no-op if already initialized."""
    if not torch.cuda.is_available():
        yield
        return

    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized():
        yield
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")  # distinct from phase1a/1b
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)

    yield


def pytest_collection_modifyitems(config, items):
    """Skip GPU-dependent tests when CUDA is unavailable, instead of erroring."""
    if not torch.cuda.is_available():
        skip_no_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords or "needs_cuda" in item.keywords:
                item.add_marker(skip_no_cuda)
