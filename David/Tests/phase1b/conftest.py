"""Shared pytest fixtures for Phase 1b tests.

Mirrors phase1a/conftest.py but is idempotent: skips re-initialization if
phase1a's session-scoped fixture already brought up the TP group.
"""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_vllm_distributed():
    """Initialize a single-rank TP group; no-op if already initialized."""
    import torch

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
    os.environ.setdefault("MASTER_PORT", "29500")
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
