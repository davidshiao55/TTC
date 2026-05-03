"""Shared pytest fixtures for Phase 1a tests."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_vllm_distributed():
    """Initialize a single-rank TP group for tests that construct vLLM
    Linear layers (their parameter constructors call get_tp_group() even
    when disable_tp=True).

    Session-scoped because re-initializing distributed in the same process
    isn't supported.
    """
    import torch

    if not torch.cuda.is_available():
        yield
        return

    # Set up the env vars init_distributed_environment expects.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    # initialize_model_parallel reads a vLLM config; set a default for the
    # init phase. Individual tests still set their own VllmConfig via
    # set_current_vllm_config when they need specific values.
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
            backend="gloo",  # tests don't run collectives
        )
        initialize_model_parallel(tensor_model_parallel_size=1)

    yield
    # Don't tear down — distributed teardown in pytest sessions is unreliable.
