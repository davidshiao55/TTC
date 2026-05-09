# SPDX-License-Identifier: Apache-2.0
# Phase 1c test fixtures. Mirrors the Phase 1b / 1a layout so the three
# suites can run side-by-side.

import os

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def init_distributed_session():
    """Phase 1a/1b tests use a TP-1 gloo init; Phase 1c follows the same
    pattern so the offloader's wrap_modules / TP-1 invariants exercise
    correctly. Single-rank, single-process — no real distribution."""
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")  # different from phase1b
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        torch.distributed.init_process_group("gloo", world_size=1, rank=0)
    yield


def pytest_collection_modifyitems(config, items):
    """Skip GPU-dependent tests when CUDA is unavailable, instead of erroring."""
    if not torch.cuda.is_available():
        skip_no_cuda = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "cuda" in item.keywords or "needs_cuda" in item.keywords:
                item.add_marker(skip_no_cuda)
