"""Native prefetch backend split: stock vs thesis-deferred."""

import inspect

import pytest
import torch

import vllm.model_executor.offloader.prefetch_defer_ops as prefetch_defer_ops
import vllm.model_executor.offloader.prefetch_ops as prefetch_ops
from vllm.config import OffloadConfig, PrefetchOffloadConfig
from vllm.model_executor.offloader.base import create_offloader
from vllm.model_executor.offloader.prefetch import (
    PrefetchOffloader,
    _ModuleOffloader,
)
from vllm.model_executor.offloader.prefetch_defer import (
    PrefetchDeferOffloader,
    _DryRunModuleOffloader,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="prefetch offloader constructors allocate CUDA streams",
)


def _config(backend: str, *, dry_run: bool = False) -> OffloadConfig:
    return OffloadConfig(
        offload_backend=backend,
        prefetch=PrefetchOffloadConfig(
            offload_group_size=2,
            offload_num_in_group=1,
            offload_prefetch_step=1,
            dry_run=dry_run,
        ),
    )


def test_prefetch_backend_factory_returns_stock_class():
    offloader = create_offloader(_config("prefetch"))

    assert type(offloader) is PrefetchOffloader
    assert not hasattr(offloader, "deferred_wraparound_index")
    assert not hasattr(offloader, "_start_deferred_prefetch")
    assert "dry_run" not in inspect.signature(PrefetchOffloader).parameters


def test_prefetch_defer_backend_factory_returns_defer_class():
    offloader = create_offloader(_config("prefetch_defer", dry_run=True))

    assert type(offloader) is PrefetchDeferOffloader
    assert offloader.dry_run is True
    assert hasattr(offloader, "deferred_wraparound_index")
    assert hasattr(offloader, "_start_deferred_prefetch")


def test_auto_prefetch_selects_stock_class():
    offloader = create_offloader(_config("auto"))

    assert type(offloader) is PrefetchOffloader


def test_stock_prefetch_has_no_deferred_hook_branch():
    source = inspect.getsource(PrefetchOffloader._hook_module_forward)
    module_source = inspect.getsource(
        __import__(
            "vllm.model_executor.offloader.prefetch",
            fromlist=["PrefetchOffloader"],
        )
    )

    assert "deferred" not in source
    assert "raw_next" not in source
    assert "start_deferred_prefetch" not in source
    assert "PrefetchDeferOffloader" not in module_source
    assert "_DryRunModuleOffloader" not in module_source


def test_stock_prefetch_ops_have_no_deferred_op():
    source = inspect.getsource(prefetch_ops)

    assert "start_deferred_prefetch" not in source
    assert "register_prefetch_defer_offloader_ops" not in source


def test_deferred_op_lives_in_defer_ops_module():
    source = inspect.getsource(prefetch_defer_ops)

    assert "start_deferred_prefetch" in source
    assert "register_prefetch_defer_offloader_ops" in source


def test_dry_run_lives_outside_stock_module_offloader():
    stock_params = inspect.signature(_ModuleOffloader).parameters
    dry_params = inspect.signature(_DryRunModuleOffloader).parameters

    assert "dry_run" not in stock_params
    assert "dry_run" not in dry_params
    assert _DryRunModuleOffloader is not _ModuleOffloader
