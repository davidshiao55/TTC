"""Native prefetch backend split: stock vs thesis-deferred."""

import inspect

import pytest
import torch

import vllm.model_executor.offloader.prefetch_defer_ops as prefetch_defer_ops
import vllm.model_executor.offloader.prefetch_ops as prefetch_ops
from vllm.config import OffloadConfig, PrefetchOffloadConfig
from vllm.model_executor.offloader.base import BaseOffloader, create_offloader
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
    # `deferred_wraparound_index` is an instance attribute set only by
    # `PrefetchDeferOffloader.__init__` (prefetch_defer.py:59), so the
    # stock instance correctly lacks it.
    assert not hasattr(offloader, "deferred_wraparound_index")
    # `_start_deferred_prefetch` is a no-op default on BaseOffloader
    # (base.py:91-94) — lifted up during phase 1b so
    # `torch.ops.vllm.start_deferred_prefetch`'s impl can call
    # `get_offloader()._start_deferred_prefetch()` unconditionally,
    # with only PrefetchDeferOffloader overriding the no-op with a real
    # body (prefetch_defer.py:153). The stock class therefore *inherits*
    # the method but must NOT override it. Asserting `not hasattr` would
    # falsely fail on the inherited no-op; assert the unbound-method
    # identity instead, which captures the actual invariant.
    assert (
        PrefetchOffloader._start_deferred_prefetch
        is BaseOffloader._start_deferred_prefetch
    ), (
        "PrefetchOffloader should NOT override _start_deferred_prefetch — "
        "that hook is thesis-specific (PrefetchDeferOffloader) machinery."
    )
    assert "dry_run" not in inspect.signature(PrefetchOffloader).parameters


def test_prefetch_defer_backend_factory_returns_defer_class():
    offloader = create_offloader(_config("prefetch_defer", dry_run=True))

    assert type(offloader) is PrefetchDeferOffloader
    assert offloader.dry_run is True
    assert hasattr(offloader, "deferred_wraparound_index")
    # PrefetchDeferOffloader DOES override the BaseOffloader no-op with a
    # real body (prefetch_defer.py:153). Symmetric to the stock-class
    # assertion above: the override identity is the load-bearing
    # invariant, not just the presence of the attribute.
    assert (
        PrefetchDeferOffloader._start_deferred_prefetch
        is not BaseOffloader._start_deferred_prefetch
    ), (
        "PrefetchDeferOffloader must override _start_deferred_prefetch "
        "with a real body, not inherit the BaseOffloader no-op."
    )


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
