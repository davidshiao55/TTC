import pytest

from vllm.config import CotsOffloadConfig, OffloadConfig
from vllm.model_executor.offloader.base import create_offloader


def test_cots_weight_modules_default_to_production_modules():
    assert CotsOffloadConfig().weight_modules == {"qkv", "mlp", "wo"}


def test_cots_weight_modules_accept_list_and_comma_separated_forms():
    assert CotsOffloadConfig(weight_modules="qkv,wo").weight_modules == {
        "qkv",
        "wo",
    }
    assert CotsOffloadConfig(weight_modules={"qkv,wo"}).weight_modules == {
        "qkv",
        "wo",
    }
    assert CotsOffloadConfig(weight_modules=["qkv", "wo"]).weight_modules == {
        "qkv",
        "wo",
    }
    assert CotsOffloadConfig(weight_modules={"QKV", "mlp"}).weight_modules == {
        "qkv",
        "mlp",
    }


def test_cots_weight_modules_reject_unknown_entries():
    with pytest.raises(ValueError, match="unsupported entries"):
        CotsOffloadConfig(weight_modules={"qkv", "attn"})


def test_cots_dispatch_table_config_accepts_valid_entries():
    cots = CotsOffloadConfig(
        f_cpu_store=0.10,
        dispatch_table={64: (0.04, 0.06), 128: (0.02, 0.08)},
    )
    config = OffloadConfig(offload_backend="cots", cots=cots)

    assert config.cots.dispatch_table == {64: (0.04, 0.06), 128: (0.02, 0.08)}


def test_cots_dispatch_table_rejects_entries_above_storage_fraction():
    with pytest.raises(ValueError, match="entry exceeds f_cpu_store"):
        CotsOffloadConfig(
            f_cpu_store=0.10,
            dispatch_table={64: (0.08, 0.04)},
        )


def test_cots_dispatch_table_rejects_incomplete_partition():
    with pytest.raises(ValueError, match="entry must sum to f_cpu_store"):
        CotsOffloadConfig(
            f_cpu_store=0.10,
            dispatch_table={64: (0.04, 0.04)},
        )


def test_cots_dispatch_table_factory_requires_all_dispatch_buckets():
    cots = CotsOffloadConfig(
        f_cpu_store=0.10,
        dispatch_table={64: (0.04, 0.06)},
    )
    offloader = create_offloader(OffloadConfig(offload_backend="cots", cots=cots))

    assert offloader._dispatch_table_factory is not None
    with pytest.raises(ValueError, match="missing dispatch buckets"):
        offloader._dispatch_table_factory([64, 128])
    assert offloader._dispatch_table_factory([64]) == {64: (0.04, 0.06)}


def test_cots_eager_dispatch_uses_explicit_dispatch_bucket_grid(monkeypatch):
    """Regression for eager collapse: no CUDA graph buckets should not force
    every COTS route to the max-token bucket."""
    from types import SimpleNamespace

    cots = CotsOffloadConfig(
        f_cpu_store=0.30,
        cpu_runner="python",
        dispatch_table={
            64: (0.30, 0.00),
            2048: (0.00, 0.30),
        },
    )
    offloader = create_offloader(OffloadConfig(offload_backend="cots", cots=cots))
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
            max_num_seqs=128,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[],
            max_cudagraph_capture_size=0,
        ),
        speculative_config=None,
        performance_mode=None,
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: vllm_config,
    )

    offloader._resolve_bucket_sets()
    offloader._build_dispatch_table()
    offloader.prepare_before_forward(64)

    assert offloader._graph_capture_buckets == ()
    assert offloader._dispatch_buckets == (64, 2048)
    assert offloader._current_bucket == 64
    assert offloader.lookup_dispatch(64) == (0.30, 0.00)
    assert offloader.lookup_dispatch(65) == (0.00, 0.30)
    if offloader._runner is not None:
        offloader._runner.close()


def test_cots_eager_default_dispatch_grid_survives_empty_graph_buckets(monkeypatch):
    from types import SimpleNamespace

    cots = CotsOffloadConfig(f_cpu_store=0.30, cpu_runner="python")
    offloader = create_offloader(OffloadConfig(offload_backend="cots", cots=cots))
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
            max_num_seqs=128,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[],
            max_cudagraph_capture_size=0,
        ),
        speculative_config=None,
        performance_mode=None,
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: vllm_config,
    )

    offloader._resolve_bucket_sets()

    assert offloader._graph_capture_buckets == ()
    assert 64 in offloader._dispatch_buckets
    assert 2048 in offloader._dispatch_buckets
    assert offloader._dispatch_bucket_for(64) == 64
    if offloader._runner is not None:
        offloader._runner.close()


def test_cots_graph_capture_buckets_must_have_dispatch_rows(monkeypatch):
    """Graph buckets are replay shapes, but every replay shape must map to
    a dispatch row so capture and replay cannot silently disagree."""
    from types import SimpleNamespace

    cots = CotsOffloadConfig(
        f_cpu_store=0.30,
        cpu_runner="python",
        dispatch_table={
            64: (0.30, 0.00),
            2048: (0.00, 0.30),
        },
    )
    offloader = create_offloader(OffloadConfig(offload_backend="cots", cots=cots))
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048,
            max_num_seqs=128,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[64, 128],
            max_cudagraph_capture_size=128,
        ),
        speculative_config=None,
        performance_mode=None,
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: vllm_config,
    )

    offloader._resolve_bucket_sets()
    assert offloader._graph_capture_buckets == (64, 128)
    assert offloader._dispatch_buckets == (64, 2048)
    with pytest.raises(ValueError, match="missing CUDA graph capture buckets"):
        offloader._build_dispatch_table()
    if offloader._runner is not None:
        offloader._runner.close()


def test_cots_decorates_batch_descriptor_with_dispatch_bucket_and_signature():
    from vllm.forward_context import BatchDescriptor
    from vllm.model_executor.offloader import cots

    cfg = CotsOffloadConfig(f_cpu_store=0.30, cpu_runner="python")
    offloader = cots.CotsOffloader(config=cfg)
    offloader._dispatch_buckets = (64, 128, 2048)

    class Handle:
        n_cpu = 10
        role = cots.QKV_ROLE
        n_prefetch_by_bucket = {64: 0, 128: 0, 2048: 2}
        n_cpu_compute_by_bucket = {64: 10, 128: 10, 2048: 8}

    offloader._handles = [Handle()]  # type: ignore[list-item]
    offloader._build_route_signatures()

    b64 = offloader.decorate_batch_descriptor(BatchDescriptor(num_tokens=64))
    b128 = offloader.decorate_batch_descriptor(BatchDescriptor(num_tokens=128))
    b2048 = offloader.decorate_batch_descriptor(BatchDescriptor(num_tokens=2048))

    assert b64.cots_dispatch_bucket == 64
    assert b128.cots_dispatch_bucket == 128
    assert b2048.cots_dispatch_bucket == 2048
    assert b64.cots_route_signature == b128.cots_route_signature
    assert b2048.cots_route_signature != b64.cots_route_signature
    if offloader._runner is not None:
        offloader._runner.close()


def test_piecewise_execution_descriptor_preserves_cots_route_metadata():
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import (
        BatchExecutionDescriptor,
        _batch_descriptor_from_execution,
    )

    desc = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE,
        num_tokens=64,
        num_reqs=None,
        cots_dispatch_bucket=64,
        cots_route_signature=3,
    )

    batch_desc = _batch_descriptor_from_execution(desc)

    assert batch_desc.num_tokens == 64
    assert batch_desc.num_reqs is None
    assert batch_desc.cots_dispatch_bucket == 64
    assert batch_desc.cots_route_signature == 3


def test_compile_wrapper_keeps_bytecode_per_cots_route_signature():
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    class Probe(TorchCompileWithNoGuardsWrapper):
        def forward(self):
            return "base"

    def route_one(self):
        return "route-one"

    def route_two(self):
        return "route-two"

    probe = object.__new__(Probe)
    probe._compiled_bytecode = None
    probe._compiled_bytecode_by_cots_signature = {}

    probe._pending_cots_bytecode_signature = 1
    probe._store_compiled_bytecode(route_one.__code__)
    probe._pending_cots_bytecode_signature = 2
    probe._store_compiled_bytecode(route_two.__code__)
    probe._pending_cots_bytecode_signature = None

    assert probe._compiled_bytecode is None
    assert set(probe._compiled_bytecode_by_cots_signature) == {1, 2}

    with probe._dispatch_to_compiled_code(
        probe._compiled_bytecode_by_cots_signature[1]
    ):
        assert probe.forward() == "route-one"
    with probe._dispatch_to_compiled_code(
        probe._compiled_bytecode_by_cots_signature[2]
    ):
        assert probe.forward() == "route-two"
    assert probe.forward() == "base"


def test_piecewise_backend_dispatches_static_subgraph_without_shape_index():
    from vllm.compilation.piecewise_backend import PiecewiseBackend, RangeEntry
    from vllm.config.utils import Range

    backend = object.__new__(PiecewiseBackend)
    compile_range = Range(start=1, end=8192)
    entry = RangeEntry(compile_range=compile_range, compiled=True)
    entry.runnable = lambda *args: ("static", args)
    backend.compile_ranges = [compile_range]
    backend.range_entries = {compile_range: entry}
    backend.sym_shape_indices = []

    assert backend("x", "y") == ("static", ("x", "y"))
