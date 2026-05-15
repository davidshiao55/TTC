import pytest

from vllm.config import CotsOffloadConfig, OffloadConfig
from vllm.model_executor.offloader.base import create_offloader


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


def test_cots_dispatch_table_factory_requires_all_capture_buckets():
    cots = CotsOffloadConfig(
        f_cpu_store=0.10,
        dispatch_table={64: (0.04, 0.06)},
    )
    offloader = create_offloader(OffloadConfig(offload_backend="cots", cots=cots))

    assert offloader._dispatch_table_factory is not None
    with pytest.raises(ValueError, match="missing captured buckets"):
        offloader._dispatch_table_factory([64, 128])
    assert offloader._dispatch_table_factory([64]) == {64: (0.04, 0.06)}
