# SPDX-License-Identifier: Apache-2.0
"""§1c.21 review-fix: native runner is incompatible with vLLM
microbatching/ubatching, until per-ubatch live counts are plumbed
(§1c.23 follow-up).

The live-token cap
(`GPUModelRunner._publish_offloader_dispatch →
BaseOffloader.set_live_num_tokens`) currently sets ONE global
`live_num_tokens` value per scheduler batch. Under ubatching, COTS
operators run on per-ubatch slices but would see the cap as the FULL
batch count, which can over-compute against the per-ubatch x_pinned
slice.

CotsOffloader.post_init hard-fails at construction when
`parallel_config.use_ubatching` is True AND `cpu_runner='native'`.
This test exercises the guard via a stub config object.
"""

from __future__ import annotations

import pytest


class _StubParallel:
    def __init__(self, use_ubatching: bool) -> None:
        self.use_ubatching = use_ubatching


class _StubModelConfig:
    def __init__(self) -> None:
        self.enforce_eager = False


class _StubVllmConfig:
    def __init__(self, use_ubatching: bool) -> None:
        self.parallel_config = _StubParallel(use_ubatching)
        self.model_config = _StubModelConfig()


def _patch_get_current_vllm_config(monkeypatch, use_ubatching: bool) -> None:
    """Inject a stub vllm_config so CotsOffloader.post_init's
    `from vllm.config import get_current_vllm_config` returns the stub."""
    import vllm.config

    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: _StubVllmConfig(use_ubatching),
    )


def test_native_runner_with_ubatching_hard_fails(monkeypatch) -> None:
    """`cpu_runner='native'` + `use_ubatching=True` → RuntimeError at
    post_init with a clear message naming §1c.23."""
    pytest.importorskip("vllm._cots_C")
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader import cots

    _patch_get_current_vllm_config(monkeypatch, use_ubatching=True)
    cfg = CotsOffloadConfig(f_cpu_store=0.05, cpu_runner="native")
    o = cots.CotsOffloader(cfg)
    # Need at least one handle for post_init to reach the runner path.
    # Cheapest: stub the handles + dispatch table to satisfy the
    # post_init prerequisites without a real model.
    o._handles = [object()]  # type: ignore[list-item]
    o._dispatch_table = {1: (0.05, 0.0)}
    o._capture_buckets = (1,)

    with pytest.raises(RuntimeError, match="ubatching|§1c.23"):
        o.post_init()


def test_native_runner_without_ubatching_passes_guard(monkeypatch) -> None:
    """The guard fires only when use_ubatching is True. Standard
    non-ubatching configs keep working — this test exercises that
    the post_init enforce_eager + ubatching branches both pass with
    use_ubatching=False (and reach the install path, where it'll
    fail on the stubs because we don't have real handles, but the
    guard itself must not be the failure)."""
    pytest.importorskip("vllm._cots_C")
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader import cots

    _patch_get_current_vllm_config(monkeypatch, use_ubatching=False)
    cfg = CotsOffloadConfig(f_cpu_store=0.05, cpu_runner="native")
    o = cots.CotsOffloader(cfg)
    o._handles = [object()]  # type: ignore[list-item]
    o._dispatch_table = {1: (0.05, 0.0)}
    o._capture_buckets = (1,)

    # post_init may fail at the install step (the stub handles aren't
    # real CotsLinearHandles) — what matters here is that the failure
    # is NOT the ubatching guard. Catch any exception and assert its
    # message doesn't mention ubatching.
    try:
        o.post_init()
    except RuntimeError as e:
        assert "ubatching" not in str(e), (
            f"ubatching guard fired unexpectedly when use_ubatching=False: "
            f"{e}"
        )
    except Exception:
        pass
