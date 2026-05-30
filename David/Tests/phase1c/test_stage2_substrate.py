# SPDX-License-Identifier: Apache-2.0
"""Stage 2 substrate tests.

Originally landed alongside Stage 2's substrate-only deliverable:
NativeCotsWeightRunner / PythonCotsWeightRunner split, `cots_ops.py` custom-op
registration + runner registry, `cpu_runner` config flag, and the
installer refactor that constructs ONE runner per offloader.

Stage 3 then wired the native runner end-to-end (operator facade
flipped to `submit_with_d2h(x, x_pinned, y_pinned, op_descriptor)` +
`wait_and_uva(...)`, slab population in `_install_runner`, dummy CUDA
anchors). The Stage-2-era native rejection in `CotsOffloader.__init__`
was DROPPED in Stage 3, so:

  * `cpu_runner='python'` is the default through Stage 2/3/4 — Phase
    1a/1b workflows are unchanged.
  * `cpu_runner='native'` constructs a real, end-to-end runnable
    NativeCotsWeightRunner post-Stage-3 (was a NotImplementedError under
    Stage 2). Stage 5 will flip the default to `"native"` once graph
    capture is verified.

These tests still gate the Stage-2 substrate invariants (registry,
factory, alias, installer refactor) — they survive Stage 3 because the
substrate didn't change shape, only what gets wired on top of it.
"""

from __future__ import annotations

import pytest
import torch


# --- cots_ops.py — custom op registration + runner registry ---------------


def test_cots_ops_imports_and_registers_ops():
    """Importing the module registers `vllm.cots_submit_gemm` and
    `vllm.cots_sync_then_uva` under torch.ops.vllm at module load."""
    from vllm.model_executor.offloader import cots_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "cots_submit_gemm")
    assert hasattr(torch.ops.vllm, "cots_sync_then_uva")


def test_infer_registry_round_trip():
    """A registered runner handle can be looked up by id; unregister drops it.

    §1c.19 split: the registry holds `CotsWeightTaskRunner` instances (the
    pybind handles), NOT `NativeCotsWeightRunner` facades. The runner only
    knows its `runner_id`."""
    from vllm.model_executor.offloader import cots_ops

    class _StubRunner:
        pass

    runner = _StubRunner()
    rid = cots_ops.register_weight_runner(runner)
    assert isinstance(rid, int)
    assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is runner

    cots_ops.unregister_weight_runner(rid)
    assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is None
    # Idempotent:
    cots_ops.unregister_weight_runner(rid)


def test_runner_registry_is_strong_ref():
    """§1c.19: the registry holds STRONG refs to `CotsWeightTaskRunner` (was a
    WeakValueDictionary in the original Stage 2 design). The runner
    facade no longer holds the pybind handle, so the registry entry is
    the sole owner — explicit `unregister_weight_runner` (or the runner's
    `close()` / `__del__`) is what drops it."""
    from vllm.model_executor.offloader import cots_ops

    class _StubRunner:
        pass

    rid = cots_ops.register_weight_runner(_StubRunner())  # no local strong ref
    import gc

    gc.collect()
    # Strong ref → still present after GC.
    assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is not None
    cots_ops.unregister_weight_runner(rid)


def testlookup_weight_runner_raises_clear_error_when_missing():
    """Calling the op impl with a stale runner_id surfaces a clear
    RuntimeError with the registry contents — not a silent NoneType."""
    from vllm.model_executor.offloader import cots_ops

    with pytest.raises(RuntimeError, match="not in registry"):
        cots_ops.lookup_weight_runner(99999, "test_op")


# --- cots.py — runner classes + factory + backwards-compat alias ----------


def test_python_cots_runner_construct_and_close():
    from vllm.model_executor.offloader import cots

    r = cots.PythonCotsWeightRunner(dry_run=False)
    assert r.kind == "python"
    # close() on a freshly-constructed runner is a no-op.
    r.close()


def test_native_cots_runner_construct_install_close():
    """Stage 3 install signature: takes a list of NativeWeightSlabSpec records
    (ordering = task_id) plus scratch sizes. The empty-list /
    zero-scratch case is the degenerate-but-valid path covering an
    offloader with no fused MLP blocks (or just a smoke test)."""
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsWeightRunner(dry_run=True)
    assert r.kind == "native"
    assert isinstance(r._runner_id, int)
    r.install(
        slab_specs=[],
        max_num_tokens=0,
    )
    # Re-install raises.
    with pytest.raises(RuntimeError, match="install\\(\\) called twice"):
        r.install(
            slab_specs=[],
            max_num_tokens=0,
        )
    r.close()


def test_native_runner_unregisters_on_close():
    """close() drops the registry entry so subsequent op calls with the
    same runner_id raise cleanly. §1c.19: the registry now holds the
    `CotsWeightTaskRunner` instance, not the runner — but the lifetime story
    from the runner facade's perspective is unchanged."""
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsWeightRunner(dry_run=False)
    rid = r._runner_id
    assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is not None
    r.close()
    assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is None


def test_deprecated_cputaskrunner_alias_removed():
    """The old Phase 1a/1b alias should not remain on the public facade."""
    from vllm.model_executor.offloader import cots

    assert not hasattr(cots, "CpuTaskRunner")


def test_make_runner_factory_picks_python():
    from vllm.model_executor.offloader import cots

    class _Cfg:
        cpu_runner = "python"
        dry_run = False

    r = cots._make_runner(_Cfg())
    assert isinstance(r, cots.PythonCotsWeightRunner)


def test_make_runner_factory_picks_native():
    from vllm.model_executor.offloader import cots

    class _Cfg:
        cpu_runner = "native"
        dry_run = False

    r = cots._make_runner(_Cfg())
    assert isinstance(r, cots.NativeCotsWeightRunner)
    r.close()


def test_make_runner_factory_rejects_unknown():
    from vllm.model_executor.offloader import cots

    class _Cfg:
        cpu_runner = "neither"
        dry_run = False

    with pytest.raises(ValueError, match="Unknown cpu_runner"):
        cots._make_runner(_Cfg())


# --- vllm/config/offload.py — cpu_runner field is plumbed -----------------


def test_cpu_runner_field_default_is_native():
    """Stage 5 production default: 'native'. The default was 'python'
    through Stage 4 to keep Phase 1a/1b workflows unchanged while the
    native runner was being wired; Stage 5 flipped it once CUDA graph
    capture was verified end-to-end."""
    from vllm.config.offload import CotsOffloadConfig

    cfg = CotsOffloadConfig()
    assert cfg.cpu_runner == "native"


def test_cpu_runner_field_accepts_native():
    from vllm.config.offload import CotsOffloadConfig

    cfg = CotsOffloadConfig(cpu_runner="native")
    assert cfg.cpu_runner == "native"


def test_cpu_runner_field_rejects_invalid():
    from pydantic import ValidationError

    from vllm.config.offload import CotsOffloadConfig

    with pytest.raises(ValidationError):
        CotsOffloadConfig(cpu_runner="bogus")  # type: ignore[arg-type]


# --- CotsOffloader installer refactor — one runner per offloader ----------


def test_offloader_no_offload_does_not_construct_runner():
    """`f_cpu_store=0` is a clean control: no runner constructed, no
    side effects. Important so the no-offload path doesn't spin up a
    worker thread for nothing."""
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader.cots import CotsOffloader

    cfg = CotsOffloadConfig(f_cpu_store=0.0)
    off = CotsOffloader(config=cfg)
    assert off._runner is None


def test_offloader_default_runner_is_native():
    """Stage 5 default: `cpu_runner='native'` → NativeCotsWeightRunner. The
    installer-refactor invariant from Stage 2 still holds: ONE runner
    per offloader, shared across all operator installs."""
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader.cots import CotsOffloader, NativeCotsWeightRunner

    cfg = CotsOffloadConfig(f_cpu_store=0.10)
    off = CotsOffloader(config=cfg)
    try:
        assert isinstance(off._runner, NativeCotsWeightRunner)
    finally:
        if off._runner is not None:
            off._runner.close()


def test_offloader_python_runner_explicit_path():
    """Explicit `cpu_runner='python'` continues to construct a
    PythonCotsWeightRunner — the kill-switch path remains valid."""
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader.cots import CotsOffloader, PythonCotsWeightRunner

    cfg = CotsOffloadConfig(f_cpu_store=0.10, cpu_runner="python")
    off = CotsOffloader(config=cfg)
    assert isinstance(off._runner, PythonCotsWeightRunner)


def test_offloader_native_runner_constructs_post_stage_3():
    """Stage 3 dropped the Stage-2 NotImplementedError barrier:
    `cpu_runner='native'` + `f_cpu_store > 0` now constructs a real
    NativeCotsWeightRunner. The runner is shared across operator install (no
    fresh runner per op). Slab population happens later in `post_init`
    so this just exercises construction, not a forward pass."""
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader.cots import CotsOffloader, NativeCotsWeightRunner

    cfg = CotsOffloadConfig(f_cpu_store=0.10, cpu_runner="native")
    off = CotsOffloader(config=cfg)
    assert isinstance(off._runner, NativeCotsWeightRunner)
    off._runner.close()


def test_make_runner_default_fallback_is_python():
    """Review finding #2: when a stub config object lacks the
    `cpu_runner` field, the factory must default to 'python' (matches
    the Stage 2 config default at vllm/config/offload.py). Defaulting
    to 'native' here would let old configs silently route through the
    unwired native path."""
    from vllm.model_executor.offloader import cots

    class _LegacyCfg:
        # No cpu_runner attribute — simulates an older config shim or
        # a test stub written against Phase 1a/1b CotsOffloadConfig.
        dry_run = False

    r = cots._make_runner(_LegacyCfg())
    assert isinstance(r, cots.PythonCotsWeightRunner)


def test_offloader_native_runner_allowed_at_zero_offload():
    """`f_cpu_store=0` short-circuits before runner construction, so
    even `cpu_runner='native'` is fine in this degenerate case (no
    runner is built either way)."""
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader.cots import CotsOffloader

    cfg = CotsOffloadConfig(f_cpu_store=0.0, cpu_runner="native")
    off = CotsOffloader(config=cfg)
    assert off._runner is None
