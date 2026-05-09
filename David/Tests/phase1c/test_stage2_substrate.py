# SPDX-License-Identifier: Apache-2.0
"""Stage 2 substrate tests.

Stage 2 lands the NativeCotsRunner / PythonCotsRunner split, the
`cots_ops.py` custom-op registration + runner registry, the
`cpu_runner` config flag, and the installer refactor that constructs
ONE runner per offloader.

Stage 2 does NOT yet wire NativeCotsRunner into operators (operators
still use the PythonCotsRunner legacy `submit_with_d2h(fn, *args)`
shape; Stage 3 flips them). So:
  * `cpu_runner='python'` is the default through Stage 2/3/4 — Phase
    1a/1b workflows are unchanged.
  * `cpu_runner='native'` is reserved; constructing a CotsOffloader
    with `f_cpu_store > 0` and `cpu_runner='native'` raises a clear
    NotImplementedError until Stage 3 lands.
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


def test_runner_registry_round_trip():
    """A registered runner can be looked up by id; unregister drops it."""
    from vllm.model_executor.offloader import cots_ops

    class _Stub:
        pass

    runner = _Stub()
    rid = cots_ops._register_runner(runner)
    assert isinstance(rid, int)
    assert cots_ops._COTS_RUNNERS.get(rid) is runner

    cots_ops._unregister_runner(rid)
    assert cots_ops._COTS_RUNNERS.get(rid) is None
    # Idempotent:
    cots_ops._unregister_runner(rid)


def test_runner_registry_is_weak_value_dict():
    """The registry must hold weak refs so a runner that's been GC'd
    auto-clears — otherwise we'd leak NativeCotsRunner instances.
    """
    from vllm.model_executor.offloader import cots_ops

    class _Stub:
        pass

    rid = cots_ops._register_runner(_Stub())  # no Python-side reference
    # `_Stub()` had no remaining strong references after _register_runner;
    # WeakValueDictionary collects it on the first GC sweep that touches
    # the entry. Force a sweep by accessing.
    import gc

    gc.collect()
    assert cots_ops._COTS_RUNNERS.get(rid) is None


def test_lookup_runner_raises_clear_error_when_missing():
    """Calling the op impl with a stale runner_id surfaces a clear
    RuntimeError with the registry contents — not a silent NoneType."""
    from vllm.model_executor.offloader import cots_ops

    with pytest.raises(RuntimeError, match="not in registry"):
        cots_ops._lookup_runner(99999, "test_op")


# --- cots.py — runner classes + factory + backwards-compat alias ----------


def test_python_cots_runner_construct_and_close():
    from vllm.model_executor.offloader import cots

    r = cots.PythonCotsRunner(dry_run=False)
    assert r.kind == "python"
    # close() on a freshly-constructed runner is a no-op.
    r.close()


def test_native_cots_runner_construct_install_close():
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsRunner(dry_run=True)
    assert r.kind == "native"
    assert isinstance(r._runner_id, int)
    # install with zero slabs is the degenerate-but-valid path.
    r.install(n_slabs=0, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)
    # Re-install raises.
    with pytest.raises(RuntimeError, match="install\\(\\) called twice"):
        r.install(n_slabs=0, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)
    r.close()


def test_native_runner_unregisters_on_close():
    """close() drops the registry entry so subsequent op calls with the
    same runner_id raise cleanly."""
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsRunner(dry_run=False)
    rid = r._runner_id
    assert cots_ops._COTS_RUNNERS.get(rid) is r
    r.close()
    assert cots_ops._COTS_RUNNERS.get(rid) is None


def test_cputaskrunner_alias_for_backwards_compat():
    """Phase 1a/1b's `CpuTaskRunner` symbol must still resolve to the
    renamed PythonCotsRunner, so any external import doesn't break."""
    from vllm.model_executor.offloader import cots

    assert cots.CpuTaskRunner is cots.PythonCotsRunner


def test_make_runner_factory_picks_python():
    from vllm.model_executor.offloader import cots

    class _Cfg:
        cpu_runner = "python"
        dry_run = False

    r = cots._make_runner(_Cfg())
    assert isinstance(r, cots.PythonCotsRunner)


def test_make_runner_factory_picks_native():
    from vllm.model_executor.offloader import cots

    class _Cfg:
        cpu_runner = "native"
        dry_run = False

    r = cots._make_runner(_Cfg())
    assert isinstance(r, cots.NativeCotsRunner)
    r.close()


def test_make_runner_factory_rejects_unknown():
    from vllm.model_executor.offloader import cots

    class _Cfg:
        cpu_runner = "neither"
        dry_run = False

    with pytest.raises(ValueError, match="Unknown cpu_runner"):
        cots._make_runner(_Cfg())


# --- vllm/config/offload.py — cpu_runner field is plumbed -----------------


def test_cpu_runner_field_default_is_python():
    """Stage 2/3/4 default: 'python', so existing Phase 1a/1b workflows
    are unchanged. Stage 5 will flip the default to 'native' once graph
    capture is verified."""
    from vllm.config.offload import CotsOffloadConfig

    cfg = CotsOffloadConfig()
    assert cfg.cpu_runner == "python"


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


def test_offloader_python_runner_default_path():
    """Default `cpu_runner='python'` constructs ONE PythonCotsRunner."""
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader.cots import CotsOffloader, PythonCotsRunner

    cfg = CotsOffloadConfig(f_cpu_store=0.10)
    off = CotsOffloader(config=cfg)
    assert isinstance(off._runner, PythonCotsRunner)
    # The legacy code constructed a fresh CpuTaskRunner per op; Stage 2's
    # invariant is one runner per offloader. We can't probe operator
    # construction here without a real model, but `off._runner` is the
    # singleton operators must use (Stage 3 flips them via the uniform
    # facade; Stage 2's _install_*_ops asserts on this same field).


def test_offloader_native_runner_blocked_until_stage_3():
    """Selecting `cpu_runner='native'` with f_cpu_store > 0 raises a
    clear NotImplementedError pointing at Stage 3 — not a silent
    fall-through to broken operator code."""
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader.cots import CotsOffloader

    cfg = CotsOffloadConfig(f_cpu_store=0.10, cpu_runner="native")
    with pytest.raises(NotImplementedError, match="Stage 3"):
        CotsOffloader(config=cfg)


def test_offloader_native_rejection_does_not_construct_runner():
    """Review finding #1: the rejection must fire BEFORE _make_runner
    runs, otherwise a NativeCotsRunner is briefly registered + a C++
    worker thread spawned + (on non-CUDA builds) `_cots_C` import
    masks the intended Stage-3 message. Confirms the registry is
    untouched on the exception path.
    """
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader import cots_ops
    from vllm.model_executor.offloader.cots import CotsOffloader

    before = dict(cots_ops._COTS_RUNNERS)
    cfg = CotsOffloadConfig(f_cpu_store=0.10, cpu_runner="native")
    with pytest.raises(NotImplementedError, match="Stage 3"):
        CotsOffloader(config=cfg)
    after = dict(cots_ops._COTS_RUNNERS)
    # Registry contents are unchanged: no NativeCotsRunner was constructed.
    assert before == after, (
        f"Stage-2 native rejection leaked a runner into the registry. "
        f"Before: {before!r}, after: {after!r}"
    )


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
    assert isinstance(r, cots.PythonCotsRunner)


def test_offloader_native_runner_allowed_at_zero_offload():
    """`f_cpu_store=0` short-circuits before runner construction, so
    even `cpu_runner='native'` is fine in this degenerate case (no
    runner is built either way)."""
    from vllm.config.offload import CotsOffloadConfig
    from vllm.model_executor.offloader.cots import CotsOffloader

    cfg = CotsOffloadConfig(f_cpu_store=0.0, cpu_runner="native")
    off = CotsOffloader(config=cfg)
    assert off._runner is None
