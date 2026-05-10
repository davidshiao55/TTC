# SPDX-License-Identifier: Apache-2.0
"""§1c.26 / §1c.27 review-fix: regression tests for the ablation-flag
hard-fail gate.

`CotsOffloader._install_ablations()` reads five env vars
(VLLM_COTS_ABLATE_HOSTFN / SUBMIT_HOSTFN / SYNC_HOSTFN / D2H / UVA)
and honors them ONLY when `dry_run=True` AND `VLLM_COTS_DIAG=1`.
Misuse — any of the env vars set without both gate conditions —
must hard-fail with `RuntimeError`. A warn-and-skip would silently
measure the NON-ablated path and produce false conclusions.

These tests stub the offloader-shaped object so we can exercise
`_install_ablations` directly, without spinning up a full vLLM
engine. The C++ side is tested implicitly via the actual
NativeCotsRunner construction.
"""

from __future__ import annotations

import os
import types
from contextlib import contextmanager

import pytest


@contextmanager
def _env(**overrides):
    """Temporary env mutations; restored on exit."""
    saved: dict[str, str | None] = {}
    try:
        for k, v in overrides.items():
            saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_native_runner_or_skip():
    """Construct a real NativeCotsRunner. Skip if the _cots_C
    extension isn't available (CPU-only build)."""
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader.cots import NativeCotsRunner

    return NativeCotsRunner(dry_run=True)


def _make_stub_offloader(runner, dry_run: bool):
    """Build a minimal stub that satisfies `_install_ablations`'s
    attribute reads. We don't need a full CotsOffloader for the
    gate-check path — just `self._runner` and `self.config.dry_run`."""
    from vllm.model_executor.offloader.cots import CotsOffloader

    obj = types.SimpleNamespace()
    obj._runner = runner
    obj.config = types.SimpleNamespace(dry_run=dry_run)
    # Bind the unbound method to the stub so `self._install_ablations()`
    # works exactly as it would on a real CotsOffloader.
    obj._install_ablations = CotsOffloader._install_ablations.__get__(obj)
    return obj


@pytest.fixture
def runner():
    r = _make_native_runner_or_skip()
    yield r
    # Best-effort cleanup of the cots_ops registry.
    from vllm.model_executor.offloader import cots_ops

    cots_ops._unregister_infer(r._runner_id)


def test_no_env_vars_set_succeeds_without_ablations(runner):
    """No env vars set → `_install_ablations()` returns silently
    (regardless of gate). This is the production default path."""
    stub = _make_stub_offloader(runner, dry_run=False)
    with _env(
        VLLM_COTS_ABLATE_HOSTFN=None,
        VLLM_COTS_ABLATE_SUBMIT_HOSTFN=None,
        VLLM_COTS_ABLATE_SYNC_HOSTFN=None,
        VLLM_COTS_ABLATE_D2H=None,
        VLLM_COTS_ABLATE_UVA=None,
    ):
        # Must not raise even though dry_run=False.
        stub._install_ablations()


@pytest.mark.parametrize(
    "env_var",
    [
        "VLLM_COTS_ABLATE_HOSTFN",
        "VLLM_COTS_ABLATE_SUBMIT_HOSTFN",
        "VLLM_COTS_ABLATE_SYNC_HOSTFN",
        "VLLM_COTS_ABLATE_D2H",
        "VLLM_COTS_ABLATE_UVA",
    ],
)
def test_env_set_without_dry_run_hard_fails(runner, env_var):
    """Each ablation env var, set without `dry_run=True`, must
    hard-fail at install. Covers the "diag-on but real-mode" misuse
    path."""
    stub = _make_stub_offloader(runner, dry_run=False)
    with _env(**{env_var: "1"}):
        # The hard-fail path imports cots_diag's ENABLED state. If
        # DIAG is also off, the failure message names BOTH gate
        # conditions; either way it raises.
        with pytest.raises(RuntimeError, match=r"§1c\.26.*§1c\.27.*gate not met"):
            stub._install_ablations()


def test_env_set_dry_run_true_diag_off_hard_fails(runner, monkeypatch):
    """dry_run=True but DIAG=0 must still hard-fail. Both gates are
    independent and BOTH must hold."""
    # Force the cots_diag ENABLED constant to False for this test.
    import vllm.utils.cots_diag

    monkeypatch.setattr(vllm.utils.cots_diag, "ENABLED", False)

    stub = _make_stub_offloader(runner, dry_run=True)
    with _env(VLLM_COTS_ABLATE_HOSTFN="1"):
        with pytest.raises(RuntimeError, match=r"VLLM_COTS_DIAG=0"):
            stub._install_ablations()


def test_uva_flag_reset_on_each_install(runner):
    """`_install_ablations()` must clear the process-global
    `_COTS_ABLATE_UVA` at the start of every native install,
    even when the current install sets no env vars. This guards
    against a prior install in the same process bleeding state
    into a non-ablating offloader."""
    from vllm.model_executor.offloader import cots_ops

    # Pretend a prior install set the UVA flag.
    cots_ops.set_uva_ablation(True)
    assert cots_ops._COTS_ABLATE_UVA is True

    # New install with NO env vars — UVA flag must be cleared.
    stub = _make_stub_offloader(runner, dry_run=False)
    with _env(
        VLLM_COTS_ABLATE_HOSTFN=None,
        VLLM_COTS_ABLATE_SUBMIT_HOSTFN=None,
        VLLM_COTS_ABLATE_SYNC_HOSTFN=None,
        VLLM_COTS_ABLATE_D2H=None,
        VLLM_COTS_ABLATE_UVA=None,
    ):
        stub._install_ablations()
    assert cots_ops._COTS_ABLATE_UVA is False, (
        "Sticky UVA flag from prior install was not reset. "
        "Subsequent non-ablating offloaders would silently inherit it."
    )
