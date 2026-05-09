# SPDX-License-Identifier: Apache-2.0
"""§1c.19 — `NativeCotsRunner` is the compile-visible facade and must be
pickleable so PyTorch's AOT compile guard serialization
(`torch._dynamo.guards.serialize_guards` →
`pickle_guards_state`) doesn't fail when it walks the runner's
attributes.

The pybind11 `CotsCpuInfer` handle is NOT pickleable (and shouldn't
be — it owns CUDA callback state, a worker thread, slabs, host-fn
userData pointers). The §1c.19 split moves the handle into the
`cots_ops._COTS_INFER` registry; the runner stores only `runner_id`,
`_task_id_for`, `_dry_run`, `_installed`. This test guards both:

1. `pickle.dumps(native_runner)` succeeds.
2. The pickle traversal does NOT touch any `vllm._cots_C.CotsCpuInfer`
   instance.
3. The runner's `__dict__` contains no `CotsCpuInfer` reference under
   any name (defensive — Dynamo's guard walker uses __dict__).
"""

from __future__ import annotations

import pickle

import pytest


def test_native_runner_is_picklable() -> None:
    """The compile-visible facade must serialize without traversing the
    pybind handle."""
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsRunner(dry_run=False)
    try:
        # Populate the task_id map a little so the dict actually has
        # data to pickle (representative of post-install state).
        r._task_id_for[(0, 1, "qkv")] = 0
        r._task_id_for[(0, 4, "qkv")] = 1
        blob = pickle.dumps(r)
        # Deserializing back doesn't matter for this test — we're
        # asserting the SERIALIZE path doesn't throw, which is what
        # PyTorch's AOT guard cache exercises.
        assert isinstance(blob, bytes) and len(blob) > 0
    finally:
        r.close()


def test_native_runner_dict_has_no_pybind_handle() -> None:
    """Defensive: walk the runner's `__dict__` and assert no value is
    a `vllm._cots_C.CotsCpuInfer` instance. Catches accidental
    re-introduction of `self._infer` (or a renamed equivalent) during
    future refactors."""
    pytest.importorskip("vllm._cots_C")
    from vllm import _cots_C
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsRunner(dry_run=False)
    try:
        for name, value in r.__dict__.items():
            assert not isinstance(value, _cots_C.CotsCpuInfer), (
                f"NativeCotsRunner.__dict__['{name}'] is a CotsCpuInfer; "
                f"§1c.19 split requires the pybind handle to live in the "
                f"cots_ops._COTS_INFER registry, not on the runner."
            )
    finally:
        r.close()


def test_native_runner_pickle_traversal_skips_cotscpuinfer() -> None:
    """Stronger gate: actually pickle and confirm the byte stream
    contains no reference to the `CotsCpuInfer` class. Prevents a
    silent regression where someone adds `self._infer` and it
    happens to be pickleable in the future (e.g., a stub gets a
    `__reduce__` added that returns None) — even then we wouldn't
    want it on the runner.
    """
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsRunner(dry_run=False)
    try:
        blob = pickle.dumps(r)
        # `pickle` records class refs as `module.qualname`; assert the
        # CotsCpuInfer class name doesn't appear anywhere in the stream.
        assert b"CotsCpuInfer" not in blob, (
            "pickle.dumps(NativeCotsRunner) reached a CotsCpuInfer "
            "instance — §1c.19 split has regressed."
        )
    finally:
        r.close()


def test_pickled_runner_can_be_used_via_registry_id() -> None:
    """The registry stores the actual handle by `runner_id`. Ensure
    that after a pickle round-trip, the unpickled facade still names
    the same registry slot — i.e., the contract is "the runner
    facade is just a tagged pointer into cots_ops._COTS_INFER."""
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsRunner(dry_run=False)
    try:
        rid = r._runner_id
        blob = pickle.dumps(r)
        r2 = pickle.loads(blob)
        # Identity is not preserved (pickle gives a new object), but the
        # registry id is — both objects refer to the same registry entry.
        assert r2._runner_id == rid
        assert cots_ops._COTS_INFER.get(rid) is not None
        # The unpickled facade must NOT have an `_infer` attribute.
        assert not hasattr(r2, "_infer"), (
            "Unpickled NativeCotsRunner has an _infer attribute — §1c.19 "
            "split requires the pybind handle to live ONLY in the registry."
        )
    finally:
        r.close()
