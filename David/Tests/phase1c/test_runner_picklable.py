# SPDX-License-Identifier: Apache-2.0
"""§1c.19 — `NativeCotsWeightRunner` is the compile-visible facade and must be
pickleable so PyTorch's AOT compile guard serialization
(`torch._dynamo.guards.serialize_guards` →
`pickle_guards_state`) doesn't fail when it walks the runner's
attributes.

The pybind11 `CotsWeightTaskRunner` handle is NOT pickleable (and shouldn't
be — it owns CUDA callback state, a worker thread, slabs, host-fn
userData pointers). The §1c.19 split moves the handle into the
`cots_ops._COTS_WEIGHT_RUNNERS` registry; the runner stores only `runner_id`,
`_task_id_for`, `_dry_run`, `_installed`. This test guards both:

1. `pickle.dumps(native_runner)` succeeds.
2. The pickle traversal does NOT touch any `vllm._cots_C.CotsWeightTaskRunner`
   instance.
3. The runner's `__dict__` contains no `CotsWeightTaskRunner` reference under
   any name (defensive — Dynamo's guard walker uses __dict__).
"""

from __future__ import annotations

import pickle

import pytest


def test_native_runner_is_picklable() -> None:
    """The compile-visible facade must serialize without traversing the
    pybind handle."""
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsWeightRunner(dry_run=False)
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
    a `vllm._cots_C.CotsWeightTaskRunner` instance. Catches accidental
    re-introduction of `self._infer` (or a renamed equivalent) during
    future refactors."""
    pytest.importorskip("vllm._cots_C")
    from vllm import _cots_C
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsWeightRunner(dry_run=False)
    try:
        for name, value in r.__dict__.items():
            assert not isinstance(value, _cots_C.CotsWeightTaskRunner), (
                f"NativeCotsWeightRunner.__dict__['{name}'] is a CotsWeightTaskRunner; "
                f"§1c.19 split requires the pybind handle to live in the "
                f"cots_ops._COTS_WEIGHT_RUNNERS registry, not on the runner."
            )
    finally:
        r.close()


def test_native_runner_pickle_traversal_skips_cots_weight_task_runner() -> None:
    """Stronger gate: actually pickle and confirm the byte stream
    contains no reference to the `CotsWeightTaskRunner` class. Prevents a
    silent regression where someone adds `self._infer` and it
    happens to be pickleable in the future (e.g., a stub gets a
    `__reduce__` added that returns None) — even then we wouldn't
    want it on the runner.
    """
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsWeightRunner(dry_run=False)
    try:
        blob = pickle.dumps(r)
        # `pickle` records class refs as `module.qualname`; assert the
        # CotsWeightTaskRunner class name doesn't appear anywhere in the stream.
        assert b"CotsWeightTaskRunner" not in blob, (
            "pickle.dumps(NativeCotsWeightRunner) reached a CotsWeightTaskRunner "
            "instance — §1c.19 split has regressed."
        )
    finally:
        r.close()


def test_pickled_runner_can_be_used_via_registry_id() -> None:
    """The registry stores the actual handle by `runner_id`. Ensure
    that after a pickle round-trip, the unpickled facade still names
    the same registry slot — i.e., the contract is "the runner
    facade is just a tagged pointer into cots_ops._COTS_WEIGHT_RUNNERS."""
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsWeightRunner(dry_run=False)
    try:
        rid = r._runner_id
        blob = pickle.dumps(r)
        r2 = pickle.loads(blob)
        # Identity is not preserved (pickle gives a new object), but the
        # registry id is — both objects refer to the same registry entry.
        assert r2._runner_id == rid
        assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is not None
        # The unpickled facade must NOT have an `_infer` attribute.
        assert not hasattr(r2, "_infer"), (
            "Unpickled NativeCotsWeightRunner has an _infer attribute — §1c.19 "
            "split requires the pybind handle to live ONLY in the registry."
        )
    finally:
        r.close()


# --- Ownership semantics — review-fix for the high-severity finding -------
#
# After §1c.19's first cut, an unpickled NativeCotsWeightRunner shared the
# original's `_runner_id` AND the same `__del__`. PyTorch's AOT guard
# cache pickles + unpickles the runner as part of guard serialization;
# GC of any unpickled copy could `unregister_weight_runner(rid)` on the live
# entry, causing a downstream `runner_id not in registry` failure on
# the next custom op call. The fix adds an ownership flag,
# `_owns_runner_registry_entry`, that `__getstate__` flips to False on
# the pickled state. `close()` / `__del__` no-op for non-owners.


def test_unpickled_copy_gc_does_not_unregister_original() -> None:
    """The exact bug the reviewer flagged: unpickle a runner, drop the
    copy, GC, and confirm the ORIGINAL's registry entry is still
    intact. Before the ownership fix this would fail."""
    import gc

    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsWeightRunner(dry_run=False)
    try:
        rid = r._runner_id
        assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is not None
        r2 = pickle.loads(pickle.dumps(r))
        assert r2._runner_id == rid
        # Drop the unpickled copy and force GC. Its `__del__` must NOT
        # unregister rid because it does not own the registry entry.
        del r2
        gc.collect()
        assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is not None, (
            "Unpickled copy's GC unregistered the live runner handle; "
            "ownership flag is missing or incorrect."
        )
    finally:
        r.close()
        assert cots_ops._COTS_WEIGHT_RUNNERS.get(r._runner_id) is None


def test_unpickled_copy_close_is_noop() -> None:
    """`r2.close()` on a non-owning copy must NOT touch the registry."""
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsWeightRunner(dry_run=False)
    try:
        rid = r._runner_id
        r2 = pickle.loads(pickle.dumps(r))
        r2.close()  # non-owning: should no-op
        assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is not None, (
            "Unpickled copy's close() unregistered the live runner handle."
        )
    finally:
        r.close()


def test_owning_runner_close_still_unregisters() -> None:
    """The owning runner's `close()` is unchanged — it drains and
    unregisters. The ownership flag must not interfere with the
    happy path."""
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsWeightRunner(dry_run=False)
    rid = r._runner_id
    assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is not None
    r.close()
    assert cots_ops._COTS_WEIGHT_RUNNERS.get(rid) is None
    # Idempotent: second close on the now-non-owning runner no-ops.
    r.close()


def test_pickled_state_marks_non_owning() -> None:
    """Inspect the pickle state directly to confirm `__getstate__` flips
    the ownership flag. This is a defense-in-depth gate: future
    refactors that add new attrs need to keep this invariant.
    Independent of `__del__` behavior."""
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsWeightRunner(dry_run=False)
    try:
        state = r.__getstate__()
        assert state["_owns_runner_registry_entry"] is False, (
            "__getstate__ must mark the pickled state as non-owning."
        )
        # Original's flag is unchanged by `__getstate__`.
        assert r._owns_runner_registry_entry is True
    finally:
        r.close()
