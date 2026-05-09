# SPDX-License-Identifier: Apache-2.0
"""§1c.20 — `CotsCpuInfer.y_pinned_view(task_id, num_tokens)` C++/pybind
contract.

This is the bridge that makes the new sync-side schema work: instead
of receiving `y_pinned` as a custom-op argument (which Inductor's
functionalization would materialize via a fresh pageable CPU clone),
the sync op's impl reaches the worker's pinned output by reconstructing
an `at::from_blob` view over the slab's `y_pinned_ptr` directly in
C++. The slab pointer was populated at install time from a real
`torch.empty(..., pin_memory=True)` allocation, so the storage is
known-pinned by construction — no Python-side check needed on the
captured-graph hot path.
"""

from __future__ import annotations

import pytest
import torch


def _make_runner_with_one_qkv_slab(
    num_tokens: int, in_dim: int, n_cpu: int
) -> tuple[object, torch.Tensor, torch.Tensor]:
    """Construct a NativeCotsRunner and populate one QKV slab pointed at
    a real pinned buffer + a CPU weight. Returns (runner, x_pin, y_pin).
    """
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader import cots

    r = cots.NativeCotsRunner(dry_run=False)
    x_pin = torch.empty(num_tokens, in_dim, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(num_tokens, n_cpu, dtype=torch.bfloat16, pin_memory=True)
    w_cpu = torch.empty(n_cpu, in_dim, dtype=torch.bfloat16)
    r.install(slab_specs=[], scratch_max_tokens=num_tokens,
              scratch_max_intermediate_per_half=0)
    # We didn't pass slab_specs to install (so the slab pool is size 0).
    # Switch strategy: use the per-task populate methods directly via
    # the cots_ops registry, which is what NativeCotsRunner.install
    # would have done if we had given it specs.
    return r, x_pin, y_pin


def _populate_qkv_slab_directly(
    runner_id: int,
    task_id: int,
    x_pin: torch.Tensor,
    y_pin: torch.Tensor,
    w_cpu: torch.Tensor,
) -> None:
    """Reach into the registry and populate task_id directly. Used for
    the test fixtures here so we don't need to go through
    `_NativeSlabSpecQkv` machinery."""
    from vllm.model_executor.offloader import cots_ops

    infer = cots_ops._lookup_infer(runner_id, "test_y_pinned_view")
    infer.populate_slab_qkv(
        task_id=task_id,
        n_threads=1,
        x_pinned_ptr=x_pin.data_ptr(),
        in_dim=int(x_pin.shape[1]),
        y_pinned_ptr=y_pin.data_ptr(),
        cpu_out_dim=int(y_pin.shape[1]),
        w_cpu_ptr=w_cpu.data_ptr(),
        w_cpu_rows=int(w_cpu.shape[0]),
    )


def test_y_pinned_view_shape_dtype_device() -> None:
    """The returned tensor has the requested (num_tokens, cpu_out_dim)
    shape, bfloat16 dtype, and is on CPU."""
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader import cots

    num_tokens, in_dim, n_cpu = 8, 16, 12
    r = cots.NativeCotsRunner(dry_run=False)
    try:
        # install with one slab; we need slab_count > 0 for the pybind
        # call to succeed (the C++ side bounds-checks task_id).
        r.install(slab_specs=[], scratch_max_tokens=num_tokens,
                  scratch_max_intermediate_per_half=0)
        # n_slabs=0 makes y_pinned_view raise (out-of-range), so do a
        # second install path: we go around `install` and call install_infer
        # directly with n_slabs=1, then populate.
    finally:
        r.close()

    # Re-do with 1 slab.
    from vllm.model_executor.offloader import cots_ops

    r = cots.NativeCotsRunner(dry_run=False)
    try:
        cots_ops.install_infer(
            r._runner_id,
            n_slabs=1,
            scratch_max_tokens=num_tokens,
            scratch_max_intermediate_per_half=0,
        )
        x_pin = torch.empty(num_tokens, in_dim, dtype=torch.bfloat16,
                            pin_memory=True)
        y_pin = torch.empty(num_tokens, n_cpu, dtype=torch.bfloat16,
                            pin_memory=True)
        w_cpu = torch.empty(n_cpu, in_dim, dtype=torch.bfloat16)
        _populate_qkv_slab_directly(r._runner_id, 0, x_pin, y_pin, w_cpu)

        infer = cots_ops._lookup_infer(r._runner_id, "test")
        view = infer.y_pinned_view(task_id=0, num_tokens=num_tokens)
        assert view.shape == (num_tokens, n_cpu)
        assert view.dtype == torch.bfloat16
        assert view.is_cpu
    finally:
        r.close()


def test_y_pinned_view_data_ptr_matches_slab() -> None:
    """The view's `data_ptr()` is the same as the y_pin tensor's
    `data_ptr()` — proving we built the view over the slab pointer,
    not a fresh allocation."""
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader import cots, cots_ops

    num_tokens, in_dim, n_cpu = 4, 8, 6
    r = cots.NativeCotsRunner(dry_run=False)
    try:
        cots_ops.install_infer(
            r._runner_id,
            n_slabs=1,
            scratch_max_tokens=num_tokens,
            scratch_max_intermediate_per_half=0,
        )
        x_pin = torch.empty(num_tokens, in_dim, dtype=torch.bfloat16,
                            pin_memory=True)
        y_pin = torch.empty(num_tokens, n_cpu, dtype=torch.bfloat16,
                            pin_memory=True)
        w_cpu = torch.empty(n_cpu, in_dim, dtype=torch.bfloat16)
        _populate_qkv_slab_directly(r._runner_id, 0, x_pin, y_pin, w_cpu)

        infer = cots_ops._lookup_infer(r._runner_id, "test")
        view = infer.y_pinned_view(task_id=0, num_tokens=num_tokens)
        assert view.data_ptr() == y_pin.data_ptr()
    finally:
        r.close()


def test_y_pinned_view_reads_worker_writes() -> None:
    """Write data into the pinned buffer (simulating the worker), then
    read via `y_pinned_view` and confirm we see the write. Closes the
    storage-correctness loop: even though the view's `is_pinned()`
    metadata bit is unset (`at::from_blob` doesn't set it for foreign
    blobs), the underlying storage IS the page-locked allocation."""
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader import cots, cots_ops

    num_tokens, in_dim, n_cpu = 4, 8, 6
    r = cots.NativeCotsRunner(dry_run=False)
    try:
        cots_ops.install_infer(
            r._runner_id,
            n_slabs=1,
            scratch_max_tokens=num_tokens,
            scratch_max_intermediate_per_half=0,
        )
        x_pin = torch.empty(num_tokens, in_dim, dtype=torch.bfloat16,
                            pin_memory=True)
        y_pin = torch.empty(num_tokens, n_cpu, dtype=torch.bfloat16,
                            pin_memory=True)
        w_cpu = torch.empty(n_cpu, in_dim, dtype=torch.bfloat16)
        _populate_qkv_slab_directly(r._runner_id, 0, x_pin, y_pin, w_cpu)

        # Simulate the worker writing a known pattern into y_pin.
        pattern = torch.arange(
            num_tokens * n_cpu, dtype=torch.bfloat16
        ).reshape(num_tokens, n_cpu)
        y_pin.copy_(pattern)

        infer = cots_ops._lookup_infer(r._runner_id, "test")
        view = infer.y_pinned_view(task_id=0, num_tokens=num_tokens)
        assert torch.equal(view, pattern)
    finally:
        r.close()


def test_y_pinned_view_rejects_out_of_range_task_id() -> None:
    """A task_id past the installed slab pool raises (defensive
    bounds check from C++)."""
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader import cots, cots_ops

    r = cots.NativeCotsRunner(dry_run=False)
    try:
        cots_ops.install_infer(
            r._runner_id, n_slabs=1, scratch_max_tokens=4,
            scratch_max_intermediate_per_half=0,
        )
        infer = cots_ops._lookup_infer(r._runner_id, "test")
        with pytest.raises(Exception, match="out of range|task_id"):
            infer.y_pinned_view(task_id=99, num_tokens=4)
    finally:
        r.close()


def test_y_pinned_view_partial_num_tokens() -> None:
    """Calling with `num_tokens < cpu_out_dim`'s natural row count
    returns a smaller view over the SAME storage prefix. This is the
    captured-forward case where each forward processes a subset of
    the slab's max-tokens preallocation."""
    pytest.importorskip("vllm._cots_C")
    from vllm.model_executor.offloader import cots, cots_ops

    max_tokens, in_dim, n_cpu = 8, 8, 6
    r = cots.NativeCotsRunner(dry_run=False)
    try:
        cots_ops.install_infer(
            r._runner_id, n_slabs=1, scratch_max_tokens=max_tokens,
            scratch_max_intermediate_per_half=0,
        )
        x_pin = torch.empty(max_tokens, in_dim, dtype=torch.bfloat16,
                            pin_memory=True)
        y_pin = torch.empty(max_tokens, n_cpu, dtype=torch.bfloat16,
                            pin_memory=True)
        w_cpu = torch.empty(n_cpu, in_dim, dtype=torch.bfloat16)
        _populate_qkv_slab_directly(r._runner_id, 0, x_pin, y_pin, w_cpu)

        infer = cots_ops._lookup_infer(r._runner_id, "test")
        for n in [1, 4, max_tokens]:
            view = infer.y_pinned_view(task_id=0, num_tokens=n)
            assert view.shape == (n, n_cpu)
            assert view.data_ptr() == y_pin.data_ptr()
    finally:
        r.close()
