# SPDX-License-Identifier: Apache-2.0
"""Stage 1: TaskQueue stress through CotsCpuInfer's submit_dryrun_burst.

The TaskQueue C++ class is not directly pybind-exposed — access goes
through CotsCpuInfer. `submit_dryrun_burst(n)` is the test-only helper
that enqueues N pure no-op closures (no slab read, no CUDA path) so we
can exercise the lock-free MPSC enqueue + drain semantics in isolation.
"""

import time

from vllm._cots_C import CotsCpuInfer


def test_drain_empty_queue_is_noop():
    ci = CotsCpuInfer()
    ci.install(n_slabs=0, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)
    ci.sync_blocking()  # should return immediately on empty queue
    assert not ci.has_error()


def test_burst_drains():
    ci = CotsCpuInfer()
    ci.install(n_slabs=0, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)
    ci.submit_dryrun_burst(1000)
    ci.sync_blocking()
    assert not ci.has_error()


def test_large_burst_drains_within_budget():
    """100k no-ops through the queue. With one worker thread, the limit
    is enqueue overhead + cv signaling. Budget is generous (2 s) — this
    is a correctness test, not a perf bench."""
    ci = CotsCpuInfer()
    ci.install(n_slabs=0, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)
    t0 = time.perf_counter()
    ci.submit_dryrun_burst(100_000)
    ci.sync_blocking()
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"drain took {elapsed:.3f}s, expected < 2s"
    assert not ci.has_error()


def test_repeated_burst_then_drain_stable():
    """Multiple submit/drain cycles on the same instance. Catches any
    bug where the queue's pending-counter or head/tail atomics drift."""
    ci = CotsCpuInfer()
    ci.install(n_slabs=0, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)
    for _ in range(10):
        ci.submit_dryrun_burst(500)
        ci.sync_blocking()
        assert not ci.has_error()


def test_destruct_drains_pending():
    """Letting CotsCpuInfer go out of scope while tasks are pending should
    not leak or hang — the destructor calls sync_blocking() before
    destructing TaskQueue."""
    for _ in range(5):
        ci = CotsCpuInfer()
        ci.install(n_slabs=0, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)
        ci.submit_dryrun_burst(200)
        # No explicit sync; rely on destructor.
        del ci
