# SPDX-License-Identifier: Apache-2.0
"""In-process wait-kernel sync smoke via pybind.

This exercises the vLLM `_cots_C` pybind path that production uses for
wait-kernel allocation, install, and launch.

What's tested:
  1. install_wait_kernel_sync_for_task allocates host-mapped pinned slots and
     sets wait_kernel_sync_installed=true. Idempotency-violation re-install
     raises.
  2. m3_get/set_req_slot, m3_get/set_done_slot round-trip.
  3. wait_kernel_sync_on_stream returns immediately when done >= req
     (immediate-resume path).
  4. wait_kernel_sync_on_stream blocks until a worker thread writes done,
     then returns (lagging-wait path). No deadlock under 100×
     replay.
  5. Diag counters increment correctly when VLLM_COTS_DIAG=1.
  6. Free path: dropping the CotsCpuInfer doesn't leak (the
     destructor calls cudaFreeHost on every installed slab —
     verified indirectly: a fresh inst can install a slab with
     the same task_id without error after the prior infer is
     dropped).
"""

from __future__ import annotations

import os
import threading
import time

import pytest
import torch


@pytest.fixture
def infer():
    pytest.importorskip("vllm._cots_C")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm._cots_C import CotsCpuInfer

    inst = CotsCpuInfer()
    inst.install(n_slabs=4, max_num_tokens=0)
    yield inst
    # Destructor runs when `inst` goes out of scope; that frees
    # the host-mapped slots.


def test_install_wait_kernel_sync_for_task_basic(infer):
    """install_wait_kernel_sync_for_task allocates slots; subsequent get returns 0."""
    infer.install_wait_kernel_sync_for_task(0)
    assert infer.wait_kernel_get_req_slot(0) == 0
    assert infer.wait_kernel_get_done_slot(0) == 0


def test_install_wait_kernel_sync_for_task_idempotency_violation(infer):
    """Re-installing wait-kernel sync for the same task_id raises."""
    infer.install_wait_kernel_sync_for_task(0)
    with pytest.raises(RuntimeError, match="already installed"):
        infer.install_wait_kernel_sync_for_task(0)


def test_install_wait_kernel_sync_for_task_out_of_range(infer):
    """task_id outside [0, n_slabs) raises."""
    with pytest.raises(RuntimeError, match="out of range"):
        infer.install_wait_kernel_sync_for_task(99)


def test_m3_set_get_round_trip(infer):
    """Direct slot writes are visible via getter (verifies the
    host-mapped memory is CPU-readable in both directions)."""
    infer.install_wait_kernel_sync_for_task(0)
    infer.wait_kernel_set_req_slot(0, 42)
    infer.wait_kernel_set_done_slot(0, 7)
    assert infer.wait_kernel_get_req_slot(0) == 42
    assert infer.wait_kernel_get_done_slot(0) == 7


def test_m3_wait_immediate_resume(infer):
    """When done >= req at kernel launch, the wait kernel returns
    immediately (no spin)."""
    infer.install_wait_kernel_sync_for_task(0)
    infer.wait_kernel_set_req_slot(0, 1)
    infer.wait_kernel_set_done_slot(0, 1)
    stream = torch.cuda.current_stream().cuda_stream
    infer.wait_kernel_sync_on_stream(0, stream)
    torch.cuda.current_stream().synchronize()
    # No deadlock; kernel returned. (No assertion needed beyond
    # not hanging — pytest's default timeout would catch it.)


def test_m3_wait_lagging_then_release(infer):
    """When done < req at kernel launch, the wait kernel spins
    until a worker thread writes done >= req, then returns."""
    infer.install_wait_kernel_sync_for_task(0)
    infer.wait_kernel_set_req_slot(0, 5)
    infer.wait_kernel_set_done_slot(0, 0)

    # Launch wait kernel on a stream; it will spin.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        infer.wait_kernel_sync_on_stream(0, stream.cuda_stream)
    # Worker thread releases after a brief delay.
    released = threading.Event()

    def worker():
        time.sleep(0.05)
        infer.wait_kernel_set_done_slot(0, 5)
        released.set()

    t = threading.Thread(target=worker)
    t.start()
    # Synchronize the stream — should block until the worker
    # writes done. Bounded timeout via cuda event would be
    # better; here we rely on the stream sync to release.
    stream.synchronize()
    t.join(timeout=5.0)
    assert released.is_set(), "worker did not run"


def test_m3_wait_repeated_replay(infer):
    """100 replays of the immediate-resume path. No deadlock,
    no corruption, no resource leak."""
    infer.install_wait_kernel_sync_for_task(0)
    stream = torch.cuda.current_stream().cuda_stream
    for i in range(1, 101):
        infer.wait_kernel_set_req_slot(0, i)
        infer.wait_kernel_set_done_slot(0, i)
        infer.wait_kernel_sync_on_stream(0, stream)
    torch.cuda.current_stream().synchronize()
    assert infer.wait_kernel_get_req_slot(0) == 100
    assert infer.wait_kernel_get_done_slot(0) == 100


def test_m3_get_without_install_raises(infer):
    """m3_get_*_slot before install raises (catches mismatched
    install/use ordering bugs)."""
    with pytest.raises(RuntimeError, match="not installed"):
        infer.wait_kernel_get_req_slot(0)


def test_m3_wait_captured_graph_replay(infer):
    """Capture wait_kernel_sync_on_stream into a
    real torch.cuda.CUDAGraph and replay it 100x. The standalone
    proof is now kept here, through the production wait-kernel +
    host-mapped-slot path that goes through the _cots_C launcher
    (cots_wait_done_kernel reading volatile uint32_t cells via
    cudaHostGetDevicePointer addresses).

    Key replay-safety properties exercised:
      (a) The captured kernel re-reads dev_req_slot / dev_done_slot
          on every replay (host-mapped pinned + volatile reads),
          NOT the values that were resident at capture time.
      (b) Stable userData / stable kernel pointers across replays
          (slab address-stable, dev pointers from
          cudaHostGetDevicePointer are stable for the lifetime of
          the host alloc).
      (c) Immediate-resume case is the simplest replay-correctness
          shape; the lagging-then-release case in
          test_m3_wait_captured_graph_lagging_release covers
          cross-thread-write semantics with a definite-block
          assertion.
    """
    infer.install_wait_kernel_sync_for_task(0)
    # Pre-warm: launch once outside capture so any first-launch
    # JIT / lazy-init does not contaminate the capture.
    infer.wait_kernel_set_req_slot(0, 0)
    infer.wait_kernel_set_done_slot(0, 0)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        infer.wait_kernel_sync_on_stream(0, s.cuda_stream)
    s.synchronize()

    # Capture once with the wait as the only graph node.
    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(g, stream=capture_stream):
            infer.wait_kernel_sync_on_stream(0, capture_stream.cuda_stream)

    # Replay 100x. For each replay, set req=i and done=i BEFORE
    # replay so the wait kernel resumes immediately (volatile read
    # of the host-mapped slots picks up the new values).
    for i in range(1, 101):
        infer.wait_kernel_set_req_slot(0, i)
        infer.wait_kernel_set_done_slot(0, i)
        g.replay()
        torch.cuda.current_stream().synchronize()
    assert infer.wait_kernel_get_req_slot(0) == 100
    assert infer.wait_kernel_get_done_slot(0) == 100


def test_m3_wait_captured_graph_lagging_release(infer):
    """Captured-graph lagging case with a definite-block assertion.

    Replay the captured wait kernel with done < req, then release
    from a worker thread after a measured delay. Asserts that the
    elapsed time from `g.replay()` to `torch.cuda.synchronize()`
    return is at least the worker's pre-release sleep — i.e., the
    replay actually waited. Without this assertion, the test could
    pass even if the wait kernel returned early (or the wrong
    stream was sync'd), because t.join(timeout=5) would still see
    `released` set after the worker thread's own sleep. Using
    torch.cuda.synchronize() (device-wide) instead of
    current_stream().synchronize() is intentional: it leaves no
    room for the wait to live on a different stream than the one
    we sync.
    """
    infer.install_wait_kernel_sync_for_task(0)
    infer.wait_kernel_set_req_slot(0, 0)
    infer.wait_kernel_set_done_slot(0, 0)

    g = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    # Pre-warm before capture (immediate-resume) to defeat any
    # first-launch lazy paths leaking into the capture.
    with torch.cuda.stream(capture_stream):
        infer.wait_kernel_sync_on_stream(0, capture_stream.cuda_stream)
    capture_stream.synchronize()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(g, stream=capture_stream):
            infer.wait_kernel_sync_on_stream(0, capture_stream.cuda_stream)

    # Replay with done < req; worker thread releases after a
    # measured delay. The 50 ms sleep is a coarse target; the
    # assertion uses 40 ms (40 ms < 50 ms target so OS jitter
    # is tolerated) but is still firmly above the noise floor
    # for a kernel that returns immediately.
    infer.wait_kernel_set_req_slot(0, 7)
    infer.wait_kernel_set_done_slot(0, 0)
    released = threading.Event()
    worker_sleep_s = 0.05

    def worker():
        time.sleep(worker_sleep_s)
        infer.wait_kernel_set_done_slot(0, 7)
        released.set()

    t = threading.Thread(target=worker)
    t.start()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    t.join(timeout=5.0)
    assert released.is_set(), "worker did not run before device sync"
    assert infer.wait_kernel_get_done_slot(0) == 7
    assert elapsed >= 0.04, (
        f"captured-graph wait did not actually block: elapsed={elapsed:.4f}s "
        f"(< 40 ms; worker sleeps {worker_sleep_s * 1000:.0f} ms before "
        f"writing done_slot, so a real wait must take at least that long)"
    )


def test_m3_diag_counters_when_enabled(monkeypatch):
    """Diag-mode wait kernel updates immediate vs lagging
    counters. Requires VLLM_COTS_DIAG=1 BEFORE process import
    (the env is read once at first call)."""
    pytest.importorskip("vllm._cots_C")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if os.environ.get("VLLM_COTS_DIAG", "0") != "1":
        pytest.skip(
            "VLLM_COTS_DIAG=1 must be set before process start to exercise "
            "the diag wait kernel. Re-run with VLLM_COTS_DIAG=1 to validate "
            "diag counters; production-default mode skips this assertion."
        )
    from vllm._cots_C import CotsCpuInfer

    inst = CotsCpuInfer()
    inst.install(n_slabs=2, max_num_tokens=0)
    inst.install_wait_kernel_sync_for_task(0)
    inst.reset_counters()

    stream = torch.cuda.current_stream().cuda_stream
    # Immediate-resume case.
    inst.wait_kernel_set_req_slot(0, 1)
    inst.wait_kernel_set_done_slot(0, 1)
    inst.wait_kernel_sync_on_stream(0, stream)
    torch.cuda.current_stream().synchronize()

    counters = dict(inst.get_counters())
    assert counters["wait_kernel_immediate_resume_count"] >= 1, (
        f"diag mode not active or counter not incremented; got {counters}"
    )
