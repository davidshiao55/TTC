# SPDX-License-Identifier: Apache-2.0
"""§1c.20 — UVA copy safety assertion is storage-level, not view-level.

Inductor's `reinterpret_tensor(...)` produces a tensor whose
`Tensor.is_pinned()` metadata bit is False, even when the underlying
storage IS page-locked (the storage came from a `torch.empty(...,
pin_memory=True)` allocation that was sliced/viewed). The Triton UVA
kernel only requires page-locked host storage; the metadata bit is
informational. Switching the safety assertion in `uva_copy_into_gpu`
to inspect the storage directly fixes the captured-forward path
without losing the pageable-CPU rejection that protects the kernel
from reading garbage.
"""

from __future__ import annotations

import pytest
import torch


def test_helper_rejects_pageable_cpu_tensor() -> None:
    """A regular `torch.empty(...)` on CPU is NOT pinned. The helper
    must reject it so the UVA kernel never sees pageable host memory."""
    from vllm.model_executor.offloader.cots import _has_pinned_host_storage

    pageable = torch.empty(8, dtype=torch.bfloat16)
    assert pageable.is_cpu
    assert not pageable.is_pinned()
    assert not _has_pinned_host_storage(pageable)


def test_helper_accepts_pinned_cpu_tensor() -> None:
    """A `torch.empty(..., pin_memory=True)` allocation has the
    metadata bit set. Helper accepts via the fast path."""
    if not torch.cuda.is_available():
        pytest.skip("pin_memory=True requires CUDA runtime")
    from vllm.model_executor.offloader.cots import _has_pinned_host_storage

    pinned = torch.empty(8, dtype=torch.bfloat16, pin_memory=True)
    assert pinned.is_pinned()
    assert _has_pinned_host_storage(pinned)


def test_helper_accepts_view_of_pinned_storage() -> None:
    """The reviewer's diagnosis: a view (slice / view / narrow) of a
    pinned allocation has the SAME storage but may report
    `is_pinned()` differently across PyTorch versions / Inductor
    paths. The helper must accept these because the storage is still
    page-locked."""
    if not torch.cuda.is_available():
        pytest.skip("pin_memory=True requires CUDA runtime")
    from vllm.model_executor.offloader.cots import _has_pinned_host_storage

    base = torch.empty(64, dtype=torch.bfloat16, pin_memory=True)
    sliced = base[:32]
    viewed = base.view(8, 8)
    # Even if `is_pinned()` reports False on the view, the storage
    # check must accept these (the storage is shared with `base` so
    # the page-locked property is preserved).
    assert _has_pinned_host_storage(sliced)
    assert _has_pinned_host_storage(viewed)


def test_helper_accepts_storage_when_metadata_bit_lies() -> None:
    """§1c.20 core case: simulate Inductor's `reinterpret_tensor`
    behavior by constructing a tensor that shares storage with a
    pinned allocation but reports `is_pinned()=False`. The helper's
    storage-level fallback must catch it.

    NB: producing this state from pure PyTorch is fragile (the
    `is_pinned()` flag is computed from storage, so on most builds a
    view does report True). We exercise both orderings: if the
    metadata bit happens to be True we test the fast path; if it
    isn't, we exercise the storage fallback. Either way, the
    assertion in `uva_copy_into_gpu` must hold."""
    if not torch.cuda.is_available():
        pytest.skip("pin_memory=True requires CUDA runtime")
    from vllm.model_executor.offloader.cots import _has_pinned_host_storage

    base = torch.empty(64, dtype=torch.bfloat16, pin_memory=True)
    # `as_strided` reinterprets the same storage with a custom layout —
    # closest pure-PyTorch analogue of Inductor's reinterpret_tensor.
    reinterpreted = torch.as_strided(base, size=(8, 8), stride=(8, 1))
    # The post-condition we care about is the storage is page-locked,
    # regardless of how Tensor.is_pinned() reports.
    storage_pinned = reinterpreted.untyped_storage().is_pinned()
    assert storage_pinned, "test scaffold broken: storage should be pinned"
    assert _has_pinned_host_storage(reinterpreted)


def test_helper_rejects_cuda_tensor() -> None:
    """A CUDA tensor isn't host memory at all — reject."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from vllm.model_executor.offloader.cots import _has_pinned_host_storage

    gpu = torch.empty(8, device="cuda", dtype=torch.bfloat16)
    assert not _has_pinned_host_storage(gpu)


def test_uva_copy_into_gpu_still_works_on_pinned_path() -> None:
    """Sanity: the existing eager UVA copy path (where `is_pinned()`
    is True) still executes correctly after the assertion swap."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from vllm.model_executor.offloader.cots import uva_copy_into_gpu

    n = 64
    src = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
    src.copy_(torch.arange(n, dtype=torch.bfloat16))
    dst = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
    uva_copy_into_gpu(src, dst)
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), src), "UVA copy lost data after assertion swap"


def test_uva_copy_into_gpu_works_on_reinterpreted_view() -> None:
    """The §1c.20 captured-forward case: a tensor that came from
    `as_strided` / `reinterpret_tensor` over pinned storage. Before
    this fix the assertion rejected it; now it must accept and the
    kernel must produce correct output."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from vllm.model_executor.offloader.cots import uva_copy_into_gpu

    base = torch.empty(64, dtype=torch.bfloat16, pin_memory=True)
    base.copy_(torch.arange(64, dtype=torch.bfloat16))
    src = torch.as_strided(base, size=(64,), stride=(1,))
    dst = torch.zeros(64, dtype=torch.bfloat16, device="cuda")
    uva_copy_into_gpu(src, dst)
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), base)


def test_uva_copy_rejects_pageable_cpu() -> None:
    """The original safety guarantee — pageable CPU tensors must NOT
    reach the Triton kernel — is preserved by the storage-level check."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    from vllm.model_executor.offloader.cots import uva_copy_into_gpu

    pageable = torch.empty(8, dtype=torch.bfloat16)
    dst = torch.zeros(8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="pinned host memory"):
        uva_copy_into_gpu(pageable, dst)
