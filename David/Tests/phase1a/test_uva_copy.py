"""Phase 1a §1 — Triton UVA copy kernel correctness (gating test).

If this fails, the cots offloader's activation-return path doesn't work,
and we'd need to fall back to a CUDA kernel (per the plan's deferred path).
"""

import pytest
import torch

from vllm.model_executor.offloader.cots import uva_copy_into_gpu


@pytest.mark.parametrize(
    "shape",
    [
        (128, 1024),  # ~256 KB at BF16
        (512, 1024),  # ~1 MB
        (1024, 2048),  # ~4 MB
        (4096, 3584),  # 7B hidden-sized batch
    ],
)
def test_uva_copy_bf16_correctness(shape):
    """Pinned host -> GPU via SM-issued UVA copy round-trips bit-identical."""
    torch.manual_seed(0)
    src = torch.randn(*shape, dtype=torch.bfloat16).pin_memory()
    dst = torch.empty(shape, dtype=torch.bfloat16, device="cuda")

    uva_copy_into_gpu(src, dst)
    torch.cuda.synchronize()

    # Bit-identical: no compute happens during the copy, just a load+store.
    assert torch.equal(dst.cpu(), src), (
        f"UVA copy mismatch at shape {shape}: max abs diff = "
        f"{(dst.cpu().float() - src.float()).abs().max().item()}"
    )


def test_uva_copy_rejects_non_pinned_src():
    """Contract: src must be pinned (otherwise UVA mapping is invalid)."""
    src = torch.randn(64, 64, dtype=torch.bfloat16)  # NOT pinned
    dst = torch.empty(64, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError):
        uva_copy_into_gpu(src, dst)


def test_uva_copy_shape_dtype_must_match():
    src = torch.randn(64, 64, dtype=torch.bfloat16).pin_memory()
    dst_wrong_shape = torch.empty(64, 32, dtype=torch.bfloat16, device="cuda")
    dst_wrong_dtype = torch.empty(64, 64, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError):
        uva_copy_into_gpu(src, dst_wrong_shape)
    with pytest.raises(AssertionError):
        uva_copy_into_gpu(src, dst_wrong_dtype)
