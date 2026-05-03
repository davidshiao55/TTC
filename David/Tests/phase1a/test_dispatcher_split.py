"""Phase 1a §3-4 — CPU-side GEMM split correctness for col & row parallel.

Exercises the CPU compute path directly via `CpuTaskRunner` +
`_cpu_gemm_into_after_event` (the worker function the QKV op uses) on
synthetic shapes — no Linear modules involved. Verifies the assembled
output matches the unsplit reference within BF16 tolerance, mirroring
`phase0/bench_split_correctness.py §A` and §C.

Storage allocation is done manually here (no `CotsLinearHandle.install`
because there's no Linear); the test owns the buffers it submits.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.offloader.cots import (
    CpuTaskRunner,
    _cpu_gemm_into_after_event,
    uva_copy_into_gpu,
)

# BF16 tolerance — see header in earlier revision: cuBLAS picks different
# tile/MMA configs for different matrix shapes (sliced vs full), producing
# FMA orderings that diverge by ~1 ULP at BF16. atol=0.5 has comfortable
# margin at unit-variance inputs with in_dim≈512.
BF16_RTOL = 5e-2
BF16_ATOL = 0.5


def _make_full(out_dim: int, in_dim: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(out_dim, in_dim, dtype=torch.bfloat16)


def _alloc_pinned(numel: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, pin_memory=True)


def _alloc_gpu(numel: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(numel, dtype=dtype, device="cuda")


def _run_cpu_gemm(
    runner: CpuTaskRunner,
    x_gpu: torch.Tensor,
    w_cpu: torch.Tensor,
    x_pinned_view: torch.Tensor,
    y_pinned_view: torch.Tensor,
    y_gpu_view: torch.Tensor,
) -> torch.Tensor:
    """Submit + wait + UVA copy. Mirrors the QKV op's inner sequence."""
    runner.submit_with_d2h(x_gpu, x_pinned_view, _cpu_gemm_into_after_event,
                           w_cpu, y_pinned_view)
    runner.wait()
    uva_copy_into_gpu(y_pinned_view, y_gpu_view)
    return y_gpu_view


@pytest.mark.parametrize("f_cpu", [0.03, 0.09, 0.22, 0.50])
@pytest.mark.parametrize("batch", [1, 8, 64])
def test_col_parallel_split(f_cpu, batch):
    """Col-parallel (MergedColumnParallelLinear / gate_up_proj): each shard
    (gate, up) has its LAST n_cpu_per_half rows on CPU.
    """
    in_dim, out_dim = 512, 768  # out_dim is even
    half = out_dim // 2
    n_cpu_per_half = round(f_cpu * half)
    n_cpu = 2 * n_cpu_per_half
    keep_gpu = half - n_cpu_per_half
    if n_cpu_per_half == 0 or n_cpu_per_half == half:
        pytest.skip(f"degenerate split at f={f_cpu}")

    full = _make_full(out_dim, in_dim)

    # CPU weight slice: [gate_cpu | up_cpu], each n_cpu_per_half rows.
    w_cpu = torch.empty(
        n_cpu, in_dim, dtype=torch.bfloat16, pin_memory=True
    )
    w_cpu[:n_cpu_per_half, :].copy_(full[half - n_cpu_per_half : half, :])
    w_cpu[n_cpu_per_half:, :].copy_(full[2 * half - n_cpu_per_half :, :])

    # GPU weight slice: [gate_gpu | up_gpu] stacked.
    gpu_weight = torch.empty(
        2 * keep_gpu, in_dim, dtype=torch.bfloat16, device="cuda"
    )
    gpu_weight[:keep_gpu, :].copy_(full[:keep_gpu, :])
    gpu_weight[keep_gpu:, :].copy_(full[half : half + keep_gpu, :])

    x_pinned_view = _alloc_pinned(batch * in_dim).view(batch, in_dim)
    y_pinned_view = _alloc_pinned(batch * n_cpu).view(batch, n_cpu)
    y_gpu_view = _alloc_gpu(batch * n_cpu).view(batch, n_cpu)

    x_gpu = torch.randn(batch, in_dim, dtype=torch.bfloat16, device="cuda")

    runner = CpuTaskRunner()
    out_gpu_slice = F.linear(x_gpu, gpu_weight, None)
    out_cpu_on_gpu = _run_cpu_gemm(
        runner, x_gpu, w_cpu, x_pinned_view, y_pinned_view, y_gpu_view
    )

    # Assemble back into [gate_full_out | up_full_out] canonical layout.
    out = torch.empty((batch, out_dim), dtype=torch.bfloat16, device="cuda")
    out[:, :keep_gpu].copy_(out_gpu_slice[:, :keep_gpu])
    out[:, keep_gpu:half].copy_(out_cpu_on_gpu[:, :n_cpu_per_half])
    out[:, half : half + keep_gpu].copy_(out_gpu_slice[:, keep_gpu:])
    out[:, half + keep_gpu :].copy_(out_cpu_on_gpu[:, n_cpu_per_half:])

    full_gpu = full.to("cuda")
    ref = F.linear(x_gpu, full_gpu, None)
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=BF16_ATOL)


@pytest.mark.parametrize("f_cpu", [0.03, 0.09, 0.22, 0.50])
@pytest.mark.parametrize("batch", [1, 8, 64])
def test_row_parallel_split(f_cpu, batch):
    """Row-parallel (MLP2-style): each device computes a partial sum on its
    input-col slice; add-reduce on GPU."""
    in_dim, out_dim = 1024, 384
    n_cpu = round(f_cpu * in_dim)
    if n_cpu == 0 or n_cpu == in_dim:
        pytest.skip(f"degenerate split at f={f_cpu}")

    full = _make_full(out_dim, in_dim)
    # CPU weight slice: last n_cpu input cols.
    w_cpu = torch.empty(
        out_dim, n_cpu, dtype=torch.bfloat16, pin_memory=True
    )
    w_cpu.copy_(full[:, in_dim - n_cpu :])

    gpu_weight = full[:, : in_dim - n_cpu].to("cuda").contiguous()

    x_pinned_view = _alloc_pinned(batch * n_cpu).view(batch, n_cpu)
    y_pinned_view = _alloc_pinned(batch * out_dim).view(batch, out_dim)
    y_gpu_view = _alloc_gpu(batch * out_dim).view(batch, out_dim)

    x_gpu = torch.randn(batch, in_dim, dtype=torch.bfloat16, device="cuda")
    x_gpu_input = x_gpu[:, : in_dim - n_cpu].contiguous()
    x_cpu_input = x_gpu[:, in_dim - n_cpu :].contiguous()

    runner = CpuTaskRunner()
    out_gpu_partial = F.linear(x_gpu_input, gpu_weight, None)
    out_cpu_partial = _run_cpu_gemm(
        runner, x_cpu_input, w_cpu, x_pinned_view, y_pinned_view, y_gpu_view
    )

    out = out_gpu_partial + out_cpu_partial
    full_gpu = full.to("cuda")
    ref = F.linear(x_gpu, full_gpu, None)
    torch.testing.assert_close(out, ref, rtol=BF16_RTOL, atol=BF16_ATOL)


def test_uva_round_trip_pins_correctly():
    """Sanity: the UVA copy round-trips bit-identical when the worker writes
    a full-shape tensor into pinned memory before the copy fires.
    """
    src = torch.randn(64, 64, dtype=torch.bfloat16, pin_memory=True)
    dst = torch.empty(64, 64, dtype=torch.bfloat16, device="cuda")
    uva_copy_into_gpu(src, dst)
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), src)
