# SPDX-License-Identifier: Apache-2.0
"""Stage 1 smoke: `vllm._cots_C` loads and `CotsCpuInfer` instantiates."""

import torch


def test_module_imports():
    import vllm._cots_C as _cots_C  # noqa: F401


def test_cotscpuinfer_constructs():
    from vllm._cots_C import CotsCpuInfer

    ci = CotsCpuInfer()
    assert ci is not None


def test_install_zero_slabs_ok():
    """install(0, 0, 0) is a valid degenerate configuration — the offloader
    may install a runner before any handles are populated."""
    from vllm._cots_C import CotsCpuInfer

    ci = CotsCpuInfer()
    ci.install(n_slabs=0, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)


def test_install_some_slabs_then_dryrun():
    from vllm._cots_C import CotsCpuInfer

    ci = CotsCpuInfer()
    ci.install(n_slabs=8, scratch_max_tokens=0, scratch_max_intermediate_per_half=0)
    for i in range(8):
        ci.populate_slab_dryrun(i, x_pinned_ptr=0, in_dim=0, y_pinned_ptr=0, cpu_out_dim=0)
    # Just confirms the no-CUDA path works (TaskQueue alone, no host callback).
    ci.submit_dryrun_burst(16)
    ci.sync_blocking()
    assert not ci.has_error()


def test_at_linear_inline_returns_correct_result():
    """Sanity that `run_at_linear_inline` (test-only helper) computes
    `at::linear` from C++ correctly. The microbench gate tests perf;
    here we just confirm correctness end-to-end through pybind."""
    from vllm._cots_C import CotsCpuInfer

    ci = CotsCpuInfer()
    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    w = torch.randn(16, 32, dtype=torch.bfloat16)
    y = torch.empty(4, 16, dtype=torch.bfloat16)
    ci.run_at_linear_inline(x, w, y)
    ref = torch.nn.functional.linear(x, w)
    # bf16 round-trip; allow 1-ULP-ish tolerance.
    assert torch.allclose(y, ref, atol=1e-2), (
        f"at::linear inline result diverges from F.linear; "
        f"max abs diff = {(y - ref).abs().max()}"
    )
