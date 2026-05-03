"""Phase 1a §5 — TP-style loader closures (col / merged-col / qkv / row).

Verifies the per-kind loader closures on `CotsLinearHandle` correctly split
each shard's `loaded_weight` into:
  * GPU portion (FIRST cols/rows) → `param.data` (already at GPU slice shape)
  * CPU portion (LAST cols/rows)  → `handle.w_cpu` at the right offset

Each test constructs a `CotsLinearHandle` directly (bypassing the factory
methods which read vLLM-specific attrs), runs `install()` to allocate slice
storage and wrap the loader, then exercises the loader with a synthetic
full `loaded_weight` and asserts the per-destination contents.
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.offloader.cots import (
    CotsLinearHandle,
    _complement,
    _qkv_kv_biased_indices,
)


def _fake_linear(out_dim: int, in_dim: int, dtype=torch.bfloat16) -> nn.Module:
    """Minimal Linear stand-in: holds a `weight` Parameter at full shape."""

    class _FakeLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(out_dim, in_dim, dtype=dtype, device="cuda")
            )

    return _FakeLinear()


def _make_handle_row(in_dim, out_dim, n_cpu, dtype=torch.bfloat16):
    linear = _fake_linear(out_dim, in_dim, dtype)
    cpu_indices = torch.arange(in_dim - n_cpu, in_dim, dtype=torch.long)
    gpu_indices = _complement(cpu_indices, in_dim)
    handle = CotsLinearHandle(
        kind="row", linear=linear, qualified_name="row.test",
        in_dim=in_dim, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=gpu_indices,
        dtype=dtype,
    )
    handle.install(torch.device("cuda"))
    return handle, linear


def _make_handle_col(half, in_dim, n_cpu_per_half, dtype=torch.bfloat16):
    out_dim = 2 * half
    n_cpu = 2 * n_cpu_per_half
    linear = _fake_linear(out_dim, in_dim, dtype)
    base = torch.arange(half - n_cpu_per_half, half, dtype=torch.long)
    cpu_indices = torch.cat([base, base + half])
    gpu_indices = _complement(cpu_indices, out_dim)
    handle = CotsLinearHandle(
        kind="col", linear=linear, qualified_name="col.test",
        in_dim=in_dim, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=gpu_indices,
        dtype=dtype, merged_partition_sizes=(half, half),
    )
    handle.install(torch.device("cuda"))
    return handle, linear


def _make_handle_qkv(q_size, kv_size, n_cpu, head_dim, in_dim,
                     dtype=torch.bfloat16):
    out_dim = q_size + 2 * kv_size
    linear = _fake_linear(out_dim, in_dim, dtype)
    cpu_indices = _qkv_kv_biased_indices(
        q_size, kv_size, n_cpu, head_dim=head_dim,
    )
    gpu_indices = _complement(cpu_indices, out_dim)
    handle = CotsLinearHandle(
        kind="qkv", linear=linear, qualified_name="qkv.test",
        in_dim=in_dim, out_dim=out_dim, n_cpu=n_cpu,
        cpu_indices=cpu_indices, gpu_indices=gpu_indices,
        dtype=dtype, q_size=q_size, kv_size=kv_size, head_dim=head_dim,
    )
    handle.install(torch.device("cuda"))
    return handle, linear


# ---------------------------------------------------------------------------
# Row loader
# ---------------------------------------------------------------------------
def test_row_loader_splits_correctly():
    in_dim, out_dim = 256, 128
    n_cpu = 32
    keep_gpu = in_dim - n_cpu
    handle, linear = _make_handle_row(in_dim, out_dim, n_cpu)

    torch.manual_seed(0)
    loaded = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device="cuda")
    linear.weight_loader(linear.weight, loaded)

    assert linear.weight.data.shape == (out_dim, keep_gpu)
    assert torch.equal(linear.weight.data, loaded[:, :keep_gpu])
    assert torch.equal(handle.w_cpu, loaded[:, keep_gpu:].cpu())


def test_row_loader_does_not_retain_loaded_weight():
    """Two consecutive loads should write through cleanly — no stale ref."""
    in_dim, out_dim = 256, 128
    n_cpu = 32
    keep_gpu = in_dim - n_cpu
    handle, linear = _make_handle_row(in_dim, out_dim, n_cpu)

    torch.manual_seed(0)
    loaded1 = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device="cuda")
    linear.weight_loader(linear.weight, loaded1)
    loaded2 = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device="cuda")
    linear.weight_loader(linear.weight, loaded2)

    assert torch.equal(linear.weight.data, loaded2[:, :keep_gpu])
    assert torch.equal(handle.w_cpu, loaded2[:, keep_gpu:].cpu())
    assert not torch.equal(handle.w_cpu, loaded1[:, keep_gpu:].cpu())


# ---------------------------------------------------------------------------
# Merged-col loader (gate_up_proj)
# ---------------------------------------------------------------------------
def test_merged_col_loader_splits_per_shard():
    half, in_dim = 256, 128
    n_cpu_per_half = 32
    keep_gpu = half - n_cpu_per_half
    handle, linear = _make_handle_col(half, in_dim, n_cpu_per_half)

    torch.manual_seed(0)
    gate = torch.randn(half, in_dim, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(half, in_dim, dtype=torch.bfloat16, device="cuda")
    linear.weight_loader(linear.weight, gate, 0)
    linear.weight_loader(linear.weight, up, 1)

    # GPU param.data: [gate_gpu | up_gpu] stacked.
    assert torch.equal(linear.weight.data[:keep_gpu, :], gate[:keep_gpu, :])
    assert torch.equal(linear.weight.data[keep_gpu:, :], up[:keep_gpu, :])
    # w_cpu: [gate_cpu | up_cpu] stacked.
    assert torch.equal(handle.w_cpu[:n_cpu_per_half, :], gate[keep_gpu:, :].cpu())
    assert torch.equal(handle.w_cpu[n_cpu_per_half:, :], up[keep_gpu:, :].cpu())


def test_merged_col_loader_rejects_bad_shard_id():
    handle, linear = _make_handle_col(half=64, in_dim=32, n_cpu_per_half=8)
    bogus = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="loaded_shard_id"):
        linear.weight_loader(linear.weight, bogus, 5)


# ---------------------------------------------------------------------------
# QKV loader (qkv_proj) — K/V-biased + head-aligned
# ---------------------------------------------------------------------------
def test_qkv_loader_typical_f_q_all_gpu():
    """At Qwen2.5-7B (q=3584, kv=512, head_dim=128), n_cpu=512 (2 KV pairs):
    all Q stays on GPU; K and V each get last 256 rows on CPU.
    """
    q_size, kv_size, head_dim = 3584, 512, 128
    n_cpu = 2 * 2 * head_dim  # 2 KV pairs (head-aligned)
    n_k = n_v = 2 * head_dim
    in_dim = 64

    handle, linear = _make_handle_qkv(q_size, kv_size, n_cpu, head_dim, in_dim)

    keep_q = q_size
    keep_k = kv_size - n_k
    keep_v = kv_size - n_v

    torch.manual_seed(0)
    q = torch.randn(q_size, in_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(kv_size, in_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(kv_size, in_dim, dtype=torch.bfloat16, device="cuda")
    linear.weight_loader(linear.weight, q, "q")
    linear.weight_loader(linear.weight, k, "k")
    linear.weight_loader(linear.weight, v, "v")

    # GPU param.data: [Q_gpu | K_gpu | V_gpu] stacked.
    assert torch.equal(linear.weight.data[:keep_q, :], q[:keep_q, :])
    assert torch.equal(
        linear.weight.data[keep_q : keep_q + keep_k, :], k[:keep_k, :]
    )
    assert torch.equal(
        linear.weight.data[keep_q + keep_k :, :], v[:keep_v, :]
    )
    # w_cpu: [Q_tail (empty) | K_cpu | V_cpu]
    assert torch.equal(handle.w_cpu[:n_k, :], k[keep_k:, :].cpu())
    assert torch.equal(handle.w_cpu[n_k : n_k + n_v, :], v[keep_v:, :].cpu())


def test_qkv_loader_high_f_with_q_tail():
    """At f≈0.5 (n_cpu=2304, head_dim=128): full K+V (1024) + Q tail
    of 1280 cols (10 whole heads). All K/V on CPU; Q's last 1280 rows on CPU.
    """
    q_size, kv_size, head_dim = 3584, 512, 128
    n_cpu = 2304  # = 1024 + 1280, all head-aligned
    n_q_tail = 1280
    in_dim = 64
    handle, linear = _make_handle_qkv(q_size, kv_size, n_cpu, head_dim, in_dim)
    keep_q = q_size - n_q_tail
    keep_k = 0
    keep_v = 0

    torch.manual_seed(0)
    q = torch.randn(q_size, in_dim, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(kv_size, in_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(kv_size, in_dim, dtype=torch.bfloat16, device="cuda")
    linear.weight_loader(linear.weight, q, "q")
    linear.weight_loader(linear.weight, k, "k")
    linear.weight_loader(linear.weight, v, "v")

    # GPU param.data: only Q[:keep_q].
    assert linear.weight.data.shape == (keep_q, in_dim)
    assert torch.equal(linear.weight.data, q[:keep_q, :])

    # w_cpu: [Q_tail | K_full | V_full]
    assert torch.equal(handle.w_cpu[:n_q_tail, :], q[keep_q:, :].cpu())
    assert torch.equal(handle.w_cpu[n_q_tail : n_q_tail + kv_size, :], k.cpu())
    assert torch.equal(handle.w_cpu[n_q_tail + kv_size :, :], v.cpu())
    del keep_k, keep_v


def test_qkv_loader_rejects_bad_shard_id():
    handle, linear = _make_handle_qkv(
        q_size=64, kv_size=16, n_cpu=4, head_dim=2, in_dim=32,
    )
    del handle
    bogus = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="loaded_shard_id"):
        linear.weight_loader(linear.weight, bogus, "z")


# ---------------------------------------------------------------------------
# Install-side checks
# ---------------------------------------------------------------------------
def test_install_pins_w_cpu_and_replaces_param():
    handle, linear = _make_handle_row(in_dim=128, out_dim=64, n_cpu=16)
    assert handle.w_cpu is not None
    assert handle.w_cpu.is_pinned()
    assert handle.w_cpu.shape == (64, 16)
    assert handle.w_cpu.dtype == torch.bfloat16
    assert handle.cpu_indices_cuda is not None
    assert handle.cpu_indices_cuda.device.type == "cuda"
    assert handle.gpu_indices_cuda is not None
    assert handle.gpu_indices_cuda.device.type == "cuda"
    # param.data was replaced with the GPU slice.
    assert linear.weight.data.shape == (64, 128 - 16)
    # The handle is reachable from the linear (used by op installers).
    assert linear._cots_handle is handle
