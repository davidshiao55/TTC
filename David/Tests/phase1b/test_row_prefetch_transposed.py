"""Phase 1b row-prefetch contention fix: tests for the transposed
pinned-CPU prefetch source.

Verifies the fix for the Bench 2 collaborative-arm slowdown:
  - Step 1: allocation only happens for kind=row with max_n_prefetch>0.
  - Step 2: loader populates w_row_prefetch_src_t such that
            src_t[:m, :].T == w_cpu[:, :m] (bit-exact).
  - Step 3: pool slot shape change for row preserves group sharing.
  - Step 4: contiguous H2D matches the old strided H2D byte-for-byte.
  - Step 5: MLP2 prefetched-GPU consumer (matmul on transposed slot)
            matches the reference math to BF16 tolerance.
  - Sentinel: at f_prefetch=0.0, no row handle has the buffer.
"""

from collections import Counter

import pytest
import torch
import torch.nn as nn

from vllm.config import (
    CompilationConfig,
    CotsOffloadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.offloader import CotsOffloader, set_offloader
from vllm.model_executor.offloader.cots import CotsPrefetchBufferPool

HIDDEN = 256
INTERMEDIATE = 1024
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
MAX_NUM_TOKENS = 64
N_LAYERS = 2

BF16_ATOL = 5e-3
BF16_RTOL = 2e-2


class _MiniMlp(nn.Module):
    def __init__(self):
        super().__init__()
        bf16 = torch.bfloat16
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=HIDDEN,
            output_sizes=[INTERMEDIATE, INTERMEDIATE],
            bias=False, disable_tp=True, params_dtype=bf16,
            prefix="mlp.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=INTERMEDIATE, output_size=HIDDEN,
            bias=False, disable_tp=True, params_dtype=bf16,
            prefix="mlp.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gu, _ = self.gate_up_proj(x)
        x = self.act_fn(gu)
        x, _ = self.down_proj(x)
        return x


class _MiniDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        bf16 = torch.bfloat16
        self.qkv_proj = QKVParallelLinear(
            hidden_size=HIDDEN, head_size=HEAD_DIM,
            total_num_heads=NUM_HEADS, total_num_kv_heads=NUM_KV_HEADS,
            bias=False, disable_tp=True, params_dtype=bf16,
            prefix="qkv_proj",
        )
        self.mlp = _MiniMlp()

    def forward(self, x):
        qkv_out, _ = self.qkv_proj(x)
        qkv_dim = NUM_HEADS * HEAD_DIM
        kv_dim = NUM_KV_HEADS * HEAD_DIM
        q = qkv_out[:, :qkv_dim]
        return x + self.mlp(q) + qkv_out[:, qkv_dim:qkv_dim + kv_dim].sum(
            dim=1, keepdim=True
        ).expand_as(x)


def _make_vllm_config():
    mc = ModelConfig.__new__(ModelConfig)
    object.__setattr__(mc, "enforce_eager", True)
    sc = SchedulerConfig.__new__(SchedulerConfig)
    object.__setattr__(sc, "max_num_batched_tokens", MAX_NUM_TOKENS)
    cc = CompilationConfig.__new__(CompilationConfig)
    object.__setattr__(cc, "cudagraph_capture_sizes", [8, 32, MAX_NUM_TOKENS])
    object.__setattr__(cc, "custom_ops", ["none"])
    object.__setattr__(cc, "enabled_custom_ops", Counter())
    object.__setattr__(cc, "disabled_custom_ops", Counter())
    pc = ParallelConfig.__new__(ParallelConfig)
    object.__setattr__(pc, "tensor_parallel_size", 1)
    vc = VllmConfig.__new__(VllmConfig)
    object.__setattr__(vc, "model_config", mc)
    object.__setattr__(vc, "scheduler_config", sc)
    object.__setattr__(vc, "compilation_config", cc)
    object.__setattr__(vc, "parallel_config", pc)
    return vc


def _populate_weights(layer):
    torch.manual_seed(0)
    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    q = torch.randn(q_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
    layer.qkv_proj.weight_loader(layer.qkv_proj.weight, q, "q")
    layer.qkv_proj.weight_loader(layer.qkv_proj.weight, k, "k")
    layer.qkv_proj.weight_loader(layer.qkv_proj.weight, v, "v")
    gate = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
    down = torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device="cuda")
    layer.mlp.gate_up_proj.weight_loader(
        layer.mlp.gate_up_proj.weight, gate, 0
    )
    layer.mlp.gate_up_proj.weight_loader(
        layer.mlp.gate_up_proj.weight, up, 1
    )
    layer.mlp.down_proj.weight_loader(layer.mlp.down_proj.weight, down)


def _build(f_cpu_store, f_prefetch):
    vc = _make_vllm_config()
    with set_current_vllm_config(vc):
        layers = [_MiniDecoderLayer().cuda() for _ in range(N_LAYERS)]
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                f_prefetch=f_prefetch,
                kv_biased=True,
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter(layers))
        for layer in layers:
            _populate_weights(layer)
        offloader.post_init()
    return layers, offloader


# ---------------------------------------------------------------------------
def test_f_prefetch_zero_does_not_allocate():
    """Sentinel: at f_prefetch=0.0, no row handle gets the transposed
    buffer. Phase 1a behavior must be bit-exactly preserved."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    _, offloader = _build(f_cpu_store=0.20, f_prefetch=0.0)
    row_handles = [h for h in offloader._handles if h.kind == "row"]
    assert row_handles, "expected at least one row handle"
    for h in row_handles:
        assert h.w_row_prefetch_src_t is None
        assert h.max_n_prefetch == 0


def test_allocation_only_for_row_with_prefetch():
    """Step 1: only kind=row with max_n_prefetch>0 gets the buffer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    _, offloader = _build(f_cpu_store=0.20, f_prefetch=0.05)
    for h in offloader._handles:
        if h.kind == "row" and h.max_n_prefetch > 0:
            assert h.w_row_prefetch_src_t is not None
            assert h.w_row_prefetch_src_t.shape == (
                h.max_n_prefetch, h.out_dim,
            )
            assert h.w_row_prefetch_src_t.dtype == h.dtype
            assert h.w_row_prefetch_src_t.is_pinned()
        else:
            assert h.w_row_prefetch_src_t is None


def test_loader_populates_transposed_src():
    """Step 2: w_row_prefetch_src_t[:m, :].T == w_cpu[:, :m] bit-exactly
    after the loader runs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    _, offloader = _build(f_cpu_store=0.20, f_prefetch=0.05)
    for h in offloader._handles:
        if h.kind != "row" or h.max_n_prefetch == 0:
            continue
        m = h.max_n_prefetch
        # Both are bf16 copies from the same loaded_weight slice; bit-exact.
        lhs = h.w_row_prefetch_src_t[:m, :].T.contiguous()
        rhs = h.w_cpu[:, :m].contiguous()
        assert torch.equal(lhs, rhs), (
            f"transposed src mismatch at {h.qualified_name}"
        )


def test_h2d_narrow_matches_old_strided():
    """Step 4: contiguous H2D from the transposed source produces the
    same prefetched values as the old strided H2D from w_cpu would have.
    Verifies the new path doesn't change the data on the GPU side."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    _, offloader = _build(f_cpu_store=0.20, f_prefetch=0.05)
    h = next(
        h for h in offloader._handles
        if h.kind == "row" and h.max_n_prefetch > 0
    )
    n = h.max_n_prefetch

    # Old path: strided H2D from w_cpu.narrow(1, 0, n).
    # The slot's transposed layout means we compare logical (out_dim, n)
    # views: slot_after_h2d.T[:n, :] should equal w_cpu[:, :n].T.
    old_dst = torch.empty(
        (h.out_dim, n), dtype=h.dtype, device="cuda"
    )
    old_dst.copy_(h.w_cpu.narrow(1, 0, n), non_blocking=False)

    # New path: contiguous H2D from w_row_prefetch_src_t.
    new_dst = torch.empty(
        (n, h.out_dim), dtype=h.dtype, device="cuda"
    )
    new_dst.copy_(
        h.w_row_prefetch_src_t.narrow(0, 0, n), non_blocking=False
    )
    torch.cuda.synchronize()

    # New layout transposed back should equal old layout (bit-exact: both
    # paths just copy bytes from the same source weight bf16 buffer).
    assert torch.equal(new_dst.T.contiguous(), old_dst.contiguous())


def test_pool_groups_unchanged_under_new_shape():
    """Step 3: pool buffer size matches the new slot shape per group.
    All N_LAYERS row handles share K=2 slots of (max_n_prefetch, out_dim)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    _, offloader = _build(f_cpu_store=0.20, f_prefetch=0.05)
    pool = offloader._prefetch_buffer_pool
    assert pool is not None and pool._buffer is not None

    # Sum of per-group slot bytes × K must equal pool.total_bytes.
    seen_groups = set()
    expected = 0
    K = CotsPrefetchBufferPool.K
    for h in offloader._handles:
        if h.max_n_prefetch == 0:
            continue
        if h.kind == "row":
            shape = (h.max_n_prefetch, h.out_dim)
        else:
            shape = (h.max_n_prefetch, h.in_dim)
        key = (h.kind, shape)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        elem = h.dtype.itemsize if hasattr(h.dtype, "itemsize") else 2
        expected += K * shape[0] * shape[1] * elem
    assert pool.total_bytes == expected

    # Verify row handles got the new transposed slot shape.
    for h in offloader._handles:
        if h.kind == "row" and h.max_n_prefetch > 0:
            for slot in h.w_prefetch_slots:
                assert slot.shape == (h.max_n_prefetch, h.out_dim)


def test_mlp2_prefetched_matmul_matches_old_flinear():
    """Step 5: the new prefetched-GPU MLP2 path uses
    `pref_silu.matmul(dn_slot[:n,:])` against a (n, out_dim) slot;
    the old path used `F.linear(pref_silu, dn_slot_old)` against a
    (out_dim, n) slot. Both compute z @ Wᵀ, just the layout differs.
    This test populates a fake slot in both layouts with the SAME bytes
    and confirms the matmul/F.linear results agree to BF16 tolerance.

    Independent of the model forward — the slot population in the
    integration tests is handled by the streamer's H2D, which we
    already verify byte-equal in test_h2d_narrow_matches_old_strided.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    T = 8
    out_dim = 256
    dn_n_pref = 32
    dt = torch.bfloat16

    pref_silu = torch.randn(T, dn_n_pref, dtype=dt, device="cuda")
    # Source weight for the prefetched MLP2.
    w = torch.randn(out_dim, dn_n_pref, dtype=dt, device="cuda")

    # Old path: F.linear with weight shape (out_dim, dn_n_pref).
    import torch.nn.functional as F
    old = F.linear(pref_silu, w, None)

    # New path: matmul with weight shape (dn_n_pref, out_dim) — i.e., w.T.
    new = pref_silu.matmul(w.T.contiguous())

    assert old.shape == (T, out_dim) == new.shape
    # cuBLAS BF16 GEMM with FP32 accum, reduction order may differ.
    torch.testing.assert_close(
        old.float(), new.float(), atol=BF16_ATOL, rtol=BF16_RTOL,
    )
