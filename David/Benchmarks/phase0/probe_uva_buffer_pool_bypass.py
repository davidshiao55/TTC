#!/usr/bin/env python3
"""Phase 0.10 — Probe whether vLLM's UvaBufferPool.copy_to_gpu bypasses CE0.

§0.10.3 root-causes prefetch's missing free regime to CE0 FIFO contention
between weight prefetch H2D and per-step input-prep H2Ds. The proposed
validation re-routes input prep through a UVA path. vLLM's
`vllm.v1.worker.gpu.buffer_utils.UvaBufferPool.copy_to_gpu` looks like
exactly the right wrapper, but its final step is
`out.copy_(uva, non_blocking=True)` — that may STILL route through CE0
since PyTorch sees a CUDA-device→CUDA-device copy.

This probe answers: does `UvaBufferPool.copy_to_gpu` actually bypass CE0?

Five NVTX-tagged phases:
    A_dma_fg_under_bg           fg = cudaMemcpyAsync, bg = 4 MB DMA → expect CE0 serial
    B_uva_kernel_fg_under_bg    fg = Triton _copy_k (proven bypass) → expect overlap
    D_uva_pool_fg_under_bg      fg = UvaBufferPool.copy_to_gpu       → ??? (this probe)
    C_solo_dma_fg               fg DMA alone (baseline)
    C_solo_uva_pool_fg          fg UVA-pool alone (baseline)

Run:
    nsys profile -o probe_uva_pool.nsys-rep --trace=cuda,nvtx --force-overwrite=true \
        python probe_uva_buffer_pool_bypass.py
    nsys stats probe_uva_pool.nsys-rep --report gputrace | head -120

What to check:
    Phase D bg+fg events: do they overlap in time, or does fg wait for bg?
    Compare engine attribution (CE0 vs SM stream) to phase A (queues on CE0)
    and phase B (overlaps on different path).

If phase D looks like phase B → UvaBufferPool.copy_to_gpu DOES bypass CE0,
we can wire it directly into vLLM's input prep with no Triton kernel.

If phase D looks like phase A → it does NOT bypass CE0, and the actual fix
in P1B needs to route input prep through a Triton-kernel-based copy
mirroring _copy_k from probe_uva_bypass.py.
"""
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool


@triton.jit
def _copy_k(in_ptr, out_ptr, n, BS: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BS + tl.arange(0, BS)
    m = off < n
    tl.store(out_ptr + off, tl.load(in_ptr + off, mask=m), mask=m)


# UvaBufferPool's CPU-staging step uses .numpy(), so dtype must be numpy-compatible.
# Real vLLM input prep tensors are int32 (input_ids, positions) — use that.
N_TOKENS = 25_000  # ~100 KB at int32 = matches §0.10 input-prep size band
n_act = N_TOKENS

# Path A: classic DMA fg
act_cpu = torch.empty(n_act, dtype=torch.int32, pin_memory=True)
act_gpu = torch.empty_like(act_cpu, device="cuda")

# Path B: Triton kernel reading UVA-mapped pinned src (proven bypass)
act_uva = get_accelerator_view_from_cpu_tensor(act_cpu)
sm_out = torch.empty_like(act_gpu)

# Path D: vLLM's UvaBufferPool wrapper
pool = UvaBufferPool(size=n_act, dtype=torch.int32, max_concurrency=2)
pool_src_cpu = torch.empty(n_act, dtype=torch.int32, pin_memory=True)
pool_dst_gpu = torch.empty_like(pool_src_cpu, device="cuda")

# Background load: continuous 4 MB DMA H2D queued on copy_stream (mirrors prefetch)
bg_n = (4 * 1024 * 1024) // 2  # 4 MB BF16
bg_cpu = torch.empty(bg_n, dtype=torch.bfloat16, pin_memory=True)
bg_gpu = torch.empty_like(bg_cpu, device="cuda")

s_act = torch.cuda.Stream()
s_bg = torch.cuda.Stream()


def uva_kernel_copy():
    grid = (triton.cdiv(n_act, 1024),)
    _copy_k[grid](act_uva, sm_out, n_act, BS=1024)


def uva_pool_copy():
    pool.copy_to_gpu(pool_src_cpu, out=pool_dst_gpu)


# Warmup all paths
torch.cuda.synchronize()
for _ in range(3):
    act_gpu.copy_(act_cpu, non_blocking=True)
    bg_gpu.copy_(bg_cpu, non_blocking=True)
    uva_kernel_copy()
    uva_pool_copy()
torch.cuda.synchronize()

# Phase A: DMA fg under bg DMA → expect CE0 FIFO serialization
nvtx.range_push("A_dma_fg_under_bg")
for _ in range(3):
    with torch.cuda.stream(s_bg):
        bg_gpu.copy_(bg_cpu, non_blocking=True)
    with torch.cuda.stream(s_act):
        act_gpu.copy_(act_cpu, non_blocking=True)
torch.cuda.synchronize()
nvtx.range_pop()

# Phase B: Triton UVA-kernel fg under bg DMA → expect SM/CE0 concurrency
nvtx.range_push("B_uva_kernel_fg_under_bg")
for _ in range(3):
    with torch.cuda.stream(s_bg):
        bg_gpu.copy_(bg_cpu, non_blocking=True)
    with torch.cuda.stream(s_act):
        uva_kernel_copy()
torch.cuda.synchronize()
nvtx.range_pop()

# Phase D: vLLM UvaBufferPool.copy_to_gpu fg under bg DMA → THIS PROBE
nvtx.range_push("D_uva_pool_fg_under_bg")
for _ in range(3):
    with torch.cuda.stream(s_bg):
        bg_gpu.copy_(bg_cpu, non_blocking=True)
    with torch.cuda.stream(s_act):
        uva_pool_copy()
torch.cuda.synchronize()
nvtx.range_pop()

# Phase C: solo baselines
nvtx.range_push("C_solo_dma_fg")
for _ in range(3):
    act_gpu.copy_(act_cpu, non_blocking=True)
torch.cuda.synchronize()
nvtx.range_pop()

nvtx.range_push("C_solo_uva_pool_fg")
for _ in range(3):
    uva_pool_copy()
torch.cuda.synchronize()
nvtx.range_pop()

print("done")
