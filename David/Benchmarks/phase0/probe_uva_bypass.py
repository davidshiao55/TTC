#!/usr/bin/env python3
"""Phase 0.5 — UVA bypass visualization probe (nsys-driven).

Demonstrates the central §0.5 finding visually: under continuous bg DMA prefetch
on CE0, fg via `cudaMemcpyAsync` queues behind bg in the H2D engine FIFO, while
fg via an SM-issued UVA copy kernel runs concurrently (different hardware path
through the GPU MMU).

Three NVTX-tagged phases let nsys pick them out of the timeline:
    A_dma_fg_under_bg       fg = cudaMemcpyAsync, bg = 4 MB DMA → expect serial
    B_uva_fg_under_bg       fg = Triton UVA copy kernel, bg = 4 MB DMA → expect overlap
    C_solo_baselines        fg only (DMA, then UVA), no bg → ground-truth durations

Run:
    nsys profile -o probe_uva.nsys-rep --trace=cuda,nvtx --force-overwrite=true \\
        python probe_uva_bypass.py
    nsys stats probe_uva.nsys-rep --report gputrace | head -60

What to look for in the trace:
    Phase A: bg memcpy on stream X runs to completion, THEN fg memcpy on stream Y
             runs. No overlap. Pure CE0 FIFO serialization.
    Phase B: bg memcpy on stream X is in flight; the Triton `_copy_k` kernel on
             stream Y starts INSIDE bg's window and overlaps for its full
             duration. Different hardware paths to the link.
    Phase C: fg kernel/memcpy run alone — gives the isolated baseline durations.

This probe is intentionally short (≈1 second of timeline) so the GUI loads
clean. Re-run only if you need to visually verify the bypass on different
hardware or after CUDA driver changes.
"""
import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


@triton.jit
def _copy_k(in_ptr, out_ptr, n, BS: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BS + tl.arange(0, BS)
    m = off < n
    tl.store(out_ptr + off, tl.load(in_ptr + off, mask=m), mask=m)


HIDDEN, N_TOKENS = 3584, 14
n_act = N_TOKENS * HIDDEN  # 98 KB BF16

act_cpu = torch.empty(n_act, dtype=torch.bfloat16, pin_memory=True)
act_gpu = torch.empty_like(act_cpu, device="cuda")
act_uva = get_accelerator_view_from_cpu_tensor(act_cpu)
sm_out = torch.empty_like(act_gpu)

bg_n = (4 * 1024 * 1024) // 2  # 4 MB
bg_cpu = torch.empty(bg_n, dtype=torch.bfloat16, pin_memory=True)
bg_gpu = torch.empty_like(bg_cpu, device="cuda")

s_act = torch.cuda.Stream()
s_bg = torch.cuda.Stream()


def uva_copy():
    grid = (triton.cdiv(n_act, 1024),)
    _copy_k[grid](act_uva, sm_out, n_act, BS=1024)


# Warmup
torch.cuda.synchronize()
for _ in range(3):
    act_gpu.copy_(act_cpu, non_blocking=True)
    bg_gpu.copy_(bg_cpu, non_blocking=True)
    uva_copy()
torch.cuda.synchronize()

# Phase A: DMA fg under bg DMA → expect CE0 serialization.
nvtx.range_push("A_dma_fg_under_bg")
for _ in range(3):
    with torch.cuda.stream(s_bg):
        bg_gpu.copy_(bg_cpu, non_blocking=True)
    with torch.cuda.stream(s_act):
        act_gpu.copy_(act_cpu, non_blocking=True)
torch.cuda.synchronize()
nvtx.range_pop()

# Phase B: UVA fg under bg DMA → expect SM/CE0 concurrency.
nvtx.range_push("B_uva_fg_under_bg")
for _ in range(3):
    with torch.cuda.stream(s_bg):
        bg_gpu.copy_(bg_cpu, non_blocking=True)
    with torch.cuda.stream(s_act):
        uva_copy()
torch.cuda.synchronize()
nvtx.range_pop()

# Phase C: solo baselines.
nvtx.range_push("C_solo_dma_fg")
for _ in range(3):
    act_gpu.copy_(act_cpu, non_blocking=True)
torch.cuda.synchronize()
nvtx.range_pop()

nvtx.range_push("C_solo_uva_fg")
for _ in range(3):
    uva_copy()
torch.cuda.synchronize()
nvtx.range_pop()

print("done")
