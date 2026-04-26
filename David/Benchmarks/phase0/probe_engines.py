#!/usr/bin/env python3
"""Phase 0.5 — Copy engine attribution probe (nsys-driven).

Verifies that RTX 4090's two async copy engines are direction-specialized
(CE0 = H2D, CE1 = D2H, neither does the other direction). This is the
mechanism behind the same-direction H2D serialization measured in §0.5.1
of phase0_findings.md.

Three NVTX-tagged phases let nsys pick them out of the timeline:
    A_2xH2D            two H2D copies on separate streams (predicted: serialize)
    B_H2D_plus_D2H     one H2D + one D2H concurrent (predicted: overlap)
    C_solo_H2D / D2H   ground-truth solo timings

Run:
    nsys profile -o probe.nsys-rep --trace=cuda,nvtx --force-overwrite=true \\
        python probe_engines.py
    nsys stats probe.nsys-rep --report gputrace | head -80

Look for:
    Phase A: stream-13 H2Ds run back-to-back, then stream-17 H2Ds run
             back-to-back. No overlap → CE0 services both streams sequentially.
    Phase B: H2D and D2H start within μs and overlap for ~full duration.
             Confirms CE0 and CE1 work concurrently on different directions.

This probe is a one-shot verification, not part of the regular benchmark
sweep — re-run only if the underlying CUDA driver / GPU changes.
"""
import torch
import torch.cuda.nvtx as nvtx

N = 4 * 1024 * 1024  # 8 MB BF16 each


def make_pair():
    cpu = torch.empty(N, dtype=torch.bfloat16, pin_memory=True)
    gpu = torch.empty_like(cpu, device="cuda")
    return cpu, gpu


a_cpu, a_gpu = make_pair()
b_cpu, b_gpu = make_pair()
c_cpu, c_gpu = make_pair()

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

torch.cuda.synchronize()
for _ in range(3):
    a_gpu.copy_(a_cpu, non_blocking=True)
    b_gpu.copy_(b_cpu, non_blocking=True)
    a_cpu.copy_(a_gpu, non_blocking=True)
torch.cuda.synchronize()

# Phase A: 2× H2D concurrent — predicted to serialize on CE0.
nvtx.range_push("A_2xH2D")
for _ in range(5):
    with torch.cuda.stream(s1):
        a_gpu.copy_(a_cpu, non_blocking=True)
    with torch.cuda.stream(s2):
        b_gpu.copy_(b_cpu, non_blocking=True)
torch.cuda.synchronize()
nvtx.range_pop()

# Phase B: H2D + D2H concurrent — predicted to overlap on CE0 + CE1.
nvtx.range_push("B_H2D_plus_D2H")
for _ in range(5):
    with torch.cuda.stream(s1):
        a_gpu.copy_(a_cpu, non_blocking=True)
    with torch.cuda.stream(s2):
        c_cpu.copy_(c_gpu, non_blocking=True)
torch.cuda.synchronize()
nvtx.range_pop()

# Phase C: ground-truth solo timings.
nvtx.range_push("C_solo_H2D")
for _ in range(5):
    a_gpu.copy_(a_cpu, non_blocking=True)
torch.cuda.synchronize()
nvtx.range_pop()

nvtx.range_push("C_solo_D2H")
for _ in range(5):
    a_cpu.copy_(a_gpu, non_blocking=True)
torch.cuda.synchronize()
nvtx.range_pop()

print("done")
