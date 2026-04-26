#!/usr/bin/env python3
"""Phase 0.3 — CPU/GPU overlap visualization probe (nsys-driven).

Visual confirmation of the §0.3.3 claim: at WQKV / f_cpu=9% / B=1 decode,
the CPU's `F.linear` call (running on the host) and the GPU's cuBLAS matmul
(running on the device) overlap in wall-clock time.

Three NVTX-tagged phases let nsys pick them out of the timeline:
    A_gpu_only    GPU runs full WQKV matmul; baseline GPU-side bar.
    B_cpu_only    CPU runs f * WQKV via F.linear; baseline CPU-side bar.
    C_concurrent  GPU runs (1-f) on stream A while host runs f * WQKV.

Run:
    nsys profile -o probe_overlap.nsys-rep \\
        --trace=cuda,nvtx --force-overwrite=true \\
        python probe_overlap.py
    nsys-ui probe_overlap.nsys-rep   # GUI inspection on host

What to look for in the GUI:
    Phase A: a single cuBLAS gemm bar on the CUDA HW row, no NVTX activity
             on the host CPU row inside the phase.
    Phase B: an NVTX `cpu_F_linear` range on the host row with no CUDA HW
             activity inside it.
    Phase C: cuBLAS gemm bar AND `cpu_F_linear` host-NVTX range overlap in
             time. Their start times are within μs and they coexist for the
             full duration of whichever finishes last.

This probe is intentionally short (≈1 second of timeline) for clean GUI
loading. Re-run only if the §0.3.3 claim needs visual re-verification on
different hardware.
"""
import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F


# Qwen2.5-7B WQKV shape: hidden=3584, qkv_out=(28+2*4)*128=4608.
HIDDEN = 3584
QKV_OUT = 4608
F_CPU = 0.09
B = 1

cpu_cols = int(QKV_OUT * F_CPU)
gpu_cols = QKV_OUT - cpu_cols

# WQKV col-parallel: GPU holds (1-f) cols, CPU holds f cols.
W_full_gpu = torch.empty(QKV_OUT, HIDDEN, dtype=torch.bfloat16, device="cuda")
W_gpu = torch.empty(gpu_cols, HIDDEN, dtype=torch.bfloat16, device="cuda")
W_cpu = torch.empty(cpu_cols, HIDDEN, dtype=torch.bfloat16)  # CPU memory
x_cpu = torch.empty(B, HIDDEN, dtype=torch.bfloat16)
x_gpu = torch.empty(B, HIDDEN, dtype=torch.bfloat16, device="cuda")

s_gpu = torch.cuda.Stream()


def gpu_full():
    with torch.cuda.stream(s_gpu):
        return F.linear(x_gpu, W_full_gpu)


def gpu_partial():
    with torch.cuda.stream(s_gpu):
        return F.linear(x_gpu, W_gpu)


def cpu_partial():
    nvtx.range_push("cpu_F_linear")
    out = F.linear(x_cpu, W_cpu)
    nvtx.range_pop()
    return out


# Warmup
torch.cuda.synchronize()
for _ in range(5):
    gpu_full()
    cpu_partial()
torch.cuda.synchronize()

# Phase A: GPU only, full WQKV. Baseline GPU-side bar.
nvtx.range_push("A_gpu_only")
for _ in range(5):
    gpu_full()
torch.cuda.synchronize()
nvtx.range_pop()

# Phase B: CPU only, f * WQKV via F.linear. Baseline CPU-side bar.
nvtx.range_push("B_cpu_only")
for _ in range(5):
    cpu_partial()
nvtx.range_pop()

# Phase C: concurrent. GPU dispatches (1-f) on s_gpu; host runs f * WQKV in
# parallel. The CUDA HW bar (cuBLAS) and the NVTX `cpu_F_linear` range should
# overlap in wall-clock time on the timeline.
nvtx.range_push("C_concurrent")
for _ in range(5):
    gpu_partial()           # GPU dispatch returns immediately (async)
    cpu_partial()           # host blocks on CPU matmul; GPU runs in parallel
torch.cuda.synchronize()
nvtx.range_pop()

print("done")
