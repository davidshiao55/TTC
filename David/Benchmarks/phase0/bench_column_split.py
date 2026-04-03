#!/usr/bin/env python3
"""Phase 0.4 — TP-Style Column Split Correctness Verification

Verifies that tensor-parallel-style column split (along output dimension)
between GPU and CPU produces numerically equivalent results to unsplit
computation. Same split dimension as standard multi-GPU TP.

GPU portion uses F.linear on CUDA, CPU portion uses F.linear on CPU
(oneDNN BF16 path). Results are concatenated and compared to the
unsplit F.linear on GPU.

Tests Qwen2.5-7B sub-module shapes at both arbitrary split (f_cpu=9%)
and Q|K|V semantic boundary split.

Run from anywhere:
    python David/Benchmarks/phase0/bench_column_split.py
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Qwen2.5-7B dimensions
# ---------------------------------------------------------------------------
HIDDEN = 3584
NUM_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
INTERMEDIATE = 18944

Q_DIM = NUM_HEADS * HEAD_DIM
K_DIM = NUM_KV_HEADS * HEAD_DIM
V_DIM = NUM_KV_HEADS * HEAD_DIM
QKV_DIM = Q_DIM + K_DIM + V_DIM
GATE_UP_DIM = 2 * INTERMEDIATE

# (name, in_dim, out_dim) — out_dim is the TP split dimension
SUBMODULES = [
    ("WQKV",  HIDDEN,       QKV_DIM),
    ("WO",    HIDDEN,       HIDDEN),
    ("MLP1",  HIDDEN,       GATE_UP_DIM),
    ("MLP2",  INTERMEDIATE, HIDDEN),
]

F_CPU = 0.09
BATCH_SIZES = [1, 8, 32]


def test_column_split(name, in_dim, out_dim, cpu_out, B):
    """Test that GPU-CPU TP split+concat == unsplit for BF16.

    Same operation as multi-GPU TP: split weight along output dimension,
    each device computes its portion, results are concatenated.
    Differences come from floating-point eval order between GPU and CPU
    kernels, not from the split itself.
    """
    gpu_out = out_dim - cpu_out

    # Weight in (out_features, in_features) format for F.linear
    W_full = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(B, in_dim, dtype=torch.bfloat16, device="cuda")

    # Unsplit: full computation on GPU (TP=1 baseline)
    y_full = F.linear(x, W_full)

    # Split: GPU gets first gpu_out rows, CPU gets rest (like TP=2)
    W_gpu = W_full[:gpu_out, :]
    W_cpu = W_full[gpu_out:, :].cpu()

    y_gpu = F.linear(x, W_gpu)
    y_cpu = F.linear(x.cpu(), W_cpu)

    # Concat results (same as TP all-gather along output dim)
    y_split = torch.cat([y_gpu, y_cpu.cuda()], dim=-1)

    # Compare
    exact = torch.equal(y_full, y_split)
    max_diff = (y_full - y_split).abs().max().item()

    # BF16 has 7-bit mantissa. Large reductions (dim=3584+) with different
    # eval order (GPU vs CPU kernel) produce diffs up to a few ULP of
    # output magnitude. Allow up to 1% relative error.
    output_scale = y_full.abs().max().item()
    close = max_diff <= output_scale * 0.01

    return exact, close, max_diff


def main():
    print("TP-Style Column Split Correctness (BF16, F.linear)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU: F.linear → oneDNN BF16 path")
    print(f"Split dimension: output columns (same as multi-GPU TP)")
    print(f"{'='*70}")

    all_passed = True

    # Test 1: Arbitrary column split (f_cpu=9%)
    print(f"\n--- Arbitrary split (f_cpu={F_CPU:.0%}) ---")
    print(f"{'Sub-module':<12} {'B':>4} {'GPU+CPU=Total':>16} {'Result':>8} {'Max diff':>12}")
    print("-" * 55)

    for name, in_dim, out_dim in SUBMODULES:
        cpu_out = max(1, int(out_dim * F_CPU))
        gpu_out = out_dim - cpu_out
        for B in BATCH_SIZES:
            exact, close, max_diff = test_column_split(
                name, in_dim, out_dim, cpu_out, B)
            if not close:
                all_passed = False
            print(f"{name:<12} {B:>4} {f'{gpu_out}+{cpu_out}={out_dim}':>16} "
                  f"{'EXACT' if exact else 'close' if close else 'FAIL':>8} "
                  f"{max_diff:>12.2e}")

    # Test 2: Q|K|V semantic boundary split
    print(f"\n--- WQKV Q|K|V semantic split ---")
    print(f"{'Sub-module':<12} {'B':>4} {'GPU+CPU=Total':>16} {'Result':>8} {'Max diff':>12}")
    print("-" * 55)
    cpu_out = K_DIM + V_DIM  # 1024
    gpu_out = Q_DIM          # 3584
    for B in BATCH_SIZES:
        exact, close, max_diff = test_column_split(
            "WQKV_QKV", HIDDEN, QKV_DIM, cpu_out, B)
        if not close:
            all_passed = False
        print(f"{'WQKV_QKV':<12} {B:>4} {f'Q={gpu_out}+KV={cpu_out}':>16} "
              f"{'EXACT' if exact else 'close' if close else 'FAIL':>8} "
              f"{max_diff:>12.2e}")

    print(f"\n{'='*70}")
    if all_passed:
        print("ALL TESTS PASSED — GPU-CPU TP split is numerically equivalent.")
        print("(Differences are from GPU vs CPU eval order, not the split.)")
    else:
        print("SOME TESTS FAILED — differences exceed tolerance.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
