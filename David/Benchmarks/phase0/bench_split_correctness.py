#!/usr/bin/env python3
"""Phase 0.2 — Tensor Split Correctness (mixed col/row per TP convention)

Verifies that per-sub-module GPU-CPU splits produce numerically equivalent
results to unsplit computation, covering the full mechanism space used by
Phase 1: col-parallel, row-parallel, and the MLP1→MLP2 col→row pipeline that
keeps the intermediate activation local to each device.

Design reference: `weight_offload_design.md §Per-Sub-Module Split Axis`.
Current axis assignment:
  WQKV → col (K/V-biased picker)
  MLP1 → col
  MLP2 → row
  WO   → col if Alt A wins in §0.4.2, else no offload (not split)

Test families:

  A. Col-parallel contiguous — WQKV / MLP1 / (WO if Alt A). CPU gets the last
     f·out_dim output rows; assembly is concat.

  B. Col-parallel K/V-biased — WQKV only. KV-head groups first, then Q heads
     from the tail. Non-contiguous CPU selection; assembly via index_copy.

  C. Row-parallel contiguous — MLP2. CPU gets the last f·in_dim input cols of
     W (and matching cols of x); each device produces a [B, out_dim] partial
     sum; assembly is `add_` (matches vLLM `RowParallelLinear.reduce_results`).

  D. MLP1→MLP2 col→row pipeline — end-to-end test that, with matching
     intermediate index selection, each device applies SwiGLU on its local
     slice and MLP2 consumes the local slice without any intermediate
     transfer. Result must match an unsplit MLP block.

Run from anywhere:

    python David/Benchmarks/phase0/bench_split_correctness.py
    python David/Benchmarks/phase0/bench_split_correctness.py --model prm1p5b
"""

import argparse
import sys

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model configs (mirror bench_cpu_gpu_overlap.py)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "qwen7b": {
        "display_name": "Qwen2.5-7B-Instruct",
        "hidden": 3584,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "intermediate": 18944,
    },
    "prm1p5b": {
        "display_name": "Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "hidden": 1536,
        "num_heads": 12,
        "num_kv_heads": 2,
        "head_dim": 128,
        "intermediate": 8960,
    },
}

F_CPU_SWEEP = [0.03, 0.09, 0.15, 0.22, 0.30, 0.50]
BATCH_SIZES = [1, 8, 32]


# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
def _max_diff(y_ref, y_test):
    """Returns (exact, close, max_diff, output_scale).

    Tolerance: 2% × output_scale. BF16 has a 7-bit mantissa (~1.5% ulp at the
    top of the exponent range). Large reductions (inner dim up to 18944) with
    different evaluation orders on GPU vs CPU can amplify this to ~2%.
    """
    exact = torch.equal(y_ref, y_test)
    max_diff = (y_ref - y_test).abs().max().item()
    output_scale = y_ref.abs().max().item()
    close = max_diff <= max(output_scale * 0.02, 1e-6)
    return exact, close, max_diff, output_scale


# ---------------------------------------------------------------------------
# K/V-biased column picker (WQKV-specific)
# ---------------------------------------------------------------------------
def kv_biased_cpu_columns(cfg, f):
    """Return CPU output column indices for WQKV at the given f_cpu.

    WQKV output layout: [Q_0..Q_{H-1} | K_0..K_{G-1} | V_0..V_{G-1}], where H
    is `num_heads`, G is `num_kv_heads`, and each head spans `head_dim` columns.

    Priority for CPU slice: **KV-head groups** (K_j and V_j paired per j)
    first, then Q heads. This keeps K_j and V_j co-located on the CPU so the
    Phase 2 CPU suffix-attention kernel can compute the head-group's attention
    without pulling the paired half back across PCIe.

    Quantization: the picker's configuration space is
        (kv_groups_on_cpu, q_heads_on_cpu) ∈ [0..G] × [0..H]
    with the priority invariant q_heads_on_cpu = 0 while kv_groups_on_cpu < G.
    The effective column count is `kv_groups_on_cpu * 2 * head_dim +
    q_heads_on_cpu * head_dim`. We pick the configuration whose effective
    column count is closest to `round(f * qkv_dim)`. This rounds toward the
    nearest valid head-group boundary — requested f values between allowed
    configurations snap to the closer one (no sub-head splits are legal,
    since they would break GQA head-group alignment).

    Returns: sorted list of column indices in [0, qkv_dim).
    """
    num_q_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim

    if f <= 0:
        return []

    target_cols = f * qkv_dim
    best_cfg = (0, 0)
    best_diff = float("inf")
    for kv_groups in range(num_kv_heads + 1):
        # Priority rule: no Q heads until all KV groups are on CPU.
        q_range = (range(num_q_heads + 1) if kv_groups == num_kv_heads
                   else [0])
        for q_heads in q_range:
            cols = kv_groups * 2 * head_dim + q_heads * head_dim
            diff = abs(cols - target_cols)
            if diff < best_diff:
                best_diff = diff
                best_cfg = (kv_groups, q_heads)
    kv_groups_on_cpu, q_heads_on_cpu = best_cfg

    cpu_idx = []
    # KV-head groups: K_j then V_j per j, j = 0..kv_groups_on_cpu-1
    for j in range(kv_groups_on_cpu):
        k_start = q_dim + j * head_dim
        v_start = q_dim + kv_dim + j * head_dim
        cpu_idx.extend(range(k_start, k_start + head_dim))
        cpu_idx.extend(range(v_start, v_start + head_dim))
    # Q heads: from the tail of the Q block (last head first). Q_0 is the
    # last to move to CPU, preserving low-index Q locality on GPU.
    for h in range(q_heads_on_cpu):
        q_start = q_dim - (h + 1) * head_dim
        cpu_idx.extend(range(q_start, q_start + head_dim))

    # Sorted indices so downstream index_select / scatter are deterministic.
    return sorted(cpu_idx)


def kv_biased_boundary(cfg):
    """f at which the strict Q|K|V boundary emerges (all K+V on CPU, all Q on GPU)."""
    q_dim = cfg["num_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_kv_heads"] * cfg["head_dim"]
    qkv_dim = q_dim + 2 * kv_dim
    return 2 * kv_dim / qkv_dim


def validate_picker(cfg):
    """Structural checks on the picker (no hardware calls).

    Invariants:
      1. At boundary f = 2*kv_dim/qkv_dim: cpu_idx == [q_dim, q_dim + 2*kv_dim).
      2. Below boundary: cpu_idx ⊂ K+V block, and K_j present ↔ V_j present.
      3. Above boundary: cpu_idx ⊃ full K+V block + whole Q heads from tail.
      4. All CPU selections are `head_dim`-aligned.
    """
    num_q_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    boundary = kv_biased_boundary(cfg)

    def present_heads(idx, block_start, block_end, head_dim):
        out = []
        s = set(idx)
        n_heads = (block_end - block_start) // head_dim
        for h in range(n_heads):
            h_start = block_start + h * head_dim
            if all((h_start + c) in s for c in range(head_dim)):
                out.append(h)
        return out

    # (1) at the boundary
    idx = kv_biased_cpu_columns(cfg, boundary)
    assert idx == list(range(q_dim, qkv_dim)), (
        f"boundary picker should be exactly [q_dim, qkv_dim), got {idx[:5]}…")

    # (2) below the boundary — KV-group pairing
    idx = kv_biased_cpu_columns(cfg, boundary * 0.5)
    assert all(q_dim <= i < qkv_dim for i in idx), (
        "below-boundary picker leaked into Q block")
    k_heads = present_heads(idx, q_dim, q_dim + kv_dim, head_dim)
    v_heads = present_heads(idx, q_dim + kv_dim, qkv_dim, head_dim)
    assert k_heads == v_heads, (
        f"KV-group pairing broken: K heads on CPU={k_heads}, "
        f"V heads on CPU={v_heads}")
    expected_groups = len(idx) // (2 * head_dim)
    assert k_heads == list(range(expected_groups)), (
        f"KV groups not contiguous from j=0: got {k_heads}")

    # (3) above the boundary — full K+V + tail Q
    over = min(boundary + (1.0 - boundary) * 0.5, 0.9)
    idx = kv_biased_cpu_columns(cfg, over)
    assert set(range(q_dim, qkv_dim)).issubset(set(idx)), (
        "above-boundary picker must contain the full K+V block")
    q_heads_on_cpu = present_heads(idx, 0, q_dim, head_dim)
    expected_q_heads = list(range(
        num_q_heads - len(q_heads_on_cpu), num_q_heads))
    assert q_heads_on_cpu == expected_q_heads, (
        f"above-boundary Q heads must come from the tail: "
        f"got {q_heads_on_cpu}, expected {expected_q_heads}")

    # (4) head alignment at all tested f values
    for f in [0.01, 0.03, 0.09, boundary, boundary + 0.1, 0.5]:
        idx = kv_biased_cpu_columns(cfg, f)
        assert len(idx) % head_dim == 0, (
            f"picker at f={f} produced non-head-aligned selection: "
            f"len(idx)={len(idx)} not divisible by head_dim={head_dim}")

    print(f"  [picker invariants]  boundary={boundary:.3f}  "
          f"num_kv_heads={num_kv_heads}  num_q_heads={num_q_heads}  OK")


# ---------------------------------------------------------------------------
# A. Col-parallel contiguous split (WQKV / MLP1 / optional WO)
# ---------------------------------------------------------------------------
def test_col_contiguous(in_dim, out_dim, f_cpu, B):
    """CPU gets the last f·out_dim output rows of W. Assembly: concat."""
    cpu_out = max(1, int(round(out_dim * f_cpu)))
    gpu_out = out_dim - cpu_out

    W_full = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(B, in_dim, dtype=torch.bfloat16, device="cuda")

    y_full = F.linear(x, W_full)

    W_gpu = W_full[:gpu_out, :]
    W_cpu = W_full[gpu_out:, :].cpu()
    y_gpu = F.linear(x, W_gpu)
    y_cpu = F.linear(x.cpu(), W_cpu)

    y_split = torch.cat([y_gpu, y_cpu.cuda()], dim=-1)
    return _max_diff(y_full, y_split)


# ---------------------------------------------------------------------------
# B. Col-parallel K/V-biased split (WQKV only)
# ---------------------------------------------------------------------------
def test_col_kv_biased(cfg, f_cpu, B):
    """CPU columns from the K/V-biased picker (possibly non-contiguous).
    Assembly via `index_copy_` into a full-shaped output."""
    hidden = cfg["hidden"]
    q_dim = cfg["num_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_kv_heads"] * cfg["head_dim"]
    qkv_dim = q_dim + 2 * kv_dim

    cpu_idx = kv_biased_cpu_columns(cfg, f_cpu)
    cpu_idx_t = torch.tensor(cpu_idx, dtype=torch.long, device="cuda")
    gpu_idx_t = torch.tensor(
        [i for i in range(qkv_dim) if i not in set(cpu_idx)],
        dtype=torch.long, device="cuda")

    W_full = torch.randn(qkv_dim, hidden, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(B, hidden, dtype=torch.bfloat16, device="cuda")

    y_full = F.linear(x, W_full)

    W_cpu = W_full.index_select(0, cpu_idx_t).cpu()
    W_gpu = W_full.index_select(0, gpu_idx_t)

    y_cpu = F.linear(x.cpu(), W_cpu)
    y_gpu = F.linear(x, W_gpu)

    y_split = torch.empty(B, qkv_dim, dtype=torch.bfloat16, device="cuda")
    y_split.index_copy_(1, gpu_idx_t, y_gpu)
    y_split.index_copy_(1, cpu_idx_t, y_cpu.cuda())

    return _max_diff(y_full, y_split), len(cpu_idx)


# ---------------------------------------------------------------------------
# C. Row-parallel contiguous split (MLP2)
# ---------------------------------------------------------------------------
def test_row_contiguous(in_dim, out_dim, f_cpu, B):
    """CPU gets the last f·in_dim input columns of W (and matching x cols).
    Each device produces a [B, out_dim] partial sum; assembly: `add_`.

    Weight is stored [out_dim, in_dim] row-major; row-splitting on the input
    dim takes a column-slice of W — materialized as a separate contiguous
    tensor (what the actual CpuComputeDispatcher does at load time).
    """
    cpu_in = max(1, int(round(in_dim * f_cpu)))
    gpu_in = in_dim - cpu_in

    W_full = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(B, in_dim, dtype=torch.bfloat16, device="cuda")

    y_full = F.linear(x, W_full)

    W_gpu = W_full[:, :gpu_in].contiguous()
    W_cpu = W_full[:, gpu_in:].contiguous().cpu()
    x_gpu = x[:, :gpu_in].contiguous()
    x_cpu = x[:, gpu_in:].contiguous().cpu()

    y_gpu_partial = F.linear(x_gpu, W_gpu)
    y_cpu_partial = F.linear(x_cpu, W_cpu)

    y_split = y_gpu_partial + y_cpu_partial.cuda()
    return _max_diff(y_full, y_split)


# ---------------------------------------------------------------------------
# D. MLP1→MLP2 col→row pipeline (end-to-end)
# ---------------------------------------------------------------------------
def test_mlp_pipeline(cfg, f_cpu, B):
    """End-to-end test of the MLP block under mixed col/row.

    Invariant: with matching intermediate index selection between MLP1's col
    split and MLP2's row split, each device applies SwiGLU on its local
    slice and MLP2 consumes the local slice directly — no intermediate
    transfer. Must match an unsplit MLP block.

    Layout (matches vLLM `MergedColumnParallelLinear` for gate_up):
      W1: shape [2·intermediate, hidden]
          rows [0, intermediate)              = gate
          rows [intermediate, 2·intermediate) = up
      W2: shape [hidden, intermediate], row-split on the input dim.

    Matching invariant: if intermediate index j is on CPU, CPU holds
      - W1 row j               (gate_j)
      - W1 row intermediate+j  (up_j)
      - W2 column j            (down col j)
    """
    hidden = cfg["hidden"]
    intermediate = cfg["intermediate"]

    cpu_inter = max(1, int(round(intermediate * f_cpu)))
    gpu_inter = intermediate - cpu_inter
    # Contiguous split: GPU holds intermediate indices [0, gpu_inter),
    # CPU holds [gpu_inter, intermediate).

    W1 = torch.randn(2 * intermediate, hidden, dtype=torch.bfloat16, device="cuda")
    W2 = torch.randn(hidden, intermediate, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(B, hidden, dtype=torch.bfloat16, device="cuda")

    # Reference: unsplit MLP block
    gate_up_full = F.linear(x, W1)
    gate_full = gate_up_full[:, :intermediate]
    up_full = gate_up_full[:, intermediate:]
    inter_full = F.silu(gate_full) * up_full
    y_full = F.linear(inter_full, W2)

    # GPU side: col MLP1 over indices [0, gpu_inter)
    W1_gate_gpu = W1[:gpu_inter, :]
    W1_up_gpu = W1[intermediate:intermediate + gpu_inter, :]
    gate_gpu = F.linear(x, W1_gate_gpu)
    up_gpu = F.linear(x, W1_up_gpu)
    inter_gpu = F.silu(gate_gpu) * up_gpu

    # CPU side: col MLP1 over indices [gpu_inter, intermediate)
    W1_gate_cpu = W1[gpu_inter:intermediate, :].cpu()
    W1_up_cpu = W1[intermediate + gpu_inter:, :].cpu()
    x_cpu = x.cpu()
    gate_cpu = F.linear(x_cpu, W1_gate_cpu)
    up_cpu = F.linear(x_cpu, W1_up_cpu)
    inter_cpu = F.silu(gate_cpu) * up_cpu

    # MLP2 row split: matching input-col selection per side.
    W2_gpu = W2[:, :gpu_inter].contiguous()
    W2_cpu = W2[:, gpu_inter:].contiguous().cpu()
    y_gpu_partial = F.linear(inter_gpu, W2_gpu)
    y_cpu_partial = F.linear(inter_cpu, W2_cpu)

    y_split = y_gpu_partial + y_cpu_partial.cuda()
    return _max_diff(y_full, y_split)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _status(exact, close):
    return "EXACT" if exact else "close" if close else "FAIL"


def run_col_contiguous(cfg, f_values, batch_sizes):
    """A. Col-parallel contiguous — WQKV, MLP1, and WO (if Alt A chosen).

    Note on WO: included here so Alt A's correctness is covered if that's what
    §0.4.2 picks. If Alt B wins, WO stays GPU-resident and this row is unused
    at runtime; the test remains a useful sanity check of the mechanism.
    """
    print("\n--- A. Col-parallel contiguous (WQKV / MLP1 / WO-Alt-A) ---")
    print(f"{'Sub-module':<10} {'B':>4} {'f_cpu':>7} "
          f"{'GPU+CPU=Total':>16} {'Result':>8} {'max_diff':>12}")
    print("-" * 64)

    gate_up = 2 * cfg["intermediate"]
    q_dim = cfg["num_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_kv_heads"] * cfg["head_dim"]
    qkv_dim = q_dim + 2 * kv_dim
    submods = [
        ("WQKV", cfg["hidden"], qkv_dim),
        ("MLP1", cfg["hidden"], gate_up),
        ("WO",   cfg["hidden"], cfg["hidden"]),
    ]

    all_pass = True
    for name, in_dim, out_dim in submods:
        for f in f_values:
            for B in batch_sizes:
                exact, close, max_diff, _ = test_col_contiguous(
                    in_dim, out_dim, f, B)
                cpu_out = max(1, int(round(out_dim * f)))
                gpu_out = out_dim - cpu_out
                all_pass &= close
                print(f"{name:<10} {B:>4} {f:>6.1%} "
                      f"{f'{gpu_out}+{cpu_out}={out_dim}':>16} "
                      f"{_status(exact, close):>8} {max_diff:>12.2e}")
    return all_pass


def run_col_kv_biased(cfg, f_values, batch_sizes):
    """B. Col-parallel K/V-biased — WQKV only."""
    q_dim = cfg["num_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_kv_heads"] * cfg["head_dim"]
    qkv_dim = q_dim + 2 * kv_dim
    boundary = kv_biased_boundary(cfg)

    print(f"\n--- B. Col-parallel K/V-biased WQKV "
          f"(boundary={boundary:.1%}, K+V block=[{q_dim}, {qkv_dim}) of {qkv_dim} cols) ---")
    print(f"{'f_cpu':>7} {'B':>4} {'cpu_cols':>9} {'region':>20} "
          f"{'Result':>8} {'max_diff':>12}")
    print("-" * 68)

    all_pass = True
    # Include the boundary value itself as a diagnostic row.
    sweep = sorted(set(list(f_values) + [boundary]))
    for f in sweep:
        region = "K+V only" if f <= boundary else "K+V + tail of Q"
        for B in batch_sizes:
            (exact, close, max_diff, _), cpu_cols = test_col_kv_biased(
                cfg, f, B)
            all_pass &= close
            print(f"{f:>6.1%} {B:>4} {cpu_cols:>9} {region:>20} "
                  f"{_status(exact, close):>8} {max_diff:>12.2e}")
    return all_pass


def run_row_contiguous(cfg, f_values, batch_sizes):
    """C. Row-parallel contiguous — MLP2 only."""
    print("\n--- C. Row-parallel contiguous (MLP2) ---")
    print(f"{'Sub-module':<10} {'B':>4} {'f_cpu':>7} "
          f"{'GPU+CPU=Total':>16} {'Result':>8} {'max_diff':>12}")
    print("-" * 64)

    submods = [
        ("MLP2", cfg["intermediate"], cfg["hidden"]),
    ]
    all_pass = True
    for name, in_dim, out_dim in submods:
        for f in f_values:
            for B in batch_sizes:
                exact, close, max_diff, _ = test_row_contiguous(
                    in_dim, out_dim, f, B)
                cpu_in = max(1, int(round(in_dim * f)))
                gpu_in = in_dim - cpu_in
                all_pass &= close
                print(f"{name:<10} {B:>4} {f:>6.1%} "
                      f"{f'{gpu_in}+{cpu_in}={in_dim}':>16} "
                      f"{_status(exact, close):>8} {max_diff:>12.2e}")
    return all_pass


def run_mlp_pipeline(cfg, f_values, batch_sizes):
    """D. MLP1→MLP2 col→row pipeline (end-to-end)."""
    print("\n--- D. MLP1→MLP2 col→row pipeline (end-to-end) ---")
    print(f"{'B':>4} {'f_cpu':>7} {'cpu_inter':>10} {'Result':>8} {'max_diff':>12}")
    print("-" * 48)

    all_pass = True
    for f in f_values:
        cpu_inter = max(1, int(round(cfg["intermediate"] * f)))
        for B in batch_sizes:
            exact, close, max_diff, _ = test_mlp_pipeline(cfg, f, B)
            all_pass &= close
            print(f"{B:>4} {f:>6.1%} {cpu_inter:>10} "
                  f"{_status(exact, close):>8} {max_diff:>12.2e}")
    return all_pass


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="qwen7b")
    p.add_argument("--f-cpu", type=float, nargs="+", default=F_CPU_SWEEP)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    args = p.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    q_dim = cfg["num_heads"] * cfg["head_dim"]
    kv_dim = cfg["num_kv_heads"] * cfg["head_dim"]
    boundary = kv_biased_boundary(cfg)

    print(f"Tensor Split Correctness — {cfg['display_name']} (BF16, F.linear)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Q|K|V dims: {q_dim} | {kv_dim} | {kv_dim}  "
          f"(K+V boundary at f={boundary:.1%})")
    print(f"{'='*78}")

    print("\n--- K/V-biased picker invariants ---")
    validate_picker(cfg)

    all_pass = True
    all_pass &= run_col_contiguous(cfg, args.f_cpu, args.batch_sizes)
    all_pass &= run_col_kv_biased(cfg, args.f_cpu, args.batch_sizes)
    all_pass &= run_row_contiguous(cfg, args.f_cpu, args.batch_sizes)
    all_pass &= run_mlp_pipeline(cfg, args.f_cpu, args.batch_sizes)

    print(f"\n{'='*78}")
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
