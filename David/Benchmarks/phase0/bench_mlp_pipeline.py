#!/usr/bin/env python3
"""Phase 0.4.1 — MLP Block Pipeline: Uniform Col vs Col→Row

Empirical wall-clock comparison of the MLP block under two split patterns:

  Uniform col (rejected): MLP1 col, MLP2 col
      CPU: [H2D x] → [MLP1 slice] → [D2H gate_up slice]
      GPU:                         → [concat] → [SwiGLU] → [H2D intermediate]
      CPU:                                                  → [MLP2 slice] → [D2H output slice]
      GPU:                                                                    → [concat]

  Col → row (chosen):     MLP1 col, MLP2 row
      CPU: [H2D x] → [MLP1 slice] → [SwiGLU on slice] → [MLP2 row partial] → [D2H partial]
      GPU:                                                                    → [add_]

The two patterns have identical CPU GEMM FLOPs; the wall-clock difference comes
from (a) the eliminated intermediate round-trip (~2.4 MB H2D + ~485 KB D2H per
block on 7B at N=64, f=10%), (b) different oneDNN microkernel selection for
col-shape vs row-shape MLP2, and (c) a small CPU-side SwiGLU in col→row.

Measurement: for each pattern, time the CPU-side serial critical path. The
GPU portions on the off-path device run in the background with lower latency;
CPU is the bottleneck in this regime (§0.3 confirms CPU GEMM is the slowest
link). GPU steps where the CPU must wait are timed via `torch.cuda.Event` and
included in the sum. This captures the Planner-relevant wall-clock.

Usage:
    python bench_mlp_pipeline.py --model qwen7b
    python bench_mlp_pipeline.py --model qwen7b --output-json out.json
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F


MODEL_CONFIGS = {
    "qwen7b": {
        "display_name": "Qwen2.5-7B-Instruct",
        "hidden": 3584,
        "intermediate": 18944,
    },
    "prm1p5b": {
        "display_name": "Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "hidden": 1536,
        "intermediate": 8960,
    },
}

NUM_TOKENS = [16, 64, 128]
SLICE_FRACS = [0.10, 0.30]

WARMUP = 5
ITERS = 20
BF16 = 2  # bytes


@dataclass
class PatternTiming:
    """Per-step ms and total PCIe bytes for one MLP-block pattern."""
    name: str
    steps: dict[str, float] = field(default_factory=dict)
    pcie_bytes: int = 0

    @property
    def total_ms(self) -> float:
        return sum(self.steps.values())


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def time_gpu_op(fn, iters=ITERS) -> float:
    """Time a GPU op on the default stream. Returns mean ms."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def time_cpu_op(fn, iters=ITERS) -> float:
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def time_h2d(shape, dtype=torch.bfloat16) -> float:
    src = torch.empty(*shape, dtype=dtype, pin_memory=True)
    dst = torch.empty(*shape, dtype=dtype, device="cuda")
    return time_gpu_op(lambda: dst.copy_(src, non_blocking=True))


def time_d2h(shape, dtype=torch.bfloat16) -> float:
    src = torch.empty(*shape, dtype=dtype, device="cuda")
    dst = torch.empty(*shape, dtype=dtype, pin_memory=True)
    return time_gpu_op(lambda: dst.copy_(src, non_blocking=True))


def bytes_of(shape, dtype_bytes=BF16) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * dtype_bytes


# ---------------------------------------------------------------------------
# Pattern A: uniform col (rejected design — measured for comparison)
# ---------------------------------------------------------------------------
def bench_uniform_col(cfg: dict, N: int, f: float) -> PatternTiming:
    """CPU critical path under uniform-col MLP (both MLP1 and MLP2 col-split).

    CPU-side serial steps:
      1. H2D x                              — CPU pinned → GPU (the input)
      2. CPU MLP1: x @ W1_cpu.T → gate_up slice shape [N, 2·inter·f]
      3. D2H gate_up CPU slice              — return to GPU for concat
      4. GPU waits: concat + SwiGLU         — GPU builds full intermediate
      5. H2D full intermediate              — send full intermediate back to CPU
      6. CPU MLP2: intermediate @ W2_cpu.T → output slice [N, hidden·f]
      7. D2H output CPU slice               — return to GPU for concat
      8. GPU concat output                  — GPU assembles final output
    """
    H, Inter = cfg["hidden"], cfg["intermediate"]
    cpu_out_mlp1 = max(1, int(2 * Inter * f))
    cpu_out_mlp2 = max(1, int(H * f))

    t = PatternTiming(name="uniform_col")
    t.steps["h2d_x"] = time_h2d((N, H))

    x_cpu = torch.randn(N, H, dtype=torch.bfloat16, device="cpu")
    W1_cpu = torch.randn(cpu_out_mlp1, H, dtype=torch.bfloat16, device="cpu")
    t.steps["cpu_mlp1"] = time_cpu_op(lambda: F.linear(x_cpu, W1_cpu))

    t.steps["d2h_gate_up"] = time_d2h((N, cpu_out_mlp1))

    # GPU concat+SwiGLU on full [N, 2·inter] — CPU must wait for this.
    gate_up_full = torch.randn(N, 2 * Inter, dtype=torch.bfloat16, device="cuda")

    def gpu_concat_swiglu():
        gate = gate_up_full[:, :Inter]
        up = gate_up_full[:, Inter:]
        return F.silu(gate) * up

    t.steps["gpu_concat_swiglu"] = time_gpu_op(gpu_concat_swiglu)

    t.steps["h2d_intermediate"] = time_h2d((N, Inter))

    inter_cpu = torch.randn(N, Inter, dtype=torch.bfloat16, device="cpu")
    W2_cpu = torch.randn(cpu_out_mlp2, Inter, dtype=torch.bfloat16, device="cpu")
    t.steps["cpu_mlp2"] = time_cpu_op(lambda: F.linear(inter_cpu, W2_cpu))

    t.steps["d2h_output"] = time_d2h((N, cpu_out_mlp2))

    out_gpu = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
    out_slice_gpu = torch.randn(N, cpu_out_mlp2, dtype=torch.bfloat16, device="cuda")
    t.steps["gpu_concat_output"] = time_gpu_op(
        lambda: torch.cat([out_gpu[:, :H - cpu_out_mlp2], out_slice_gpu], dim=-1))

    t.pcie_bytes = (
        bytes_of((N, H))                 # h2d x
        + bytes_of((N, cpu_out_mlp1))    # d2h gate_up slice
        + bytes_of((N, Inter))           # h2d full intermediate
        + bytes_of((N, cpu_out_mlp2))    # d2h output slice
    )
    return t


# ---------------------------------------------------------------------------
# Pattern B: col → row (chosen design)
# ---------------------------------------------------------------------------
def bench_col_to_row(cfg: dict, N: int, f: float) -> PatternTiming:
    """CPU critical path under col→row MLP (MLP1 col, MLP2 row).

    CPU-side serial steps:
      1. H2D x                              — CPU pinned → GPU (the input)
      2. CPU MLP1: x @ W1_cpu.T → gate_up slice [N, 2·inter·f]
      3. CPU SwiGLU on slice → intermediate_cpu slice [N, inter·f]
      4. CPU MLP2 (row):   slice @ W2_cpu.T → partial sum [N, hidden]
      5. D2H partial sum                    — full [N, hidden]
      6. GPU .add_(cpu_partial)             — assembly
    """
    H, Inter = cfg["hidden"], cfg["intermediate"]
    cpu_inter = max(1, int(Inter * f))

    t = PatternTiming(name="col_to_row")
    t.steps["h2d_x"] = time_h2d((N, H))

    # MLP1 col slice = the fused gate_up weight at the CPU's intermediate
    # indices, stored contiguously (matches vLLM's MergedColumnParallelLinear).
    # Shape: [2·cpu_inter, H] — same per-FLOP cost as uniform_col's cpu_mlp1.
    x_cpu = torch.randn(N, H, dtype=torch.bfloat16, device="cpu")
    W1_cpu = torch.randn(2 * cpu_inter, H, dtype=torch.bfloat16, device="cpu")
    t.steps["cpu_mlp1"] = time_cpu_op(lambda: F.linear(x_cpu, W1_cpu))

    # Local SwiGLU on the slice — elementwise on gate half × up half.
    gate_up = torch.randn(N, 2 * cpu_inter, dtype=torch.bfloat16, device="cpu")
    def cpu_swiglu():
        g = gate_up[:, :cpu_inter]
        u = gate_up[:, cpu_inter:]
        return F.silu(g) * u
    t.steps["cpu_swiglu"] = time_cpu_op(cpu_swiglu)

    # MLP2 row-split: CPU slice of input dim → full output rows (partial sum)
    inter_slice_cpu = torch.randn(N, cpu_inter, dtype=torch.bfloat16, device="cpu")
    W2_cpu = torch.randn(H, cpu_inter, dtype=torch.bfloat16, device="cpu")
    t.steps["cpu_mlp2_row"] = time_cpu_op(lambda: F.linear(inter_slice_cpu, W2_cpu))

    t.steps["d2h_partial"] = time_d2h((N, H))

    out_gpu = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
    partial_gpu = torch.randn(N, H, dtype=torch.bfloat16, device="cuda")
    t.steps["gpu_add"] = time_gpu_op(lambda: out_gpu.add_(partial_gpu))

    t.pcie_bytes = (
        bytes_of((N, H))                 # h2d x
        + bytes_of((N, H))               # d2h full partial sum
    )
    return t


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_pattern(t: PatternTiming):
    print(f"  [{t.name}] total={t.total_ms:7.3f} ms  PCIe bytes={t.pcie_bytes/1e6:6.2f} MB")
    for step, ms in t.steps.items():
        print(f"     {step:<22}  {ms:7.3f} ms")


def print_summary(rows):
    print(f"\n{'='*96}")
    print(f"  Summary — uniform col vs col→row per MLP block")
    print(f"{'='*96}")
    hdr = (f"{'N':>4} {'f':>5} {'col total':>11} {'row total':>11} "
           f"{'Δ (ms)':>9} {'col PCIe':>10} {'row PCIe':>10} {'PCIe ratio':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        col = r["uniform_col"]
        row = r["col_to_row"]
        ratio = col.pcie_bytes / max(1, row.pcie_bytes)
        print(f"{r['N']:>4} {r['f']:>5.1%} "
              f"{col.total_ms:>10.3f}m {row.total_ms:>10.3f}m "
              f"{col.total_ms - row.total_ms:>+8.3f}m "
              f"{col.pcie_bytes/1e6:>8.2f}MB {row.pcie_bytes/1e6:>8.2f}MB "
              f"{ratio:>10.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="qwen7b")
    p.add_argument("--num-tokens", type=int, nargs="+", default=NUM_TOKENS)
    p.add_argument("--slice-fracs", type=float, nargs="+", default=SLICE_FRACS)
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    print(f"Phase 0.4.1 — MLP Block Pipeline  ({cfg['display_name']})")
    print(f"GPU: {torch.cuda.get_device_name(0)}  CPU threads: {torch.get_num_threads()}")
    print(f"N={args.num_tokens}  f={args.slice_fracs}")

    rows = []
    for N in args.num_tokens:
        for f in args.slice_fracs:
            print(f"\nN={N}, f={f:.1%}")
            col = bench_uniform_col(cfg, N, f)
            print_pattern(col)
            row = bench_col_to_row(cfg, N, f)
            print_pattern(row)
            rows.append({"N": N, "f": f, "uniform_col": col, "col_to_row": row})

    print_summary(rows)

    if args.output_json:
        out = {
            "model": args.model,
            "model_cfg": cfg,
            "config": {
                "num_tokens": args.num_tokens,
                "slice_fracs": args.slice_fracs,
                "warmup": WARMUP, "iters": ITERS,
            },
            "rows": [
                {
                    "N": r["N"], "f": r["f"],
                    "uniform_col": {
                        "total_ms": r["uniform_col"].total_ms,
                        "pcie_bytes": r["uniform_col"].pcie_bytes,
                        "steps": r["uniform_col"].steps,
                    },
                    "col_to_row": {
                        "total_ms": r["col_to_row"].total_ms,
                        "pcie_bytes": r["col_to_row"].pcie_bytes,
                        "steps": r["col_to_row"].steps,
                    },
                } for r in rows
            ],
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
