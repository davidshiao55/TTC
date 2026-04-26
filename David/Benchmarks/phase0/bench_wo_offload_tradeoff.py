#!/usr/bin/env python3
"""Phase 0.4.2 — WO Offload Alt A vs Alt B Tradeoff

Phase 2 exposes two clean alternatives for handling WO after the online-softmax
merge. This benchmark measures the per-layer critical-path differential
between them so we can pick one at design time.

  Alt A — WO col-split (weight offloaded):

    GPU: [prefix attn] → [merge] ─┬─ [GPU WO reduced]          ┐
    CPU: [suffix attn] → D2H ─────┤                              │ wait for both
                                  └─ H2D → [CPU WO] → D2H      ┘
                                                   → [concat on GPU]

  Alt B — No WO offload (WO stays GPU-resident):

    GPU: [prefix attn] → [merge] → [GPU WO full]
    CPU: [suffix attn] → D2H

The attention / merge phases are identical in both alternatives (CPU attn is
the bottleneck in Phase 2 and we do nothing different before merge), so they
cancel out of the comparison. The differential critical path is only the
post-merge region:

    Δ_latency(f, N) = max(t_wo_reduced_gpu(f, N),
                          H2D_merged + t_wo_cpu(f, N) + D2H_partial)
                      − t_wo_full_gpu(N)

A negative or zero Δ means Alt A hides its transfers behind GPU-reduced WO
and matches Alt B's latency — no reason to prefer Alt B on speed. A positive
Δ is the latency tax Alt A charges to save `f × WO_bytes × num_layers` of
GPU memory.

Outputs:
  - Per (N, f): measured primitives, Δ_latency, GPU bytes saved.
  - Summary table: latency cost vs memory saved.
  - Recommended default given the measured regime.

Usage:
    python bench_wo_offload_tradeoff.py --model qwen7b
    python bench_wo_offload_tradeoff.py --model qwen7b --output-json out.json
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


MODEL_CONFIGS = {
    "qwen7b": {
        "display_name": "Qwen2.5-7B-Instruct",
        "hidden": 3584,
        "num_layers": 28,
    },
    "prm1p5b": {
        "display_name": "Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "hidden": 1536,
        "num_layers": 28,
    },
}

# Decode-realistic num_tokens for Phase 2 (batch size == num_tokens in uniform decode).
NUM_TOKENS = [1, 16, 64, 128]
SLICE_FRACS = [0.10, 0.20, 0.30, 0.50]

WARMUP = 10
ITERS = 50
BF16 = 2  # bytes


@dataclass
class Primitives:
    """Measured per-(N, f) timings, in ms, and transfer bytes."""
    N: int
    f: float
    t_wo_full_gpu: float     # [N, H] @ [H, H]
    t_wo_reduced_gpu: float  # [N, H] @ [H, (1-f)·H]
    t_wo_cpu: float          # [N, H] @ [H, f·H]  (col-split CPU)
    h2d_merged_ms: float     # transfer [N, H] bf16 host→device (merged attn_out)
    d2h_partial_ms: float    # transfer [N, f·H] bf16 device→host (CPU WO partial)

    def alt_a_postmerge_ms(self) -> float:
        cpu_chain = self.h2d_merged_ms + self.t_wo_cpu + self.d2h_partial_ms
        return max(self.t_wo_reduced_gpu, cpu_chain)

    def alt_b_postmerge_ms(self) -> float:
        return self.t_wo_full_gpu

    def delta_ms(self) -> float:
        return self.alt_a_postmerge_ms() - self.alt_b_postmerge_ms()


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------
def _time_gpu_linear(x: torch.Tensor, W: torch.Tensor,
                     warmup: int = WARMUP, iters: int = ITERS) -> float:
    for _ in range(warmup):
        F.linear(x, W)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        F.linear(x, W)
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def _time_cpu_linear(x: torch.Tensor, W: torch.Tensor,
                     warmup: int = WARMUP, iters: int = ITERS) -> float:
    for _ in range(warmup):
        F.linear(x, W)
    t0 = time.perf_counter()
    for _ in range(iters):
        F.linear(x, W)
    return (time.perf_counter() - t0) / iters * 1000.0


def _time_pcie_transfer(nbytes: int, direction: str,
                        warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Time a pinned-memory PCIe transfer of `nbytes`. Returns ms."""
    assert direction in {"h2d", "d2h"}
    n_elem = max(1, nbytes // BF16)
    if direction == "h2d":
        src = torch.empty(n_elem, dtype=torch.bfloat16, pin_memory=True)
        dst = torch.empty(n_elem, dtype=torch.bfloat16, device="cuda")
    else:
        src = torch.empty(n_elem, dtype=torch.bfloat16, device="cuda")
        dst = torch.empty(n_elem, dtype=torch.bfloat16, pin_memory=True)

    for _ in range(warmup):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        dst.copy_(src, non_blocking=True)
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def measure(cfg, num_tokens: int, f: float) -> Primitives:
    H = cfg["hidden"]
    cpu_out = max(1, int(H * f))
    gpu_out = H - cpu_out

    # GPU weights + activations
    W_full_gpu = torch.randn(H, H, dtype=torch.bfloat16, device="cuda")
    W_reduced_gpu = torch.randn(gpu_out, H, dtype=torch.bfloat16, device="cuda")
    x_gpu = torch.randn(num_tokens, H, dtype=torch.bfloat16, device="cuda")

    # CPU weights + activations
    W_cpu = torch.randn(cpu_out, H, dtype=torch.bfloat16, device="cpu")
    x_cpu = torch.randn(num_tokens, H, dtype=torch.bfloat16, device="cpu")

    t_wo_full_gpu = _time_gpu_linear(x_gpu, W_full_gpu)
    t_wo_reduced_gpu = _time_gpu_linear(x_gpu, W_reduced_gpu)
    t_wo_cpu = _time_cpu_linear(x_cpu, W_cpu)

    merged_bytes = num_tokens * H * BF16
    partial_bytes = num_tokens * cpu_out * BF16
    h2d_ms = _time_pcie_transfer(merged_bytes, "h2d")
    d2h_ms = _time_pcie_transfer(partial_bytes, "d2h")

    return Primitives(
        N=num_tokens, f=f,
        t_wo_full_gpu=t_wo_full_gpu,
        t_wo_reduced_gpu=t_wo_reduced_gpu,
        t_wo_cpu=t_wo_cpu,
        h2d_merged_ms=h2d_ms,
        d2h_partial_ms=d2h_ms,
    )


def gpu_bytes_saved(cfg, f: float) -> int:
    """WO weight bytes that move off GPU under Alt A, across all layers."""
    H = cfg["hidden"]
    return int(f * H * H * BF16 * cfg["num_layers"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_detail(cfg, results: list[Primitives]):
    print(f"\n{'='*96}")
    print(f"  Per-(N, f) primitive timings and Δ — {cfg['display_name']}")
    print(f"{'='*96}")
    hdr = (f"{'N':>4} {'f':>5} "
           f"{'WO_full':>8} {'WO_red':>8} {'WO_cpu':>8} "
           f"{'H2D':>7} {'D2H':>7} "
           f"{'AltA_post':>10} {'AltB_post':>10} {'Δ':>8}")
    print(hdr)
    print("-" * len(hdr))
    for p in results:
        print(f"{p.N:>4} {p.f:>5.1%} "
              f"{p.t_wo_full_gpu:>7.3f}m {p.t_wo_reduced_gpu:>7.3f}m "
              f"{p.t_wo_cpu:>7.3f}m "
              f"{p.h2d_merged_ms:>6.3f}m {p.d2h_partial_ms:>6.3f}m "
              f"{p.alt_a_postmerge_ms():>9.3f}m {p.alt_b_postmerge_ms():>9.3f}m "
              f"{p.delta_ms():>+7.3f}m")


def print_summary(cfg, results: list[Primitives], num_layers: int):
    print(f"\n{'='*96}")
    print(f"  Summary — per-decode-step latency tax vs GPU memory saved")
    print(f"  ({cfg['display_name']}, {num_layers} layers)")
    print(f"{'='*96}")
    # Group by f; report the worst-case Δ across N (what the Planner plans for)
    by_f: dict[float, list[Primitives]] = {}
    for p in results:
        by_f.setdefault(p.f, []).append(p)

    hdr = (f"{'f':>6} {'max_Δ/layer':>14} {'max_Δ/step':>14} "
           f"{'GPU_saved':>12} {'verdict':<28}")
    print(hdr)
    print("-" * len(hdr))
    for f in sorted(by_f):
        deltas = [p.delta_ms() for p in by_f[f]]
        max_d = max(deltas)
        saved_mb = gpu_bytes_saved(cfg, f) / 1e6
        # Verdict classes the Δ against rough decode-step budget:
        #   tight (< 0.05 ms/layer → hidden in noise)
        #   mild  (< 0.5 ms/layer → ~1-3% of typical decode step)
        #   heavy (≥ 0.5 ms/layer)
        if max_d < 0.05:
            verdict = "free (in noise)"
        elif max_d < 0.5:
            verdict = "mild latency tax"
        else:
            verdict = "heavy latency tax"
        print(f"{f:>5.1%} {max_d:>+12.3f}m "
              f"{max_d * num_layers:>+12.2f}m "
              f"{saved_mb:>9.0f}MB {verdict:<28}")

    # Recommendation
    print()
    any_free = any(max(p.delta_ms() for p in by_f[f]) < 0.05 for f in by_f)
    if any_free:
        print("Recommendation: Alt A is viable at the lowest 'free' f — pick Alt A if memory")
        print("                saved is meaningful relative to overall budget; else Alt B.")
    else:
        print("Recommendation: Alt A charges a non-zero latency tax at every tested f.")
        print("                Default to Alt B unless GPU memory is binding.")


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
    print(f"Phase 0.4.2 — WO Offload Alt A vs Alt B  ({cfg['display_name']})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"num_tokens={args.num_tokens}  slice_fracs={args.slice_fracs}")

    results: list[Primitives] = []
    for N in args.num_tokens:
        for f in args.slice_fracs:
            results.append(measure(cfg, N, f))

    print_detail(cfg, results)
    print_summary(cfg, results, cfg["num_layers"])

    if args.output_json:
        out = {
            "model": args.model,
            "model_cfg": cfg,
            "config": {
                "num_tokens": args.num_tokens,
                "slice_fracs": args.slice_fracs,
                "warmup": WARMUP, "iters": ITERS,
            },
            "primitives": [p.__dict__ for p in results],
            "derived": [
                {
                    "N": p.N, "f": p.f,
                    "alt_a_postmerge_ms": p.alt_a_postmerge_ms(),
                    "alt_b_postmerge_ms": p.alt_b_postmerge_ms(),
                    "delta_ms": p.delta_ms(),
                    "gpu_bytes_saved": gpu_bytes_saved(cfg, p.f),
                }
                for p in results
            ],
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
