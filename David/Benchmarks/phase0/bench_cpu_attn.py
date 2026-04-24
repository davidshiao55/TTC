#!/usr/bin/env python3
"""Phase 0.5 — CPU Attention Latency (PyTorch reference implementation)

Measures decode-time suffix attention latency on CPU as a function of
`(batch_size, suffix_context_len)` for the generator (Qwen2.5-7B) and verifier
(Skywork-PRM-1.5B) head configs. Feeds `cpu_attn_curve[B, S]` in the Planner's
profile schema (`profiler_design.md §1.4`).

Why PyTorch reference, not vLLM's C++ kernel:
  vLLM's optimized `cpu_attention_with_kv_cache` is only compiled into builds
  where `VLLM_TARGET_DEVICE=cpu`; our CUDA build does not include it. Using
  `F.scaled_dot_product_attention` on CPU is an **upper bound** on the real
  kernel's latency — it uses native PyTorch dispatches (oneDNN / MKL paths),
  not vLLM's hand-optimized AMX/vec kernels. Real numbers will land when
  Phase 2 integrates the C++ kernel (engineering gap #1 in
  `implementation_roadmap.md §2`; also needs the per-head LSE output mod).

For Phase 0 / Planner bootstrap this is sufficient: the Planner treats the
resulting curve as a conservative upper bound, which naturally makes it
under-estimate the benefit of attention offload. The sign is safe: if Phase 2
beats this curve (likely, since the C++ kernel is optimized), actual
performance is better than the Planner predicted.

Workload per (B, S) cell:
  - q: [B, num_query_heads, 1, head_dim]     (single decode token per seq)
  - k: [B, num_kv_heads,    S, head_dim]     (suffix KV cache)
  - v: [B, num_kv_heads,    S, head_dim]
  - attention is causal within the suffix
  - GQA is expanded via `enable_gqa=True`

Usage:
    python bench_cpu_attn.py --model qwen7b
    python bench_cpu_attn.py --model prm1p5b
    python bench_cpu_attn.py --model qwen7b --output-json cpu_attn_qwen7b.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


MODEL_CONFIGS = {
    "qwen7b": {
        "display_name": "Qwen2.5-7B-Instruct",
        "num_query_heads": 28,
        "num_kv_heads":    4,
        "head_dim":        128,
    },
    "prm1p5b": {
        "display_name": "Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
        "num_query_heads": 12,
        "num_kv_heads":    2,
        "head_dim":        128,
    },
}

BATCH_SIZES = [4, 8, 16, 32]
SUFFIX_LENS = [100, 500, 1000, 2000, 4000]

WARMUP = 5
ITERS = 20   # CPU attention is slow; keep iters low


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_cpu_attention(B, S, cfg, dtype=torch.bfloat16,
                      warmup=WARMUP, iters=ITERS):
    """Time one (B, S) cell. Returns mean, p50, p95 in ms."""
    Hq = cfg["num_query_heads"]
    Hkv = cfg["num_kv_heads"]
    D = cfg["head_dim"]

    q = torch.randn(B, Hq,  1, D, dtype=dtype)
    k = torch.randn(B, Hkv, S, D, dtype=dtype)
    v = torch.randn(B, Hkv, S, D, dtype=dtype)

    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=False)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=False)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "mean_ms":   sum(times) / len(times),
        "median_ms": times[len(times) // 2],
        "p95_ms":    times[int(len(times) * 0.95)],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="qwen7b")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    p.add_argument("--suffix-lens", type=int, nargs="+", default=SUFFIX_LENS)
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--threads", type=int, default=None,
                   help="Override torch.set_num_threads (default: all cores)")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = MODEL_CONFIGS[args.model]
    dtype = getattr(torch, args.dtype)

    print(f"Phase 0.5 — CPU Attention Latency (reference, PyTorch SDPA)")
    print(f"Model: {cfg['display_name']}")
    print(f"  num_query_heads={cfg['num_query_heads']}  "
          f"num_kv_heads={cfg['num_kv_heads']}  head_dim={cfg['head_dim']}")
    print(f"dtype={args.dtype}  CPU threads={torch.get_num_threads()}")
    print(f"Note: PyTorch reference (not vLLM's AMX/vec kernel) — upper bound")
    print()

    header = f"  {'B':>4} " + " ".join(
        f"{'S='+str(S):>10}" for S in args.suffix_lens)
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    for B in args.batch_sizes:
        results[B] = {}
        cells = []
        for S in args.suffix_lens:
            r = time_cpu_attention(B, S, cfg, dtype=dtype)
            results[B][S] = {k: round(v, 4) for k, v in r.items()}
            cells.append(f"{r['mean_ms']:8.2f}ms")
        print(f"  {B:>4} " + " ".join(cells))

    # Per-cell breakdown: effective tokens/ms = B × S / mean_ms
    print(f"\n  Arithmetic intensity check — tokens/ms at each (B, S):")
    for B in args.batch_sizes:
        row = []
        for S in args.suffix_lens:
            tps = B * S / results[B][S]["mean_ms"]
            row.append(f"{tps:>8.0f}")
        print(f"  B={B:>3}: " + " ".join(row))

    if args.output_json:
        out = {
            "schema_version": 1,
            "model_key": args.model,
            "model": cfg,
            "dtype": args.dtype,
            "cpu_threads": torch.get_num_threads(),
            "note": ("PyTorch F.scaled_dot_product_attention reference (not "
                     "vLLM's C++ kernel — upper-bound estimate only)"),
            "config": {
                "batch_sizes": args.batch_sizes,
                "suffix_lens": args.suffix_lens,
                "warmup": WARMUP,
                "iters": ITERS,
            },
            "cpu_attn_curve": {
                str(B): {str(S): results[B][S] for S in args.suffix_lens}
                for B in args.batch_sizes
            },
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
