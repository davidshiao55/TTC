#!/usr/bin/env python3
"""Phase 0.9 — PCIe Contention and Stream Concurrency

Measures whether concurrent PCIe transfers on separate CUDA streams contend
for bandwidth, and whether GPU compute on one stream is correctly overlapped
with PCIe traffic on another. Feeds the Phase 3 sub-layer pipeline design
(see `pcie_bandwidth_allocation_design.md`).

  §0.9a — Activation mechanism: explicit copy vs UVA.
          Both variants use `fg_first` submission order (the realistic pipelined
          inference pattern). Answers: which mechanism should the CPU-compute
          activation return path use?
            copy — explicit cudaMemcpyAsync (pinned → GDDR6X) then matmul
            uva  — matmul reads pinned CPU memory via UVA during kernel exec

  §0.9b — Submission order (anti-pattern note).
          Path is fixed to `copy`. Compares submission orders:
            fg_first       — compute submitted before bg prefetch queue (good)
            bg_first       — bg queued first, compute after (anti-pattern)
            explicit_event — bg records an event; demonstrates that events
                             alone do NOT fix bg-first ordering — the submission
                             order itself is what drives the serialization
          Answers: what happens if implementations accidentally queue prefetch
          before current-layer compute?

  §0.9c — Bidirectional H2D + D2H.
          PCIe 4.0 x16 is full-duplex. Run weight-prefetch H2D and KV-spill
          D2H concurrently; measure effective BW on each direction vs. isolated.
          Answers: does Phase 2's KV D2H compete with weight prefetch?

Usage:
    python bench_contention.py
    python bench_contention.py --output-json out.json
"""

import argparse
import json
import time
from pathlib import Path

import torch

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor


HIDDEN = 3584
N_TOKENS = 14                             # ~100 KB activation at BF16
WEIGHT_BYTES = 4 * 1024 * 1024            # 4 MB — representative prefetch slice
D2H_KV_BYTES = 100 * 1024                 # 100 KB — representative KV spill chunk

QUEUE_DEPTHS = [0, 1, 2, 4]               # realistic sub-layer-pipeline depths
WARMUP = 20
ITERS = 100


# ---------------------------------------------------------------------------
# Shared buffers
# ---------------------------------------------------------------------------
def make_buffers():
    act_cpu = torch.empty(N_TOKENS * HIDDEN, dtype=torch.bfloat16, pin_memory=True)
    act_gpu = torch.empty_like(act_cpu, device="cuda")
    act_gpu_2d = act_gpu.view(N_TOKENS, HIDDEN)
    # UVA view of the same pinned activation — for the "uva" path variant.
    act_uva = get_accelerator_view_from_cpu_tensor(act_cpu).view(N_TOKENS, HIDDEN)
    wt_cpu = torch.empty(WEIGHT_BYTES // 2, dtype=torch.bfloat16, pin_memory=True)
    wt_gpu = torch.empty_like(wt_cpu, device="cuda")
    W = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16, device="cuda")
    mm_out = torch.empty(N_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    return dict(act_cpu=act_cpu, act_gpu=act_gpu, act_gpu_2d=act_gpu_2d,
                act_uva=act_uva,
                wt_cpu=wt_cpu, wt_gpu=wt_gpu, W=W, mm_out=mm_out)


# ---------------------------------------------------------------------------
# Shared helpers — fg/bg ops used by both §0.9a and §0.9b
# ---------------------------------------------------------------------------
def _make_fg_ops(bufs):
    def fg_op_copy():
        bufs["act_gpu"].copy_(bufs["act_cpu"], non_blocking=True)
        torch.matmul(bufs["act_gpu_2d"], bufs["W"].t(), out=bufs["mm_out"])

    def fg_op_uva():
        # No explicit copy — matmul reads pinned memory via UVA mapping.
        torch.matmul(bufs["act_uva"], bufs["W"].t(), out=bufs["mm_out"])

    return {"copy": fg_op_copy, "uva": fg_op_uva}


def _make_bg_op(bufs):
    def bg_op_n(N):
        for _ in range(N):
            bufs["wt_gpu"].copy_(bufs["wt_cpu"], non_blocking=True)
    return bg_op_n


def _measure_one(mode, N, fg_op, bg_op_n, s_weight, s_act):
    for _ in range(WARMUP):
        _submit(mode, N, s_weight, s_act, fg_op, bg_op_n)
        torch.cuda.synchronize()

    fg_times, bg_times, wall_times = [], [], []
    for _ in range(ITERS):
        torch.cuda.synchronize()
        fg_s = torch.cuda.Event(enable_timing=True)
        fg_e = torch.cuda.Event(enable_timing=True)
        bg_s = torch.cuda.Event(enable_timing=True)
        bg_e = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        _submit(mode, N, s_weight, s_act, fg_op, bg_op_n,
                fg_events=(fg_s, fg_e), bg_events=(bg_s, bg_e))
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - t0) * 1000)
        fg_times.append(fg_s.elapsed_time(fg_e))
        if N > 0:
            bg_times.append(bg_s.elapsed_time(bg_e))

    return {
        "fg_mean_ms":   round(sum(fg_times)/len(fg_times), 4),
        "bg_mean_ms":   round(sum(bg_times)/len(bg_times), 4) if bg_times else 0.0,
        "wall_mean_ms": round(sum(wall_times)/len(wall_times), 4),
    }


# ---------------------------------------------------------------------------
# §0.9a — Activation mechanism: explicit copy vs UVA (fg_first only)
# ---------------------------------------------------------------------------
def bench_activation_mechanism(bufs, queue_depths):
    """Headline comparison: should the CPU-compute activation return path use
    explicit cudaMemcpyAsync or UVA? Submission order is fixed to fg_first
    (the realistic pipelined pattern).
    """
    s_weight = torch.cuda.Stream()
    s_act    = torch.cuda.Stream()
    fg_ops = _make_fg_ops(bufs)
    bg_op_n = _make_bg_op(bufs)

    results = {}
    for path in ["copy", "uva"]:
        per_depth = []
        for N in queue_depths:
            per_depth.append({"queue_depth": N,
                              **_measure_one("fg_first", N, fg_ops[path],
                                             bg_op_n, s_weight, s_act)})
        results[path] = per_depth
    return results


# ---------------------------------------------------------------------------
# §0.9b — Submission order (anti-pattern note, path=copy only)
# ---------------------------------------------------------------------------
def bench_submission_order(bufs, queue_depths):
    """What happens if implementations accidentally queue prefetches before the
    current-layer compute? Path is fixed to `copy` since §0.9a already settled
    the UVA question.
    """
    s_weight = torch.cuda.Stream()
    s_act    = torch.cuda.Stream()
    fg_op = _make_fg_ops(bufs)["copy"]
    bg_op_n = _make_bg_op(bufs)

    results = {}
    for mode in ["fg_first", "bg_first", "explicit_event"]:
        per_depth = []
        for N in queue_depths:
            per_depth.append({"queue_depth": N,
                              **_measure_one(mode, N, fg_op, bg_op_n,
                                             s_weight, s_act)})
        results[mode] = per_depth
    return results


def _submit(mode, N, s_weight, s_act, fg_op, bg_op_n,
            fg_events=(None, None), bg_events=(None, None)):
    """Submit fg + bg work on separate streams following the given pattern."""
    fg_s, fg_e = fg_events
    bg_s, bg_e = bg_events

    if mode == "fg_first":
        with torch.cuda.stream(s_act):
            if fg_s: fg_s.record(s_act)
            fg_op()
            if fg_e: fg_e.record(s_act)
        with torch.cuda.stream(s_weight):
            if bg_s: bg_s.record(s_weight)
            bg_op_n(N)
            if bg_e: bg_e.record(s_weight)

    elif mode == "bg_first":
        with torch.cuda.stream(s_weight):
            if bg_s: bg_s.record(s_weight)
            bg_op_n(N)
            if bg_e: bg_e.record(s_weight)
        with torch.cuda.stream(s_act):
            if fg_s: fg_s.record(s_act)
            fg_op()
            if fg_e: fg_e.record(s_act)

    elif mode == "explicit_event":
        # Production pattern: both streams submitted in any order, but s_act
        # declares an explicit dependency only on the specific prefetch event
        # it needs (here: none — fg is independent of bg). The key property
        # is that we do NOT rely on implicit ordering between streams.
        prefetch_done = torch.cuda.Event()
        with torch.cuda.stream(s_weight):
            if bg_s: bg_s.record(s_weight)
            bg_op_n(N)
            prefetch_done.record(s_weight)
            if bg_e: bg_e.record(s_weight)
        with torch.cuda.stream(s_act):
            if fg_s: fg_s.record(s_act)
            # fg doesn't need the prefetch; matmul uses a pre-resident W.
            # We still illustrate the pattern: s_act would wait_event here if
            # it needed the prefetched weight.
            # s_act.wait_event(prefetch_done)   # uncomment if fg depends on bg
            fg_op()
            if fg_e: fg_e.record(s_act)

    else:
        raise ValueError(f"unknown mode: {mode}")


# ---------------------------------------------------------------------------
# §0.9b — Bidirectional H2D + D2H
# ---------------------------------------------------------------------------
def bench_bidirectional(bufs):
    s_h2d = torch.cuda.Stream()
    s_d2h = torch.cuda.Stream()

    # H2D work: 4 MB weight prefetch (reuse bufs["wt_cpu"] / wt_gpu).
    wt_cpu, wt_gpu = bufs["wt_cpu"], bufs["wt_gpu"]
    # D2H work: KV spill — 100 KB from GPU to pinned CPU.
    kv_nelts = D2H_KV_BYTES // 2
    kv_gpu = torch.empty(kv_nelts, dtype=torch.bfloat16, device="cuda")
    kv_cpu = torch.empty(kv_nelts, dtype=torch.bfloat16, pin_memory=True)

    def measure(h2d_iters, d2h_iters):
        """Time one round: `h2d_iters` H2Ds on s_h2d + `d2h_iters` D2Hs on s_d2h."""
        for _ in range(WARMUP):
            with torch.cuda.stream(s_h2d):
                for _ in range(h2d_iters):
                    wt_gpu.copy_(wt_cpu, non_blocking=True)
            with torch.cuda.stream(s_d2h):
                for _ in range(d2h_iters):
                    kv_cpu.copy_(kv_gpu, non_blocking=True)
            torch.cuda.synchronize()

        h2d_times, d2h_times = [], []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            h2d_s = torch.cuda.Event(enable_timing=True)
            h2d_e = torch.cuda.Event(enable_timing=True)
            d2h_s = torch.cuda.Event(enable_timing=True)
            d2h_e = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(s_h2d):
                h2d_s.record(s_h2d)
                for _ in range(h2d_iters):
                    wt_gpu.copy_(wt_cpu, non_blocking=True)
                h2d_e.record(s_h2d)
            with torch.cuda.stream(s_d2h):
                d2h_s.record(s_d2h)
                for _ in range(d2h_iters):
                    kv_cpu.copy_(kv_gpu, non_blocking=True)
                d2h_e.record(s_d2h)
            torch.cuda.synchronize()
            if h2d_iters > 0:
                h2d_times.append(h2d_s.elapsed_time(h2d_e))
            if d2h_iters > 0:
                d2h_times.append(d2h_s.elapsed_time(d2h_e))

        h2d_mean = sum(h2d_times)/len(h2d_times) if h2d_times else 0.0
        d2h_mean = sum(d2h_times)/len(d2h_times) if d2h_times else 0.0
        h2d_gbps = (h2d_iters * WEIGHT_BYTES / 1e9) / (h2d_mean / 1000) if h2d_mean else 0
        d2h_gbps = (d2h_iters * D2H_KV_BYTES / 1e9) / (d2h_mean / 1000) if d2h_mean else 0
        return h2d_mean, d2h_mean, h2d_gbps, d2h_gbps

    scenarios = [
        ("h2d_only",  4, 0),
        ("d2h_only",  0, 4),
        ("both_4",    4, 4),
        ("both_4_16", 4, 16),  # D2H-heavy (many small KV spills during one prefetch)
    ]
    out = []
    for name, h_n, d_n in scenarios:
        h_ms, d_ms, h_gbps, d_gbps = measure(h_n, d_n)
        out.append({
            "scenario": name, "h2d_iters": h_n, "d2h_iters": d_n,
            "h2d_ms": round(h_ms, 4), "d2h_ms": round(d_ms, 4),
            "h2d_gbps": round(h_gbps, 3), "d2h_gbps": round(d_gbps, 3),
        })
    return out


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------
def print_activation_mechanism(results):
    print(f"\n§0.9a — Activation mechanism: explicit copy vs UVA (fg_first)")
    print(f"  activation: {N_TOKENS} tokens × {HIDDEN} hidden "
          f"(BF16, ~{N_TOKENS*HIDDEN*2/1024:.0f} KB)")
    print(f"  prefetch:   {WEIGHT_BYTES/1e6:.1f} MB × queue_depth")
    print(f"  {'path':<5} {'N':>3} {'fg_ms':>9} {'bg_ms':>9} {'wall_ms':>10}")
    print(f"  {'-'*5} {'-'*3} {'-'*9} {'-'*9} {'-'*10}")
    for path, rows in results.items():
        for r in rows:
            print(f"  {path:<5} {r['queue_depth']:>3} "
                  f"{r['fg_mean_ms']:>8.4f} {r['bg_mean_ms']:>8.4f} "
                  f"{r['wall_mean_ms']:>9.4f}")


def print_submission_order(results):
    print(f"\n§0.9b — Submission order (path=copy)")
    print(f"  {'mode':<18} {'N':>3} {'fg_ms':>9} {'bg_ms':>9} {'wall_ms':>10}")
    print(f"  {'-'*18} {'-'*3} {'-'*9} {'-'*9} {'-'*10}")
    for mode, rows in results.items():
        for r in rows:
            print(f"  {mode:<18} {r['queue_depth']:>3} "
                  f"{r['fg_mean_ms']:>8.4f} {r['bg_mean_ms']:>8.4f} "
                  f"{r['wall_mean_ms']:>9.4f}")


def print_bidirectional(results):
    print(f"\n§0.9c — Bidirectional H2D + D2H")
    print(f"  h2d: {WEIGHT_BYTES/1e6:.1f} MB × h2d_iters | d2h: {D2H_KV_BYTES/1024:.0f} KB × d2h_iters")
    print(f"  {'scenario':<12} {'h2d_iters':>10} {'d2h_iters':>10} "
          f"{'h2d_ms':>9} {'d2h_ms':>9} {'h2d_gbps':>10} {'d2h_gbps':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['scenario']:<12} {r['h2d_iters']:>10} {r['d2h_iters']:>10} "
              f"{r['h2d_ms']:>8.3f} {r['d2h_ms']:>8.3f} "
              f"{r['h2d_gbps']:>9.2f} {r['d2h_gbps']:>9.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--queue-depths", type=int, nargs="+", default=QUEUE_DEPTHS)
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args()

    print(f"Phase 0.9 — PCIe Contention & Stream Concurrency")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    bufs = make_buffers()

    mechanism_res = bench_activation_mechanism(bufs, args.queue_depths)
    print_activation_mechanism(mechanism_res)

    submission_res = bench_submission_order(bufs, args.queue_depths)
    print_submission_order(submission_res)

    bidir_res = bench_bidirectional(bufs)
    print_bidirectional(bidir_res)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({
                "schema_version": 3,
                "gpu": torch.cuda.get_device_name(0),
                "config": {
                    "hidden": HIDDEN,
                    "n_tokens": N_TOKENS,
                    "weight_bytes": WEIGHT_BYTES,
                    "d2h_kv_bytes": D2H_KV_BYTES,
                },
                "activation_mechanism": mechanism_res,
                "submission_order":     submission_res,
                "bidirectional":        bidir_res,
            }, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
