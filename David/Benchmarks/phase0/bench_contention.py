#!/usr/bin/env python3
"""Phase 0.5 — nsys-driven PCIe contention benchmarks.

Two modes:
  --mode=orchestrator (default): runs self under nsys, parses SQLite, outputs JSON.
  --mode=workload                : runs NVTX-tagged GPU work; child of nsys.

nsys-driven (not CUDA-event-driven) to avoid event-firing-semantics ambiguity
for streams without a compute-tail. All timings are derived from CUPTI activity
records in the nsys SQLite export.

Experiments (numbered to match phase0_findings.md §0.5):
  §0.5.1 — Same-direction H2D serializes on CE0 (1/4/16/64 MB sweep).
  §0.5.2 — Bidirectional H2D + D2H on CE0 + CE1.
  §0.5.3 — DMA copy vs UVA copy kernel, isolated and under bg DMA.
  §0.5.4 — Full zero-copy validation (4 fg variants × 4 bg chunk sizes).

The JSON is the only persistent output. Intermediate .nsys-rep and .sqlite
files are written to a temp dir and deleted after parsing — the bench is
reproducible, so the trace itself isn't a deliverable. For visual inspection
of specific behaviors, use focused probes (e.g. `probe_engines.py`,
`probe_uva_bypass.py`) — short, single-purpose, clean to view in the GUI.

Usage:
    python bench_contention.py
    python bench_contention.py --output-json out.json
    python bench_contention.py --mode=workload   # for nsys to wrap
"""

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


NSYS = "/usr/local/bin/nsys"

HIDDEN = 3584
N_TOKENS = 14
ACT_BYTES = N_TOKENS * HIDDEN * 2          # 98 KB activation
WEIGHT_BYTES = 4 * 1024 * 1024              # 4 MB representative prefetch slice
D2H_KV_BYTES = 100 * 1024                   # 100 KB representative KV chunk

WARMUP = 10
ITERS = 50

H2D_PURE_SIZES_MB = [1, 4, 16, 64]
BIDIR_SCENARIOS = [
    ("h2d_only", 4, 0),
    ("d2h_only", 0, 4),
    ("both_4",   4, 4),
    ("both_4_16", 4, 16),
]
ZC_VARIANTS = ["dma_copy", "uva_copy_kernel", "uva_matmul", "dma_into_matmul"]
ZC_BG_CHUNK_SIZES = [64*1024, 256*1024, 1024*1024, 4*1024*1024]


# ============================================================================
#   WORKLOAD MODE — invoked under nsys
# ============================================================================
def workload_main():
    import torch
    import torch.cuda.nvtx as nvtx
    import triton
    import triton.language as tl
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    @triton.jit
    def _uva_copy_k(in_ptr, out_ptr, n, BS: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BS + tl.arange(0, BS)
        m = off < n
        tl.store(out_ptr + off, tl.load(in_ptr + off, mask=m), mask=m)

    def uva_copy(uva_view, gpu_out, BS=1024):
        n = gpu_out.numel()
        grid = (triton.cdiv(n, BS),)
        _uva_copy_k[grid](uva_view, gpu_out, n, BS=BS)

    # ---- Shared buffers ----
    act_cpu = torch.empty(N_TOKENS * HIDDEN, dtype=torch.bfloat16, pin_memory=True)
    act_gpu = torch.empty_like(act_cpu, device="cuda")
    act_gpu_2d = act_gpu.view(N_TOKENS, HIDDEN)
    act_uva_flat = get_accelerator_view_from_cpu_tensor(act_cpu)
    act_uva_2d = act_uva_flat.view(N_TOKENS, HIDDEN)
    sm_out = torch.empty_like(act_gpu)
    W = torch.randn(HIDDEN, HIDDEN, dtype=torch.bfloat16, device="cuda")
    mm_out = torch.empty(N_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")

    wt_cpu = torch.empty(WEIGHT_BYTES // 2, dtype=torch.bfloat16, pin_memory=True)
    wt_gpu = torch.empty_like(wt_cpu, device="cuda")

    s_a = torch.cuda.Stream()
    s_b = torch.cuda.Stream()

    def iter_range(body):
        """Run WARMUP iters (no NVTX) then ITERS iters under inner NVTX ranges."""
        for _ in range(WARMUP):
            body()
        for i in range(ITERS):
            with nvtx.range(f"iter_{i}"):
                body()

    print(f"[workload] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[workload] WARMUP={WARMUP} ITERS={ITERS}", flush=True)

    # ------------------- §0.5.1 — Equal-size H2D pure contention -------------------
    for size_mb in H2D_PURE_SIZES_MB:
        n = (size_mb * 1024 * 1024) // 2
        a_cpu = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        a_gpu = torch.empty_like(a_cpu, device="cuda")
        b_cpu = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        b_gpu = torch.empty_like(b_cpu, device="cuda")

        with nvtx.range(f"0.5.1_iso_{size_mb}MB"):
            def body_iso():
                with torch.cuda.stream(s_a):
                    a_gpu.copy_(a_cpu, non_blocking=True)
                torch.cuda.synchronize()
            iter_range(body_iso)

        with nvtx.range(f"0.5.1_co_{size_mb}MB"):
            def body_co():
                with nvtx.range("submit_fg"):
                    with torch.cuda.stream(s_a):
                        a_gpu.copy_(a_cpu, non_blocking=True)
                with nvtx.range("submit_bg"):
                    with torch.cuda.stream(s_b):
                        b_gpu.copy_(b_cpu, non_blocking=True)
                torch.cuda.synchronize()
            iter_range(body_co)

        del a_cpu, a_gpu, b_cpu, b_gpu

    # ------------------- §0.5.2 — Bidirectional H2D + D2H -------------------
    kv_n = D2H_KV_BYTES // 2
    kv_gpu = torch.empty(kv_n, dtype=torch.bfloat16, device="cuda")
    kv_cpu = torch.empty(kv_n, dtype=torch.bfloat16, pin_memory=True)

    for name, h_n, d_n in BIDIR_SCENARIOS:
        with nvtx.range(f"0.5.2_{name}"):
            def body(h_n=h_n, d_n=d_n):
                with nvtx.range("submit_h2d"):
                    with torch.cuda.stream(s_a):
                        for _ in range(h_n):
                            wt_gpu.copy_(wt_cpu, non_blocking=True)
                with nvtx.range("submit_d2h"):
                    with torch.cuda.stream(s_b):
                        for _ in range(d_n):
                            kv_cpu.copy_(kv_gpu, non_blocking=True)
                torch.cuda.synchronize()
            iter_range(body)

    # ------------------- §0.5.3 — DMA vs UVA, isolated and with bg -------------------
    fg_ops = {
        "dma": lambda: act_gpu.copy_(act_cpu, non_blocking=True),
        "uva": lambda: uva_copy(act_uva_flat, sm_out),
    }

    for fg_name, fg_op in fg_ops.items():
        with nvtx.range(f"0.5.3_{fg_name}_no_bg"):
            def body(op=fg_op):
                with nvtx.range("submit_fg"):
                    with torch.cuda.stream(s_a):
                        op()
                torch.cuda.synchronize()
            iter_range(body)

        with nvtx.range(f"0.5.3_{fg_name}_with_bg"):
            def body(op=fg_op):
                with nvtx.range("submit_bg"):
                    with torch.cuda.stream(s_b):
                        wt_gpu.copy_(wt_cpu, non_blocking=True)
                with nvtx.range("submit_fg"):
                    with torch.cuda.stream(s_a):
                        op()
                torch.cuda.synchronize()
            iter_range(body)

    # ------------------- §0.5.4 — Full zero-copy validation -------------------
    fg_variants = {
        "dma_copy":        lambda: act_gpu.copy_(act_cpu, non_blocking=True),
        "uva_copy_kernel": lambda: uva_copy(act_uva_flat, sm_out),
        "uva_matmul":      lambda: torch.matmul(act_uva_2d, W.t(), out=mm_out),
        "dma_into_matmul": lambda: (act_gpu.copy_(act_cpu, non_blocking=True),
                                    torch.matmul(act_gpu_2d, W.t(), out=mm_out)),
    }
    total_bg_bytes = 4 * 1024 * 1024
    for variant_name, fg_op in fg_variants.items():
        for chunk_bytes in ZC_BG_CHUNK_SIZES:
            n_per = chunk_bytes // 2
            n_chunks = max(1, total_bg_bytes // chunk_bytes)
            bg_cpu = torch.empty(n_per, dtype=torch.bfloat16, pin_memory=True)
            bg_gpu = torch.empty_like(bg_cpu, device="cuda")
            with nvtx.range(f"0.5.4_{variant_name}_chunk{chunk_bytes//1024}K"):
                def body(op=fg_op, gpu=bg_gpu, cpu=bg_cpu, n=n_chunks):
                    with nvtx.range("submit_bg"):
                        with torch.cuda.stream(s_b):
                            for _ in range(n):
                                gpu.copy_(cpu, non_blocking=True)
                    with nvtx.range("submit_fg"):
                        with torch.cuda.stream(s_a):
                            op()
                    torch.cuda.synchronize()
                iter_range(body)
            del bg_cpu, bg_gpu

    print("[workload] done", flush=True)


# ============================================================================
#   ORCHESTRATOR — runs nsys, parses SQLite, outputs JSON
# ============================================================================
def orchestrator_main(args):
    with tempfile.TemporaryDirectory(prefix="bench_contention_") as td:
        td = Path(td)
        trace_base = td / "trace"

        print(f"[orchestrator] Profiling under nsys (temp dir: {td})")
        subprocess.run([
            NSYS, "profile",
            "-o", str(trace_base),
            "--trace=cuda,nvtx",
            "--force-overwrite=true",
            sys.executable, str(Path(__file__).resolve()), "--mode=workload",
        ], check=True)

        sqlite_path = str(trace_base) + ".sqlite"
        print(f"[orchestrator] Exporting SQLite")
        subprocess.run([
            NSYS, "export", "--type=sqlite",
            "-o", sqlite_path,
            str(trace_base) + ".nsys-rep",
        ], check=True)

        print(f"[orchestrator] Parsing trace …")
        results = parse_trace(sqlite_path)
        results["meta"] = {
            "workload_warmup": WARMUP,
            "workload_iters": ITERS,
        }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[orchestrator] JSON → {args.output_json}")

    print_summary(results)


# ============================================================================
#   PARSER — SQLite → per-experiment metrics
# ============================================================================
def parse_trace(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    nvtx_rows = conn.execute(
        "SELECT start, end, text FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL AND text IS NOT NULL ORDER BY start"
    ).fetchall()
    nvtx = [(r["start"], r["end"], r["text"]) for r in nvtx_rows]

    memcpy_rows = conn.execute(
        "SELECT start, end, streamId, bytes, copyKind, correlationId "
        "FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
    ).fetchall()
    memcpys = [(r["start"], r["end"], r["streamId"], r["bytes"],
                r["copyKind"], r["correlationId"]) for r in memcpy_rows]

    kernels = []
    try:
        kernel_rows = conn.execute(
            "SELECT start, end, streamId, correlationId, demangledName "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        ).fetchall()
        kernels = [(r["start"], r["end"], r["streamId"],
                    r["correlationId"], r["demangledName"]) for r in kernel_rows]
    except sqlite3.OperationalError:
        pass

    runtime_rows = conn.execute(
        "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchall()
    runtime_by_corr = {r["correlationId"]: (r["start"], r["end"])
                       for r in runtime_rows if r["correlationId"] is not None}

    exp_ranges = [(s, e, t) for (s, e, t) in nvtx if t.startswith("0.5.")]
    iter_ranges = [(s, e, t) for (s, e, t) in nvtx if t.startswith("iter_")]
    submit_ranges = [(s, e, t) for (s, e, t) in nvtx if t.startswith("submit_")]

    out = {}
    for exp_s, exp_e, exp_text in exp_ranges:
        iters = [(s, e) for (s, e, _) in iter_ranges if s >= exp_s and e <= exp_e]
        if not iters:
            continue

        per_iter = []
        for iter_s, iter_e in iters:
            submits = [(s, e, t) for (s, e, t) in submit_ranges
                       if s >= iter_s and e <= iter_e]

            role_intervals = defaultdict(list)
            for s, e, t in submits:
                role = t.replace("submit_", "")
                role_intervals[role].append((s, e))

            iter_memcpys = []
            iter_kernels = []
            for ms, me, sid, bytes_, ckind, corr in memcpys:
                rt = runtime_by_corr.get(corr)
                if rt and iter_s <= rt[0] <= iter_e:
                    iter_memcpys.append({
                        "start": ms, "end": me, "duration": me - ms,
                        "streamId": sid, "bytes": bytes_, "copyKind": ckind,
                        "role": _role_for(rt, role_intervals),
                    })
            for ks, ke, sid, corr, kname in kernels:
                rt = runtime_by_corr.get(corr)
                if rt and iter_s <= rt[0] <= iter_e:
                    iter_kernels.append({
                        "start": ks, "end": ke, "duration": ke - ks,
                        "streamId": sid, "name": kname,
                        "role": _role_for(rt, role_intervals),
                    })

            iter_data = {
                "iter_wall_ns": iter_e - iter_s,
                "memcpys": iter_memcpys,
                "kernels": iter_kernels,
            }
            for role in role_intervals:
                role_events = [m for m in iter_memcpys if m["role"] == role] + \
                              [k for k in iter_kernels if k["role"] == role]
                if role_events:
                    starts = [e["start"] for e in role_events]
                    ends = [e["end"] for e in role_events]
                    iter_data[f"{role}_wall_ns"] = max(ends) - min(starts)
                    iter_data[f"{role}_active_ns"] = sum(e["duration"] for e in role_events)
                    earliest_submit = min(s for s, _ in role_intervals[role])
                    iter_data[f"{role}_submit_to_complete_ns"] = max(ends) - earliest_submit

            all_events = iter_memcpys + iter_kernels
            if all_events:
                iter_data["gpu_wall_ns"] = (max(e["end"] for e in all_events)
                                             - min(e["start"] for e in all_events))
            per_iter.append(iter_data)

        out[exp_text] = _aggregate(per_iter)
    return out


def _role_for(host_interval, role_intervals):
    hs, _ = host_interval
    for role, ivs in role_intervals.items():
        for s, e in ivs:
            if s <= hs <= e:
                return role
    return None


def _aggregate(per_iter):
    if not per_iter:
        return {"per_iter": [], "n_iters": 0}
    fields = defaultdict(list)
    for it in per_iter:
        for k, v in it.items():
            if isinstance(v, (int, float)):
                fields[k].append(v)
    summary = {"n_iters": len(per_iter)}
    for k, vs in fields.items():
        vs_sorted = sorted(vs)
        summary[k] = {
            "mean_us":   round(mean(vs) / 1000, 4),
            "median_us": round(median(vs) / 1000, 4),
            "p99_us":    round(vs_sorted[int(len(vs_sorted) * 0.99)] / 1000, 4),
        }
    return summary


# ============================================================================
#   Pretty-printer
# ============================================================================
def _us(d, key="median_us"):
    if isinstance(d, dict) and key in d:
        return f"{d[key]:7.2f}"
    return "       "


def print_summary(results):
    print()
    print("§0.5.1 — Same-direction H2D contention (median μs)")
    print(f"  {'size':<8} {'iso':>10} {'co fg':>10} {'co bg':>10} {'wall':>10}")
    for size_mb in H2D_PURE_SIZES_MB:
        iso = results.get(f"0.5.1_iso_{size_mb}MB", {})
        co  = results.get(f"0.5.1_co_{size_mb}MB", {})
        iso_active = iso.get("gpu_wall_ns") or iso.get("iter_wall_ns")
        co_fg = co.get("fg_active_ns") or co.get("fg_wall_ns")
        co_bg = co.get("bg_active_ns") or co.get("bg_wall_ns")
        co_wall = co.get("gpu_wall_ns")
        print(f"  {str(size_mb)+'MB':<8} {_us(iso_active)} "
              f"{_us(co_fg)} {_us(co_bg)} {_us(co_wall)}")

    print("\n§0.5.2 — Bidirectional H2D + D2H (median μs)")
    print(f"  {'scenario':<14} {'h2d wall':>10} {'d2h wall':>10} {'iter wall':>10}")
    for name, _, _ in BIDIR_SCENARIOS:
        r = results.get(f"0.5.2_{name}", {})
        print(f"  {name:<14} {_us(r.get('h2d_wall_ns'))} "
              f"{_us(r.get('d2h_wall_ns'))} {_us(r.get('iter_wall_ns'))}")

    print("\n§0.5.3 — DMA vs UVA, isolated and with bg (median μs)")
    print(f"  fg_s2c = fg submission→complete (includes any DMA-queue wait)")
    print(f"  {'fg':<6} {'bg':<10} {'fg s2c':>10} {'fg active':>10} "
          f"{'bg active':>10} {'iter wall':>10}")
    for fg in ["dma", "uva"]:
        for bg in ["no_bg", "with_bg"]:
            r = results.get(f"0.5.3_{fg}_{bg}", {})
            print(f"  {fg:<6} {bg:<10} "
                  f"{_us(r.get('fg_submit_to_complete_ns'))} "
                  f"{_us(r.get('fg_active_ns'))} "
                  f"{_us(r.get('bg_active_ns'))} "
                  f"{_us(r.get('iter_wall_ns'))}")

    print("\n§0.5.4 — Full zero-copy validation (median μs)")
    print(f"  fg_s2c = fg submission→complete latency under continuous bg DMA")
    print(f"  {'variant':<18} {'chunk':>7} {'fg s2c':>10} {'fg active':>10} "
          f"{'bg s2c':>10} {'iter wall':>10}")
    for variant in ZC_VARIANTS:
        for chunk in ZC_BG_CHUNK_SIZES:
            r = results.get(f"0.5.4_{variant}_chunk{chunk//1024}K", {})
            chunk_str = f"{chunk//1024}K" if chunk < 1024*1024 else f"{chunk//(1024*1024)}M"
            print(f"  {variant:<18} {chunk_str:>7} "
                  f"{_us(r.get('fg_submit_to_complete_ns'))} "
                  f"{_us(r.get('fg_active_ns'))} "
                  f"{_us(r.get('bg_submit_to_complete_ns'))} "
                  f"{_us(r.get('iter_wall_ns'))}")


# ============================================================================
#   ENTRY POINT
# ============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["workload", "orchestrator"],
                   default="orchestrator")
    p.add_argument("--output-json",
                   default="results/0.5_pcie/contention.json")
    args = p.parse_args()

    if args.mode == "workload":
        workload_main()
    else:
        orchestrator_main(args)


if __name__ == "__main__":
    main()
