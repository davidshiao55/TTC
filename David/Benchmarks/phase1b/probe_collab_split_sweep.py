#!/usr/bin/env python3
"""Phase 1b — collaborative-split sweep at the losing batches.

Bench 2 shows C_collaborative at f_cpu_store=0.30, f_pref=0.15 (50/50
split) loses to B_prefetch_only at B=16 and B=64. The hypothesis: at
large B the CPU GEMM scales with batch but PCIe time is constant, so
the optimal split shifts toward prefetch-heavy. With the right
f_prefetch (and tiny f_cpu_compute), collaborative should still beat
pure prefetch at every B — just the ratio changes.

Linear cost model from Bench 3 per-path baselines
(`A_per_path(f_cpu=0.15)` and `B_per_path(f_pref=0.15)`):

  T_pre(f_pref)        ≈ 10 s × (f_pref / 0.15)        (PCIe; const w.r.t. B)
  T_cpu(f_cpu, B=16)   ≈ 28.27 s × (f_cpu / 0.15)
  T_cpu(f_cpu, B=64)   ≈ 93.09 s × (f_cpu / 0.15)
  constraint:          f_cpu + f_pref = f_cpu_store = 0.30

Balanced-overlap optimum (T_cpu == T_pre):

  B=16: f_pref ≈ 0.22, predicted C ≈ 17 s   (vs B_arm 20 s → -15%)
  B=64: f_pref ≈ 0.27, predicted C ≈ 19 s   (vs B_arm 20 s → -5%)

This sweep brackets both: f_pref ∈ {0.20, 0.25, 0.28} at f_cpu_store=0.30.
At f_pref=0.30 we already have the pure-prefetch (B_arm) baseline from
Bench 2. At f_pref=0.15 we have the 50/50 C arm (loses at large B).

Workload: input=8, output=128. Cells at B={16, 64} only — small B is
already a clear collaborative win in Bench 2.

Outputs go to David/Benchmarks/phase1b/results/collab_split_sweep/.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PHASE1B_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE1B_DIR / "results" / "collab_split_sweep"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
WARMUP_ITERS = 2
BENCH_ITERS = 3
F_CPU_STORE = 0.30
SWEEP_F_PREF = (0.20, 0.25, 0.28)
SWEEP_BATCHES = (16, 64)


def run(name: str, flags: list[str]) -> Path:
    out_json = RESULTS_DIR / f"{name}.json"
    out_log = RESULTS_DIR / f"{name}.log"
    if out_json.exists():
        print(f"  [skip] {name} (cached)")
        return out_json

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", MODEL, "--dtype", DTYPE,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--num-iters-warmup", str(WARMUP_ITERS),
        "--num-iters", str(BENCH_ITERS),
        "--enforce-eager",
        "--output-json", str(out_json),
        *flags,
    ]
    t0 = time.perf_counter()
    with open(out_log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    dur = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-15:])
        print(f"  [FAIL] {name} rc={proc.returncode} ({dur:.1f}s)\n        {tail}")
    else:
        print(f"  [ok]  {name} ({dur:.1f}s)")
    return out_json


def parse(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("avg_latency")
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cells: list[tuple[str, list[str]]] = []
    for B in SWEEP_BATCHES:
        for f_pref in SWEEP_F_PREF:
            cells.append((
                f"C_split_b{B}_fpref{f_pref:.2f}",
                [
                    "--batch-size", str(B),
                    "--offload-backend", "cots",
                    "--cots-f-cpu-store", str(F_CPU_STORE),
                    "--cots-f-prefetch", str(f_pref),
                ],
            ))

    print(
        f"[setup] {len(cells)} cells, f_cpu_store={F_CPU_STORE}, "
        f"f_pref ∈ {SWEEP_F_PREF}, B ∈ {SWEEP_BATCHES}"
    )
    for name, flags in cells:
        run(name, flags)

    # Existing endpoints from Bench 2 results (cached at f_pref=0.15 and 0.30).
    BENCH2_DIR = PHASE1B_DIR / "results" / "collaborative"
    print(f"\n=== collab-split sweep at f_cpu_store={F_CPU_STORE} ===")
    print(f"{'B':<4} " + " ".join(
        f"{'f_pref=' + f'{x:.2f}':>14}" for x in (0.15, *SWEEP_F_PREF, 0.30)
    ))
    print("-" * (4 + 1 + 15 * (len(SWEEP_F_PREF) + 2)))
    for B in SWEEP_BATCHES:
        row = f"{B:<4} "
        # 0.15 = Bench 2 C arm
        c15 = parse(BENCH2_DIR / f"C_collaborative_b{B}.json")
        row += f"{c15:>13.3f}s " if c15 else f"{'—':>14}"
        for f_pref in SWEEP_F_PREF:
            v = parse(RESULTS_DIR / f"C_split_b{B}_fpref{f_pref:.2f}.json")
            row += f"{v:>13.3f}s " if v else f"{'—':>14}"
        # 0.30 = Bench 2 B arm (pure prefetch)
        c30 = parse(BENCH2_DIR / f"B_prefetch_only_b{B}.json")
        row += f"{c30:>13.3f}s " if c30 else f"{'—':>14}"
        print(row)

    # Find the per-batch winner.
    print("\n[per-batch optimum]")
    for B in SWEEP_BATCHES:
        cands: list[tuple[str, float]] = []
        c15 = parse(BENCH2_DIR / f"C_collaborative_b{B}.json")
        if c15:
            cands.append(("f_pref=0.15", c15))
        for f_pref in SWEEP_F_PREF:
            v = parse(RESULTS_DIR / f"C_split_b{B}_fpref{f_pref:.2f}.json")
            if v:
                cands.append((f"f_pref={f_pref:.2f}", v))
        c30 = parse(BENCH2_DIR / f"B_prefetch_only_b{B}.json")
        if c30:
            cands.append(("f_pref=0.30 (=B_arm)", c30))
        if cands:
            best_name, best_val = min(cands, key=lambda kv: kv[1])
            b_arm = c30 if c30 else float("inf")
            margin = (best_val - b_arm) / b_arm * 100
            tag = "WIN_vs_B" if margin < 0 else "TIE_vs_B"
            print(
                f"  B={B:<2}: best = {best_name:>20s} at {best_val:.3f}s  "
                f"(B_arm={b_arm:.3f}s, margin={margin:+.1f}%) [{tag}]"
            )


if __name__ == "__main__":
    main()
