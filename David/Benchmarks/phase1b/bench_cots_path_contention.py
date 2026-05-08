#!/usr/bin/env python3
"""Phase 1b §12 Bench 3 — matched per-path resource (contention diagnostic).

**Question this bench answers:** when the CPU-compute and GPU-prefetch
paths are run concurrently in the collaborative arm C, do they
*contend* (PCIe ↔ host dispatch ↔ cache ↔ scheduler) or do they
overlap cleanly?

To diagnose contention you need each path's baseline at the EXACT load
C exposes on it:

  A_per_path : pure CPU at f_cpu_store=f_collab     (matches C's CPU-compute load)
  B_per_path : pure prefetch at f_prefetch=f_collab (matches C's PCIe load)
  C          : f_cpu_store=2*f_collab, f_prefetch=f_collab
                                                    (same C as Bench 2)

Default f_collab = 0.15, so the matched baselines are at f_cpu_store=0.15
(half of C's total budget) and C is at f_cpu_store=0.30 — same C arm
that Bench 2 reports. A_per_path and B_per_path move the same bytes
on each path that C does, just without the *other* path running
concurrently.

**Contention metric** (per batch):

    contention_s = T_C - max(T_A_per_path, T_B_per_path)

  ~0  → perfect overlap; the second path is free (collaboration ideal).
  > 0 → paths interfere when concurrent — the cost in seconds is the
        contention overhead. Possible sources: PCIe shared with CPU
        DMA reads of pinned host memory, host-dispatch saturation,
        thread scheduling.
  < 0 → matched-per-path baselines are themselves over-saturated
        relative to C's per-path loads (e.g., a thread-count knob is
        pessimal at higher load). Should be rare.

The fixed-depth comparison (Bench 2) cannot make this statement
because A(f_cpu=0.30) and B(f_pref=0.30) move twice C's per-path bytes
on each path — the comparison conflates "different per-path load" with
"contention".

**Question this bench does NOT answer:** the headline performance claim
("at a fixed offload budget, does collaborative beat the pure paths").
That's Bench 2's job — total offload depth differs across the arms
here (A/B at 0.15 free less GPU memory than C at 0.30), so this bench
is intentionally not a fair offload-budget comparison.

Workload: decode-heavy (input=8, output=128); batches {1, 4, 16, 64}.

Outputs go to `David/Benchmarks/phase1b/results/path_contention/`.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

PHASE1B_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE1B_DIR / "results" / "path_contention"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
WARMUP_ITERS = 2
BENCH_ITERS = 3
DEFAULT_BATCHES = [1, 4, 16, 64]
DEFAULT_F_COLLAB = 0.15  # C's f_prefetch (and C's f_cpu_compute, by symmetry)


def build_arms(f_collab: float) -> dict[str, dict]:
    """Per-path baselines + the C arm.

    f_collab is C's f_prefetch (== C's f_cpu_compute, by the collaborative
    half-and-half split). C's f_cpu_store is `2 * f_collab`. The per-path
    baselines have f_cpu_store=f_collab so they expose the same bytes on
    a single path that C exposes on its corresponding path.
    """
    f_cpu_store_C = 2 * f_collab
    arms = {
        "none": {"flags": [], "family": "none"},
        "A_per_path_cpu": {
            # Pure CPU at C's CPU-compute load: f_cpu_compute = f_collab.
            "flags": [
                "--offload-backend", "cots",
                "--cots-f-cpu-store", str(f_collab),
                "--cots-f-prefetch", "0.0",
            ],
            "family": "cots",
        },
        "B_per_path_prefetch": {
            # Pure prefetch at C's PCIe load: f_prefetch = f_collab.
            # (f_cpu_store == f_prefetch → all CPU-stored bytes stream
            # back via H2D, no CPU compute path.)
            "flags": [
                "--offload-backend", "cots",
                "--cots-f-cpu-store", str(f_collab),
                "--cots-f-prefetch", str(f_collab),
            ],
            "family": "cots",
        },
        "C_collaborative": {
            # Same C as Bench 2 — collaborative split at total offload
            # depth f_cpu_store = 2 * f_collab.
            "flags": [
                "--offload-backend", "cots",
                "--cots-f-cpu-store", str(f_cpu_store_C),
                "--cots-f-prefetch", str(f_collab),
            ],
            "family": "cots",
        },
    }
    return arms


def run_cell(arm: str, flags: list[str], batch: int) -> Path:
    out_json = RESULTS_DIR / f"{arm}_b{batch}.json"
    out_log = RESULTS_DIR / f"{arm}_b{batch}.log"
    if out_json.exists():
        print(f"  [skip] {arm} b={batch} (cached)")
        return out_json

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", MODEL, "--dtype", DTYPE,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--batch-size", str(batch),
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
        print(f"  [FAIL] {arm} b={batch} rc={proc.returncode} ({dur:.1f}s)\n        {tail}")
    else:
        print(f"  [ok]  {arm} b={batch} ({dur:.1f}s)")
    return out_json


def parse_avg(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("avg_latency")
    except (json.JSONDecodeError, OSError):
        return None


def summarize(arms: dict, batches: list[int]) -> dict:
    out = {}
    for arm, spec in arms.items():
        per_b = {}
        for B in batches:
            avg = parse_avg(RESULTS_DIR / f"{arm}_b{B}.json")
            per_b[B] = (
                None if avg is None
                else {
                    "avg_latency_s": round(avg, 4),
                    "tokens_per_s": round(B * OUTPUT_LEN / avg, 1),
                }
            )
        out[arm] = {
            "flags": spec["flags"],
            "family": spec["family"],
            "by_batch": per_b,
        }
    return out


def env_info() -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        })
    except Exception:
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--f-collab", type=float, default=DEFAULT_F_COLLAB,
        help="C's f_prefetch (== C's f_cpu_compute). C's f_cpu_store is "
             "2 * f_collab. Per-path baselines run at f_cpu_store=f_collab.",
    )
    ap.add_argument("--only-arms", nargs="*", default=None)
    ap.add_argument("--batches", type=int, nargs="*", default=DEFAULT_BATCHES)
    ap.add_argument("--exp", action="store_true",
                    help="Run benchmarks (else just summarize cached results).")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = build_arms(args.f_collab)
    run_arms = arms if not args.only_arms else {
        name: arms[name] for name in args.only_arms if name in arms
    }

    print(
        f"[setup] f_collab={args.f_collab} (C uses f_cpu_store="
        f"{2 * args.f_collab}, f_prefetch={args.f_collab}), "
        f"arms: {len(run_arms)}, batches: {args.batches}, "
        f"input={INPUT_LEN}, output={OUTPUT_LEN}"
    )

    if args.exp:
        for arm, spec in run_arms.items():
            for B in args.batches:
                run_cell(arm, spec["flags"], B)

    summary = summarize(arms, args.batches)
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(
        {"env": env_info(), "model": MODEL,
         "f_collab": args.f_collab,
         "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
         "summary": summary},
        indent=2,
    ))
    print(f"[summary] wrote {summary_path}")

    print("\n" + "=" * 80)
    header = f"{'arm':<22}  " + "  ".join(
        f"{f'B={B} (s)':>10}" for B in args.batches
    )
    print(header)
    print("-" * len(header))
    for arm, spec in summary.items():
        row = f"{arm:<22}  "
        for B in args.batches:
            cell = spec["by_batch"][B]
            row += f"{cell['avg_latency_s']:>10.3f}  " if cell else f"{'—':>10}  "
        print(row)
    print("=" * len(header))

    # Contention metric: T_C - max(T_A_per_path, T_B_per_path).
    print(f"\n[path contention @ f_collab={args.f_collab}]")
    a, b, c = "A_per_path_cpu", "B_per_path_prefetch", "C_collaborative"
    if all(arm in summary for arm in (a, b, c)):
        for B in args.batches:
            la = summary[a]["by_batch"][B]
            lb = summary[b]["by_batch"][B]
            lc = summary[c]["by_batch"][B]
            if la and lb and lc:
                ml = max(la["avg_latency_s"], lb["avg_latency_s"])
                contention = lc["avg_latency_s"] - ml
                marker = (
                    "OVERLAP" if contention <= 0.5
                    else "MILD"  if contention <= 2.0
                    else "CONTENT"
                )
                print(
                    f"  B={B:<2}: A_per_path={la['avg_latency_s']:.3f}  "
                    f"B_per_path={lb['avg_latency_s']:.3f}  "
                    f"C={lc['avg_latency_s']:.3f}  "
                    f"contention=C-max(A,B): {contention:+.3f}s  [{marker}]"
                )


if __name__ == "__main__":
    main()
