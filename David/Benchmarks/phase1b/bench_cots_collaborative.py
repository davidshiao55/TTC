#!/usr/bin/env python3
"""Phase 1b §12 Bench 2 — matched total offload depth (headline).

**Question this bench answers:** at a fixed *total* offload budget
(f_cpu_store=0.30 → 30% of every weight on CPU), which dispatch
strategy minimizes latency?

  A) cpu-only      : f_cpu_store=0.30, f_prefetch=0.0   (Phase 1a — pure CPU)
  B) prefetch-only : f_cpu_store=0.30, f_prefetch=0.30  (pure PCIe stream)
  C) collaborative : f_cpu_store=0.30, f_prefetch=0.15  (CPU + prefetch mix)

If C beats min(A, B) → collaborative dispatch is the right choice at
this budget. The headline thesis claim.

**Question this bench does NOT answer:** whether the two paths *contend*
when run concurrently. C and {A, B} expose the *same total bytes* on
the offload boundary but *different per-path bytes* (C splits the
budget; A and B saturate one path with the full budget). So C beating
min(A, B) could be due to either (a) clean overlap or (b) A/B being
over-saturated relative to C's per-path loads. To isolate path
contention you need matched-per-path baselines — see Bench 3
(`bench_cots_path_contention.py`).

Workload: decode-heavy (input=8, output=128); batches {1, 4, 16, 64}.

Outputs go to `David/Benchmarks/phase1b/results/collaborative/`.
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
RESULTS_DIR = PHASE1B_DIR / "results" / "collaborative"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
WARMUP_ITERS = 2
BENCH_ITERS = 3
DEFAULT_BATCHES = [1, 4, 16, 64]
DEFAULT_DEPTH = 0.30  # f_cpu_store


def build_arms(f_cpu_store: float) -> dict[str, dict]:
    """Three matched-depth arms."""
    f_pref_collab = f_cpu_store / 2
    arms = {
        "none": {"flags": [], "family": "none"},
        "A_cpu_only": {
            "flags": [
                "--offload-backend", "cots",
                "--cots-f-cpu-store", str(f_cpu_store),
                "--cots-f-prefetch", "0.0",
            ],
            "family": "cots",
        },
        "B_prefetch_only": {
            "flags": [
                "--offload-backend", "cots",
                "--cots-f-cpu-store", str(f_cpu_store),
                "--cots-f-prefetch", str(f_cpu_store),
            ],
            "family": "cots",
        },
        "C_collaborative": {
            "flags": [
                "--offload-backend", "cots",
                "--cots-f-cpu-store", str(f_cpu_store),
                "--cots-f-prefetch", str(f_pref_collab),
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
    ap.add_argument("--depth", type=float, default=DEFAULT_DEPTH,
                    help="f_cpu_store for the matched-depth arms.")
    ap.add_argument("--only-arms", nargs="*", default=None)
    ap.add_argument("--batches", type=int, nargs="*", default=DEFAULT_BATCHES)
    ap.add_argument("--exp", action="store_true",
                    help="Run benchmarks (else just summarize cached results).")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = build_arms(args.depth)
    run_arms = arms if not args.only_arms else {
        name: arms[name] for name in args.only_arms if name in arms
    }

    print(f"[setup] depth={args.depth}, arms: {len(run_arms)}, "
          f"batches: {args.batches}, input={INPUT_LEN}, output={OUTPUT_LEN}")

    if args.exp:
        for arm, spec in run_arms.items():
            for B in args.batches:
                run_cell(arm, spec["flags"], B)

    summary = summarize(arms, args.batches)
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(
        {"env": env_info(), "model": MODEL,
         "depth_f_cpu_store": args.depth,
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

    # Headline: does collaborative C beat both pure A and pure B?
    print(f"\n[collaborative vs pure @ depth={args.depth}]")
    a, b, c = "A_cpu_only", "B_prefetch_only", "C_collaborative"
    if all(arm in summary for arm in (a, b, c)):
        for B in args.batches:
            la = summary[a]["by_batch"][B]
            lb = summary[b]["by_batch"][B]
            lc = summary[c]["by_batch"][B]
            if la and lb and lc:
                ml = min(la["avg_latency_s"], lb["avg_latency_s"])
                pct = (lc["avg_latency_s"] - ml) / ml * 100
                marker = "WIN" if pct < 0 else "LOSE"
                print(f"  B={B:<2}: A={la['avg_latency_s']:.3f}  "
                      f"B={lb['avg_latency_s']:.3f}  "
                      f"C={lc['avg_latency_s']:.3f}  "
                      f"C vs min(A,B): {pct:+.1f}%  [{marker}]")


if __name__ == "__main__":
    main()
