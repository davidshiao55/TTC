#!/usr/bin/env python3
"""Phase 1a §1.14 (Claim A) — Wrapper-only COTS vs none vs full COTS.

Three-arm decode-heavy bench at the f=0.05 B=1 "free regime" cell. Isolates
the cost of the COTS dispatcher path from the cost of the CPU GEMM itself:

    none              wrappers absent, no offload                  → baseline
    cots_005_dryrun   wrappers installed, worker GEMM is a noop    → orchestration
    cots_005_real     wrappers installed, real CPU GEMM runs       → end-to-end

Decomposition of the 2.50s real-vs-none gap:

    cots_005_real  − cots_005_dryrun  =  active CPU-work penalty
                                        (GEMM time + oneDNN/runtime interference;
                                         upper bound on post-Phase-1c floor)
    cots_005_dryrun − none            =  pure host orchestration / Python wrapper

`dryrun` requires the `--cots-dry-run` flag (added in offload.py / arg_utils.py;
see `phase1a_findings.md §1.14`). Token output is garbage in dryrun mode — we
only care about wall clock.

Outputs go to ``David/Benchmarks/phase1/results/dryrun_vs_none/``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE1_DIR / "results" / "dryrun_vs_none"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
WARMUP_ITERS = 2
BENCH_ITERS = 3
DEFAULT_BATCHES = [1, 4]
DEFAULT_F = 0.05
DEFAULT_THREADS = 16


def arms_for(threads: int) -> dict[str, list[str]]:
    """COTS arms parameterized by `--cots-cpu-num-threads`. `none` is
    independent of `t` (no cots wrappers). Default `t=16` matches the
    existing `cots_vs_native_decode` baseline so the t=16 cell can reuse
    those results without re-running."""
    cots_base = ["--offload-backend", "cots", "--cots-f-cpu-store", str(DEFAULT_F)]
    if threads != DEFAULT_THREADS:
        cots_base += ["--cots-cpu-num-threads", str(threads)]
    return {
        "none": [],
        "cots_005_dryrun": cots_base + ["--cots-dry-run"],
        "cots_005_real": cots_base,
    }


def cell_path(arm: str, batch: int, threads: int) -> Path:
    """`none` is t-invariant — its path never carries a `_t{N}` suffix
    (otherwise the summarizer can't find it for thread sweeps where the
    first `--threads` value isn't 16). For cots arms, t=16 keeps the
    original filename for backward compatibility with the initial run;
    other t values get a `_t{N}` suffix."""
    if arm == "none" or threads == DEFAULT_THREADS:
        return RESULTS_DIR / f"{arm}_b{batch}.json"
    return RESULTS_DIR / f"{arm}_b{batch}_t{threads}.json"


def run_cell(arm: str, flags: list[str], batch: int, threads: int) -> Path:
    out_json = cell_path(arm, batch, threads)
    out_log = out_json.with_suffix(".log")
    if out_json.exists():
        print(f"  [skip] {arm} b={batch} (cached)")
        return out_json
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", MODEL, "--dtype", DTYPE,
        "--input-len", str(INPUT_LEN), "--output-len", str(OUTPUT_LEN),
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
        avg = json.loads(out_json.read_text()).get("avg_latency")
        print(f"  [ok]  {arm} b={batch}: avg={avg:.4f}s ({dur:.1f}s)")
    return out_json


def parse_avg(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("avg_latency")
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, nargs="*", default=DEFAULT_BATCHES)
    ap.add_argument("--threads", type=int, nargs="*", default=[DEFAULT_THREADS])
    ap.add_argument("--only-arms", nargs="*", default=None)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] threads={args.threads}, batches={args.batches}, "
          f"f={DEFAULT_F}, input={INPUT_LEN}, output={OUTPUT_LEN}")

    for t in args.threads:
        arms = arms_for(t)
        run_arms = arms if not args.only_arms else {
            n: arms[n] for n in args.only_arms if n in arms
        }
        # `none` is t-invariant — only run it once (at the first t).
        if t != args.threads[0] and "none" in run_arms:
            del run_arms["none"]
        print(f"\n[t={t}] arms={list(run_arms)}")
        for arm, flags in run_arms.items():
            for B in args.batches:
                run_cell(arm, flags, B, t)

    # Build summary table: rows = (arm, t), cols = batches.
    print("\n" + "=" * 80)
    print(f"{'arm':<22} {'t':>3}  " + "  ".join(
        f"{f'B={B} (s)':>11}" for B in args.batches
    ))
    print("-" * 80)
    none_by_b = {B: parse_avg(cell_path("none", B, args.threads[0]))
                 for B in args.batches}
    rows: dict = {}
    for t in args.threads:
        for arm in ("cots_005_dryrun", "cots_005_real"):
            row = {B: parse_avg(cell_path(arm, B, t)) for B in args.batches}
            rows[(arm, t)] = row
            cells = "  ".join(
                f"{row[B]:>11.4f}" if row[B] is not None else f"{'—':>11}"
                for B in args.batches
            )
            print(f"{arm:<22} {t:>3}  {cells}")
    cells = "  ".join(
        f"{none_by_b[B]:>11.4f}" if none_by_b[B] is not None else f"{'—':>11}"
        for B in args.batches
    )
    print(f"{'none':<22} {'-':>3}  {cells}")
    print("=" * 80)

    print("\n=== Decomposition by t (s, per generate) ===")
    for t in args.threads:
        for B in args.batches:
            none = none_by_b[B]
            dryrun = rows[("cots_005_dryrun", t)][B]
            real = rows[("cots_005_real", t)][B]
            if None in (none, dryrun, real):
                continue
            wrap = dryrun - none
            gemm = real - dryrun
            gap = real - none
            print(f"  t={t} B={B}:  orch={wrap:+.4f}s  "
                  f"cpu-work={gemm:+.4f}s  total={gap:+.4f}s "
                  f"(orch={wrap / gap:.0%})")

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps({
        "model": MODEL, "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
        "f": DEFAULT_F, "batches": args.batches, "threads": args.threads,
        "none": {str(B): v for B, v in none_by_b.items()},
        "cots": {
            f"{arm}_t{t}": {str(B): v for B, v in rows[(arm, t)].items()}
            for (arm, t) in rows
        },
    }, indent=2))
    print(f"\n[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
