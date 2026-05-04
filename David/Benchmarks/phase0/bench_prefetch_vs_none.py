#!/usr/bin/env python3
"""Phase 0 §0.10.3 — Prefetch vs none, finding the prefetch free regime.

Three-arm sweep that mirrors the COTS-vs-none methodology of
`David/Benchmarks/phase1/bench_cots_dryrun_vs_none.py` (`phase1a_findings.md
§1.14`), applied to the native vLLM `PrefetchOffloader`. Decomposes the
prefetch-vs-`none` gap into pure host orchestration (Python wrappers +
`wait_prefetch` / `start_prefetch` custom op dispatch + stream/event sync)
versus the actual unhidden H2D transfer cost.

    none                          no offload                                → baseline
    prefetch_dryrun_G{g}          all wrappers, copy skipped via            → orchestration
                                  `--prefetch-dry-run` flag
    prefetch_real_G{g}            real H2D copy                             → end-to-end

Decomposition:

    prefetch_real − prefetch_dryrun  =  unhidden PCIe (the actual H2D cost
                                        not absorbed by the GPU layer budget)
    prefetch_dryrun − none           =  pure host orchestration
                                        (what `_ModuleOffloader` overhead costs
                                         even with the copy itself removed)

Coverage axis: N=1, K=1 (the empirically best prefetch knob choice per
§0.10's "denser uniform spacing dominates clustering" finding); G varies
across divisors of Qwen2.5-7B's 28 decoder layers so coverage = 1/G is
exact at every cell.

Two workload grids:

    decode_heavy      input=8,   output=128, B ∈ {1, 4, 16, 64}
                      Matches §1.14's COTS condition exactly so the prefetch
                      and COTS gap tables can be overlaid apples-to-apples.

    pf_match          input=256, output=32,  B ∈ {1, 16, 64}
                      Matches `bench_prefetch_knobs.py` so the
                      `prefetch_real_G14` cell can be cross-checked against
                      the existing knob-sweep numbers.

The "free regime" is the (B, G) cell where `total = real − none ≤ ~0`
within run-to-run noise.

`--prefetch-dry-run` requires the field added to `PrefetchOffloadConfig`
in `vllm/config/offload.py` and the skip guard in
`vllm/model_executor/offloader/prefetch.py:_ModuleOffloader.start_onload_to_static`.

Outputs go to ``David/Benchmarks/phase0/results/0.10_prefetch_vs_none/``
under per-grid sub-dirs.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PHASE0_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE0_DIR / "results" / "0.10_prefetch_vs_none"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
WARMUP_ITERS = 2
BENCH_ITERS = 3

# Qwen2.5-7B has 28 decoder layers — G must divide 28 for clean uniform
# grouping. Full divisor set: {1, 2, 4, 7, 14, 28}; coverage = 1/G ∈
# {100%, 50%, 25%, 14.3%, 7.1%, 3.6%}.
DEFAULT_GS = [1, 2, 4, 7, 14, 28]

GRIDS: dict[str, dict] = {
    "decode_heavy": {
        "input_len": 8,
        "output_len": 128,
        "batches": [1, 4, 16, 64],
    },
    "pf_match": {
        "input_len": 256,
        "output_len": 32,
        "batches": [1, 16, 64],
    },
}


def arms_for(G: int) -> dict[str, list[str]]:
    """Arms parameterized by G. `none` is G-invariant (no prefetch flags)
    — handled separately so its JSON path doesn't carry a `_g{G}` suffix.

    `prefetch_real_g{G}` uses the default `defer_wraparound=True` (the fix
    landed alongside §0.10.5). `prefetch_real_unfixed_g{G}` toggles the fix
    off via `--no-prefetch-defer-wraparound` so the same workload is
    measured under the legacy end-of-iter-N prefetch ordering — gives an
    in-script before/after at matched conditions instead of relying on the
    `0.10_prefetch_vs_none-unfixed/` snapshot from a different driver day.
    """
    base = [
        "--offload-group-size", str(G),
        "--offload-num-in-group", "1",
        "--offload-prefetch-step", "1",
    ]
    return {
        f"prefetch_dryrun_g{G}": base + ["--prefetch-dry-run"],
        f"prefetch_real_g{G}": base,
        f"prefetch_real_unfixed_g{G}": base + ["--no-prefetch-defer-wraparound"],
    }


def cell_path(grid: str, arm: str, batch: int) -> Path:
    return RESULTS_DIR / grid / f"{arm}_b{batch}.json"


def run_cell(grid: str, arm: str, flags: list[str], batch: int) -> Path:
    spec = GRIDS[grid]
    out_json = cell_path(grid, arm, batch)
    out_log = out_json.with_suffix(".log")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists():
        print(f"  [skip] {grid}/{arm} b={batch} (cached)")
        return out_json
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", MODEL, "--dtype", DTYPE,
        "--input-len", str(spec["input_len"]),
        "--output-len", str(spec["output_len"]),
        "--batch-size", str(batch),
        "--num-iters-warmup", str(WARMUP_ITERS),
        "--num-iters", str(BENCH_ITERS),
        "--enforce-eager",
        "--output-json", str(out_json),
        *flags,
    ]
    t0 = time.perf_counter()
    # 5 min covers the largest legitimate cell (B=64, ~30s wall) with margin.
    # vLLM v1 spawns an engine-core worker; on timeout we must kill the whole
    # process group or the worker keeps the GPU busy and breaks later cells.
    PER_CELL_TIMEOUT = 300
    with open(out_log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=PER_CELL_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
            dur = time.perf_counter() - t0
            if out_json.exists():
                try:
                    avg = json.loads(out_json.read_text()).get("avg_latency")
                    print(f"  [hung-but-ok] {grid}/{arm} b={batch}: avg={avg:.4f}s ({dur:.1f}s, killed)")
                    return out_json
                except (json.JSONDecodeError, OSError):
                    pass
            print(f"  [TIMEOUT] {grid}/{arm} b={batch} ({dur:.1f}s)")
            return out_json
    dur = time.perf_counter() - t0
    if rc != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-15:])
        print(f"  [FAIL] {grid}/{arm} b={batch} rc={rc} ({dur:.1f}s)\n        {tail}")
    else:
        avg = json.loads(out_json.read_text()).get("avg_latency")
        print(f"  [ok]  {grid}/{arm} b={batch}: avg={avg:.4f}s ({dur:.1f}s)")
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
    ap.add_argument("--grids", nargs="*", default=list(GRIDS),
                    choices=list(GRIDS),
                    help="Workload grids to run; default both.")
    ap.add_argument("--gs", type=int, nargs="*", default=DEFAULT_GS,
                    help="G values to sweep (must divide 28).")
    ap.add_argument("--batches", type=int, nargs="*", default=None,
                    help="Override per-grid batches (applies to all selected grids).")
    ap.add_argument("--only-arms", nargs="*", default=None,
                    choices=["none", "prefetch_dryrun", "prefetch_real",
                             "prefetch_real_unfixed"],
                    help="Restrict which arms to run (G is appended at runtime).")
    args = ap.parse_args()

    bad_g = [g for g in args.gs if 28 % g != 0]
    if bad_g:
        sys.exit(f"G values must divide 28; bad: {bad_g}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] grids={args.grids}, gs={args.gs}, "
          f"batches={'(per-grid)' if args.batches is None else args.batches}, "
          f"only_arms={args.only_arms}")

    # Run each grid.
    for grid in args.grids:
        spec = GRIDS[grid]
        batches = args.batches if args.batches is not None else spec["batches"]
        print(f"\n[grid={grid}] input={spec['input_len']} "
              f"output={spec['output_len']} batches={batches}")

        # `none` is G-invariant — run once per grid per batch.
        if args.only_arms is None or "none" in args.only_arms:
            for B in batches:
                run_cell(grid, "none", [], B)

        for G in args.gs:
            arms = arms_for(G)
            if args.only_arms is not None:
                wanted_prefixes = [a for a in args.only_arms if a != "none"]
                # Match `prefix_g{G}` exactly so e.g. "prefetch_real" doesn't
                # also pick up "prefetch_real_unfixed_g{G}".
                arms = {n: f for n, f in arms.items()
                        if any(n.startswith(p + "_g") for p in wanted_prefixes)}
            for arm, flags in arms.items():
                for B in batches:
                    run_cell(grid, arm, flags, B)

    # Summary table per grid.
    summary: dict = {
        "model": MODEL,
        "grids": {},
    }
    for grid in args.grids:
        spec = GRIDS[grid]
        batches = args.batches if args.batches is not None else spec["batches"]
        none_by_b = {B: parse_avg(cell_path(grid, "none", B)) for B in batches}

        print("\n" + "=" * 88)
        print(f"GRID: {grid}  (input={spec['input_len']}, output={spec['output_len']})")
        print(f"{'arm':<26} {'G':>3}  " + "  ".join(
            f"{f'B={B} (s)':>11}" for B in batches
        ))
        print("-" * 88)
        rows: dict = {}
        for G in args.gs:
            for arm_prefix in ("prefetch_dryrun", "prefetch_real",
                               "prefetch_real_unfixed"):
                arm = f"{arm_prefix}_g{G}"
                row = {B: parse_avg(cell_path(grid, arm, B)) for B in batches}
                rows[(arm_prefix, G)] = row
                cells = "  ".join(
                    f"{row[B]:>11.4f}" if row[B] is not None else f"{'—':>11}"
                    for B in batches
                )
                print(f"{arm_prefix:<26} {G:>3}  {cells}")
        cells = "  ".join(
            f"{none_by_b[B]:>11.4f}" if none_by_b[B] is not None else f"{'—':>11}"
            for B in batches
        )
        print(f"{'none':<26} {'-':>3}  {cells}")
        print("=" * 88)

        # Decomposition.
        print(f"\n=== {grid}: decomposition (s, per generate) ===")
        for G in args.gs:
            cov_pct = 100.0 / G
            for B in batches:
                none = none_by_b[B]
                dryrun = rows[("prefetch_dryrun", G)][B]
                real = rows[("prefetch_real", G)][B]
                unfixed = rows[("prefetch_real_unfixed", G)][B]
                if None in (none, dryrun, real):
                    continue
                orch = dryrun - none
                pcie = real - dryrun
                gap = real - none
                orch_pct = orch / gap if gap not in (0.0, None) and abs(gap) > 1e-6 else float("nan")
                free_marker = "  ← FREE" if gap <= 0.05 else ""  # within 50ms noise
                # Fix delta: how much the deferred-wraparound fix saved vs
                # the legacy prefetch ordering at matched conditions.
                fix_str = ""
                if unfixed is not None:
                    saved = unfixed - real
                    fix_str = f"  fix Δ={saved:+.4f}s"
                print(f"  G={G:>2} (cov {cov_pct:>5.1f}%) B={B:>2}: "
                      f"orch={orch:+.4f}s  pcie={pcie:+.4f}s  total={gap:+.4f}s "
                      f"(orch={orch_pct:.0%}){fix_str}{free_marker}")

        summary["grids"][grid] = {
            "input_len": spec["input_len"],
            "output_len": spec["output_len"],
            "batches": batches,
            "gs": args.gs,
            "none": {str(B): v for B, v in none_by_b.items()},
            "prefetch": {
                f"{prefix}_g{G}": {str(B): v for B, v in rows[(prefix, G)].items()}
                for (prefix, G) in rows
            },
        }

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
