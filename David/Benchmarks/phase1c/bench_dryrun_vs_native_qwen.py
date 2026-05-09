#!/usr/bin/env python3
"""Phase 1c real-model §1.14 anchor — Qwen2.5-7B + FastTTS-equivalent decode.

The thesis-locked absolute orch number. Phase 1a's
`bench_cots_dryrun_vs_none.py` ported with two changes:

  1. A new `cots_005_native_capture_dryrun` arm — `cpu_runner=native`,
     no `--enforce-eager`. Captured graph replay re-issues
     `cudaLaunchHostFunc` host-callback nodes only; per-operator Python
     orchestration is gone.
  2. Native + eager + dryrun arm — for substrate-vs-capture
     decomposition (Stage 2 substrate gate vs Stage 5 collapse).

§1.14 absolute target on Qwen2.5-7B BF16 decode-heavy
(input=8, output=128):

    orch_python_eager = T(cots_005_python_eager_dryrun) − T(none)
                      ≈ 0.45 s/generate  (Phase 1a baseline)
    orch_native_capture = T(cots_005_native_capture_dryrun) − T(none)
                      target ≤ 0.05 s/generate  (Stage 5 collapse)

The synthetic shape-collapse bench
(`bench_dryrun_vs_real_native.py`) confirmed the SHAPE on a stub
workload (ratio 0.477). THIS bench locks the absolute on the real
model — different layer count (28 vs 8), different HIDDEN (3584 vs
256), real attention + MLP between QKV calls.

Outputs go to `David/Benchmarks/phase1c/results/dryrun_vs_native_qwen/`.
First-run cells take ~30s each; the full grid (5 arms × default 2
batches) is ~5 minutes. Set `--only-arms` to subset; previously-saved
JSON cells are skipped.

Run examples:
    # full grid, default batches [1, 4]
    /opt/conda/envs/thesis/bin/python bench_dryrun_vs_native_qwen.py

    # just the §1.14 collapse comparison
    /opt/conda/envs/thesis/bin/python bench_dryrun_vs_native_qwen.py \\
        --only-arms none cots_005_python_eager_dryrun \\
        cots_005_native_capture_dryrun

    # check a smaller grid first
    /opt/conda/envs/thesis/bin/python bench_dryrun_vs_native_qwen.py \\
        --batches 1 --num-iters 1 --num-iters-warmup 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PHASE1C_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE1C_DIR / "results" / "dryrun_vs_native_qwen"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
DEFAULT_BATCHES = [1, 4]
DEFAULT_F = 0.05
DEFAULT_THREADS = 16


# Five arms. `none` is the no-offload baseline (eager). The four cots
# arms span (runner ∈ {python, native}) × (eager-vs-graph) × (dryrun
# vs real). Capture mode is only legal under cpu_runner=native (Phase
# 1c Stage 5 conditional check); python+graph would hard-fail at
# post_init. The "real" cells under native+capture are the production
# Phase 1c path; "dryrun" cells under that path measure pure orch.
def arms_for(threads: int) -> dict[str, list[str]]:
    cots_base = [
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        str(DEFAULT_F),
    ]
    if threads != DEFAULT_THREADS:
        cots_base += ["--cots-cpu-num-threads", str(threads)]
    return {
        # No-offload baseline. Eager — the offloader doesn't construct
        # so the runner choice is moot; matches phase1a's `none`.
        "none": [],
        # Phase 1a baseline orch: python runner, eager, dryrun. The
        # 0.45 s/generate §1.14 reference number.
        "cots_005_python_eager_dryrun": cots_base
        + ["--cots-cpu-runner", "python", "--cots-dry-run"],
        # Stage 2 substrate-gate cell on the real model: native+eager+dryrun.
        # Validates that at-model-scale the C++ host-callback round-trip
        # is no slower than python's executor.submit/future.result.
        "cots_005_native_eager_dryrun": cots_base
        + ["--cots-cpu-runner", "native", "--cots-dry-run"],
        # THE Stage 5 HEADLINE: native+capture+dryrun. Captured graph
        # replay re-issues only host-callback + GPU-kernel nodes; no
        # Python operator-body traversal between forwards. Target:
        # T(this) − T(none) ≤ 0.05 s/generate.
        "cots_005_native_capture_dryrun": cots_base
        + ["--cots-cpu-runner", "native", "--cots-dry-run"],
        # Production path: native+capture, real CPU GEMM. Where
        # FastTTS will land.
        "cots_005_native_capture_real": cots_base
        + ["--cots-cpu-runner", "native"],
    }


# Arms that need --enforce-eager (the python-runner-graph hard-fail
# is what Stage 5 enforces; eager is required for python and for the
# substrate-gate native_eager arm).
_EAGER_ARMS = frozenset({
    "none",
    "cots_005_python_eager_dryrun",
    "cots_005_native_eager_dryrun",
})


def cell_path(arm: str, batch: int, threads: int) -> Path:
    if arm == "none" or threads == DEFAULT_THREADS:
        return RESULTS_DIR / f"{arm}_b{batch}.json"
    return RESULTS_DIR / f"{arm}_b{batch}_t{threads}.json"


def run_cell(
    arm: str,
    flags: list[str],
    batch: int,
    threads: int,
    *,
    num_iters: int,
    num_iters_warmup: int,
    extra_flags: list[str],
) -> Path:
    out_json = cell_path(arm, batch, threads)
    out_log = out_json.with_suffix(".log")
    if out_json.exists():
        print(f"  [skip] {arm} b={batch} (cached)")
        return out_json
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model",
        MODEL,
        "--dtype",
        DTYPE,
        "--input-len",
        str(INPUT_LEN),
        "--output-len",
        str(OUTPUT_LEN),
        "--batch-size",
        str(batch),
        "--num-iters-warmup",
        str(num_iters_warmup),
        "--num-iters",
        str(num_iters),
        "--output-json",
        str(out_json),
        *flags,
        *extra_flags,
    ]
    if arm in _EAGER_ARMS:
        cmd.append("--enforce-eager")
    t0 = time.perf_counter()
    with open(out_log, "w") as fh:
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, check=False
        )
    dur = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-15:])
        print(
            f"  [FAIL] {arm} b={batch} rc={proc.returncode} ({dur:.1f}s)\n"
            f"        {tail}"
        )
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
    ap.add_argument(
        "--only-arms",
        nargs="*",
        default=None,
        help="Subset of arms to run; default = all 5",
    )
    ap.add_argument("--num-iters", type=int, default=3)
    ap.add_argument("--num-iters-warmup", type=int, default=2)
    ap.add_argument(
        "--extra-flag",
        action="append",
        default=[],
        help="Pass-through flag to `vllm bench latency` (e.g., "
        "--max-model-len 1024). Repeatable.",
    )
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] threads={args.threads}, batches={args.batches}, "
        f"f={DEFAULT_F}, input={INPUT_LEN}, output={OUTPUT_LEN}, "
        f"iters={args.num_iters} (warmup {args.num_iters_warmup})"
    )

    for t in args.threads:
        arms = arms_for(t)
        run_arms = (
            arms if not args.only_arms
            else {n: arms[n] for n in args.only_arms if n in arms}
        )
        # `none` is t-invariant — run only at the first t.
        if t != args.threads[0] and "none" in run_arms:
            del run_arms["none"]
        print(f"\n[t={t}] arms={list(run_arms)}")
        for arm, flags in run_arms.items():
            for B in args.batches:
                run_cell(
                    arm,
                    flags,
                    B,
                    t,
                    num_iters=args.num_iters,
                    num_iters_warmup=args.num_iters_warmup,
                    extra_flags=args.extra_flag,
                )

    # Summary table.
    print("\n" + "=" * 88)
    print(
        f"{'arm':<36} {'t':>3}  "
        + "  ".join(f"{f'B={B} (s)':>11}" for B in args.batches)
    )
    print("-" * 88)
    none_by_b = {
        B: parse_avg(cell_path("none", B, args.threads[0]))
        for B in args.batches
    }
    cots_arms = [
        "cots_005_python_eager_dryrun",
        "cots_005_native_eager_dryrun",
        "cots_005_native_capture_dryrun",
        "cots_005_native_capture_real",
    ]
    rows: dict = {}
    for t in args.threads:
        for arm in cots_arms:
            row = {B: parse_avg(cell_path(arm, B, t)) for B in args.batches}
            rows[(arm, t)] = row
            cells = "  ".join(
                f"{row[B]:>11.4f}" if row[B] is not None else f"{'—':>11}"
                for B in args.batches
            )
            print(f"{arm:<36} {t:>3}  {cells}")
    cells = "  ".join(
        f"{none_by_b[B]:>11.4f}" if none_by_b[B] is not None else f"{'—':>11}"
        for B in args.batches
    )
    print(f"{'none':<36} {'-':>3}  {cells}")
    print("=" * 88)

    print("\n=== §1.14 orch decomposition (s, per generate) ===")
    print(
        "    target: orch_native_capture ≤ 0.05 s/generate "
        "(down from python_eager ≈ 0.45 s)"
    )
    print()
    for t in args.threads:
        for B in args.batches:
            none = none_by_b[B]
            py = rows[("cots_005_python_eager_dryrun", t)][B]
            nat_eager = rows[("cots_005_native_eager_dryrun", t)][B]
            nat_cap = rows[("cots_005_native_capture_dryrun", t)][B]
            real = rows[("cots_005_native_capture_real", t)][B]
            print(f"  t={t} B={B}:")
            if none is not None:
                if py is not None:
                    print(
                        f"    orch_python_eager      = {py - none:+.4f}s "
                        f"(§1.14 baseline ≈ 0.45s/generate)"
                    )
                if nat_eager is not None:
                    print(
                        f"    orch_native_eager      = {nat_eager - none:+.4f}s "
                        f"(Stage 2 substrate gate)"
                    )
                if nat_cap is not None:
                    delta = nat_cap - none
                    target = "PASS" if delta <= 0.05 else "FAIL"
                    print(
                        f"    orch_native_capture    = {delta:+.4f}s   "
                        f"§1.14 target ≤ 0.050s: {target}"
                    )
                if real is not None and nat_cap is not None:
                    print(
                        f"    cpu_work_native_capture= "
                        f"{real - nat_cap:+.4f}s (real GEMM time on "
                        f"capture path)"
                    )

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "input_len": INPUT_LEN,
                "output_len": OUTPUT_LEN,
                "f": DEFAULT_F,
                "batches": args.batches,
                "threads": args.threads,
                "num_iters": args.num_iters,
                "none": {str(B): v for B, v in none_by_b.items()},
                "cots": {
                    f"{arm}_t{t}": {str(B): v for B, v in rows[(arm, t)].items()}
                    for (arm, t) in rows
                },
            },
            indent=2,
        )
    )
    print(f"\n[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
