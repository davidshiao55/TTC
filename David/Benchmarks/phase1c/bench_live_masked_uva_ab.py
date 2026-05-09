"""§1c.23 A/B: live-masked UVA prototype vs baseline + matched none_capture.

NOTE — FROZEN ARTIFACT FOR A FAILED PROTOTYPE.
The §1c.23 implementation (`--cots-live-masked-uva`,
`CotsOffloadConfig.live_masked_uva`, `cots_sync_then_uva_masked`,
`_uva_copy_kernel_masked`) was REVERTED from the thesis branch
after failing the §1c.22 decision gate (improvement ≈ −0.007
s/gen, well below the +0.12 s/gen threshold). The
implementation lives on the
`phase1c23-live-masked-uva-experiment` branch in the vllm
submodule for future revisits if the input-D2H side is
patched. This bench script will only run against that branch
(or any future re-introduction of the flag); it is kept here
as the reproducible methodology for the failed prototype, NOT
as a runnable bench against the production thesis branch.

Three measurements per run, default capture sizes, B=1, input_len=8,
output_len=128, Qwen2.5-7B BF16, f_cpu_store=0.05:

    A. native_capture_real, live_masked_uva = False  (baseline)
    B. native_capture_real, live_masked_uva = True   (prototype)
    C. none_capture                                    (matched baseline)

Decision gate (§1c.22 measured ~0.24 s/gen bucket-sensitive component):
    delta_off  = A − C
    delta_on   = B − C
    improvement = delta_off − delta_on
    LAND iff   improvement >= 0.12 s/gen   (≥50% of 0.24)

Run from /TTC/David/Benchmarks/phase1c with the thesis env active
AND the vllm submodule checked out to the experiment branch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "diag_live_masked_uva"
RESULTS.mkdir(parents=True, exist_ok=True)

PY = "/opt/conda/envs/thesis/bin/python"


def run_bench(label: str, env_overrides: dict[str, str], extra_args: list[str]) -> dict:
    """Invoke `vllm bench latency` with the given env + args. Returns
    {wall_clock_s, label, log_path}."""
    out_json = RESULTS / f"{label}.json"
    out_log = RESULTS / f"{label}.log"
    cmd = [
        PY,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model",
        "Qwen/Qwen2.5-7B-Instruct",
        "--dtype",
        "bfloat16",
        "--input-len",
        "8",
        "--output-len",
        "128",
        "--batch-size",
        "1",
        "--num-iters-warmup",
        "0",
        "--num-iters",
        "1",
        "--output-json",
        str(out_json),
        *extra_args,
    ]
    env = os.environ.copy()
    env.update(env_overrides)
    print(f"[{label}] launching ...", flush=True)
    t0 = time.time()
    with open(out_log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    print(f"[{label}] done in {dt:.1f} s wall", flush=True)
    if proc.returncode != 0:
        print(f"[{label}] FAILED — see {out_log}", flush=True)
        return {"label": label, "wall_clock_s": float("nan"), "log": str(out_log)}
    # Parse the JSON the bench dropped.
    with open(out_json) as f:
        data = json.load(f)
    return {
        "label": label,
        "wall_clock_s": float(data["avg_latency"]),
        "log": str(out_log),
        "json": str(out_json),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reset-counters",
        action="store_true",
        help="Set VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE=1 for clean counter dumps",
    )
    args = ap.parse_args()

    base_env = {}
    if args.reset_counters:
        base_env["VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE"] = "1"
        base_env["VLLM_COTS_DUMP_COUNTERS"] = "1"

    cots_off_args = [
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        "0.05",
    ]
    cots_on_args = cots_off_args + ["--cots-live-masked-uva"]
    none_args: list[str] = []

    runs = [
        run_bench("C_none_capture", base_env, none_args),
        run_bench("A_native_capture_real_off", base_env, cots_off_args),
        run_bench("B_native_capture_real_on", base_env, cots_on_args),
    ]

    # Print summary.
    print()
    print("=" * 70)
    print("§1c.23 A/B summary")
    print("=" * 70)
    for r in runs:
        print(f"  {r['label']:35s}  {r['wall_clock_s']:.4f} s")

    by_label = {r["label"]: r["wall_clock_s"] for r in runs}
    a = by_label.get("A_native_capture_real_off", float("nan"))
    b = by_label.get("B_native_capture_real_on", float("nan"))
    c = by_label.get("C_none_capture", float("nan"))

    print()
    delta_off = a - c
    delta_on = b - c
    improvement = delta_off - delta_on
    print(f"  COTS delta (off):    {delta_off:+.4f} s")
    print(f"  COTS delta (on):     {delta_on:+.4f} s")
    print(f"  Improvement:         {improvement:+.4f} s")
    print(f"  Decision gate:       {'PASS' if improvement >= 0.12 else 'FAIL'} "
          f"(need >= 0.12 s/gen = 50% of 0.24 s bucket-sensitive component)")
    print("=" * 70)

    summary_path = RESULTS / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "runs": runs,
                "delta_off": delta_off,
                "delta_on": delta_on,
                "improvement": improvement,
                "passes_gate": bool(improvement >= 0.12),
            },
            f,
            indent=2,
        )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
