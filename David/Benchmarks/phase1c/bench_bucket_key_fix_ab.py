"""Bucket-key fix benchmark: native task selection with live-token cap.

After the native task-id fix, the captured graph references the correct
per-bucket slab (e.g., slab_1 for B=1 decode). The live-token cap is now
part of the default design, so
this script runs the default production path. Historical result filenames
are kept in `results/bucket_key_fix_ab/` for comparison.

Compares against the §1c.21 historical anchor: 2.76s (post-fix) vs
119.3s (pre-fix regression). A result near 2.7-3.0s confirms task
selection is fixed and CPU work remains live-row capped.

Single arm: `cots_005_native_capture_real` at B=1, f=0.05, t=16,
input=8, output=128. Mirrors `bench_dryrun_vs_native_qwen.py`'s
production cell.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results" / "bucket_key_fix_ab"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "Qwen/Qwen2.5-7B-Instruct"


def run_arm(name: str, env_extra: dict[str, str], num_iters: int, warmup: int) -> dict:
    out_json = RESULTS_DIR / f"{name}.json"
    out_log = RESULTS_DIR / f"{name}.log"
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model", MODEL,
        "--dtype", "bfloat16",
        "--input-len", "8",
        "--output-len", "128",
        "--batch-size", "1",
        "--num-iters-warmup", str(warmup),
        "--num-iters", str(num_iters),
        "--output-json", str(out_json),
        "--offload-backend", "cots",
        "--cots-f-cpu-store", "0.05",
        "--cots-cpu-runner", "native",
    ]
    env = os.environ.copy()
    env.update(env_extra)
    env_summary = " ".join(f"{k}={v}" for k, v in env_extra.items()) or "(default)"
    print(f"\n[arm] {name}   env: {env_summary}", flush=True)
    t0 = time.perf_counter()
    with open(out_log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=False)
    dur = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-20:])
        print(f"  [FAIL] rc={proc.returncode} ({dur:.1f}s)\n        {tail}", flush=True)
        return {"name": name, "avg_latency": None, "error": proc.returncode, "duration": dur}
    data = json.loads(out_json.read_text())
    avg = data.get("avg_latency")
    print(f"  [ok]  avg_latency={avg:.4f}s  wall={dur:.1f}s", flush=True)
    return {"name": name, "avg_latency": avg, "duration": dur}


def main() -> None:
    # Phase 1c bench uses iters=3, warmup=2. Keep parity.
    iters, warmup = 3, 2
    results = []
    results.append(run_arm(
        "live_cap_default",
        env_extra={},
        num_iters=iters,
        warmup=warmup,
    ))

    print("\n" + "=" * 60)
    print("SUMMARY  (anchors: pre-§1c.21 regression 119.33s, post-§1c.21 2.76s)")
    print("=" * 60)
    for r in results:
        avg = r.get("avg_latency")
        avg_str = f"{avg:.4f}s" if avg is not None else "FAILED"
        print(f"  {r['name']:<30}  avg_latency={avg_str:<12}  (wall={r['duration']:.1f}s)")

    avg = results[0].get("avg_latency")
    if avg is not None:
        print("\n  CONCLUSION: default live-token-capped path completed")


if __name__ == "__main__":
    if os.path.abspath(os.getcwd()) == "/TTC":
        raise RuntimeError("Don't run from /TTC — vllm import gotcha. cd elsewhere first.")
    main()
