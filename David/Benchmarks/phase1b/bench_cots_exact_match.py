#!/usr/bin/env python3
"""Phase 1b Bench 1 — COTS pure-prefetch vs vLLM `PrefetchDeferOffloader`.

**Question this bench answers:** at matched offloaded GiB, does COTS
in pure-prefetch mode (`f_prefetch == f_cpu_store`, every layer
offloaded a fraction) match the densest-spread native baseline?

The COTS arm offloads a **fraction** of EVERY layer. The native arm
offloads ENTIRE layers, picked uniformly via the canonical picker
(`module_index % G >= G - N` with N=1). At matched offloaded GiB the
arms move the same total bytes per forward — the question is whether
COTS' tensor-granularity layout is byte-for-byte equivalent to
native's layer-granularity layout in PCIe time, or whether one loses
to the other due to scheduling / kernel-launch overhead.

Native arm: **`prefetch_defer`** — `PrefetchDeferOffloader` from
`vllm/model_executor/offloader/prefetch_defer.py`, the thesis-
optimized variant that defers the last wrap-around prefetch out of
the CE0 FIFO bottleneck (`phase0_findings.md §0.10.3`). The shipped
factory `prefetch` backend is the unoptimized baseline that
`phase0_findings.md §0.10.3` already characterizes; the apples-to-
apples comparison for COTS is against the *best* native baseline,
which is `prefetch_defer`.

For Qwen2.5-7B (28 layers, ~12.15 GiB BF16 weights), each layer is
~444.5 MiB → 0.434 GiB. Pairs `(n_layers_offloaded, native_G, cots_f)`
where `cots_f = n_layers / 28`:

| n_layers | native G | cots_f | offloaded GiB |
|---:|---:|---:|---:|
| 1  | 28 | 0.0357 | 0.43 |
| 2  | 14 | 0.0714 | 0.87 |
| 4  |  7 | 0.1429 | 1.74 |
| 7  |  4 | 0.2500 | 3.04 |
| 14 |  2 | 0.5000 | 6.08 |

Workload: decode-heavy (input=8, output=128). Batches {1, 64} —
B=1 captures the small-batch regime where COTS' tensor-granularity
should be neutral; B=64 stresses the per-forward dispatch overhead
of COTS' 28 partial-layer prefetches vs native's `n_layers` full-
layer prefetches.

Outputs go to `David/Benchmarks/phase1b/results/exact_match/`.
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
RESULTS_DIR = PHASE1B_DIR / "results" / "exact_match"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
WARMUP_ITERS = 2
BENCH_ITERS = 3
DEFAULT_BATCHES = [1, 64]
N_LAYERS = 28
PER_LAYER_GIB = 12.15 / N_LAYERS  # ~0.434

# Pairs at matched offloaded GiB. cots_f = n_offloaded_layers / N_LAYERS.
PAIRS: list[tuple[int, int, float, str]] = [
    # (n_layers, native_G, cots_f, label)
    (1,  28, 1 / N_LAYERS,  "01L"),
    (2,  14, 2 / N_LAYERS,  "02L"),
    (4,  7,  4 / N_LAYERS,  "04L"),
    (7,  4,  7 / N_LAYERS,  "07L"),
    (14, 2,  14 / N_LAYERS, "14L"),
]


def build_arms() -> dict[str, dict]:
    """COTS arm + native arms at K=1 and K=2 per depth pair.

    K=1 is the apples-to-apples lookahead match for COTS' pre-compute
    schedule. K=2 is native's empirical optimum at uniform spacing
    (`phase0_findings.md §0.10.1d`); included for the more pessimistic
    comparison against COTS.
    """
    arms: dict[str, dict] = {
        "none": {"flags": [], "family": "none"},
    }
    for n_layers, G, cots_f, label in PAIRS:
        depth_gib = round(n_layers * PER_LAYER_GIB, 3)
        arms[f"cots_{label}"] = {
            "flags": [
                "--offload-backend", "cots",
                "--cots-f-cpu-store", f"{cots_f:.4f}",
                "--cots-f-prefetch", f"{cots_f:.4f}",
            ],
            "family": "cots",
            "n_layers": n_layers,
            "depth_gib": depth_gib,
        }
        arms[f"native_k1_{label}"] = {
            "flags": [
                "--offload-backend", "prefetch_defer",
                "--offload-group-size", str(G),
                "--offload-num-in-group", "1",
                "--offload-prefetch-step", "1",
            ],
            "family": "prefetch_defer_k1",
            "n_layers": n_layers,
            "depth_gib": depth_gib,
        }
        arms[f"native_k2_{label}"] = {
            "flags": [
                "--offload-backend", "prefetch_defer",
                "--offload-group-size", str(G),
                "--offload-num-in-group", "1",
                "--offload-prefetch-step", "2",
            ],
            "family": "prefetch_defer_k2",
            "n_layers": n_layers,
            "depth_gib": depth_gib,
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
        entry = {
            "flags": spec["flags"],
            "family": spec["family"],
            "by_batch": per_b,
        }
        if "n_layers" in spec:
            entry["n_layers"] = spec["n_layers"]
            entry["depth_gib"] = spec["depth_gib"]
        out[arm] = entry
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
    ap.add_argument("--only-arms", nargs="*", default=None)
    ap.add_argument("--batches", type=int, nargs="*", default=DEFAULT_BATCHES)
    ap.add_argument("--exp", action="store_true",
                    help="Run benchmarks (else just summarize cached results).")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = build_arms()
    run_arms = arms if not args.only_arms else {
        name: arms[name] for name in args.only_arms if name in arms
    }

    print(f"[setup] arms: {len(run_arms)}, batches: {args.batches}, "
          f"input={INPUT_LEN}, output={OUTPUT_LEN}")

    if args.exp:
        for arm, spec in run_arms.items():
            for B in args.batches:
                run_cell(arm, spec["flags"], B)

    summary = summarize(arms, args.batches)
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(
        {"env": env_info(), "model": MODEL,
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

    print("\n[cots vs prefetch_defer @ matched offloaded GiB]")
    tag = lambda x: "MATCH" if abs(x) <= 5 else ("COTS+" if x > 0 else "COTS-")
    for n_layers, _G, _cots_f, label in PAIRS:
        gib = round(n_layers * PER_LAYER_GIB, 2)
        for B in args.batches:
            c = summary.get(f"cots_{label}", {}).get("by_batch", {}).get(B)
            k1 = summary.get(f"native_k1_{label}", {}).get("by_batch", {}).get(B)
            k2 = summary.get(f"native_k2_{label}", {}).get("by_batch", {}).get(B)
            if not (c and k1 and k2):
                continue
            pct1 = (c["avg_latency_s"] - k1["avg_latency_s"]) / k1["avg_latency_s"] * 100
            pct2 = (c["avg_latency_s"] - k2["avg_latency_s"]) / k2["avg_latency_s"] * 100
            print(
                f"  {label} ({gib:.2f} GiB) B={B:<2}: "
                f"cots={c['avg_latency_s']:.3f}  "
                f"k1={k1['avg_latency_s']:.3f}  "
                f"k2={k2['avg_latency_s']:.3f}  | "
                f"vs k1: {pct1:+.1f}% [{tag(pct1)}]  "
                f"vs k2: {pct2:+.1f}% [{tag(pct2)}]"
            )


if __name__ == "__main__":
    main()
