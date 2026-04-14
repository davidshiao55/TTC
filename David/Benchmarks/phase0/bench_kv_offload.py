#!/usr/bin/env python3
"""
Phase 0.7 - KV Offload Impact on FastTTS (Proof of Concept)

A/B test: FastTTS (all optimizations on) with and without stock V1 vLLM
`kv_offload` on the memory-tightest 7B+1.5B configuration. The question is
whether spilling evicted GPU prefix-cache blocks to pinned CPU memory
captures enough of the beam-search eviction footprint to produce a
measurable end-to-end speedup on the thesis target hardware.

Sweep
-----
    generator : 7B-instruct (spec-prefix split 0.74 / 0.16)
    dataset   : aime (math500 deferred to a later phase)
    method    : {fasttts, fasttts_kvoff}
    n         : {4, 16, 64, 256}
    --> 8 runs total

The two yaml families under
``FastTTS-thesis/benchmarks/configs/7B-instruct/aime/{fasttts,fasttts_kvoff}/``
differ only in the presence of ``kv_offloading_size`` under
generator/verifier (32 GiB / 8 GiB respectively). Everything else — SBE,
DPAS, asymmetric memory allocation, beam width, num_iterations,
temperature, max_model_len — is held constant.

Hypothesis
----------
kv_offload should help most where KV demand / KV supply is worst. The 7B
generator has ~2x per-token KV vs 1.5B while leaving less KV budget after
weights, so this is the regime where a CPU prefix-cache extension has the
most to catch. Expected direction: lower latency and higher goodput for
fasttts_kvoff, especially at large N.

Usage
-----
    conda activate thesis
    python David/Benchmarks/phase0/bench_kv_offload.py --exp --plot

Flags
-----
    --exp     Run the 8-cell sweep (sequential; each cell is a fresh
              Python subprocess so vllm engines are cleanly torn down).
    --plot    Collect jsonl results and write summary.json, a latency /
              goodput / speedup comparison PDF, and a terminal table.

Outputs
-------
    David/Benchmarks/phase0/results/kv_offload/
        {method}_n{N}.log              per-run stdout/stderr
        summary.json                   parsed metrics, both variants
        kv_offload_comparison.pdf      3-panel comparison plot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# We want to reuse parse_jsonl_folder from FastTTS-thesis so this script
# stays in sync with how the rest of the project interprets benchmark
# output — but that module pulls in matplotlib/pandas/seaborn at import
# time. Defer the import until --plot is actually requested so --exp can
# run on a lighter environment.
FASTTTS_DIR = Path("/TTC/FastTTS-thesis")


# ----------------------------------------------------------------------------
# POC configuration
# ----------------------------------------------------------------------------

PHASE0_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE0_DIR / "results" / "kv_offload"

GENERATOR = "7B-instruct"
DATASET = "aime"
METHODS = ["fasttts", "fasttts_kvoff"]
N_VALUES = [4, 16, 64, 256]

METHOD_LABELS = {
    "fasttts": "FastTTS (no offload)",
    "fasttts_kvoff": "FastTTS + kv_offload (32G / 8V)",
}
METHOD_COLORS = {
    "fasttts": "tab:blue",
    "fasttts_kvoff": "tab:orange",
}


# ----------------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------------

def run_experiments() -> None:
    """Run each (method, n) cell as an isolated subprocess."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_dir = FASTTTS_DIR / "benchmarks"

    print("=" * 72)
    print(f"  Phase 0.7: KV Offload POC - {GENERATOR} / {DATASET}")
    print("=" * 72)

    for method in METHODS:
        for n in N_VALUES:
            cfg_rel = f"configs/{GENERATOR}/{DATASET}/{method}/n{n}.yaml"
            cfg_abs = bench_dir / cfg_rel
            if not cfg_abs.exists():
                print(f"  [skip] missing config: {cfg_rel}")
                continue

            label = f"{method}_n{n}"
            log_path = RESULTS_DIR / f"{label}.log"

            print()
            print("-" * 72)
            print(f"  Run:    {label}")
            print(f"  Config: {cfg_rel}")
            print(f"  Log:    {log_path}")
            print("-" * 72)

            # Launch run_benchmarks.py in a fresh subprocess from the
            # benchmarks/ dir so the yaml's relative output_dir resolves
            # inside FastTTS-thesis/benchmarks/benchmark_results/ — same
            # convention the existing run_all_experiments.py follows.
            with open(log_path, "w") as log_fp:
                proc = subprocess.run(
                    [sys.executable, "run_benchmarks.py", cfg_rel],
                    cwd=str(bench_dir),
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if proc.returncode != 0:
                print(f"  [error] {label} exited with code {proc.returncode}")
                print(f"          see {log_path} for details")


# ----------------------------------------------------------------------------
# Result collection and plotting
# ----------------------------------------------------------------------------

def collect_results() -> dict:
    """Parse jsonl output of both variants into a nested dict."""
    if str(FASTTTS_DIR) not in sys.path:
        sys.path.insert(0, str(FASTTTS_DIR))
    from run_all_experiments import parse_jsonl_folder  # lazy — heavy deps

    bench_results = FASTTTS_DIR / "benchmarks" / "benchmark_results"
    out: dict[str, dict] = {}
    for method in METHODS:
        folder = bench_results / GENERATOR / DATASET / method
        out[method] = parse_jsonl_folder(folder, DATASET)
    return out


def print_table(results: dict) -> None:
    """Dump a human-readable comparison table to stdout."""
    print()
    print("=" * 92)
    print(f"  Results: {GENERATOR} / {DATASET}")
    print("=" * 92)
    header = (
        f"{'N':>5} | {'method':<16} | {'total_lat':>10} | "
        f"{'gen_lat':>9} | {'ver_lat':>9} | {'goodput':>9} | {'tok/beam':>9}"
    )
    print(header)
    print("-" * len(header))
    for n in N_VALUES:
        for method in METHODS:
            r = results.get(method, {}).get(str(n))
            if r is None:
                print(f"{n:>5} | {method:<16} | {'(missing)':>10}")
                continue
            print(
                f"{n:>5} | {method:<16} | "
                f"{r['mean_total_latency']:>10.2f} | "
                f"{r['mean_generator_latency']:>9.2f} | "
                f"{r['mean_verifier_latency']:>9.2f} | "
                f"{r['mean_precise_goodput']:>9.2f} | "
                f"{r['mean_average_tokens_per_completion']:>9.1f}"
            )
        print("-" * len(header))

    # Paired latency speedup
    print()
    print("=" * 52)
    print("  Paired latency speedup: fasttts_kvoff vs fasttts")
    print("=" * 52)
    print(f"{'N':>5} | {'off (s)':>10} | {'on (s)':>10} | {'speedup':>10}")
    print("-" * 46)
    for n in N_VALUES:
        f_r = results.get("fasttts", {}).get(str(n))
        k_r = results.get("fasttts_kvoff", {}).get(str(n))
        if f_r is None or k_r is None:
            print(f"{n:>5} | {'(incomplete)':>10}")
            continue
        lat_off = f_r["mean_total_latency"]
        lat_on = k_r["mean_total_latency"]
        speedup = lat_off / lat_on
        print(f"{n:>5} | {lat_off:>10.2f} | {lat_on:>10.2f} | {speedup:>9.3f}x")
    print()


def plot_results(results: dict, output_path: Path) -> None:
    """Two-panel comparison: latency curves and paired latency speedup vs N."""
    import matplotlib.pyplot as plt

    fig, (ax_lat, ax_speedup) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel 1: latency vs N ---
    for method in METHODS:
        ns_present = sorted(int(n) for n in results.get(method, {}).keys())
        if not ns_present:
            continue
        lat = [results[method][str(n)]["mean_total_latency"] for n in ns_present]
        ax_lat.plot(
            ns_present, lat,
            marker="o", linewidth=2.2, markersize=9,
            label=METHOD_LABELS[method], color=METHOD_COLORS[method],
        )

    # --- Panel 2: paired latency speedup ---
    speedup_ns = []
    lat_speedups = []
    for n in N_VALUES:
        f_r = results.get("fasttts", {}).get(str(n))
        k_r = results.get("fasttts_kvoff", {}).get(str(n))
        if f_r is None or k_r is None:
            continue
        speedup_ns.append(n)
        lat_speedups.append(f_r["mean_total_latency"] / k_r["mean_total_latency"])

    if speedup_ns:
        ax_speedup.plot(
            speedup_ns, lat_speedups,
            marker="s", linewidth=2.2, markersize=9,
            label="Latency speedup", color="tab:green",
        )
    ax_speedup.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5)

    # --- Common axis formatting ---
    for ax in (ax_lat, ax_speedup):
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES])
        ax.set_xlabel("N (completions)")
        ax.grid(True, which="both", ls="--", c="0.85")
        ax.legend(loc="best", frameon=True, framealpha=0.92)

    ax_lat.set_ylabel("Mean total latency (s / problem)")
    ax_lat.set_title(f"Latency — {GENERATOR} / {DATASET}")

    ax_speedup.set_ylabel("Speedup (kvoff / off)")
    ax_speedup.set_title("Paired kv_offload latency speedup")

    fig.suptitle(
        "Phase 0.7 - KV Offload Impact on FastTTS",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0.7 KV offload POC for FastTTS (7B + aime)"
    )
    parser.add_argument("--exp", action="store_true", help="Run the 8-cell sweep")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Collect jsonl results and generate comparison plot + table",
    )
    args = parser.parse_args()

    if not (args.exp or args.plot):
        parser.print_help()
        print("\nPlease specify --exp and/or --plot")
        sys.exit(1)

    if args.exp:
        run_experiments()

    if args.plot:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results = collect_results()
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print_table(results)
        plot_results(results, RESULTS_DIR / "kv_offload_comparison.pdf")


if __name__ == "__main__":
    main()
