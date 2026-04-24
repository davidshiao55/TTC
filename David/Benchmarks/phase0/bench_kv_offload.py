#!/usr/bin/env python3
"""
Phase 0.7 - KV Offload Impact on FastTTS (Proof of Concept)

A/B test: fasttts (SBE + prefix-aware scheduling on) vs. fasttts_kvoff
(same plus CPU prefix-cache extension). Directly measures whether
kv_offload helps the thesis-deployment config.

Sweep
-----
    generator : 7B-instruct  (spec-prefix split 0.74 / 0.16)
    dataset   : {aime, math500}
    method    : {fasttts, fasttts_kvoff}
    n         : {4, 16, 64, 256}
    --> 16 runs total

Hypothesis
----------
kv_offload should help most where KV demand / KV supply is worst. The 7B
generator has ~2x per-token KV vs 1.5B while leaving less KV budget after
weights, so this is the regime where a CPU prefix-cache extension has the
most to catch.

Run stats sidecar
-----------------
Each run dumps a ``.runstats.json`` sidecar next to the jsonl results
with per-engine totals: GPU/CPU prefix cache counters, PCIe transfer
bytes/time, and per-step batch-size histogram. This lets kv_offload's
latency effect be attributed to CPU-tier hits vs. overhead, and exposes
whether the scheduler is actually running all N beams concurrently.

Usage
-----
    conda activate thesis
    python David/Benchmarks/phase0/bench_kv_offload.py --exp --plot

Outputs
-------
    David/Benchmarks/phase0/results/kv_offload/
        {dataset}_{method}_n{N}.log      per-run stdout/stderr
        summary.json                     parsed metrics, both methods, both datasets
        kv_offload_comparison.pdf        rows=datasets, cols={latency,
                                         goodput, paired speedup,
                                         gen hit-rate, ver hit-rate}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

FASTTTS_DIR = Path("/TTC/FastTTS-thesis")


# ----------------------------------------------------------------------------
# POC configuration
# ----------------------------------------------------------------------------

PHASE0_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE0_DIR / "results" / "kv_offload"

GENERATOR = "7B-instruct"
DATASETS = ["aime", "math500"]
METHODS = ["fasttts", "fasttts_kvoff"]
N_VALUES = [4, 16, 64, 256]

# Paired comparison — (no-offload variant, kvoff variant).
# Used for the speedup plot and the "kvoff vs no-offload" diagnostic column.
METHOD_PAIRS = [
    ("fasttts", "fasttts_kvoff"),
]

METHOD_LABELS = {
    "fasttts": "FastTTS (no offload)",
    "fasttts_kvoff": "FastTTS + kv_offload (32G / 8V)",
}
METHOD_COLORS = {
    "fasttts": "tab:blue",
    "fasttts_kvoff": "tab:orange",
}
PAIR_STYLES = {
    ("fasttts", "fasttts_kvoff"): {"color": "tab:orange", "marker": "^"},
}


# ----------------------------------------------------------------------------
# Experiment runner
# ----------------------------------------------------------------------------

def run_experiments(
    only_methods: list[str] | None = None,
    only_datasets: list[str] | None = None,
) -> None:
    """Run each (dataset, method, n) cell as an isolated subprocess.

    ``only_methods`` / ``only_datasets`` filter the sweep; unknown names
    raise to surface typos rather than silently skipping.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bench_dir = FASTTTS_DIR / "benchmarks"

    datasets = only_datasets or DATASETS
    methods = only_methods or METHODS
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}; expected one of {METHODS}")
    for d in datasets:
        if d not in DATASETS:
            raise ValueError(f"unknown dataset {d!r}; expected one of {DATASETS}")

    print("=" * 72)
    print(f"  Phase 0.7: KV Offload POC - {GENERATOR} / {datasets} / {methods}")
    print("=" * 72)

    for dataset in datasets:
        for method in methods:
            for n in N_VALUES:
                cfg_rel = f"configs/{GENERATOR}/{dataset}/beam_search/{method}/n{n}.yaml"
                cfg_abs = bench_dir / cfg_rel
                if not cfg_abs.exists():
                    print(f"  [skip] missing config: {cfg_rel}")
                    continue

                label = f"{dataset}_{method}_n{n}"
                log_path = RESULTS_DIR / f"{label}.log"

                print()
                print("-" * 72)
                print(f"  Run:    {label}")
                print(f"  Config: {cfg_rel}")
                print(f"  Log:    {log_path}")
                print("-" * 72)

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
# Result collection
# ----------------------------------------------------------------------------

def _hit_rate(stats: dict | None) -> float | None:
    if not stats:
        return None
    q = stats.get("queries") or 0
    return (stats.get("hits") or 0) / q if q > 0 else None


def _batch_fields(engine_stats: dict | None) -> dict:
    """Extract batch fields (mean/max running, throttling stats) from one engine's sidecar."""
    if not engine_stats:
        return {}
    b = engine_stats.get("batch")
    if not b:
        return {}
    return {
        "mean_running": b.get("mean_running"),
        "max_running": b.get("max_running"),
        "max_waiting": b.get("max_waiting"),
        "mean_waiting_when_queued": b.get("mean_waiting_when_queued"),
        "frac_steps_queued": b.get("frac_steps_queued"),
        "max_kv_usage": b.get("max_kv_usage"),
        "histogram": b.get("histogram"),
    }


def _load_run_sidecars(folder: Path) -> dict[str, dict]:
    """Return ``{N_str: run_stats_dict}`` for every *.runstats.json in *folder*.

    Also reads legacy ``*.cachestats.json`` so folders that haven't been
    regenerated still show cache-hit data (batch/transfer fields will be
    absent for those).

    Each run_stats_dict has keys ``gen_gpu_hit``, ``gen_cpu_hit``,
    ``ver_gpu_hit``, ``ver_cpu_hit`` (each either a float in [0,1] or None),
    plus generator-side batch stats ``gen_mean_running``, ``gen_max_running``,
    ``gen_max_waiting``, ``gen_max_kv_usage``, ``gen_batch_hist`` when the
    sidecar was written by a build that captures them.
    """
    import re
    out: dict[str, dict] = {}
    if not folder.exists():
        return out
    n_re = re.compile(r"_n(\d+)_")
    # Prefer the new filename when both are present.
    candidates: dict[str, Path] = {}
    for p in sorted(folder.glob("*.cachestats.json")) + sorted(
        folder.glob("*.runstats.json")
    ):
        m = n_re.search(p.name)
        if not m:
            continue
        candidates[m.group(1)] = p
    for n_key, p in candidates.items():
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        gen = data.get("generator") or {}
        ver = data.get("verifier") or {}
        gen_batch = _batch_fields(gen)
        out[n_key] = {
            "gen_gpu_hit": _hit_rate(gen.get("gpu")),
            "gen_cpu_hit": _hit_rate(gen.get("cpu")),
            "ver_gpu_hit": _hit_rate(ver.get("gpu")),
            "ver_cpu_hit": _hit_rate(ver.get("cpu")),
            "gen_mean_running": gen_batch.get("mean_running"),
            "gen_max_running": gen_batch.get("max_running"),
            "gen_max_waiting": gen_batch.get("max_waiting"),
            "gen_mean_waiting_when_queued": gen_batch.get("mean_waiting_when_queued"),
            "gen_frac_steps_queued": gen_batch.get("frac_steps_queued"),
            "gen_max_kv_usage": gen_batch.get("max_kv_usage"),
            "gen_batch_hist": gen_batch.get("histogram"),
        }
    return out


def collect_results() -> dict:
    """Parse jsonl + runstats sidecars into ``results[dataset][method][N] = metrics``."""
    if str(FASTTTS_DIR) not in sys.path:
        sys.path.insert(0, str(FASTTTS_DIR))
    from run_all_experiments import parse_jsonl_folder  # lazy — heavy deps

    bench_results = FASTTTS_DIR / "benchmarks" / "benchmark_results"
    out: dict[str, dict] = {}
    for dataset in DATASETS:
        out[dataset] = {}
        for method in METHODS:
            folder = bench_results / GENERATOR / dataset / "beam_search" / method
            per_n = parse_jsonl_folder(folder, dataset)
            cache = _load_run_sidecars(folder)
            for n_key, metrics in per_n.items():
                metrics.update(cache.get(n_key, {}))
            out[dataset][method] = per_n
    return out


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def _fmt_pct(v: float | None) -> str:
    return "   —  " if v is None else f"{v * 100:5.1f}%"


def print_table(results: dict) -> None:
    """Dump a human-readable comparison table to stdout, one block per dataset."""
    for dataset in DATASETS:
        print()
        print("=" * 108)
        print(f"  Results: {GENERATOR} / {dataset}")
        print("=" * 108)
        header = (
            f"{'N':>4} | {'method':<15} | {'lat':>7} | {'gen':>7} | {'ver':>7} | "
            f"{'goodput':>8} | {'gGPU':>6} | {'gCPU':>6} | {'vGPU':>6} | {'vCPU':>6}"
        )
        print(header)
        print("-" * len(header))
        ds_data = results.get(dataset, {})
        for n in N_VALUES:
            for method in METHODS:
                r = ds_data.get(method, {}).get(str(n))
                if r is None:
                    print(f"{n:>4} | {method:<15} | {'(missing)':>7}")
                    continue
                print(
                    f"{n:>4} | {method:<15} | "
                    f"{r['mean_total_latency']:>7.2f} | "
                    f"{r['mean_generator_latency']:>7.2f} | "
                    f"{r['mean_verifier_latency']:>7.2f} | "
                    f"{r['mean_precise_goodput']:>8.2f} | "
                    f"{_fmt_pct(r.get('gen_gpu_hit'))} | "
                    f"{_fmt_pct(r.get('gen_cpu_hit'))} | "
                    f"{_fmt_pct(r.get('ver_gpu_hit'))} | "
                    f"{_fmt_pct(r.get('ver_cpu_hit'))}"
                )
            print("-" * len(header))

        # Generator batch-size stats (from the _BatchStatsAcc sidecar field).
        # Rows with no batch data (older sidecars) are skipped so the section
        # collapses when nothing is available.
        batch_rows = []
        for n in N_VALUES:
            for method in METHODS:
                r = ds_data.get(method, {}).get(str(n))
                if r is None or r.get("gen_max_running") is None:
                    continue
                batch_rows.append((n, method, r))
        if batch_rows:
            print()
            print("-" * 110)
            print(f"  Generator batch stats   [{dataset}]   "
                  "(per engine step; queued% = throttling signal)")
            print("-" * 110)
            hdr = (
                f"{'N':>4} | {'method':<15} | "
                f"{'run mean':>8} | {'run max':>7} | "
                f"{'queued%':>7} | {'wait mean|q':>11} | {'wait max':>8} | "
                f"{'kv peak':>7} | steps by bucket"
            )
            print(hdr)
            print("-" * len(hdr))
            for n, method, r in batch_rows:
                hist = r.get("gen_batch_hist") or {}
                hist_str = ", ".join(
                    f"{k}:{v}" for k, v in hist.items() if v > 0
                ) or "—"
                kv_peak = r.get("gen_max_kv_usage")
                kv_peak_str = f"{kv_peak * 100:5.1f}%" if kv_peak is not None else "   —  "
                frac_q = r.get("gen_frac_steps_queued")
                frac_q_str = f"{frac_q * 100:5.1f}%" if frac_q is not None else "   —  "
                mean_w = r.get("gen_mean_waiting_when_queued")
                mean_w_str = f"{mean_w:9.1f}  " if mean_w is not None else "     —     "
                print(
                    f"{n:>4} | {method:<15} | "
                    f"{r.get('gen_mean_running', 0.0):>8.1f} | "
                    f"{r.get('gen_max_running', 0):>7d} | "
                    f"{frac_q_str} | {mean_w_str} | "
                    f"{r.get('gen_max_waiting', 0):>8d} | "
                    f"{kv_peak_str} | {hist_str}"
                )

        # Paired speedup (fasttts vs fasttts_kvoff).
        for base, kv in METHOD_PAIRS:
            print()
            print("-" * 60)
            print(f"  Paired speedup ({kv} vs {base})   [{dataset}]")
            print("-" * 60)
            print(f"{'N':>4} | {'latency speedup':>16} | {'goodput speedup':>16}")
            for n in N_VALUES:
                b_r = ds_data.get(base, {}).get(str(n))
                k_r = ds_data.get(kv, {}).get(str(n))
                if b_r is None or k_r is None:
                    print(f"{n:>4} | {'(incomplete)':>16}")
                    continue
                lat_speedup = b_r["mean_total_latency"] / k_r["mean_total_latency"]
                good_speedup = (
                    k_r["mean_precise_goodput"] / b_r["mean_precise_goodput"]
                    if b_r["mean_precise_goodput"] > 0
                    else float("nan")
                )
                print(f"{n:>4} | {lat_speedup:>15.3f}x | {good_speedup:>15.3f}x")
    print()


def plot_results(results: dict, output_path: Path) -> None:
    """10-panel plot: rows = datasets, cols = {latency, goodput, speedup, gen hit, ver hit}."""
    import matplotlib.pyplot as plt

    nrows = len(DATASETS)
    fig, axes = plt.subplots(nrows, 5, figsize=(30, 5.5 * nrows), squeeze=False)

    for row, dataset in enumerate(DATASETS):
        ax_lat, ax_good, ax_speedup, ax_gen_hit, ax_ver_hit = axes[row]
        ds_data = results.get(dataset, {})

        # Panels 1+2: absolute latency and goodput per method
        for method in METHODS:
            ns_present = sorted(int(n) for n in ds_data.get(method, {}).keys())
            if not ns_present:
                continue
            lat = [ds_data[method][str(n)]["mean_total_latency"] for n in ns_present]
            good = [ds_data[method][str(n)]["mean_precise_goodput"] for n in ns_present]
            ax_lat.plot(
                ns_present, lat,
                marker="o", linewidth=2.2, markersize=9,
                label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            )
            ax_good.plot(
                ns_present, good,
                marker="o", linewidth=2.2, markersize=9,
                label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            )

        # Panel 3: paired latency speedup — one curve per (base, kvoff) pair.
        for base, kv in METHOD_PAIRS:
            ns_out: list[int] = []
            lat: list[float] = []
            for n in N_VALUES:
                b_r = ds_data.get(base, {}).get(str(n))
                k_r = ds_data.get(kv, {}).get(str(n))
                if b_r is None or k_r is None:
                    continue
                ns_out.append(n)
                lat.append(b_r["mean_total_latency"] / k_r["mean_total_latency"])
            if ns_out:
                style = PAIR_STYLES[(base, kv)]
                ax_speedup.plot(
                    ns_out, lat,
                    marker=style["marker"], linewidth=2.2, markersize=9,
                    label=f"{kv} / {base}",
                    color=style["color"],
                )
        ax_speedup.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5)

        # Panels 4+5: per-engine prefix cache hit rates.
        # For each (base, kvoff) pair we draw three lines: base GPU hit,
        # kvoff GPU hit, kvoff CPU hit (conditional on GPU miss). All four
        # methods' GPU hits appear, so the pair GPU lines share the family
        # color but differ by marker to stay legible.
        def _series(method: str, key: str) -> tuple[list[int], list[float]]:
            ns_out, ys = [], []
            for n in N_VALUES:
                r = ds_data.get(method, {}).get(str(n))
                if r is None or r.get(key) is None:
                    continue
                ns_out.append(n)
                ys.append(r[key] * 100)
            return ns_out, ys

        def _plot_hits(ax, gpu_key: str, cpu_key: str, engine_label: str) -> None:
            for base, kv in METHOD_PAIRS:
                ns_b, y_b = _series(base, gpu_key)
                ns_kg, y_kg = _series(kv, gpu_key)
                ns_kc, y_kc = _series(kv, cpu_key)
                if ns_b:
                    ax.plot(
                        ns_b, y_b, marker="o", linewidth=2.0, markersize=8,
                        label=f"{base} GPU hit",
                        color=METHOD_COLORS[base],
                    )
                if ns_kg:
                    ax.plot(
                        ns_kg, y_kg, marker="o", linewidth=2.0, markersize=8,
                        label=f"{kv} GPU hit",
                        color=METHOD_COLORS[kv],
                    )
                if ns_kc:
                    ax.plot(
                        ns_kc, y_kc,
                        marker="^", linestyle="--", linewidth=2.0, markersize=8,
                        label=f"{kv} CPU hit (| GPU miss)",
                        color=METHOD_COLORS[kv],
                    )
            ax.set_ylabel(f"{engine_label} prefix cache hit rate (%)")
            ax.set_ylim(0, 100)
            ax.set_title(f"{engine_label} hit rate — {dataset}")

        _plot_hits(ax_gen_hit, "gen_gpu_hit", "gen_cpu_hit", "Generator")
        _plot_hits(ax_ver_hit, "ver_gpu_hit", "ver_cpu_hit", "Verifier")

        for ax in (ax_lat, ax_good, ax_speedup, ax_gen_hit, ax_ver_hit):
            ax.set_xscale("log", base=2)
            ax.set_xticks(N_VALUES)
            ax.set_xticklabels([str(n) for n in N_VALUES])
            ax.set_xlabel("N (completions)")
            ax.grid(True, which="both", ls="--", c="0.85")
            ax.legend(loc="best", frameon=True, framealpha=0.92)

        ax_lat.set_ylabel("Mean total latency (s / problem)")
        ax_lat.set_title(f"Latency — {GENERATOR} / {dataset}")

        ax_good.set_ylabel("Mean precise goodput (tokens/s)")
        ax_good.set_title(f"Goodput — {GENERATOR} / {dataset}")

        ax_speedup.set_ylabel("Speedup (kvoff / off)")
        ax_speedup.set_title(f"Paired kv_offload speedup — {dataset}")

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
        description="Phase 0.7 KV offload POC for FastTTS (7B + {aime, math500})"
    )
    parser.add_argument(
        "--exp", action="store_true", help="Run the full sweep (16 cells)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Collect jsonl + runstats sidecars, write summary.json + comparison PDF",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=METHODS,
        help="Restrict --exp to a subset of methods (e.g. fasttts_kvoff)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=DATASETS,
        help="Restrict --exp to a subset of datasets",
    )
    args = parser.parse_args()

    if not (args.exp or args.plot):
        parser.print_help()
        print("\nPlease specify --exp and/or --plot")
        sys.exit(1)

    if args.exp:
        run_experiments(only_methods=args.methods, only_datasets=args.datasets)

    if args.plot:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results = collect_results()
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print_table(results)
        plot_results(results, RESULTS_DIR / "kv_offload_comparison.pdf")


if __name__ == "__main__":
    main()
