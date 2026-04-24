#!/usr/bin/env python3
"""Run all experiments and generate figures for the FastTTS paper.

Usage:
    python run_all_experiments.py --exp                      # Run experiments
    python run_all_experiments.py --plot                     # Generate plots
    python run_all_experiments.py --exp --plot               # Both
    python run_all_experiments.py --dir /path/to/data        # Custom data dir
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


# ============================================================================
# Configuration — thesis experimental setup
# ============================================================================

GENERATORS = ["7B-instruct"]
DATASETS = ["aime", "math500"]

# Two experimental axes. Each run is a (search_strategy, optimization) pair.
# Config layout: configs/{gen}/{dataset}/{strategy}/{optimization}/n{N}.yaml
#
# Only the fasttts optimization variant runs: (beam_search, baseline) is the
# un-optimized reference for full fasttts, and comparing against it would
# just duplicate the thesis's own internal ablation. The thesis's primary
# comparison is FastTTS (beam_search + SBE + prefix-aware + asymmetric
# memory) vs. BoN. (best_of_n, baseline) is also omitted: for BoN the only
# difference between baseline and fasttts yamls is the memory split, since
# SBE and prefix-aware scheduling are no-ops for a single-call strategy.
COMBO_ORDER_KEYS: List[Tuple[str, str]] = [
    ("beam_search", "fasttts"),
    ("best_of_n", "fasttts"),
]

# Display labels shown in plot legends and DataFrame columns. Both runs
# apply the fasttts optimization; the axis that actually differs is the
# search strategy, so labels name the strategy.
COMBO_DISPLAY_MAP: Dict[Tuple[str, str], str] = {
    ("beam_search", "fasttts"): "Beam Search",
    ("best_of_n", "fasttts"): "BoN",
}
COMBO_ORDER: List[str] = [COMBO_DISPLAY_MAP[k] for k in COMBO_ORDER_KEYS]

# N=1 is the "no TTC" reference point (single CoT chain). For beam_search
# at N=1 there's no pruning; for best_of_n at N=1 it's a single greedy sample.
N_VALUES = [1, 4, 16, 64, 256]

BENCHMARK_DIR = Path(__file__).parent / "benchmarks"
DEFAULT_DATA_DIR = BENCHMARK_DIR / "benchmark_results"
FIGURES_DIR = Path(__file__).parent / "figures"
ACCURACY_EVAL_DIR = Path(__file__).parent / "accuracy_evaluation"

PROBLEM_LIMITS = {"aime": 30, "amc": 40, "math500": 500}

_N_IN_FILENAME = re.compile(r"_n(\d+)_")


def _combo_key(strategy: str, optimization: str) -> str:
    """Flat string key used in nested result dicts (JSON-friendly)."""
    return f"{strategy}/{optimization}"


def _combo_from_key(key: str) -> Tuple[str, str]:
    """Inverse of :func:`_combo_key`."""
    strategy, optimization = key.split("/", 1)
    return strategy, optimization


# ============================================================================
# Plot styling — keep all matplotlib rcParam knobs in one place
# ============================================================================

_PLOT_STYLE_BASE = {
    "font.size": 30,
    "axes.titlesize": 36,
    "axes.labelsize": 36,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 32,
    "legend.title_fontsize": 32,
    "figure.titlesize": 38,
    "lines.markersize": 16,
}

PLOT_STYLE_GOODPUT = dict(_PLOT_STYLE_BASE)

PLOT_STYLE_LATENCY = {
    **_PLOT_STYLE_BASE,
    "axes.titlesize": 32,
    "axes.labelsize": 32,
    "xtick.labelsize": 27,
    "ytick.labelsize": 30,
    "legend.title_fontsize": 34,
    "figure.titlesize": 36,
    "axes.titlepad": 8,
    "axes.labelpad": 6,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
}

PLOT_STYLE_ACCURACY = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 9,
}


# ============================================================================
# Experiment runner
# ============================================================================

def _planned_runs():
    for generator in GENERATORS:
        for dataset in DATASETS:
            for strategy, optimization in COMBO_ORDER_KEYS:
                for n in N_VALUES:
                    yield (generator, dataset, strategy, optimization, n)


def run_experiments(data_dir: Path):
    """Run all benchmark experiments based on the planned configs."""
    print("=" * 60)
    print(f"Running all experiments (saving to {data_dir})...")
    print("=" * 60)

    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(BENCHMARK_DIR)

    for generator, dataset, strategy, optimization, n in _planned_runs():
        config_path = f"configs/{generator}/{dataset}/{strategy}/{optimization}/n{n}.yaml"
        if not Path(config_path).exists():
            print(f"Config not found: {config_path}, skipping...")
            continue

        label = f"{generator}_{dataset}_{strategy}_{optimization}_n{n}"
        log_file = log_dir / f"{label}.log"
        print(f"\n{'=' * 60}")
        print(f"Running: {generator}/{dataset}/{strategy}/{optimization}/n={n}")
        print(f"Log:     {log_file}")
        print(f"{'=' * 60}")

        output_dir = data_dir / generator / dataset / strategy / optimization
        output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["BENCHMARK_OUTPUT_DIR"] = str(output_dir)
        try:
            with open(log_file, "w") as log_fp:
                subprocess.run(
                    [sys.executable, "run_benchmarks.py", config_path],
                    check=True, env=env,
                    stdout=log_fp, stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark: {e} (see {log_file})")
            continue

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print(f"Logs saved to: {log_dir}")
    print("=" * 60)


# ============================================================================
# JSONL parsing
# ============================================================================

def _extract_n_from_filename(file_name: str) -> int | None:
    match = _N_IN_FILENAME.search(file_name)
    return int(match.group(1)) if match else None


def _load_jsonl_records(file_path: Path) -> List[dict]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _compute_folder_metrics(records: List[dict]) -> Dict[str, float]:
    """Aggregate per-problem JSONL records into mean metrics."""
    gen_latencies = [r["solutions"]["total_generator_latency_s"] for r in records]
    ver_latencies = [r["solutions"]["total_verifier_latency_s"] for r in records]
    total_latencies = [g + v for g, v in zip(gen_latencies, ver_latencies)]
    avg_tokens_per_completion = [
        _mean(r["solutions"]["effective_num_tokens"][0]) for r in records
    ]
    total_tokens = [r["solutions"]["total_num_tokens"] for r in records]
    n_completion_tokens = [r["solutions"]["n_completion_tokens"] for r in records]
    avg_completion_times = [
        _mean(r["solutions"]["completion_time"][0]) for r in records
    ]
    return {
        "mean_generator_latency": _mean(gen_latencies),
        "mean_verifier_latency": _mean(ver_latencies),
        "mean_total_latency": _mean(total_latencies),
        "mean_average_tokens_per_completion": _mean(avg_tokens_per_completion),
        "mean_total_tokens": _mean(total_tokens),
        "mean_n_completion_tokens": _mean(n_completion_tokens),
        "mean_average_completion_times": _mean(avg_completion_times),
        # Precise Goodput, per FastTTS paper §6.2:
        #   Precise Goodput := avg tokens per beam / avg beam completion time
        "mean_precise_goodput": sum(avg_tokens_per_completion) / sum(avg_completion_times),
    }


def parse_jsonl_folder(folder_path: Path, dataset: str) -> Dict[str, Dict[str, float]]:
    """Parse all ``*_results.jsonl`` files in ``folder_path`` → ``{n: metrics}``."""
    results: Dict[str, Dict[str, float]] = {}
    if not folder_path.exists():
        return results

    problem_limit = PROBLEM_LIMITS.get(dataset, 0)
    for file in folder_path.iterdir():
        if not file.name.endswith(".jsonl"):
            continue
        n = _extract_n_from_filename(file.name)
        if n is None:
            continue

        records = _load_jsonl_records(file)
        if not records:
            continue
        if len(records) < problem_limit:
            print(f"File {file} has {len(records)}/{problem_limit} problems")

        try:
            results[str(n)] = _compute_folder_metrics(records)
        except (KeyError, IndexError, ZeroDivisionError) as e:
            print(f"Error parsing {file}: {e}")

    return results


def collect_results(data_dir: Path):
    """Collect results from benchmark output files and compute metrics.

    Output shape:
        results[dataset][generator][combo_key][n] -> metrics

    where ``combo_key`` = ``"{strategy}/{optimization}"``.
    """
    print(f"Collecting results from {data_dir}...")
    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        for generator in GENERATORS:
            results[dataset][generator] = {}
            for strategy, optimization in COMBO_ORDER_KEYS:
                folder = data_dir / generator / dataset / strategy / optimization
                key = _combo_key(strategy, optimization)
                results[dataset][generator][key] = parse_jsonl_folder(folder, dataset)
    return results


# ============================================================================
# Accuracy evaluation driver
# ============================================================================

def run_accuracy_evaluation(eval_script: Path, result_file: Path,
                            agg_strategy: str = "last"):
    """Invoke the multi-metric evaluator on a single result file."""
    output_path = result_file.with_suffix(".eval.json")
    cmd = [
        sys.executable, str(eval_script),
        "--data_name", "math",
        "--file_path", str(result_file),
        "--agg_strategy", agg_strategy,
        "--output", str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(eval_script.parent),
        )
        if result.returncode != 0:
            print(f"Evaluation failed: {result.stderr[:500]}")
            return None
        if output_path.exists():
            with open(output_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"Error evaluating accuracy: {e}")
    return None


def evaluate_accuracy(data_dir: Path):
    """Evaluate accuracy for every (generator, dataset, combo, N) combination.

    Output shape mirrors ``collect_results``:
        accuracy_results[dataset][generator][combo_key] -> List[per-N dict]
    """
    print(f"Evaluating accuracy from {data_dir}...")
    accuracy_json_path = data_dir / "accuracy.json"
    if accuracy_json_path.exists():
        print(f"Loading existing accuracy data from {accuracy_json_path}")
        with open(accuracy_json_path, "r") as f:
            return json.load(f)

    accuracy_results: Dict[str, Dict[str, Dict[str, List[dict]]]] = {}
    eval_script = ACCURACY_EVAL_DIR / "evaluation" / "evaluate.py"

    for dataset in DATASETS:
        accuracy_results[dataset] = {}
        for generator in GENERATORS:
            accuracy_results[dataset][generator] = {}
            for strategy, optimization in COMBO_ORDER_KEYS:
                combo_results = _evaluate_combo(
                    data_dir, eval_script, dataset, generator, strategy, optimization,
                )
                if combo_results:
                    accuracy_results[dataset][generator][_combo_key(strategy, optimization)] = combo_results

    if accuracy_results:
        os.makedirs(data_dir, exist_ok=True)
        with open(accuracy_json_path, "w") as f:
            json.dump(accuracy_results, f, indent=2)
        print(f"Saved accuracy results to {accuracy_json_path}")

    return accuracy_results


def _evaluate_combo(
    data_dir: Path, eval_script: Path, dataset: str, generator: str,
    strategy: str, optimization: str,
) -> List[dict]:
    """Glob all per-N result files for one (strategy, optimization) combo and
    run the accuracy evaluator on each.
    """
    result_dir = data_dir / generator / dataset / strategy / optimization
    combo_results = []

    # `_specdiff` suffix only fires when SBE actually activated at runtime —
    # that's beam_search/fasttts only. BoN/fasttts keeps enable_spec_diff=false
    # (SBE is a no-op without an iteration loop), so its filenames carry no
    # suffix. `_iter*` widens across beam_search's iter10 vs BoN's default
    # iter40; `_n{n}_` keeps each file uniquely identified by N.
    is_sbe_active = (strategy == "beam_search" and optimization == "fasttts")
    suffix = "_specdiff" if is_sbe_active else ""

    for n in N_VALUES:
        glob_pat = f"{dataset}_bw*_n{n}_iter*{suffix}_results.jsonl"
        matches = list(result_dir.glob(glob_pat))
        if not matches:
            print(f"Result not found: {result_dir}/{glob_pat}")
            continue
        eval_data = run_accuracy_evaluation(eval_script, matches[0])
        if eval_data and "result" in eval_data:
            r = eval_data["result"]
            combo_results.append(r)
            print(
                f"{dataset}/{generator}/{strategy}/{optimization} N={r['n']}: "
                f"pass@n={r['pass_at_n']}% pass@1={r['pass_at_1']}%"
            )
    return combo_results


# ============================================================================
# Shared plot helpers
# ============================================================================

def _build_records_df(
    data: Dict[str, Any],
    metric_extractor,
    min_n_values: int = 2,
) -> pd.DataFrame:
    """Flatten ``data[dataset][generator][combo_key][n] -> metrics`` into a DataFrame.

    Columns: Dataset, Combination (= generator), Strategy, Optimization,
    Method (= display label), n, plus whatever ``metric_extractor`` returns.
    """
    records = []
    for dataset, generators in data.items():
        for generator, combos in generators.items():
            for combo_key, n_values in combos.items():
                try:
                    strategy, optimization = _combo_from_key(combo_key)
                except ValueError:
                    continue
                display = COMBO_DISPLAY_MAP.get((strategy, optimization))
                if display is None:
                    continue
                if len(n_values) < min_n_values:
                    continue
                for n, metrics in n_values.items():
                    record = {
                        "Dataset": dataset.upper(),
                        "Combination": generator,
                        "Strategy": strategy,
                        "Optimization": optimization,
                        "Method": display,
                        "n": int(n),
                    }
                    record.update(metric_extractor(metrics))
                    records.append(record)
    return pd.DataFrame(records)


def _combos_per_dataset(df: pd.DataFrame, datasets: List[str]) -> List[List[str]]:
    return [sorted(df[df["Dataset"] == d]["Combination"].unique()) for d in datasets]


def _draw_dataset_separators(fig, axes, split_indices, x_offset: float) -> None:
    """Draw vertical separator lines between dataset groups."""
    for split_idx in split_indices:
        ax = axes[split_idx - 1]
        bbox = ax.get_position()
        x = bbox.x1 + x_offset
        line = plt.Line2D(
            [x, x], [0.08, 0.98],
            color="black", linestyle="-", linewidth=2, alpha=0.85,
            transform=fig.transFigure, zorder=100,
        )
        fig.add_artist(line)


def _save_figure(fig_path: Path | str) -> None:
    os.makedirs(os.path.dirname(str(fig_path)), exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")
    plt.close()


def _lighten(color, amount=0.45) -> Tuple[float, float, float]:
    c = np.array(to_rgb(color))
    return tuple(1 - amount * (1 - c))


def _combo_palette() -> Dict[str, Tuple[float, float, float]]:
    """Display label → base color, one per (strategy, optimization) combo."""
    palette = sns.color_palette(n_colors=len(COMBO_ORDER))
    return {label: palette[i] for i, label in enumerate(COMBO_ORDER)}


# ============================================================================
# Plot: goodput
# ============================================================================

def plot_goodput(data, output_path):
    """Throughput (tokens/s per completion) across the N sweep."""
    df = _build_records_df(
        data,
        lambda m: {"Goodput": m.get("mean_precise_goodput", 0)},
    )
    if df.empty:
        print("No data for goodput plot")
        return

    datasets = sorted(df["Dataset"].unique())
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PLOT_STYLE_GOODPUT)

    combos_per_dataset = _combos_per_dataset(df, datasets)
    total_combos = sum(len(c) for c in combos_per_dataset)

    fig, axes = plt.subplots(
        1, total_combos, figsize=(7 * total_combos, 8),
        sharey=True, sharex=False, squeeze=False,
    )
    axes = axes.flatten()

    color_palette = _combo_palette()

    ax_idx = 0
    for i, dataset in enumerate(datasets):
        dataset_df = df[df["Dataset"] == dataset]
        combos = combos_per_dataset[i]
        for j, combo in enumerate(combos):
            ax = axes[ax_idx]
            subset = dataset_df[dataset_df["Combination"] == combo].sort_values("n")
            sns.lineplot(
                data=subset, x="n", y="Goodput", hue="Method", style="Method",
                ax=ax, marker="o", markersize=16, linewidth=4.5, legend=False,
                hue_order=COMBO_ORDER, style_order=COMBO_ORDER, palette=color_palette,
            )
            if j == len(combos) // 2:
                ax.set_title(f"{dataset}", fontsize=32, fontweight="bold")
            ax.set_xlabel(combo)
            ax.set_ylabel("Goodput (tokens/s)" if ax_idx == 0 else "")
            ax.set_xscale("log", base=2)
            ax.set_xticks(N_VALUES)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.tick_params(axis="x", rotation=0)
            ax.grid(True, which="both", ls="--", c="0.7")
            ax_idx += 1

    dataset_split_indices = np.cumsum([len(c) for c in combos_per_dataset])[:-1]
    _draw_dataset_separators(fig, axes, dataset_split_indices, x_offset=0.013)

    legend_elements = [
        Patch(facecolor=color_palette[label], edgecolor="k", label=label)
        for label in COMBO_ORDER
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        bbox_to_anchor=(0.515, -0.07), ncol=len(COMBO_ORDER), title="",
        fontsize=30,
    )
    plt.tight_layout()
    _save_figure(output_path)


# ============================================================================
# Plot: latency
# ============================================================================

def plot_latency(data, output_path):
    """Generator + verifier latency per N as a grouped bar chart.

    For each N-group on the x-axis, we draw ``2*k`` bars where
    ``k = len(COMBO_ORDER)``: one generator bar per combo, then one verifier
    bar per combo (the verifier is twin-colored and hatched).
    """
    df = _build_records_df(
        data,
        lambda m: {
            "mean_generator_latency": m.get("mean_generator_latency", 0),
            "mean_verifier_latency": m.get("mean_verifier_latency", 0),
        },
    )
    if df.empty:
        print("No data for latency plot")
        return

    datasets = sorted(df["Dataset"].unique())
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(PLOT_STYLE_LATENCY)

    color_palette = _combo_palette()
    k = len(COMBO_ORDER)

    # Half-integer offsets centered on 0 for 2*k bars per group.
    offsets = np.linspace(-(2 * k - 1) / 2, (2 * k - 1) / 2, 2 * k)
    bar_width = min(0.18, 0.8 / (2 * k))

    # Stack order: gen-combo1, gen-combo2, …, ver-combo1, ver-combo2, …
    bar_colors = [color_palette[label] for label in COMBO_ORDER] + \
                 [_lighten(color_palette[label]) for label in COMBO_ORDER]
    hatches = [None] * k + ["//"] * k
    labels = [f"{label} Generator" for label in COMBO_ORDER] + \
             [f"{label} Verifier" for label in COMBO_ORDER]

    combos_per_dataset = _combos_per_dataset(df, datasets)
    total_axes = sum(len(c) for c in combos_per_dataset)

    fig, axes = plt.subplots(
        1, total_axes, figsize=(6 * total_axes, 6.5),
        sharey=True, sharex=False, squeeze=False,
    )
    axes = axes.flatten()

    ax_idx = 0
    for dataset_idx, dataset in enumerate(datasets):
        dataset_df = df[df["Dataset"] == dataset]
        combos = combos_per_dataset[dataset_idx]
        for j, combo in enumerate(combos):
            ax = axes[ax_idx]
            subset = dataset_df[dataset_df["Combination"] == combo].sort_values("n")
            n_vals = sorted(subset["n"].unique())
            x = np.arange(len(n_vals))

            bar_vals: List[np.ndarray] = []
            for method in COMBO_ORDER:  # generator bars
                d = subset[subset["Method"] == method].set_index("n").reindex(n_vals)
                bar_vals.append(d["mean_generator_latency"].values)
            for method in COMBO_ORDER:  # verifier bars
                d = subset[subset["Method"] == method].set_index("n").reindex(n_vals)
                bar_vals.append(d["mean_verifier_latency"].values)

            for bi in range(2 * k):
                ax.bar(
                    x + offsets[bi] * bar_width,
                    bar_vals[bi],
                    width=bar_width,
                    color=bar_colors[bi],
                    edgecolor="black",
                    linewidth=1.2,
                    hatch=hatches[bi],
                    label=labels[bi] if ax_idx == 0 else None,
                    zorder=2,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in n_vals])
            ax.set_xlabel(combo, labelpad=4)
            if j == len(combos) // 2:
                ax.set_title(f"{dataset}", pad=6, fontsize=32, fontweight="bold")
            ax.set_ylabel("Latency (s/completion)" if ax_idx == 0 else "", labelpad=4)
            ax.grid(True, which="both", ls="--", c="0.7", zorder=0)
            ax.tick_params(axis="x", rotation=0, pad=2)
            ax.tick_params(axis="y", pad=2)
            ax.set_yscale("symlog", linthresh=10)
            ax.minorticks_off()
            ax.margins(x=0.04, y=0.04)
            ax_idx += 1

    dataset_split_indices = np.cumsum([len(c) for c in combos_per_dataset])[:-1]
    _draw_dataset_separators(fig, axes, dataset_split_indices, x_offset=0.0155)

    legend_elements = [
        Patch(facecolor=color_palette[label], edgecolor="k", label=label)
        for label in COMBO_ORDER
    ] + [
        Patch(facecolor="w", edgecolor="k", label="Generator"),
        Patch(facecolor="w", edgecolor="k", hatch="//", label="Verifier"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        bbox_to_anchor=(0.525, -0.06), ncol=len(legend_elements), frameon=False,
        borderaxespad=0.2, handletextpad=0.6, columnspacing=0.8, fontsize=26,
    )
    plt.tight_layout()
    _save_figure(output_path)


# ============================================================================
# Plot: accuracy scaling
# ============================================================================

_ACCURACY_METRIC_LABELS = {
    "pass_at_n": "pass@n",
    "pass_at_1": "pass@1",
}
_ACCURACY_METRIC_COLORS = {
    "pass_at_n": "tab:gray",
    "pass_at_1": "tab:green",
}
_ACCURACY_METRIC_MARKERS = {
    "pass_at_n": "s",
    "pass_at_1": "D",
}

# Visual distinction per (strategy, optimization) combo. The `fill` flag
# controls whether the marker face is filled with the metric color (True) or
# left hollow/white (False). Line style distinguishes the combos at a glance.
_ACCURACY_COMBO_STYLES: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("beam_search", "fasttts"):  dict(linestyle="-",  linewidth=2.5, markersize=8, fill=True),
    ("best_of_n", "fasttts"):    dict(linestyle="-.", linewidth=2.5, markersize=8, fill=True),
}


def _collect_accuracy_panels(
    accuracy_data,
) -> List[Tuple[str, str, Dict[Tuple[str, str], List[dict]]]]:
    """Flatten ``accuracy_data`` → one (dataset, generator, combo_results) triple per panel.

    ``combo_results`` maps ``(strategy, optimization)`` → list of per-N eval dicts.
    """
    panels = []
    for dataset in accuracy_data:
        for generator in accuracy_data[dataset]:
            combo_results: Dict[Tuple[str, str], List[dict]] = {}
            for combo_key, results in accuracy_data[dataset][generator].items():
                if not results:
                    continue
                try:
                    combo = _combo_from_key(combo_key)
                except ValueError:
                    continue
                if combo in COMBO_DISPLAY_MAP:
                    combo_results[combo] = results
            if combo_results:
                panels.append((dataset, generator, combo_results))
    return panels


def _draw_accuracy_panel(
    ax, combo_results: Dict[Tuple[str, str], List[dict]],
) -> List[int]:
    all_n_vals = set()
    for combo_key in COMBO_ORDER_KEYS:  # draw in consistent order
        results = combo_results.get(combo_key)
        if not results:
            continue
        results = sorted(results, key=lambda r: r["n"])
        n_vals = [r["n"] for r in results]
        all_n_vals.update(n_vals)
        mstyle = _ACCURACY_COMBO_STYLES[combo_key]
        combo_label = COMBO_DISPLAY_MAP[combo_key]
        for metric_key, metric_label in _ACCURACY_METRIC_LABELS.items():
            vals = [r[metric_key] for r in results]
            kwargs = {k: v for k, v in mstyle.items() if k != "fill"}
            kwargs["markerfacecolor"] = (
                _ACCURACY_METRIC_COLORS[metric_key] if mstyle["fill"] else "white"
            )
            ax.plot(
                n_vals, vals,
                color=_ACCURACY_METRIC_COLORS[metric_key],
                marker=_ACCURACY_METRIC_MARKERS[metric_key],
                markeredgecolor=_ACCURACY_METRIC_COLORS[metric_key],
                label=f"{metric_label} ({combo_label})",
                **kwargs,
            )
    return sorted(all_n_vals)


def plot_accuracy(accuracy_data, output_path):
    """One panel per (dataset, generator), overlaying pass@n + pass@1 per combo."""
    plt.rcParams.update(PLOT_STYLE_ACCURACY)

    panels = _collect_accuracy_panels(accuracy_data)
    if not panels:
        print("No scaling data for accuracy plot")
        return

    n_panels = len(panels)
    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, (dataset, generator, combo_results) in zip(axes, panels):
        sorted_n = _draw_accuracy_panel(ax, combo_results)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N (completions)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{dataset.upper()} — {generator}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", framealpha=0.9, ncol=2, fontsize=7)
        ax.set_xticks(sorted_n)
        ax.set_xticklabels([str(n) for n in sorted_n])

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    _save_figure(output_path)


# ============================================================================
# Entry points
# ============================================================================

def has_valid_data(data) -> bool:
    """True if ``data`` contains any non-empty leaf dict."""
    for combos in data.values():
        for methods in combos.values():
            for n_values in methods.values():
                if n_values:
                    return True
    return False


def generate_plots(data_dir: Path):
    """Generate all plots from existing data."""
    print("=" * 60)
    print(f"Generating plots (data from {data_dir})...")
    print("=" * 60)

    main_results_path = data_dir / "main_results.json"
    data = None
    if main_results_path.exists():
        print(f"Loading data from {main_results_path}")
        with open(main_results_path, "r") as f:
            data = json.load(f)
        if not has_valid_data(data):
            print("Loaded data is empty, will collect from benchmark outputs...")
            data = None

    if data is None:
        print("Collecting results from benchmark outputs...")
        data = collect_results(data_dir)
        if has_valid_data(data):
            os.makedirs(data_dir, exist_ok=True)
            with open(main_results_path, "w") as f:
                json.dump(data, f)
            print(f"Saved results to {main_results_path}")
        else:
            print("No valid benchmark results found in output files.")

    figs_dir = data_dir / "figs"
    os.makedirs(figs_dir, exist_ok=True)

    plot_goodput(data, figs_dir / "main_results_combined.pdf")
    plot_latency(data, figs_dir / "latency_combined.pdf")

    accuracy_data = evaluate_accuracy(data_dir)
    if accuracy_data:
        plot_accuracy(accuracy_data, figs_dir / "acc.pdf")

    print("\n" + "=" * 60)
    print("All plots generated!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments and generate plots for FastTTS",
    )
    parser.add_argument("--exp", action="store_true", help="Run all experiments")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots from existing/collected data")
    parser.add_argument("--dir", type=str, default=None,
                        help=f"Data directory (default: {DEFAULT_DATA_DIR})")
    args = parser.parse_args()

    if not args.exp and not args.plot:
        parser.print_help()
        print("\nPlease specify --exp and/or --plot")
        sys.exit(1)

    data_dir = (Path(args.dir) if args.dir else DEFAULT_DATA_DIR).resolve()
    print(f"Using data directory: {data_dir}")

    if args.exp:
        run_experiments(data_dir)
    if args.plot:
        generate_plots(data_dir)


if __name__ == "__main__":
    main()
