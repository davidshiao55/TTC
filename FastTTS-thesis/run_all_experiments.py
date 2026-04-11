#!/usr/bin/env python3
"""
Script to run all experiments and generate figures for FastTTS paper.

Usage:
    python run_all_experiments.py --exp                      # Run experiments (default data dir)
    python run_all_experiments.py --plot                     # Generate plots (default data dir)
    python run_all_experiments.py --exp --plot               # Run experiments and generate plots
    python run_all_experiments.py --dir /path/to/data         # Custom data directory
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import ConnectionPatch, Patch, Rectangle

# ============================================================================
# Configuration — thesis experimental setup
# ============================================================================
#
# Generators:  7B-instruct, 1.5B-instruct (Qwen2.5-*-Instruct, 32K context)
# Verifier:    Skywork-PRM-1.5B (fixed)
# Datasets:    math500 (primary, 500 problems), aime (hard subset, 30 problems)
# Methods:     fasttts (FastTTS w/ all opts), baseline (vanilla beam search)
# N sweep:     [4, 16, 64, 256] — matches Liu et al. (compute-optimal-tts)
# ============================================================================

GENERATORS = ["7B-instruct", "1.5B-instruct"]
DATASETS = ["math500", "aime"]
# fasttts is the primary method; baseline is only run at N=64 as a sanity check
METHODS = ["fasttts", "baseline"]
N_VALUES = [4, 16, 64, 256]

# N values where the baseline sanity check is run (only one point — confirms
# fasttts and baseline give equivalent accuracy on the same setup)
BASELINE_SANITY_N = [64]
# Restrict baseline runs to math500 only (saves compute on the noisy AIME)
BASELINE_SANITY_DATASETS = ["math500"]

BENCHMARK_DIR = Path(__file__).parent / "benchmarks"
DEFAULT_DATA_DIR = BENCHMARK_DIR / "benchmark_results"
FIGURES_DIR = Path(__file__).parent / "figures"
ACCURACY_EVAL_DIR = Path(__file__).parent / "accuracy_evaluation"


def _planned_runs():
    """Yield (generator, dataset, method, n) tuples for all planned experiments."""
    for generator in GENERATORS:
        for dataset in DATASETS:
            # fasttts: full sweep
            for n in N_VALUES:
                yield (generator, dataset, "fasttts", n)
            # baseline: only at the sanity-check N values, restricted datasets
            if dataset in BASELINE_SANITY_DATASETS:
                for n in BASELINE_SANITY_N:
                    yield (generator, dataset, "baseline", n)


def run_experiments(data_dir: Path):
    """Run all benchmark experiments based on the planned configs."""
    print("=" * 60)
    print(f"Running all experiments (saving to {data_dir})...")
    print("=" * 60)

    os.chdir(BENCHMARK_DIR)

    for generator, dataset, method, n in _planned_runs():
        config_path = f"configs/{generator}/{dataset}/{method}/n{n}.yaml"

        if not Path(config_path).exists():
            print(f"Config not found: {config_path}, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {generator}/{dataset}/{method}/n={n}")
        print(f"{'='*60}")

        output_dir = data_dir / generator / dataset / method
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "run_benchmarks.py", config_path]
        env = os.environ.copy()
        env["BENCHMARK_OUTPUT_DIR"] = str(output_dir)
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark: {e}")
            continue

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


def parse_jsonl_folder(folder_path: Path, dataset: str):
    """
    Parse all jsonl files in the given folder, extract the 'n' value from each filename,
    and return a dict with n as keys and metrics as values.
    Based on stats.ipynb parsing logic.
    """
    import re

    results = {}
    n_pattern = re.compile(r"_n(\d+)_")
    # Expected problem counts for sanity warnings only
    problem_limit = {"aime": 30, "amc": 40, "math500": 500}.get(dataset, 0)

    if not folder_path.exists():
        return results

    for file in folder_path.iterdir():
        if not file.name.endswith(".jsonl"):
            continue

        match = n_pattern.search(file.name)
        if not match:
            continue

        n_value = int(match.group(1))

        data = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if len(data) == 0:
            continue

        if len(data) < problem_limit:
            print(f"File {file} has {len(data)}/{problem_limit} problems")

        try:
            generator_latencies = [d["solutions"]["total_generator_latency_s"] for d in data]
            verifier_latencies = [d["solutions"]["total_verifier_latency_s"] for d in data]
            total_latencies = [g + v for g, v in zip(generator_latencies, verifier_latencies)]
            average_tokens_per_completion = [
                sum(d["solutions"]["effective_num_tokens"][0])
                / len(d["solutions"]["effective_num_tokens"][0])
                for d in data
            ]
            total_tokens = [d["solutions"]["total_num_tokens"] for d in data]
            n_completion_tokens = [d["solutions"]["n_completion_tokens"] for d in data]
            n_generator_latencies = [d["solutions"]["n_generator_latency_s"] for d in data]
            n_verifier_latencies = [d["solutions"]["n_verifier_latency_s"] for d in data]
            average_completion_times = [
                sum(d["solutions"]["completion_time"][0])
                / len(d["solutions"]["completion_time"][0])
                for d in data
            ]

            results[str(n_value)] = {
                "mean_generator_latency": sum(generator_latencies) / len(generator_latencies),
                "mean_verifier_latency": sum(verifier_latencies) / len(verifier_latencies),
                "mean_total_latency": sum(total_latencies) / len(total_latencies),
                "mean_average_tokens_per_completion": sum(average_tokens_per_completion)
                / len(average_tokens_per_completion),
                "mean_total_tokens": sum(total_tokens) / len(total_tokens),
                "mean_n_completion_tokens": sum(n_completion_tokens) / len(n_completion_tokens),
                "mean_n_generator_latencies": sum(n_generator_latencies)
                / len(n_generator_latencies),
                "mean_n_verifier_latencies": sum(n_verifier_latencies) / len(n_verifier_latencies),
                "mean_average_completion_times": sum(average_completion_times)
                / len(average_completion_times),
                "mean_precise_goodput": sum(average_tokens_per_completion)
                / sum(average_completion_times),
                "mean_goodput": sum(average_tokens_per_completion) / sum(total_latencies),
            }
        except (KeyError, IndexError, ZeroDivisionError) as e:
            print(f"Error parsing {file}: {e}")
            continue

    return results


def collect_results(data_dir: Path):
    """Collect results from benchmark output files and compute metrics."""
    print(f"Collecting results from {data_dir}...")

    results = {}
    for dataset in DATASETS:
        results[dataset] = {}
        for generator in GENERATORS:
            results[dataset][generator] = {}
            for method in METHODS:
                result_dir = data_dir / generator / dataset / method
                folder_results = parse_jsonl_folder(result_dir, dataset)
                if folder_results:
                    results[dataset][generator][method] = folder_results
                else:
                    results[dataset][generator][method] = {}

    return results


def run_accuracy_evaluation(eval_script: Path, result_file: Path,
                            agg_strategy: str = "last"):
    """Run the multi-metric evaluation script on a single result file."""
    output_path = result_file.with_suffix(".eval.json")
    cmd = [
        sys.executable,
        str(eval_script),
        "--data_name", "math",
        "--file_path", str(result_file),
        "--agg_strategy", agg_strategy,
        "--output", str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(eval_script.parent),
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
    """Evaluate accuracy for all experiments.

    Evaluates each N-value result file independently (separate beam search
    runs), then aggregates into a scaling curve.

    For `fasttts`: sweeps all N_VALUES.
    For `baseline`: only the BASELINE_SANITY_N values (sanity check).
    """
    print(f"Evaluating accuracy from {data_dir}...")

    accuracy_json_path = data_dir / "accuracy.json"

    if accuracy_json_path.exists():
        print(f"Loading existing accuracy data from {accuracy_json_path}")
        with open(accuracy_json_path, "r") as f:
            return json.load(f)

    accuracy_results = {}
    eval_script = ACCURACY_EVAL_DIR / "evaluation" / "evaluate.py"

    for dataset in DATASETS:
        accuracy_results[dataset] = {}
        for generator in GENERATORS:
            accuracy_results[dataset][generator] = {}
            for method in METHODS:
                result_dir = data_dir / generator / dataset / method

                # Decide which N values to evaluate for this method
                if method == "baseline":
                    if dataset not in BASELINE_SANITY_DATASETS:
                        continue
                    ns_to_eval = BASELINE_SANITY_N
                else:
                    ns_to_eval = N_VALUES

                method_results = []
                for n in ns_to_eval:
                    pattern = f"{dataset}_bw4_n{n}_iter10"
                    if method == "fasttts":
                        pattern += "_specdiff"
                    pattern += "_results.jsonl"

                    result_file = result_dir / pattern
                    if not result_file.exists():
                        print(f"Result not found: {result_file}")
                        continue

                    eval_data = run_accuracy_evaluation(eval_script, result_file)
                    if eval_data and "result" in eval_data:
                        r = eval_data["result"]
                        method_results.append(r)
                        print(
                            f"{dataset}/{generator}/{method} N={r['n']}: "
                            f"Pass@N={r['pass_at_n']}% "
                            f"MajVote={r['majority_vote']}% "
                            f"PRM-Vote={r['prm_vote']}%"
                        )

                if method_results:
                    accuracy_results[dataset][generator][method] = method_results

    if accuracy_results:
        os.makedirs(data_dir, exist_ok=True)
        with open(accuracy_json_path, "w") as f:
            json.dump(accuracy_results, f, indent=2)
        print(f"Saved accuracy results to {accuracy_json_path}")

    return accuracy_results


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_goodput(data, output_path):
    """Generate goodput figure similar to main_results.ipynb."""
    method_map = {"baseline": "Baseline", "fasttts": "FastTTS"}

    # Parse data to DataFrame
    records = []
    for dataset, combinations in data.items():
        for combo, methods in combinations.items():
            for method, n_values in methods.items():
                if method not in method_map:
                    continue
                for n, metrics in n_values.items():
                    records.append(
                        {
                            "Dataset": dataset.upper(),
                            "Combination": combo.replace("-", "+"),
                            "Method": method_map[method],
                            "n": int(n),
                            "Goodput": metrics.get("mean_precise_goodput", 0),
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        print("No data for goodput plot")
        return

    datasets = sorted(df["Dataset"].unique())
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
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
    )

    combos_per_dataset = [
        sorted(df[df["Dataset"] == dataset]["Combination"].unique())
        for dataset in datasets
    ]
    n_combos = [len(c) for c in combos_per_dataset]
    total_combos = sum(n_combos)

    fig, axes = plt.subplots(
        1, total_combos, figsize=(7 * total_combos, 8), sharey=True, sharex=False, squeeze=False
    )
    axes = axes.flatten()

    method_order = ["Baseline", "FastTTS"]
    color_palette = {
        "Baseline": sns.color_palette()[0],
        "FastTTS": sns.color_palette()[1],
    }

    dataset_split_indices = np.cumsum(n_combos)[:-1]
    ax_idx = 0

    for i, dataset in enumerate(datasets):
        dataset_df = df[df["Dataset"] == dataset]
        combinations = combos_per_dataset[i]
        for j, combo in enumerate(combinations):
            ax = axes[ax_idx]
            subset = dataset_df[dataset_df["Combination"] == combo].sort_values("n")

            sns.lineplot(
                data=subset,
                x="n",
                y="Goodput",
                hue="Method",
                style="Method",
                ax=ax,
                marker="o",
                markersize=16,
                linewidth=4.5,
                legend=False,
                hue_order=method_order,
                style_order=method_order,
                palette=color_palette,
            )

            if j == 1:
                ax.set_title(f"{dataset}", fontsize=32, fontweight="bold")
            ax.set_xlabel(combo)
            ax.set_ylabel("Goodput (tokens/s)" if ax_idx == 0 else "")
            ax.set_xscale("log", base=2)
            ax.set_xticks(N_VALUES)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.tick_params(axis="x", rotation=0)
            ax.grid(True, which="both", ls="--", c="0.7")

            # Inset plot for the largest N
            zoom_n = max(N_VALUES)
            zoom_data = subset[subset["n"] == zoom_n]
            if not zoom_data.empty:
                inset_ax = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
                bar_width = 0.7
                x_vals = np.arange(len(method_order))
                for idx, method in enumerate(method_order):
                    row = zoom_data[zoom_data["Method"] == method]
                    if not row.empty:
                        val = row["Goodput"].values[0]
                        inset_ax.bar(
                            x_vals[idx],
                            val,
                            width=bar_width,
                            color=color_palette[method],
                            edgecolor="black",
                            linewidth=2.0,
                            zorder=2,
                        )
                inset_ax.set_xticks(x_vals)
                inset_ax.set_xticklabels(["", ""])
                inset_ax.tick_params(bottom=False)
                inset_ax.set_xlabel("")
                inset_ax.set_ylabel("")
                inset_ax.set_ylim(0, zoom_data["Goodput"].max() * 1.2)

                rect_x = zoom_n * 0.78
                rect_width = zoom_n * 0.4
                rect_height = max(zoom_data["Goodput"]) * 1.5
                rect = Rectangle(
                    (rect_x, 0),
                    rect_width,
                    rect_height,
                    edgecolor="gray",
                    facecolor="none",
                    linestyle="--",
                    linewidth=2.2,
                    zorder=0,
                )
                ax.add_patch(rect)

                arrow = ConnectionPatch(
                    xyA=(rect_x + rect_width / 2, rect_height),
                    coordsA=ax.transData,
                    xyB=(0.5, 0.0),
                    coordsB=inset_ax.transAxes,
                    arrowstyle="->",
                    linestyle="--",
                    color="gray",
                    linewidth=2.2,
                    mutation_scale=32,
                )
                fig.add_artist(arrow)

            ax_idx += 1

    # Dataset separator lines
    for split_idx in dataset_split_indices:
        ax = axes[split_idx - 1]
        bbox = ax.get_position()
        x = bbox.x1 + 0.013
        line = plt.Line2D(
            [x, x],
            [0.08, 0.98],
            color="black",
            linestyle="-",
            linewidth=2,
            alpha=0.85,
            transform=fig.transFigure,
            zorder=100,
        )
        fig.add_artist(line)

    legend_elements = [
        Patch(facecolor=sns.color_palette()[0], edgecolor="k", label="Baseline"),
        Patch(facecolor=sns.color_palette()[1], edgecolor="k", label="FastTTS"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.515, -0.07),
        ncol=2,
        title="",
        fontsize=35,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved goodput figure to {output_path}")
    plt.close()


def plot_latency(data, output_path):
    """Generate latency figure similar to main_results.ipynb."""
    method_map = {"baseline": "Baseline", "fasttts": "FastTTS"}

    records = []
    for dataset, combinations in data.items():
        for combo, methods in combinations.items():
            for method, n_values in methods.items():
                if method not in method_map:
                    continue
                for n, metrics in n_values.items():
                    records.append(
                        {
                            "Dataset": dataset.upper(),
                            "Combination": combo.replace("-", "+"),
                            "Method": method_map[method],
                            "n": int(n),
                            "mean_generator_latency": metrics.get("mean_generator_latency", 0),
                            "mean_verifier_latency": metrics.get("mean_verifier_latency", 0),
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        print("No data for latency plot")
        return

    datasets = sorted(df["Dataset"].unique())
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 30,
            "axes.titlesize": 32,
            "axes.labelsize": 32,
            "xtick.labelsize": 27,
            "ytick.labelsize": 30,
            "legend.fontsize": 32,
            "legend.title_fontsize": 34,
            "figure.titlesize": 36,
            "axes.titlepad": 8,
            "axes.labelpad": 6,
            "xtick.major.pad": 2,
            "ytick.major.pad": 2,
        }
    )

    from matplotlib.colors import to_rgb

    def lighten(color, amount=0.45):
        c = np.array(to_rgb(color))
        return tuple(1 - amount * (1 - c))

    base_color = sns.color_palette()[0]
    fasttts_color = sns.color_palette()[1]
    colors = [base_color, lighten(base_color, 0.45), fasttts_color, lighten(fasttts_color, 0.45)]

    dataset_combos = [
        sorted(df[df["Dataset"] == dataset]["Combination"].unique()) for dataset in datasets
    ]
    num_combos_per_dataset = [len(combos) for combos in dataset_combos]
    total_axes = sum(num_combos_per_dataset)

    fig_width = 6 * total_axes
    fig_height = 6.5
    fig, axes = plt.subplots(
        1, total_axes, figsize=(fig_width, fig_height), sharey=True, sharex=False, squeeze=False
    )
    axes = axes.flatten()

    ax_idx = 0
    for dataset_idx, dataset in enumerate(datasets):
        dataset_df = df[df["Dataset"] == dataset]
        combinations = dataset_combos[dataset_idx]
        for j, combo in enumerate(combinations):
            ax = axes[ax_idx]
            subset = dataset_df[dataset_df["Combination"] == combo].sort_values("n")
            method_order = ["Baseline", "FastTTS"]
            n_values = sorted(subset["n"].unique())
            bar_width = 0.18
            x = np.arange(len(n_values))

            bar_vals = []
            for method in method_order:
                method_data = subset[subset["Method"] == method].set_index("n").reindex(n_values)
                bar_vals.append(method_data["mean_generator_latency"].values)
            for method in method_order:
                method_data = subset[subset["Method"] == method].set_index("n").reindex(n_values)
                bar_vals.append(method_data["mean_verifier_latency"].values)

            new_offsets = [-1.5, -0.5, 0.5, 1.5]
            new_colors = [colors[0], colors[2], colors[1], colors[3]]
            new_hatches = [None, None, "//", "//"]
            new_labels = [
                "Baseline Generator",
                "FastTTS Generator",
                "Baseline Verifier",
                "FastTTS Verifier",
            ]

            for k in range(4):
                bar_pos = x + new_offsets[k] * bar_width
                ax.bar(
                    bar_pos,
                    bar_vals[k],
                    width=bar_width,
                    color=new_colors[k],
                    edgecolor="black",
                    linewidth=1.2,
                    hatch=new_hatches[k],
                    label=new_labels[k] if ax_idx == 0 else None,
                    zorder=2,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in n_values])
            ax.set_xlabel(combo, labelpad=4)
            if j == 1:
                ax.set_title(f"{dataset}", pad=6, fontsize=32, fontweight="bold")
            ax.set_ylabel("Latency (s/completion)" if ax_idx == 0 else "", labelpad=4)
            ax.grid(True, which="both", ls="--", c="0.7", zorder=0)
            ax.tick_params(axis="x", rotation=0, pad=2)
            ax.tick_params(axis="y", pad=2)
            ax.set_yscale("symlog", linthresh=10)
            ax.minorticks_off()
            ax.margins(x=0.04, y=0.04)

            ax_idx += 1

    # Dataset separator lines
    dataset_split_indices = np.cumsum(num_combos_per_dataset)[:-1]
    for split_idx in dataset_split_indices:
        ax = axes[split_idx - 1]
        bbox = ax.get_position()
        x = bbox.x1 + 0.0155
        line = plt.Line2D(
            [x, x],
            [0.08, 0.98],
            color="black",
            linestyle="-",
            linewidth=2,
            alpha=0.85,
            transform=fig.transFigure,
            zorder=100,
        )
        fig.add_artist(line)

    legend_elements = [
        Patch(facecolor=base_color, edgecolor="k", label="Baseline"),
        Patch(facecolor=fasttts_color, edgecolor="k", label="FastTTS"),
        Patch(facecolor="w", edgecolor="k", label="Generator"),
        Patch(facecolor="w", edgecolor="k", hatch="//", label="Verifier"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.525, -0.06),
        ncol=4,
        frameon=False,
        borderaxespad=0.2,
        handletextpad=0.6,
        columnspacing=0.8,
        fontsize=30,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved latency figure to {output_path}")
    plt.close()


def plot_accuracy(accuracy_data, output_path):
    """Generate accuracy-vs-N scaling curves.

    Primary thesis figure: shows how accuracy scales with the number of
    completions N, for each metric (Pass@N, MajVote, PRM-Max, PRM-Vote).
    """
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )

    metric_labels = {
        "pass_at_n": "Pass@N",
        "majority_vote": "Majority Vote",
        "prm_max": "PRM-Max",
        "prm_vote": "PRM-Vote",
    }
    metric_styles = {
        "pass_at_n": {"color": "tab:gray", "linestyle": "--", "marker": "s"},
        "majority_vote": {"color": "tab:blue", "linestyle": "-", "marker": "o"},
        "prm_max": {"color": "tab:orange", "linestyle": "-", "marker": "^"},
        "prm_vote": {"color": "tab:green", "linestyle": "-", "marker": "D"},
    }

    # Collect all (dataset, combo, method) combinations that have results
    panels = []
    for dataset in accuracy_data:
        for combo in accuracy_data[dataset]:
            for method in accuracy_data[dataset][combo]:
                results = accuracy_data[dataset][combo][method]
                if isinstance(results, list) and results:
                    panels.append((dataset, combo, method, results))

    if not panels:
        print("No scaling data for accuracy plot")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax, (dataset, combo, method, results) in zip(axes, panels):
        n_vals = [r["n"] for r in results]

        for metric_key, label in metric_labels.items():
            vals = [r[metric_key] for r in results]
            style = metric_styles[metric_key]
            ax.plot(n_vals, vals, label=label, linewidth=2, markersize=6, **style)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("N (completions)")
        ax.set_ylabel("Accuracy (%)")
        method_label = "FastTTS" if method == "fasttts" else "Baseline"
        ax.set_title(f"{dataset.upper()} / {combo} / {method_label}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

        # Set x-ticks to actual N values
        ax.set_xticks(n_vals)
        ax.set_xticklabels([str(n) for n in n_vals], rotation=45)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved accuracy scaling figure to {output_path}")
    plt.close()


def has_valid_data(data):
    """Check if data dict contains actual metrics (not just empty nested dicts)."""
    for dataset, combos in data.items():
        for combo, methods in combos.items():
            for method, n_values in methods.items():
                if n_values:  # Non-empty dict with n values
                    return True
    return False


def generate_plots(data_dir: Path):
    """Generate all plots from existing data."""
    print("=" * 60)
    print(f"Generating plots (data from {data_dir})...")
    print("=" * 60)

    # Try to load existing main_results.json first (from data_dir)
    main_results_path = data_dir / "main_results.json"
    data = None

    if main_results_path.exists():
        print(f"Loading data from {main_results_path}")
        with open(main_results_path, "r") as f:
            data = json.load(f)
        # Check if data is valid (not empty)
        if not has_valid_data(data):
            print("Loaded data is empty, will collect from benchmark outputs...")
            data = None

    if data is None:
        print("Collecting results from benchmark outputs...")
        data = collect_results(data_dir)
        if has_valid_data(data):
            # Save collected results to data_dir
            os.makedirs(data_dir, exist_ok=True)
            with open(main_results_path, "w") as f:
                json.dump(data, f)
            print(f"Saved results to {main_results_path}")
        else:
            print("No valid benchmark results found in output files.")

    # Create figures directory inside data_dir
    figs_dir = data_dir / "figs"
    os.makedirs(figs_dir, exist_ok=True)

    # Plot goodput
    plot_goodput(data, figs_dir / "main_results_combined.pdf")

    # Plot latency
    plot_latency(data, figs_dir / "latency_combined.pdf")

    # Evaluate and plot accuracy
    accuracy_data = evaluate_accuracy(data_dir)
    if accuracy_data:
        plot_accuracy(accuracy_data, figs_dir / "acc.pdf")

    print("\n" + "=" * 60)
    print("All plots generated!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments and generate plots for FastTTS"
    )
    parser.add_argument(
        "--exp",
        action="store_true",
        help="Run all experiments",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots from existing/collected data",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help=f"Data directory for saving/loading results (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()

    if not args.exp and not args.plot:
        parser.print_help()
        print("\nPlease specify --exp and/or --plot")
        sys.exit(1)

    # Resolve data directory
    data_dir = Path(args.dir) if args.dir else DEFAULT_DATA_DIR
    data_dir = data_dir.resolve()
    print(f"Using data directory: {data_dir}")

    if args.exp:
        run_experiments(data_dir)

    if args.plot:
        generate_plots(data_dir)


if __name__ == "__main__":
    main()

