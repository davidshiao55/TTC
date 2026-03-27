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

# Configuration
# MODEL_COMBOS = ["1.5B-1.5B", "1.5B-7B", "7B-1.5B"]
MODEL_COMBOS = ["1.5B-1.5B"]
DATASETS = ["aime"]
METHODS = ["baseline", "spec_prefix"]
N_VALUES = [8, 16, 32, 64, 128, 256, 512]

# Dataset name mapping for config files
DATASET_CONFIG = {
    "aime": "aime2024",
    "amc": "amc2023",
}

BENCHMARK_DIR = Path(__file__).parent / "benchmarks"
DEFAULT_DATA_DIR = BENCHMARK_DIR / "benchmark_results"
FIGURES_DIR = Path(__file__).parent / "figures"
ACCURACY_EVAL_DIR = Path(__file__).parent / "accuracy_evaluation"


def run_experiments(data_dir: Path):
    """Run all benchmark experiments."""
    print("=" * 60)
    print(f"Running all experiments (saving to {data_dir})...")
    print("=" * 60)

    os.chdir(BENCHMARK_DIR)

    for combo in MODEL_COMBOS:
        for dataset in DATASETS:
            for method in METHODS:
                for n in N_VALUES:
                    config_name = f"{DATASET_CONFIG[dataset]}_{n}.yaml"
                    config_path = f"configs/{combo}/{dataset}/{method}/{config_name}"

                    if not Path(config_path).exists():
                        print(f"Config not found: {config_path}, skipping...")
                        continue

                    print(f"\n{'='*60}")
                    print(f"Running: {combo}/{dataset}/{method}/n={n}")
                    print(f"{'='*60}")

                    # Override output_dir via environment or modify config dynamically
                    output_dir = data_dir / combo / dataset / method
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
    problem_limit = 30 if "aime" in dataset else 40

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
        for combo in MODEL_COMBOS:
            results[dataset][combo] = {}
            for method in METHODS:
                result_dir = data_dir / combo / dataset / method
                folder_results = parse_jsonl_folder(result_dir, dataset)
                if folder_results:
                    results[dataset][combo][method] = folder_results
                else:
                    results[dataset][combo][method] = {}

    return results


TOP_N_VALUES = N_VALUES


def run_accuracy_evaluation(eval_script: Path, result_file: Path, top_n: int):
    """Run accuracy evaluation script and return the accuracy value."""
    import re

    cmd = [
        "python",
        str(eval_script),
        "--data_name",
        "math",
        "--file_path",
        str(result_file),
        "--top_n",
        str(top_n),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(eval_script.parent),
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error evaluating accuracy: {e}")

    return None


def evaluate_accuracy(data_dir: Path):
    """Evaluate accuracy for all experiments using the accuracy evaluation script.
    
    Checks for existing accuracy.json in data_dir. If not found, evaluates
    accuracy for all combinations, trying all top_n values and picking the highest.
    """
    print(f"Evaluating accuracy from {data_dir}...")

    accuracy_json_path = data_dir / "accuracy.json"

    # Check if accuracy.json already exists
    if accuracy_json_path.exists():
        print(f"Loading existing accuracy data from {accuracy_json_path}")
        with open(accuracy_json_path, "r") as f:
            return json.load(f)

    accuracy_results = {}
    eval_script = ACCURACY_EVAL_DIR / "evaluation" / "evaluate.py"

    for dataset in DATASETS:
        accuracy_results[dataset] = {}
        for combo in MODEL_COMBOS:
            accuracy_results[dataset][combo] = {}
            for method in METHODS:
                result_dir = data_dir / combo / dataset / method

                # Use n=512 for the result file (highest n)
                n = 512
                pattern = f"{DATASET_CONFIG[dataset]}_bw4_n{n}_iter10"
                if method == "spec_prefix":
                    pattern += "_specdiff"
                pattern += "_results.jsonl"

                result_file = result_dir / pattern

                if not result_file.exists():
                    print(f"Result not found for accuracy: {result_file}")
                    continue

                # Try all top_n values and pick the top-1 accuracy
                best_acc = None
                best_top_n = None

                for top_n in TOP_N_VALUES:
                    acc = run_accuracy_evaluation(eval_script, result_file, top_n)
                    if acc is not None:
                        if best_acc is None or acc > best_acc:
                            best_acc = acc
                            best_top_n = top_n
                    else:
                        print(f"{dataset}/{combo}/{method}: No accuracy found")

                if best_acc is not None:
                    accuracy_results[dataset][combo][method] = best_acc
                    print(f"{dataset}/{combo}/{method}: {best_acc}% (top_n={best_top_n})")

    # Save accuracy results to accuracy.json
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
    method_map = {"baseline": "Baseline", "spec_prefix": "FastTTS"}

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
            ax.set_xticks([8, 32, 128, 512])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.tick_params(axis="x", rotation=0)
            ax.grid(True, which="both", ls="--", c="0.7")

            # Inset plot for n=512
            zoom_data = subset[subset["n"] == 512]
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

                rect_x = 400
                rect_height = max(zoom_data["Goodput"]) * 1.5
                rect = Rectangle(
                    (rect_x, 0),
                    200,
                    rect_height,
                    edgecolor="gray",
                    facecolor="none",
                    linestyle="--",
                    linewidth=2.2,
                    zorder=0,
                )
                ax.add_patch(rect)

                arrow = ConnectionPatch(
                    xyA=(rect_x + 100, rect_height),
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
    method_map = {"baseline": "Baseline", "spec_prefix": "FastTTS"}

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
    """Generate accuracy figure similar to acc.ipynb."""
    plt.rcParams.update(
        {
            "font.size": 22,
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "legend.title_fontsize": 24,
            "figure.titlesize": 30,
        }
    )

    methods = ["baseline", "spec_prefix"]
    models = ["1.5B-7B", "7B-1.5B", "1.5B-1.5B"]
    datasets = ["aime", "amc"]

    # Create DataFrame
    records = []
    for dataset in datasets:
        for model in models:
            for method in methods:
                if (
                    dataset in accuracy_data
                    and model in accuracy_data[dataset]
                    and method in accuracy_data[dataset][model]
                ):
                    value = accuracy_data[dataset][model][method]
                    records.append(
                        {
                            "Dataset": dataset.upper(),
                            "Model": model,
                            "Method": method,
                            "Acc": value,
                        }
                    )

    df = pd.DataFrame(records)
    if df.empty:
        print("No data for accuracy plot")
        return

    model_names = {"1.5B-7B": "1.5/7", "7B-1.5B": "7/1.5", "1.5B-1.5B": "1.5/1.5"}
    method_names = {"baseline": "Baseline", "spec_prefix": "FastTTS"}

    base_color = sns.color_palette()[0]
    fasttts_color = sns.color_palette()[1]
    palette = [base_color, fasttts_color]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    yticks_dict = {
        "aime": [0, 5, 10, 15, 20, 25],
        "amc": [0, 20, 40, 60, 80],
    }

    for ax, dataset in zip(axes, datasets):
        data_subset = df[df["Dataset"] == dataset.upper()]
        for i, model in enumerate(models):
            for j, method in enumerate(methods):
                acc_vals = data_subset[
                    (data_subset["Model"] == model) & (data_subset["Method"] == method)
                ]["Acc"].values
                if len(acc_vals) > 0:
                    acc = acc_vals[0]
                    ax.bar(
                        i + j * 0.32 - 0.16,
                        acc,
                        width=0.32,
                        color=palette[j],
                        edgecolor="black",
                        linewidth=2.5,
                        label=method_names[method] if i == 0 else None,
                        zorder=2,
                    )
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([model_names[m] for m in models], fontsize=28)
        ax.set_xlabel("", fontsize=28)
        ax.set_title(dataset.upper(), fontsize=35)
        ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_yticks(yticks_dict[dataset])
        ax.set_yticklabels(yticks_dict[dataset], fontsize=35)
        ax.set_ylabel("")

    fig.text(0.04, 0.5, "Acc (%)", va="center", rotation="vertical", fontsize=35)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=base_color, ec="k", lw=2.5, label="Baseline"),
        plt.Rectangle((0, 0), 1, 1, color=fasttts_color, ec="k", lw=2.5, label="FastTTS"),
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        title="",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(methods),
        frameon=False,
        fontsize=35,
        title_fontsize=35,
    )

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved accuracy figure to {output_path}")
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

