# FastTTS

This is the repository to reproduce the experiment results for FastTTS.

## TL;DR

One-line setup and run all experiments and reproduce the main results:
```bash
conda env create -f environment.yml && conda activate FastTTS && pip install -e . && (cd modified-skywork-o1-prm-inference && pip install -e . && cd ..) && python run_all_experiments.py --exp --plot
```

Figure 12 (Goodput), Figure 13 (Latency), and Figure 14a (Accuracy) are generated in `benchmarks/benchmark_results/figs/` by default.

## Installation

### 1. Create Conda Environment

First, create and activate the conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate FastTTS
```

### 2. Install FastTTS Package

Install the FastTTS package in development mode:

```bash
pip install -e .
```

### 3. Install Modified Skywork O1 PRM Inference

Navigate to the `modified-skywork-o1-prm-inference` directory and install it:

```bash
cd modified-skywork-o1-prm-inference
pip install -e .
cd ..
```

## Reproducing Experiment Results

The `run_all_experiments.py` script automates running all benchmarks and generating figures for the paper.

### Usage

```bash
# Run all experiments (saves results to default directory: benchmarks/benchmark_results/)
python run_all_experiments.py --exp

# Generate plots from existing results
python run_all_experiments.py --plot

# Run experiments and generate plots
python run_all_experiments.py --exp --plot

# Use a custom data directory
python run_all_experiments.py --exp --plot --dir /path/to/custom/results
```

### Flags

- `--exp`: Run all benchmark experiments for:
  - 2 generators: `7B-instruct`, `1.5B-instruct` (both `Qwen2.5-*-Instruct`, `max_model_len=8192`) paired with the fixed Skywork-PRM-1.5B verifier
  - 2 datasets: MATH-500 (primary, 500 problems), AIME 2024 (hard subset, 30 problems)
  - 2 methods: `fasttts` (all optimisations enabled) and `baseline`
  - 5 N values: `1, 4, 16, 64, 256` — N=1 is the "no TTC" reference point (`beam_width=1`); the other four match Liu et al.'s compute-optimal-tts sweep

- `--plot`: Generate figures from existing or newly collected data:
  - Goodput figure (`main_results_combined.pdf`)
  - Latency figure (`latency_combined.pdf`)
  - Accuracy scaling figure (`acc.pdf`) — four metrics per panel: Pass@N, MajVote, PRM-Max, PRM-Vote

- `--dir <path>`: Specify a custom directory for saving/loading results (default: `benchmarks/benchmark_results/`)

### Output Locations

When running with `--exp`, experiment results are saved as `.jsonl` files under:
```
<data_dir>/<generator>/<dataset>/<method>/
```

For example:
```
benchmarks/benchmark_results/7B-instruct/math500/fasttts/math500_bw4_n16_iter10_specdiff_results.jsonl
```

When running with `--plot`, the script:
1. Collects metrics from the `.jsonl` files and saves to `<data_dir>/main_results.json`
2. Evaluates accuracy (if result files exist) and saves to `<data_dir>/accuracy.json`
3. Generates figures in `<data_dir>/figs/`:
   - `main_results_combined.pdf` - Goodput comparison
   - `latency_combined.pdf` - Latency breakdown (generator/verifier)
   - `acc.pdf` - Accuracy comparison

