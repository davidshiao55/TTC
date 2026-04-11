"""Task 2: Context length accuracy experiment — 7B generator at 4096 vs 8192.

Runs both benchmark configs, then evaluates accuracy using the existing
evaluation pipeline and prints a comparison table.

Run from /TTC/FastTTS-thesis with ``conda activate thesis``:

    python migration_verification/run_ctx_experiment.py
"""

import json
import subprocess
import sys
from pathlib import Path
import os

CONFIGS = [
    {
        "label": "7B-Instruct ctx=4096",
        "config": "benchmarks/configs/7B-1.5B/aime/ctx4096/aime2024_8.yaml",
        "output_dir": "benchmark_results/7B-1.5B/aime/ctx4096",
    },
    {
        "label": "7B-Instruct ctx=8192",
        "config": "benchmarks/configs/7B-1.5B/aime/ctx8192/aime2024_8.yaml",
        "output_dir": "benchmark_results/7B-1.5B/aime/ctx8192",
    },
    {
        "label": "7B-Math ctx=4096",
        "config": "benchmarks/configs/7B-1.5B/aime/ctx4096/aime2024_8_math.yaml",
        "output_dir": "benchmark_results/7B-1.5B/aime/ctx4096_math",
    },
]

EVAL_SCRIPT = "accuracy_evaluation/evaluation/evaluate.py"
DATA_NAME = "aime"  # for extract_answer format


def find_result_file(output_dir: str) -> Path:
    """Find the single .jsonl result file in the output directory."""
    output_path = Path(FASTTTS_DIR) / output_dir
    jsonl_files = list(output_path.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files in {output_dir}")
    return jsonl_files[0]


FASTTTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_benchmark(config_path: str) -> None:
    """Run a benchmark via subprocess."""
    cmd = [sys.executable, "benchmarks/run_benchmarks.py", config_path]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=FASTTTS_DIR)


def evaluate_results(result_file: Path) -> dict:
    """Evaluate a result file using the existing evaluation pipeline."""
    cmd = [
        sys.executable, "evaluate.py",
        "--data_name", DATA_NAME,
        "--file_path", str(result_file),
        "--top_n", "8",
    ]
    print(f"  Evaluating: {result_file.name}")
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=os.path.join(FASTTTS_DIR, "accuracy_evaluation", "evaluation"),
    )
    if result.returncode != 0:
        print(f"  Evaluation stderr:\n{result.stderr}")
        raise RuntimeError(f"Evaluation failed for {result_file}")

    # The evaluate.py script prints the accuracy as the last line
    acc_line = result.stdout.strip().split("\n")[-1]
    try:
        acc = float(acc_line)
    except ValueError:
        print(f"  Could not parse accuracy from: {acc_line!r}")
        print(f"  Full stdout:\n{result.stdout}")
        acc = None
    return {"accuracy": acc, "file": str(result_file)}


def collect_stats(result_file: Path) -> dict:
    """Extract summary stats from a JSONL result file."""
    problems = []
    with open(result_file) as f:
        for line in f:
            problems.append(json.loads(line))

    total_beams = 0
    total_gen_latency = 0.0
    total_ver_latency = 0.0
    truncation_warnings = 0

    for p in problems:
        sol = p.get("solutions", {})
        completions = sol.get("completions", [[]])[0]
        total_beams += len(completions)
        total_gen_latency += sol.get("total_generator_latency_s", 0)
        total_ver_latency += sol.get("total_verifier_latency_s", 0)

    n = len(problems)
    return {
        "num_problems": n,
        "avg_beams_per_problem": total_beams / n if n else 0,
        "avg_gen_latency_s": total_gen_latency / n if n else 0,
        "avg_ver_latency_s": total_ver_latency / n if n else 0,
    }


def main():
    print("=" * 64)
    print("Context Length Experiment: 7B Generator (4096 vs 8192)")
    print("=" * 64)

    results = {}

    # Phase 1: Run benchmarks
    for cfg in CONFIGS:
        print(f"\n--- Benchmark: {cfg['label']} ---")
        run_benchmark(cfg["config"])

    # Phase 2: Evaluate and collect stats
    print("\n" + "=" * 64)
    print("Results")
    print("=" * 64)

    for cfg in CONFIGS:
        result_file = find_result_file(cfg["output_dir"])
        eval_result = evaluate_results(result_file)
        stats = collect_stats(result_file)
        results[cfg["label"]] = {**eval_result, **stats}

    # Phase 3: Comparison table
    labels = [cfg["label"] for cfg in CONFIGS]
    header = f"{'Metric':<30}" + "".join(f"{l:>20}" for l in labels)
    print(f"\n{header}")
    print("-" * len(header))
    for metric in ["accuracy", "num_problems", "avg_beams_per_problem",
                    "avg_gen_latency_s", "avg_ver_latency_s"]:
        row = f"{metric:<30}"
        for label in labels:
            v = results[label].get(metric)
            if isinstance(v, float):
                row += f"{v:>20.2f}"
            else:
                row += f"{str(v):>20}"
        print(row)

    print(f"\nResult files:")
    for cfg in CONFIGS:
        print(f"  {cfg['label']}: {results[cfg['label']]['file']}")


if __name__ == "__main__":
    main()
