"""
Two separate sweeps to measure batch size vs throughput on vLLM:
  1. Sweep max-num-seqs (fixed gpu-memory-utilization=0.9)
  2. Sweep gpu-memory-utilization (fixed max-num-seqs=256)

Results saved under ./results/ with a combined CSV summary.
"""

import subprocess
import json
import csv
import os

# ── Configuration ──────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
NUM_PROMPTS = 512
INPUT_LEN = 512
OUTPUT_LEN = 128
DTYPE = "bfloat16"
CONDA_ENV = "vllm"

# Sweep 1: vary max-num-seqs, fix gpu mem
SEQS_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256]
FIXED_GPU_MEM = 0.9

# Sweep 2: vary gpu-memory-utilization, fix max-num-seqs
GPU_MEM_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIXED_MAX_SEQS = 256

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# ───────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_benchmark(max_seqs, gpu_mem, tag):
    json_path = os.path.join(RESULTS_DIR, f"{tag}.json")

    print(f"\n{'='*60}")
    print(f"Running: max-num-seqs={max_seqs}  gpu-memory-utilization={gpu_mem}")
    print(f"{'='*60}\n")

    bench_cmd = (
        f"conda activate {CONDA_ENV} && "
        f"vllm bench throughput "
        f"--model {MODEL} "
        f"--max-num-seqs {max_seqs} "
        f"--num-prompts {NUM_PROMPTS} "
        f"--dataset-name random "
        f"--random-input-len {INPUT_LEN} "
        f"--random-output-len {OUTPUT_LEN} "
        f"--dtype {DTYPE} "
        f"--gpu-memory-utilization {gpu_mem} "
        f"--output-json {json_path}"
    )

    try:
        subprocess.run(["bash", "-ic", bench_cmd], check=True)
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {tag} (exit code {e.returncode})")
        return {
            "gpu_memory_utilization": gpu_mem,
            "max_num_seqs": max_seqs,
            "elapsed_time": None,
            "num_requests": None,
            "requests_per_second": None,
            "tokens_per_second": None,
            "status": "FAILED",
        }

    with open(json_path) as f:
        data = json.load(f)

    print(f"  -> {data['tokens_per_second']:.1f} tok/s  "
          f"({data['requests_per_second']:.2f} req/s)")

    return {
        "gpu_memory_utilization": gpu_mem,
        "max_num_seqs": max_seqs,
        "elapsed_time": data["elapsed_time"],
        "num_requests": data["num_requests"],
        "requests_per_second": data["requests_per_second"],
        "tokens_per_second": data["tokens_per_second"],
        "status": "OK",
    }


def write_csv(rows, filename):
    fields = [
        "gpu_memory_utilization", "max_num_seqs",
        "elapsed_time", "num_requests",
        "requests_per_second", "tokens_per_second", "status",
    ]
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ── Sweep 1: max-num-seqs ─────────────────────────────────────
print("\n" + "#"*60)
print("# SWEEP 1: max-num-seqs  (gpu-memory-utilization fixed at "
      f"{FIXED_GPU_MEM})")
print("#"*60)

seqs_rows = []
for seqs in SEQS_SWEEP:
    tag = f"sweep_seqs_{seqs}"
    result = run_benchmark(seqs, FIXED_GPU_MEM, tag)
    seqs_rows.append(result)

write_csv(seqs_rows, "sweep_max_num_seqs.csv")

# ── Sweep 2: gpu-memory-utilization ───────────────────────────
print("\n" + "#"*60)
print("# SWEEP 2: gpu-memory-utilization  (max-num-seqs fixed at "
      f"{FIXED_MAX_SEQS})")
print("#"*60)

mem_rows = []
for mem in GPU_MEM_SWEEP:
    tag = f"sweep_mem_{mem}"
    result = run_benchmark(FIXED_MAX_SEQS, mem, tag)
    mem_rows.append(result)

write_csv(mem_rows, "sweep_gpu_mem_util.csv")

print(f"\n{'='*60}")
print("All done. Results in ./results/")
print(f"  sweep_max_num_seqs.csv  — {len(seqs_rows)} runs")
print(f"  sweep_gpu_mem_util.csv  — {len(mem_rows)} runs")
print(f"{'='*60}")
