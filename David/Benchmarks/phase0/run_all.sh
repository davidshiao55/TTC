#!/bin/bash
# Phase 0 — Master Runner
#
# Runs all Phase 0 benchmarks in order. Each benchmark is independent
# and can also be run standalone.
#
# Usage:
#   conda activate thesis
#   cd /TTC/David/Benchmarks/phase0
#   bash run_all.sh              # run all
#   bash run_all.sh 0.1 0.3      # run specific benchmarks only
#
# Benchmarks:
#   0.1  CPU/GPU overlap feasibility  (Python, BF16 F.linear, needs GPU)
#   0.3  PCIe bandwidth sweep         (Python, needs GPU)
#   0.4  Column-split correctness     (Python, needs GPU)
#   0.5  CPU attention latency        (shell, needs vLLM CPU backend — SKIP for now)
#   0.6  CUDA graph impact            (shell, needs vLLM + model)
#   0.7  KV offload impact            (shell, needs vLLM + model)
#   0.8  FastTTS baseline             (shell, needs FastTTS + models)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR"

# If specific benchmarks requested, only run those
REQUESTED=("$@")

should_run() {
    if [ ${#REQUESTED[@]} -eq 0 ]; then
        return 0  # run all
    fi
    for r in "${REQUESTED[@]}"; do
        if [ "$r" = "$1" ]; then
            return 0
        fi
    done
    return 1
}

run_benchmark() {
    local id="$1"
    local name="$2"
    local cmd="$3"

    if ! should_run "$id"; then
        return
    fi

    echo ""
    echo "################################################################"
    echo "  Phase $id: $name"
    echo "  $(date)"
    echo "################################################################"
    echo ""

    eval "$cmd" 2>&1 | tee "${RESULTS_DIR}/bench_${id}.log"

    echo ""
    echo "  Phase $id complete. Log: ${RESULTS_DIR}/bench_${id}.log"
}

echo "Phase 0 — Pre-Implementation Benchmarking"
echo "=========================================="
echo "Start: $(date)"
echo "Results dir: $RESULTS_DIR"

# --- Micro-benchmarks (no vLLM engine needed) ---

run_benchmark "0.1" "CPU/GPU Overlap Feasibility (BF16 F.linear)" \
    "python ${SCRIPT_DIR}/bench_cpu_gpu_overlap.py --output-json ${RESULTS_DIR}/cpu_gpu_overlap.json"

run_benchmark "0.3" "PCIe Bandwidth Sweep" \
    "python ${SCRIPT_DIR}/bench_pcie_sweep.py --output-json ${RESULTS_DIR}/pcie_sweep.json"

run_benchmark "0.4" "Column-Split Correctness" \
    "python ${SCRIPT_DIR}/bench_column_split.py"

# --- vLLM-level benchmarks (need model weights) ---

# 0.5 SKIPPED — requires CPU attention kernel (not compiled in GPU build)
# Will benchmark when we build CPU ops for Phase 1b+2

run_benchmark "0.6" "CUDA Graph Impact" \
    "bash ${SCRIPT_DIR}/bench_cuda_graph.sh"

run_benchmark "0.7" "KV Offload Impact" \
    "python ${SCRIPT_DIR}/bench_kv_offload.py --exp --plot"

# --- Application-level benchmark ---

run_benchmark "0.8" "FastTTS V1 Baseline" \
    "bash ${SCRIPT_DIR}/bench_fasttts_baseline.sh"

echo ""
echo "=========================================="
echo "Phase 0 complete: $(date)"
echo "All results in: $RESULTS_DIR"
echo "=========================================="
