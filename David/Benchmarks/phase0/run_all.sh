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
# Benchmarks (numbering matches phase0_findings.md):
#   0.1    num_tokens dispatch axis        (Python, GPU + CPU)
#   0.2    Tensor split correctness        (Python, needs GPU)
#   0.3    CPU/GPU compute characterization (Python, BF16 F.linear, needs GPU)
#   0.4.1  MLP block col→row pipeline      (Python, needs GPU)
#   0.4.2  WO offload Alt A vs Alt B       (Python, needs GPU)
#   0.5.0  PCIe bandwidth sweep            (Python, needs GPU)
#   0.5    PCIe contention (nsys-driven)   (Python, needs nsys)
#   0.6    CPU attention latency           (Python, needs vLLM CPU backend — SKIP)
#   0.7    CUDA graph impact               (shell, needs vLLM + model)
#   0.8/9  V1 baseline + KV offload impact (Python, needs FastTTS + models)
#   0.10   vLLM native weight offloader baseline (Python, needs vLLM CLI + models)

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

run_benchmark "0.1" "num_tokens dispatch axis" \
    "python ${SCRIPT_DIR}/bench_num_tokens_axis.py --model qwen7b  --output-json ${RESULTS_DIR}/0.1_num_tokens/qwen7b.json && \
     python ${SCRIPT_DIR}/bench_num_tokens_axis.py --model prm1p5b --output-json ${RESULTS_DIR}/0.1_num_tokens/prm1p5b.json"

run_benchmark "0.2" "Tensor Split Correctness (mixed col/row)" \
    "python ${SCRIPT_DIR}/bench_split_correctness.py"

run_benchmark "0.3" "CPU/GPU Compute Characterization (BF16 F.linear)" \
    "python ${SCRIPT_DIR}/bench_cpu_gpu_overlap.py --model qwen7b  --output-json ${RESULTS_DIR}/0.3_cpu_gpu_overlap/qwen7b.json && \
     python ${SCRIPT_DIR}/bench_cpu_gpu_overlap.py --model prm1p5b --output-json ${RESULTS_DIR}/0.3_cpu_gpu_overlap/prm1p5b.json"

run_benchmark "0.4.1" "MLP Block Pipeline: uniform col vs col→row" \
    "python ${SCRIPT_DIR}/bench_mlp_pipeline.py --model qwen7b  --output-json ${RESULTS_DIR}/0.4_split_axis/mlp_pipeline_qwen7b.json && \
     python ${SCRIPT_DIR}/bench_mlp_pipeline.py --model prm1p5b --output-json ${RESULTS_DIR}/0.4_split_axis/mlp_pipeline_prm1p5b.json"

run_benchmark "0.4.2" "WO Offload Alt A vs Alt B" \
    "python ${SCRIPT_DIR}/bench_wo_offload_tradeoff.py --model qwen7b  --output-json ${RESULTS_DIR}/0.4_split_axis/wo_offload_qwen7b.json && \
     python ${SCRIPT_DIR}/bench_wo_offload_tradeoff.py --model prm1p5b --output-json ${RESULTS_DIR}/0.4_split_axis/wo_offload_prm1p5b.json"

run_benchmark "0.5.0" "PCIe Bandwidth Sweep" \
    "python ${SCRIPT_DIR}/bench_pcie_sweep.py --output-json ${RESULTS_DIR}/0.5_pcie/pcie_bw.json"

run_benchmark "0.5" "PCIe Contention (nsys-driven)" \
    "python ${SCRIPT_DIR}/bench_contention.py --output-json ${RESULTS_DIR}/0.5_pcie/contention.json"

# --- vLLM-level benchmarks (need model weights) ---

# 0.6 SKIPPED — requires CPU attention kernel (not compiled in GPU build).
# Will benchmark when we build CPU ops for Phase 1b+2.

run_benchmark "0.7" "CUDA Graph Impact" \
    "bash ${SCRIPT_DIR}/bench_cuda_graph.sh"

# --- Application-level benchmark (V1 baseline §0.8 + KV offload ablation §0.9) ---

run_benchmark "0.8/0.9" "V1 baseline + KV offload Impact" \
    "python ${SCRIPT_DIR}/bench_kv_offload.py --exp --plot"

run_benchmark "0.10.1" "UVA vs Prefetch (head-to-head)" \
    "python ${SCRIPT_DIR}/bench_uva_vs_prefetch.py --exp --plot"

run_benchmark "0.10.2" "PrefetchOffloader knob sweep (G, N, K)" \
    "python ${SCRIPT_DIR}/bench_prefetch_knobs.py --exp --plot"

run_benchmark "0.10.probe" "Native offloader overlap probe (nsys)" \
    "python ${SCRIPT_DIR}/probe_native_offload_overlap.py --arm all"

echo ""
echo "=========================================="
echo "Phase 0 complete: $(date)"
echo "All results in: $RESULTS_DIR"
echo "=========================================="
