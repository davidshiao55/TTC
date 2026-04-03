#!/bin/bash
# Phase 0.8 — FastTTS V1 Baseline
#
# Runs FastTTS-thesis end-to-end on 7B to establish the V1 baseline.
#
# Usage: conda activate thesis && bash David/Benchmarks/phase0/bench_fasttts_baseline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results/fasttts_baseline"
mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "  Phase 0.8: FastTTS V1 Baseline"
echo "================================================================"

# Run from FastTTS-thesis directory (avoids vllm path conflict)
cd /TTC/FastTTS-thesis

echo "Running experiments..."
python run_all_experiments.py \
    --exp \
    --plot \
    --dir "$RESULTS_DIR" 2>&1 | tee "${RESULTS_DIR}/run.log"

echo ""
echo "================================================================"
echo "Results saved to: $RESULTS_DIR"
echo "================================================================"
