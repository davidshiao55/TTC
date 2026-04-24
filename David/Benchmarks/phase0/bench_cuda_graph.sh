#!/bin/bash
# Phase 0.6 — CUDA Graph Impact Benchmark
#
# Measures decode latency with CUDA Graphs enabled vs disabled (enforce_eager).
# Quantifies the performance cost of prototyping without CUDA Graphs.
#
# Usage: conda activate thesis && cd /TTC/David/Benchmarks/phase0 && bash bench_cuda_graph.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR"

MODEL="Qwen/Qwen2.5-7B-Instruct"
DTYPE="bfloat16"
INPUT_LEN=512
OUTPUT_LEN=128
NUM_ITERS=10
NUM_WARMUP=3

# Focus on the decode-critical range (matches Phase 3 sub-layer pipeline).
# Skipping B=256+ — capture cost scales with batch count, and we've
# already validated BW at B=64 represents the memory-bound regime.
BATCH_SIZES=(1 4 16 64)

OUTFILE="${RESULTS_DIR}/cuda_graph_impact.csv"
echo -e "batch_size\tmode\tavg_latency_ms\tp50_ms\tp99_ms" > "$OUTFILE"

echo "================================================================"
echo "  Phase 0.6: CUDA Graph Impact"
echo "  Model: $MODEL"
echo "  Input/Output: $INPUT_LEN / $OUTPUT_LEN tokens"
echo "================================================================"

for B in "${BATCH_SIZES[@]}"; do
    for MODE in "cuda_graph" "eager"; do
        echo ""
        echo "--- B=$B, mode=$MODE ---"

        EXTRA_ARGS=""
        if [ "$MODE" = "eager" ]; then
            EXTRA_ARGS="--enforce-eager"
        fi

        JSON_OUT="${RESULTS_DIR}/latency_${MODE}_B${B}.json"

        # Run from /tmp to avoid vllm path conflict
        cd /tmp && vllm bench latency \
            --model "$MODEL" \
            --dtype "$DTYPE" \
            --batch-size "$B" \
            --input-len "$INPUT_LEN" \
            --output-len "$OUTPUT_LEN" \
            --num-iters-warmup "$NUM_WARMUP" \
            --num-iters "$NUM_ITERS" \
            --output-json "$JSON_OUT" \
            $EXTRA_ARGS 2>&1 | tail -5

        # Parse results
        if [ -f "$JSON_OUT" ]; then
            AVG=$(python3 -c "import json; d=json.load(open('$JSON_OUT')); print(f\"{d['avg_latency']*1000:.2f}\")" 2>/dev/null || echo "N/A")
            P50=$(python3 -c "import json; d=json.load(open('$JSON_OUT')); print(f\"{d['percentiles']['p50']*1000:.2f}\")" 2>/dev/null || echo "N/A")
            P99=$(python3 -c "import json; d=json.load(open('$JSON_OUT')); print(f\"{d['percentiles']['p99']*1000:.2f}\")" 2>/dev/null || echo "N/A")
        else
            AVG="FAIL"; P50="FAIL"; P99="FAIL"
        fi

        echo -e "${B}\t${MODE}\t${AVG}\t${P50}\t${P99}" >> "$OUTFILE"
        echo "  avg=${AVG}ms  p50=${P50}ms  p99=${P99}ms"
    done
done

echo ""
echo "================================================================"
echo "  Summary: CUDA Graph vs Eager"
echo "================================================================"

# Print comparison table
python3 - "$RESULTS_DIR" << 'PYEOF'
import csv, sys
results_dir = sys.argv[1]
rows = []
with open(f"{results_dir}/cuda_graph_impact.csv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

print(f"{'B':>4} {'Graph (ms)':>12} {'Eager (ms)':>12} {'Overhead':>10}")
print("-" * 42)

by_batch = {}
for r in rows:
    B = r["batch_size"]
    if B not in by_batch:
        by_batch[B] = {}
    by_batch[B][r["mode"]] = r["avg_latency_ms"]

for B in sorted(by_batch.keys(), key=int):
    g = by_batch[B].get("cuda_graph", "N/A")
    e = by_batch[B].get("eager", "N/A")
    try:
        overhead = f"{(float(e)/float(g) - 1)*100:+.1f}%"
    except (ValueError, ZeroDivisionError):
        overhead = "N/A"
    print(f"{B:>4} {g:>12} {e:>12} {overhead:>10}")
PYEOF

echo ""
echo "Raw results: $OUTFILE"
echo "Per-run JSONs: ${RESULTS_DIR}/latency_*.json"
