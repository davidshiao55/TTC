#!/bin/bash
# Phase 0.5 — CPU Attention Latency
#
# Sweeps batch size × suffix length for Qwen2.5-7B attention config.
# Uses vLLM's existing benchmark_cpu_attn.py.
#
# Usage: cd /TTC/David/Benchmarks/phase0 && bash bench_cpu_attn.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR"

BENCH_SCRIPT="/TTC/vllm/benchmarks/kernels/cpu/benchmark_cpu_attn.py"

# Qwen2.5-7B attention config
NUM_QUERY_HEADS=28
NUM_KV_HEADS=4
HEAD_SIZE=128
BLOCK_SIZE=32

# Sweep parameters
BATCH_SIZES=(4 8 16 32)
# kv-len represents suffix length (the part on CPU)
KV_LENS=(100 500 1000 2000)

OUTFILE="${RESULTS_DIR}/cpu_attn_latency.csv"
echo -e "batch_size\tkv_len\tmean_ms\tmedian_ms\tstd_ms" > "$OUTFILE"

echo "================================================================"
echo "  Phase 0.5: CPU Attention Latency (Qwen2.5-7B config)"
echo "  heads=$NUM_QUERY_HEADS, kv_heads=$NUM_KV_HEADS, head_dim=$HEAD_SIZE"
echo "================================================================"

for B in "${BATCH_SIZES[@]}"; do
    for S in "${KV_LENS[@]}"; do
        echo ""
        echo "--- B=$B, suffix_len=$S ---"

        # Run benchmark, capture output
        OUTPUT=$(cd /tmp && python "$BENCH_SCRIPT" \
            --batch-size "$B" \
            --q-len-min 1 --q-len-max 1 \
            --kv-len-min "$S" --kv-len-max "$S" \
            --num-query-heads "$NUM_QUERY_HEADS" \
            --num-kv-heads "$NUM_KV_HEADS" \
            --head-size "$HEAD_SIZE" \
            --block-size "$BLOCK_SIZE" \
            --dtype bfloat16 \
            --iters 30 2>&1) || true

        echo "$OUTPUT"

        # Parse mean and median from output (format varies, try to extract)
        MEAN=$(echo "$OUTPUT" | grep -i "mean" | head -1 | grep -oP '[\d.]+' | head -1 || echo "N/A")
        MEDIAN=$(echo "$OUTPUT" | grep -i "median" | head -1 | grep -oP '[\d.]+' | head -1 || echo "N/A")
        STD=$(echo "$OUTPUT" | grep -i "std" | head -1 | grep -oP '[\d.]+' | head -1 || echo "N/A")

        echo -e "${B}\t${S}\t${MEAN}\t${MEDIAN}\t${STD}" >> "$OUTFILE"
    done
done

echo ""
echo "================================================================"
echo "Results saved to: $OUTFILE"
cat "$OUTFILE"
