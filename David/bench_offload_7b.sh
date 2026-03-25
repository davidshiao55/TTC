#!/bin/bash
# Test prefetch offloading on 7B model:
# Fix offload-num-in-group=1, vary offload-group-size to control fraction offloaded.
# Qwen2.5-Math-7B has 28 layers.
#   group-size=28 → offload 1/28 layers
#   group-size=14 → offload 2/28 layers
#   group-size=7  → offload 4/28 layers
#   group-size=4  → offload 7/28 layers
#   group-size=2  → offload 14/28 layers
#   group-size=1  → offload 28/28 layers
#
# Usage: conda activate vllm && bash David/bench_offload_7b.sh

set -e

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
NUM_PROMPTS=1024
INPUT_LEN=512
OUTPUT_LEN=128
DTYPE="bfloat16"
GPU_MEM=0.75
MAX_SEQS=256

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_offload_7b"
mkdir -p "$RESULTS_DIR"

CSV_FILE="${RESULTS_DIR}/sweep_offload.csv"
echo "offload_group_size,layers_offloaded,elapsed_time,num_requests,requests_per_second,tokens_per_second,status" \
    > "$CSV_FILE"

# Baseline: no offloading
echo ""
echo "============================================================"
echo "Running: NO offloading (baseline)"
echo "============================================================"
echo ""

JSON_PATH="${RESULTS_DIR}/offload_baseline.json"
if vllm bench throughput \
    --model "$MODEL" \
    --max-num-seqs "$MAX_SEQS" \
    --num-prompts "$NUM_PROMPTS" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEM" \
    --output-json "$JSON_PATH"; then

    python3 -c "
import json
with open('${JSON_PATH}') as f:
    d = json.load(f)
print(f'0,0,{d[\"elapsed_time\"]},{d[\"num_requests\"]},{d[\"requests_per_second\"]},{d[\"tokens_per_second\"]},OK')
" >> "$CSV_FILE"
    echo "  -> OK"
else
    echo "0,0,,,,,FAILED" >> "$CSV_FILE"
    echo "  -> FAILED"
fi

# Sweep: offload-group-size with offload-num-in-group=1
for GS in 28 14 7 4; do
    LAYERS_OFFLOADED=$((28 / GS))
    TAG="offload_gs${GS}"
    JSON_PATH="${RESULTS_DIR}/${TAG}.json"

    echo ""
    echo "============================================================"
    echo "Running: offload-group-size=${GS} (${LAYERS_OFFLOADED}/28 layers offloaded)"
    echo "============================================================"
    echo ""

    if vllm bench throughput \
        --model "$MODEL" \
        --max-num-seqs "$MAX_SEQS" \
        --num-prompts "$NUM_PROMPTS" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$GPU_MEM" \
        --offload-group-size "$GS" \
        --offload-num-in-group 1 \
        --output-json "$JSON_PATH"; then

        python3 -c "
import json
with open('${JSON_PATH}') as f:
    d = json.load(f)
print(f'${GS},${LAYERS_OFFLOADED},{d[\"elapsed_time\"]},{d[\"num_requests\"]},{d[\"requests_per_second\"]},{d[\"tokens_per_second\"]},OK')
" >> "$CSV_FILE"
        echo "  -> OK"
    else
        echo "${GS},${LAYERS_OFFLOADED},,,,,FAILED" >> "$CSV_FILE"
        echo "  -> FAILED"
    fi
done

echo ""
echo "============================================================"
echo "All done. Results in ${RESULTS_DIR}/"
echo "  sweep_offload.csv"
echo "============================================================"
