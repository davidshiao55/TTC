#!/bin/bash
# Phase 0.7 — KV Cache CPU Offload Impact
#
# Tests how KV cache pressure and CPU offloading affect performance.
# Two approaches:
#   1. Vary gpu-memory-utilization to control KV cache size (forces scheduler limits)
#   2. Test vLLM's KV offload if available (check vllm CLI flags)
#
# Usage: conda activate thesis && cd /TTC/David/Benchmarks/phase0 && bash bench_kv_offload.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR"

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
DTYPE="bfloat16"
NUM_PROMPTS=256
INPUT_LEN=512
OUTPUT_LEN=128
MAX_NUM_SEQS=256

# ---- Part 1: GPU memory utilization sweep (KV pressure) ----
# This replicates and extends the findings from vllm_benchmarking_findings.md on V1

GPU_MEMS=(0.70 0.75 0.80 0.85 0.90 0.95)

OUTFILE="${RESULTS_DIR}/kv_pressure.csv"
echo -e "gpu_mem_util\ttokens_per_sec\trequests_per_sec\telapsed_time" > "$OUTFILE"

echo "================================================================"
echo "  Phase 0.7a: KV Cache Pressure (GPU memory utilization sweep)"
echo "  Model: $MODEL | Prompts: $NUM_PROMPTS | max-num-seqs: $MAX_NUM_SEQS"
echo "================================================================"

for GPU_MEM in "${GPU_MEMS[@]}"; do
    echo ""
    echo "--- gpu-memory-utilization=$GPU_MEM ---"

    JSON_OUT="${RESULTS_DIR}/kv_pressure_gm${GPU_MEM}.json"

    cd /tmp && vllm bench throughput \
        --model "$MODEL" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$GPU_MEM" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --num-prompts "$NUM_PROMPTS" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --output-json "$JSON_OUT" 2>&1 | tail -3

    if [ -f "$JSON_OUT" ]; then
        TOK=$(python3 -c "import json; d=json.load(open('$JSON_OUT')); print(f\"{d['tokens_per_second']:.1f}\")" 2>/dev/null || echo "N/A")
        REQ=$(python3 -c "import json; d=json.load(open('$JSON_OUT')); print(f\"{d['requests_per_second']:.2f}\")" 2>/dev/null || echo "N/A")
        TIME=$(python3 -c "import json; d=json.load(open('$JSON_OUT')); print(f\"{d['elapsed_time']:.2f}\")" 2>/dev/null || echo "N/A")
    else
        TOK="FAIL"; REQ="FAIL"; TIME="FAIL"
    fi

    echo -e "${GPU_MEM}\t${TOK}\t${REQ}\t${TIME}" >> "$OUTFILE"
    echo "  tokens/s=$TOK  requests/s=$REQ  elapsed=${TIME}s"
done

# ---- Part 2: KV offload to CPU (if supported) ----
# vLLM V1 may support KV offloading via --kv-cache-offload or similar flags.
# Check available flags and test if present.

echo ""
echo "================================================================"
echo "  Phase 0.7b: KV Offload to CPU"
echo "================================================================"

# Check if kv offload flags exist
KV_OFFLOAD_HELP=$(cd /tmp && vllm bench throughput --help 2>&1 || true)

if echo "$KV_OFFLOAD_HELP" | grep -qi "kv.*offload\|offload.*kv\|kv-transfer"; then
    echo "KV offload flags detected. Testing..."

    # Try common flag patterns
    for FLAG in "--kv-cache-offload-percent 0.5" "--kv-transfer-config '{\"kv_connector\":\"PyNcclConnector\"}'" ; do
        echo "  Trying: $FLAG"
        JSON_OUT="${RESULTS_DIR}/kv_offload_test.json"
        cd /tmp && timeout 120 vllm bench throughput \
            --model "$MODEL" \
            --dtype "$DTYPE" \
            --gpu-memory-utilization 0.80 \
            --max-num-seqs 128 \
            --num-prompts 64 \
            --dataset-name random \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --output-json "$JSON_OUT" \
            $FLAG 2>&1 | tail -5 || echo "  Flag not supported or failed"
    done
else
    echo "No KV offload CLI flags found in vllm bench throughput --help."
    echo "KV offloading may need to be configured programmatically."
    echo "Skipping Part 2."
    echo ""
    echo "To test manually, check:"
    echo "  - vllm/v1/kv_offload/ for configuration options"
    echo "  - vllm serve --help | grep -i offload"
fi

# ---- Part 3: Weight offload impact on KV cache (re-test on V1) ----
# The V0 findings showed catastrophic slowdown. Re-test on V1.

echo ""
echo "================================================================"
echo "  Phase 0.7c: Weight Offload → KV Cache Impact (V1 re-test)"
echo "================================================================"

OFFLOAD_CONFIGS=("28 1" "14 1" "7 1" "4 1")  # group_size num_in_group

OUTFILE2="${RESULTS_DIR}/weight_offload_kv_impact.csv"
echo -e "group_size\tnum_in_group\tlayers_offloaded\ttokens_per_sec\telapsed_time" > "$OUTFILE2"

for CONFIG in "${OFFLOAD_CONFIGS[@]}"; do
    read -r GS NIG <<< "$CONFIG"
    LAYERS_OFF=$((28 / GS * NIG))

    echo ""
    echo "--- group_size=$GS, num_in_group=$NIG → ${LAYERS_OFF}/28 layers offloaded ---"

    JSON_OUT="${RESULTS_DIR}/weight_offload_G${GS}_M${NIG}.json"

    cd /tmp && timeout 300 vllm bench throughput \
        --model "$MODEL" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization 0.80 \
        --max-num-seqs 128 \
        --num-prompts 64 \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --offload-group-size "$GS" \
        --offload-num-in-group "$NIG" \
        --output-json "$JSON_OUT" 2>&1 | tail -3 || echo "  FAILED or TIMED OUT"

    if [ -f "$JSON_OUT" ]; then
        TOK=$(python3 -c "import json; d=json.load(open('$JSON_OUT')); print(f\"{d['tokens_per_second']:.1f}\")" 2>/dev/null || echo "N/A")
        TIME=$(python3 -c "import json; d=json.load(open('$JSON_OUT')); print(f\"{d['elapsed_time']:.2f}\")" 2>/dev/null || echo "N/A")
    else
        TOK="FAIL"; TIME="FAIL"
    fi

    echo -e "${GS}\t${NIG}\t${LAYERS_OFF}\t${TOK}\t${TIME}" >> "$OUTFILE2"
    echo "  tokens/s=$TOK  elapsed=${TIME}s"
done

echo ""
echo "================================================================"
echo "Results:"
echo "  KV pressure:     $OUTFILE"
echo "  Weight offload:  $OUTFILE2"
echo "================================================================"
