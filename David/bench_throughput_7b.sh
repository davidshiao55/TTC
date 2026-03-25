#!/bin/bash
# Batch size vs throughput sweeps for vLLM — 7B model
# Usage: conda activate vllm && bash David/bench_throughput_7b.sh

set -e

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
NUM_PROMPTS=1024
INPUT_LEN=512
OUTPUT_LEN=128
DTYPE="bfloat16"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_7b"
mkdir -p "$RESULTS_DIR"

# CSV headers
echo "gpu_memory_utilization,max_num_seqs,elapsed_time,num_requests,requests_per_second,tokens_per_second,status" \
    > "$RESULTS_DIR/sweep_max_num_seqs.csv"
echo "gpu_memory_utilization,max_num_seqs,elapsed_time,num_requests,requests_per_second,tokens_per_second,status" \
    > "$RESULTS_DIR/sweep_gpu_mem_util.csv"

run_bench() {
    local max_seqs=$1
    local gpu_mem=$2
    local tag=$3
    local csv_file=$4
    local json_path="${RESULTS_DIR}/${tag}.json"

    echo ""
    echo "============================================================"
    echo "Running: max-num-seqs=${max_seqs}  gpu-memory-utilization=${gpu_mem}"
    echo "============================================================"
    echo ""

    if vllm bench throughput \
        --model "$MODEL" \
        --max-num-seqs "$max_seqs" \
        --num-prompts "$NUM_PROMPTS" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --dtype "$DTYPE" \
        --gpu-memory-utilization "$gpu_mem" \
        --output-json "$json_path"; then

        # Parse JSON and append to CSV
        python3 -c "
import json
with open('${json_path}') as f:
    d = json.load(f)
print(f'${gpu_mem},${max_seqs},{d[\"elapsed_time\"]},{d[\"num_requests\"]},{d[\"requests_per_second\"]},{d[\"tokens_per_second\"]},OK')
" >> "$csv_file"

        echo "  -> OK"
    else
        echo "${gpu_mem},${max_seqs},,,,,FAILED" >> "$csv_file"
        echo "  -> FAILED"
    fi
}

# ── Sweep 1: max-num-seqs (gpu-memory-utilization fixed at 0.9) ──
echo ""
echo "############################################################"
echo "# SWEEP 1: max-num-seqs  (gpu-memory-utilization fixed at 0.9)"
echo "############################################################"

for SEQS in 1 2 4 8 16 32 64 128 256; do
    run_bench "$SEQS" 0.9 "sweep_seqs_${SEQS}" "$RESULTS_DIR/sweep_max_num_seqs.csv"
done

# ── Sweep 2: gpu-memory-utilization (max-num-seqs fixed at 256) ──
echo ""
echo "############################################################"
echo "# SWEEP 2: gpu-memory-utilization  (max-num-seqs fixed at 256)"
echo "############################################################"

for MEM in 0.5 0.6 0.7 0.8 0.9; do
    run_bench 256 "$MEM" "sweep_mem_${MEM}" "$RESULTS_DIR/sweep_gpu_mem_util.csv"
done

echo ""
echo "============================================================"
echo "All done. Results in ${RESULTS_DIR}/"
echo "  sweep_max_num_seqs.csv"
echo "  sweep_gpu_mem_util.csv"
echo "============================================================"
