# Dispatch Validation Surprise Report

Date: 2026-06-01

Status: Planner implementation is paused. This report is a handoff note for a
separate investigation of COTS weight-offloading performance.

Update: the follow-up investigation in
`dispatch_validation_investigation_report.md` found that the graph-mode
decode-only pure-CPU endpoint in this report is invalid. Native COTS CPU custom
ops were absent from the compiled graph for the nonuniform dispatch table, so
the reported "pure CPU decode" latency did not execute CPU work. Keep this file
as the historical surprise/handoff note, not as Planner ground truth.

## Executive Summary

The initial dispatch-validation runs suggested that hybrid CPU plus prefetch
dispatch rarely beats endpoint policies, and that large buckets should prefer
pure prefetch. That conclusion was misleading because the validation harness
used the same dispatch split for every captured bucket. In vLLM latency
benchmarks, this means prefill and decode both used the same CPU/prefetch split.
Prefill is much more compute-dense, so CPU compute in prefill can dominate the
measured end-to-end latency and hide the decode-side benefit of CPU compute.

After adding a `decode-only` validation layout, the result changed
substantially:

- non-decode buckets were fixed to pure prefetch;
- only the measured decode bucket varied across `(f_cpu, f_prefetch)`;
- decode CPU compute became strongly beneficial;
- pure CPU decode beat all hybrid splits through `f_cpu_store = 0.40` for
  Qwen2.5-7B-Instruct, B=16 and B=64, graph mode.

This is surprising because the earlier mental model predicted hybrid should
eventually beat pure CPU by balancing CPU compute against H2D prefetch. In the
measured decode-only workload, that crossover did not appear up to 40% CPU
storage. Any nonzero decode prefetch remained slower than assigning those bytes
to CPU compute.

## Code Checkpoint

Relevant implementation files:

- `FastTTS-thesis/planner.py`
  - Adds `DispatchProblem`, `DispatchEntry`, `DispatchSolveResult`, and
    `solve_per_bucket_dispatch`.
- `David/Tests/fasttts/test_planner.py`
  - Adds synthetic profile tests for the per-bucket dispatch solver.
- `David/Benchmarks/planner/bench_dispatch_model_validation.py`
  - Runtime validation harness for forced COTS dispatch tables.
  - Supports `--dispatch-layout uniform` and `--dispatch-layout decode-only`.
  - Supports `--cell-timeout-s`, summary-on-interrupt, and failed-cell reporting.
- `David/Docs/planner_design.md`
  - Updated with the validation caveat: bucket is a shape proxy, not a semantic
    prefill/decode phase label.

## Harness Behavior

The harness calls:

```bash
python -m vllm.entrypoints.cli.main bench latency ...
```

For each cell it sets:

```text
--offload-backend cots
--cots-f-cpu-store <fixed f_cpu_store>
--cots-f-prefetch 0.0
--cots-dispatch-table <forced table>
```

For a candidate split:

```text
f_prefetch = f_cpu_store - f_cpu
```

### Uniform Layout

`--dispatch-layout uniform` assigns the candidate pair to every captured
bucket:

```text
dispatch[bucket] = (f_cpu, f_prefetch) for all buckets
```

This is useful for testing the runtime, but it mixes prefill and decode effects.

### Decode-Only Layout

`--dispatch-layout decode-only` assigns pure prefetch to every bucket, then
overrides only the benchmark batch's decode bucket:

```text
dispatch[bucket] = (0.0, f_cpu_store) for all buckets
dispatch[bucket_for(batch)] = (f_cpu, f_prefetch)
```

For the current latency benchmark with `input_len=8`, prefill maps to a larger
bucket than decode. For example, B=16 decode maps to bucket 16, while prefill
maps roughly to bucket 128. This is an approximation, not a true semantic phase
key, but it isolates the effect we wanted in this benchmark.

## Key Results

All runs below used:

```text
model: Qwen/Qwen2.5-7B-Instruct
dtype: bfloat16
mode: graph
input_len: 8
output_len: 32
gpu_memory_utilization: 0.75
num_iters_warmup: 1
num_iters: 2
```

### Uniform 15% Sweep

Result path:

```text
/TTC/results/planner/dispatch_model_validation/20260601T150936Z/summary.md
```

Uniform dispatch suggested CPU was bad at large batch:

| B | pure prefetch | best hybrid | pure CPU | best |
|---:|---:|---:|---:|---|
| 16 | 2.8412 s | 2.6564 s | 2.6070 s | pure CPU |
| 64 | 2.8381 s | 4.1033 s | 8.0724 s | pure prefetch |

This result is now considered phase-contaminated. The B=64 pure CPU cell forces
CPU compute in prefill and decode, so it is not a clean measurement of decode
CPU viability.

### Decode-Only 15% Sweep

Command:

```bash
TTC_DOCKER_WORKDIR=/TTC/FastTTS-thesis scripts/ttc-docker-env.sh thesis \
  'python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
    --exp --dispatch-layout decode-only --modes graph \
    --batches 16 64 \
    --f-cpu-store-values 0.15 \
    --f-cpu-ratios 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 \
    --output-len 32 --num-iters-warmup 1 --num-iters 2 \
    --repeat 1 --cell-timeout-s 300 --keep-going'
```

Result path:

```text
/TTC/results/planner/dispatch_model_validation/20260601T161515Z/summary.md
```

| B | pure prefetch | 50/50 hybrid | near-CPU hybrid | pure CPU decode | best |
|---:|---:|---:|---:|---:|---|
| 16 | 2.8168 s | 1.4892 s | 0.7477 s | 0.7288 s | pure CPU |
| 64 | 2.7982 s | 1.5584 s | 0.8364 s | 0.7921 s | pure CPU |

Interpretation:

- CPU compute in decode is strongly beneficial once prefill is kept off CPU.
- Hybrid improves greatly over pure prefetch.
- Pure CPU decode still wins at 15%.

### Dry-Run Control at 15%

Command:

```bash
TTC_DOCKER_WORKDIR=/TTC/FastTTS-thesis scripts/ttc-docker-env.sh thesis \
  'python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
    --exp --dispatch-layout decode-only --modes graph \
    --batches 16 64 \
    --f-cpu-store-values 0.15 \
    --f-cpu-ratios 0.875 1 \
    --output-len 32 --num-iters-warmup 1 --num-iters 2 \
    --repeat 1 --cell-timeout-s 300 --keep-going \
    --extra-vllm-args --cots-dry-run'
```

Result path:

```text
/TTC/results/planner/dispatch_model_validation/20260601T163920Z/summary.md
```

Near-CPU hybrid and pure CPU became essentially identical in dry-run:

| B | near-CPU hybrid dry-run | pure CPU dry-run | dry-run gap |
|---:|---:|---:|---:|
| 16 | 0.4818 s | 0.4818 s | ~0.08 ms |
| 64 | 0.5559 s | 0.5533 s | ~2.6 ms |

In the real run, near-CPU hybrid was slower than pure CPU by:

| B | real near-CPU hybrid | real pure CPU | real gap |
|---:|---:|---:|---:|
| 16 | 0.7477 s | 0.7288 s | +18.9 ms |
| 64 | 0.8364 s | 0.7921 s | +44.3 ms |

Interpretation:

- The hybrid penalty is not dispatch-table lookup or graph/control overhead.
- The gap comes from real prefetch work: H2D, prefetch wait, and prefetched-slice
  GPU GEMM/addmm.

### Decode-Only Crossover Sweep

Command:

```bash
TTC_DOCKER_WORKDIR=/TTC/FastTTS-thesis scripts/ttc-docker-env.sh thesis \
  'python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
    --exp --dispatch-layout decode-only --modes graph \
    --batches 16 64 \
    --f-cpu-store-values 0.25 0.30 0.40 \
    --f-cpu-ratios 0 0.25 0.5 0.75 0.875 1 \
    --output-len 32 --num-iters-warmup 1 --num-iters 2 \
    --repeat 1 --cell-timeout-s 300 --keep-going'
```

Result path:

```text
/TTC/results/planner/dispatch_model_validation/20260601T170208Z/summary.md
```

Pure CPU decode won through 40% storage:

| B | `f_cpu_store` | pure prefetch | best hybrid | pure CPU decode | best |
|---:|---:|---:|---:|---:|---|
| 16 | 0.25 | 4.6601 s | 1.0027 s | 0.8751 s | pure CPU |
| 16 | 0.30 | 5.5098 s | 1.1521 s | 0.9181 s | pure CPU |
| 16 | 0.40 | 7.2146 s | 1.4892 s | 1.2517 s | pure CPU |
| 64 | 0.25 | 4.6613 s | 1.1382 s | 0.9285 s | pure CPU |
| 64 | 0.30 | 5.5381 s | 1.2236 s | 1.0127 s | pure CPU |
| 64 | 0.40 | 7.3241 s | 1.5847 s | 1.1869 s | pure CPU |

Best hybrid was always near the CPU endpoint, but still slower than pure CPU:

| B | `f_cpu_store` | best hybrid split | hybrid vs pure CPU |
|---:|---:|---|---:|
| 16 | 0.25 | 0.21875 / 0.03125 | +14.6% |
| 16 | 0.30 | 0.26250 / 0.03750 | +25.5% |
| 16 | 0.40 | 0.35000 / 0.05000 | +19.0% |
| 64 | 0.25 | 0.21875 / 0.03125 | +22.6% |
| 64 | 0.30 | 0.26250 / 0.03750 | +20.8% |
| 64 | 0.40 | 0.35000 / 0.05000 | +33.5% |

## Current Interpretation

The old simple model was:

```text
penalty ~= max(f_cpu, f_cpu_store - f_cpu)
```

This predicts an interior hybrid optimum when CPU compute and prefetch are both
useful resources. The decode-only data does not support that model in the
measured regime. Instead, the measured curve is monotonic toward more CPU:

```text
more f_cpu, less f_prefetch -> lower latency
```

This suggests:

1. Decode CPU compute is still largely hidden or at least cheaper than prefetch
   through `f_cpu_store=0.40`.
2. Layer-ahead prefetch has a large real cost in this COTS path.
3. The cost of even a small residual prefetch slice is larger than the CPU
   compute it replaces in this decode workload.
4. Uniform prefill+decode validation can invert conclusions because prefill
   CPU compute is much more expensive than decode CPU compute.

## Runtime Paths To Inspect

Relevant vLLM files:

- `vllm/model_executor/offloader/cots_offloader.py`
  - `_hook_layer_forward`: emits `wait_prefetch` and `start_prefetch` around
    layer forward calls.
  - `on_dispatch`: sets active bucket, syncs copy stream, and publishes live
    token count to native COTS.
- `vllm/model_executor/offloader/cots_storage.py`
  - `WeightPrefetchStreamer.start`: issues layer H2D copies for handles whose
    active bucket has `n_prefetch > 0`.
  - `WeightPrefetchStreamer.wait`: compute stream waits for copy completion.
- `vllm/model_executor/offloader/cots_operators.py`
  - output-split QKV path runs GPU permanent GEMM, optional prefetched-slice
    GEMM, optional CPU submit/wait/UVA.
  - fused MLP path similarly runs optional prefetched MLP block plus optional
    CPU block.
- `vllm/model_executor/offloader/cots_ops.py`
  - native custom ops for submit, wait, and UVA copy.

Important detail: in `decode-only` mode, the prefetch machinery is still
installed globally because prefill buckets use pure prefetch. Pure CPU decode
does not remove the hooks; it makes the active decode bucket's `n_prefetch = 0`,
so per-layer H2D and prefetched-slice GEMMs are skipped for decode. This makes
near-CPU hybrid vs pure CPU a relatively clean comparison of actual residual
prefetch work.

## Open Questions For The Weight-Offloading Investigation

1. Why is decode prefetch so expensive relative to CPU compute?

   The pure-prefetch decode-only latency at 15% is ~2.8 s, while pure CPU decode
   is ~0.7 to 0.8 s. At 40%, pure prefetch grows to ~7.2 s, while pure CPU is
   ~1.2 s. This is a large gap.

2. Is prefetch H2D actually overlapping with compute?

   We need Nsight Systems traces with NVTX for:

   - pure prefetch decode bucket,
   - near-CPU hybrid,
   - pure CPU decode,
   - maybe dry-run control.

   Look for unhidden H2D, `wait_prefetch` stalls, copy stream ordering, and
   layer-boundary synchronization.

3. Is the layer-ahead prefetch schedule doing too much work per decode step?

   The prefetch path may be effectively serializing around every layer, or
   suffering from layer-0 repair / wraparound / copy-stream drain effects.

4. Is the prefetched-slice GPU GEMM inefficient at small irregular slices?

   Hybrid pays extra small GEMMs/addmms for the prefetched slice. If these are
   launch-bound, poorly shaped, or not fused well, they can dominate.

5. Is the CPU path unexpectedly good because it avoids a GPU-side bottleneck?

   CPU compute returns via pinned output/UVA and may overlap with GPU permanent
   GEMMs. The native runner plus thread policy might be good enough that CPU is
   not the bottleneck for decode even at 40%.

6. Are we measuring a latency benchmark artifact?

   The benchmark uses input/output 8/32 and averages only two iterations in the
   validation sweeps. Repeat selected cells with more iterations and possibly
   longer output lengths before promoting numbers to final thesis claims.

7. Does the result hold for other models and modules?

   Qwen2.5-7B is the current model. Llama 8B or larger Qwen generators may have
   different KV pressure, layer shapes, and GPU/CPU balance.

## Suggested Next Experiments

Run these from `/TTC/FastTTS-thesis` inside the thesis Docker environment.

### 1. Repeat High-Signal Cells With More Iterations

```bash
python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
  --exp --dispatch-layout decode-only --modes graph \
  --batches 16 64 \
  --f-cpu-store-values 0.15 0.30 0.40 \
  --f-cpu-ratios 0 0.5 0.875 1 \
  --output-len 128 \
  --num-iters-warmup 2 --num-iters 5 \
  --repeat 2 \
  --cell-timeout-s 600 --keep-going
```

### 2. Nsight Trace Pure Prefetch vs Pure CPU Decode

Start with B=64 and `f_cpu_store=0.30`.

Trace cells:

- pure prefetch: `(f_cpu=0.0, f_prefetch=0.30)`
- midpoint: `(f_cpu=0.15, f_prefetch=0.15)`
- near CPU: `(f_cpu=0.2625, f_prefetch=0.0375)`
- pure CPU: `(f_cpu=0.30, f_prefetch=0.0)`

Questions:

- How much H2D is on the critical path?
- Does `wait_prefetch` stall every layer?
- Are prefetched-slice GEMMs launch-bound?
- Does CPU work finish before the GPU permanent path?

### 3. Module Subset Isolation

Use `--cots-weight-modules` to isolate:

```text
qkv only
mlp only
qkv mlp
```

The suspected outcomes:

- QKV may have K/V placement interactions.
- MLP may dominate prefetch volume.
- If one module group causes the prefetch tax, the Planner may need
  module-group dispatch sooner than expected.

### 4. Eager Mode Control

Run a small decode-only sweep in eager mode to see whether graph capture or
piecewise graph boundaries contribute to the prefetch tax. Eager mode exposes
different buckets and uses scalar thread fallback, so compare qualitatively.

### 5. Longer Decode

Increase `output_len` to 128 or 256. If the effect is mostly initial
prefill/graph/first-token overhead, the per-token average may change. If it is
per-layer/per-token prefetch cost, the ordering should persist.

## Planner Implications For Now

Do not finalize the Planner's performance model until the weight-offloading
investigation explains the prefetch cost. For the current checkpoint:

- Keep the per-bucket solver as a reference abstraction.
- Keep `decode-only` validation harness support.
- Treat prefill-heavy and decode buckets separately.
- Do not assume hybrid should beat pure CPU in decode.
- Do not use the old uniform free-zone analysis as evidence about decode CPU
  free zone.

The next planner pass should consume whatever the weight-offloading
investigation finds as calibrated profile terms, likely including:

```text
T_h2d_effective(bucket, f_prefetch, phase_class)
T_prefetch_gpu_gemm(bucket, f_prefetch, module_group)
T_cpu_effective(bucket, f_cpu, phase_class)
O_wait_prefetch(bucket, f_prefetch)
O_sync_uva(bucket, f_cpu)
```

The surprising empirical fact to explain is simple:

```text
For Qwen2.5-7B decode-only validation, pure CPU decode is faster than any
hybrid split through f_cpu_store = 0.40 at B=16 and B=64.
```
