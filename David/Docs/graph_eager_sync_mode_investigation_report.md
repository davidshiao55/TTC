# Graph/Eager Sync Mode Investigation Report

Date: 2026-06-02

Status: handoff note. This was discovered while verifying the structural
nonuniform COTS graph-routing fix. Forced logprob parity is intentionally
deferred until this graph/eager performance question is understood.

## Executive Summary

The original graph-mode nonuniform dispatch bug is fixed: decode-only graph
routes now execute native COTS CPU work after CUDA graph capture, and the
uniform dispatch validation cells did not show a meaningful regression.

However, a larger issue appeared during the verification pass. In the current
planner validation cell (`B=64`, `f_cpu_store=0.30`, decode-only CPU route,
short decode), piecewise graph mode is slower than eager mode for COTS routes.
That contradicts the expected purpose of graph replay: graph should reduce
launch/Python overhead and should be the production path once COTS custom-op
submit/sync boundaries are graph compatible.

The purpose of `weight_capture_sync_mode="wait_kernel"` is also unclear under
the current piecewise graph design. It was previously useful in the full-graph
COTS capture gap experiments, but in this new short decode cell, host callback
synchronization is faster than wait-kernel synchronization in graph mode. Eager
mode with wait-kernel synchronization is also slower than normal eager
host-callback synchronization.

This should be treated as a separate performance investigation before returning
to strict output/logprob parity.

## Current Evidence

### Structural route fix is active

The fixed graph run used:

```text
/TTC/results/planner/bug_fix_verify_20260602/graph_nonuniform_static_piece_fix
```

For `B=64`, `f_cpu_store=0.30`, decode-only pure CPU route:

```text
submit_count_qkv = 84
submit_count_mlp = 84
worker_run_count = 168
```

The counters were reset after CUDA graph capture, so these counts are measured
replay-time work, not capture-time artifacts. The run also compiled separate
COTS route signatures, including the nonuniform CPU route.

### Uniform dispatch did not regress

The mini-sweep compared the new code against the previously recorded uniform
15% sweep at:

```text
old: /TTC/results/planner/dispatch_model_validation/20260601T150936Z
new: /TTC/results/planner/verify_route_fix_20260602
```

| cell | old latency | new latency | delta |
|---|---:|---:|---:|
| B16 none | 0.5376 s | 0.5415 s | +0.73% |
| B16 prefetch | 2.8412 s | 2.7932 s | -1.69% |
| B16 hybrid50 | 2.9555 s | 2.9597 s | +0.14% |
| B16 CPU | 2.6070 s | 2.6286 s | +0.83% |
| B64 none | 0.6281 s | 0.6324 s | +0.69% |
| B64 prefetch | 2.8381 s | 2.9092 s | +2.50% |
| B64 hybrid50 | 5.7345 s | 5.7306 s | -0.07% |
| B64 CPU | 8.0724 s | 8.1762 s | +1.29% |

This is small enough that the graph-routing changes are not the obvious cause
of the new graph/eager concern.

### Sync-mode experiment

All cells used:

```text
model: Qwen/Qwen2.5-7B-Instruct
mode: eager or graph
dispatch layout: decode-only
batch: 64
f_cpu_store: 0.30
output_len: 4
num_iters_warmup: 0
num_iters: 1
thread policy: scalar
cots_cpu_num_threads: 24
```

Result root:

```text
/TTC/results/planner/sync_mode_experiment_20260602
```

| mode | sync mode | none | pure prefetch | pure CPU decode |
|---|---|---:|---:|---:|
| eager | host callback | 0.173344 s | 0.791364 s | 1.697724 s |
| eager | wait kernel | 0.149741 s | 0.790401 s | 1.758447 s |
| graph piecewise | host callback | 0.142263 s | 1.023624 s | 1.768793 s |
| graph piecewise | wait kernel | 0.132464 s | 1.015946 s | 1.885605 s |

Interpretation:

- Graph is faster than eager for the no-offload baseline.
- Graph is slower than eager for pure prefetch in this cell.
- Graph is slower than eager for the decode-only pure CPU route in this cell.
- Graph host callback is faster than graph wait kernel for the pure CPU route.
- Eager wait kernel is slower than normal eager host callback for the pure CPU
  route, so switching eager to wait kernel for consistency likely hurts.

CPU-route counters matched across the sync-mode cells:

```text
submit_count_qkv = 84
submit_count_mlp = 84
worker_run_count = 168
```

That makes the graph/eager comparison a real runtime question, not the earlier
"CPU work did not execute" bug.

## Why This Matters

The thesis graph story is supposed to be:

```text
eager path: correctness and A/B fallback
piecewise graph path: production path with lower replay overhead
wait kernel: graph-safe synchronization point for native COTS CPU work
```

The current short-decode validation cell does not support that story. If graph
only wins in a narrow B=1 long-decode regime, but loses in planner-relevant
larger batch decode buckets, the Planner cannot treat graph mode as a free
runtime improvement. The dispatch profiler also needs to know which runtime
mode it is measuring.

## Previous Graph Win

The older Phase 1c capture-gap gate was a different regime:

```text
/TTC/results/regression_phase1_phase2_20260530/phase1c_capture_gap/summary.json
```

Configuration:

```text
B = 1
output_len = 128
f_cpu_store = 0.05
cpu_threads = 16
```

Recorded result:

```text
native_eager_real mean = 2.442040 s
piecewise_cots_split_wait_kernel_real mean = 2.288399 s
graph advantage ~= 153.6 ms
```

So the new result does not prove graph is always worse. It shows the old graph
win is not enough evidence for the current planner validation regime.

## Working Hypotheses

1. Piecewise graph boundary/replay overhead may dominate in short decode cells.
   The old B=1, `output_len=128` win may have amortized this cost.
2. Wait-kernel launch/spin overhead may not be hidden under piecewise graph
   replay. Host callback synchronization may be cheaper for these cells.
3. Eager and graph may be operating at different padded capacities or live-token
   caps. Counter logs showed matching submit counts, but live-token values and
   recorded bytes should still be audited.
4. Pure prefetch graph losing to eager suggests this is not only CPU runner
   synchronization. The graph/piecewise path may serialize or wait around H2D
   prefetch differently.
5. Route-signature compilation is structurally correct but may create more
   graph fragments or replay transitions than the original uniform/full-graph
   path.

## Recommended Next Experiments

1. Re-run the exact old Phase 1c capture-gap gate after the route-signature
   changes:

```text
B=1, output_len=128, f_cpu_store=0.05, cpu_threads=16
native eager vs piecewise wait-kernel vs piecewise host-callback
```

2. Repeat the sync-mode matrix with at least three repeats across:

```text
B in {1, 16, 64}
output_len in {4, 16, 32, 128}
f_cpu_store in {0.05, 0.15, 0.30}
routes: none, pure prefetch, pure CPU
```

3. Use Nsight Systems on graph host-callback vs graph wait-kernel for the
   `B=64`, `f=0.30` pure CPU route. Inspect:

```text
CUDA graph replay segments
cots_submit_gemm / cots_sync_then_uva placement
wait-kernel duration and stream occupancy
host callback scheduling delay
H2D and UVA memcpy timing
GPU idle gaps around piecewise boundaries
```

4. Add a dry-run COTS route to separate graph/piecewise overhead from real CPU
   and prefetch work.

5. Keep counter reset after CUDA graph capture enabled for graph measurements:

```text
VLLM_COTS_COUNTERS=1
VLLM_COTS_DUMP_COUNTERS_ON_SHUTDOWN=1
VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE=1
```

6. Return to forced-context/logprob parity only after this graph/eager runtime
   issue is classified. Free-generation token parity has already proven too
   brittle as a correctness oracle for these COTS route comparisons.

## Reproduction Commands

Graph wait-kernel default:

```bash
TTC_DOCKER_WORKDIR=/TTC/FastTTS-thesis scripts/ttc-docker-env.sh thesis \
  'VLLM_CACHE_ROOT=/tmp/ttc-sync-mode-graph-wait \
   VLLM_COTS_COUNTERS=1 \
   VLLM_COTS_DUMP_COUNTERS_ON_SHUTDOWN=1 \
   VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE=1 \
   python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
   --exp --force \
   --results-dir /TTC/results/planner/sync_mode_experiment_20260602/graph_piecewise_wait_decode_cpu_b64_f030 \
   --modes graph --dispatch-layout decode-only --batches 64 \
   --f-cpu-store-values 0.30 --f-cpu-ratios 1 \
   --output-len 4 --num-iters-warmup 0 --num-iters 1 \
   --thread-policy scalar --cots-cpu-num-threads 24 \
   --cell-timeout-s 900 --keep-going'
```

Graph host callback used an explicit splitting-ops override and:

```text
--no-cots-auto-graph-split
--cots-weight-capture-sync-mode host_callback
```

Eager host callback:

```bash
TTC_DOCKER_WORKDIR=/TTC/FastTTS-thesis scripts/ttc-docker-env.sh thesis \
  'VLLM_CACHE_ROOT=/tmp/ttc-sync-mode-eager-host \
   VLLM_COTS_COUNTERS=1 \
   VLLM_COTS_DUMP_COUNTERS_ON_SHUTDOWN=1 \
   python /TTC/David/Benchmarks/planner/bench_dispatch_model_validation.py \
   --exp --force \
   --results-dir /TTC/results/planner/sync_mode_experiment_20260602/eager_host_decode_cpu_b64_f030 \
   --modes eager --dispatch-layout decode-only --batches 64 \
   --f-cpu-store-values 0.30 --f-cpu-ratios 1 \
   --output-len 4 --num-iters-warmup 0 --num-iters 1 \
   --thread-policy scalar --cots-cpu-num-threads 24 \
   --cell-timeout-s 900 --keep-going'
```

Eager wait-kernel was run only as a temporary diagnostic by relaxing the
configuration guard. That guard was removed after the experiment; do not keep
wait-kernel eager mode enabled by default.
