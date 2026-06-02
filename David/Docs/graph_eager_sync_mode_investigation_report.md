# Final Report: Graph/Eager COTS Sync-Mode Investigation

Date: 2026-06-02

Status: final session report. The full-graph zero-work bug is fixed, the
piecewise graph COTS fast path was optimized, the current host/wait/full/eager
matrix was rerun, and the default policy decision is recorded below.

## Final Answer

The production COTS weight-offload default should be:

- **Graph mode:** piecewise graph with all COTS split points enabled and
  `weight_capture_sync_mode="wait_kernel"`.
- **Eager mode:** `weight_capture_sync_mode="host_callback"`.
- **Full graph:** restored as a valid A/B experiment, but **not** the default.
- **No per-bucket sync matrix by default.** Keep one graph fast path. Add a
  future Planner/Profiler fallback only if a measured workload shows a material
  eager win outside noise.

The strict slogan "graph capture always beats eager" is still too strong as a
mathematical claim. The practical status is better:

- B=64 pure-prefetch graph/eager is restored to parity.
- B=1 long-decode CPU graph still clearly beats eager.
- B=64 short-decode CPU no-counter production runs are within-noise parity, not
  a proven eager win.
- Full graph is now counter-valid for native COTS CPU routes, but valid full
  graph is slower than piecewise in the tested B=64 and B=1 regimes.

So the session resolves the original blockers without adding default
overengineering: use piecewise graph + wait-kernel as the graph fast path,
host-callback for eager, and keep full graph/profile-gated eager as explicit
diagnostic or future policy tools.

## What Was Fixed

### 1. Full Graph Now Executes Native COTS CPU Work

Old symptom:

- A B=64 full-graph CPU run reported a very fast `0.5322 s` latency.
- Replay counters after CUDA graph capture were all zero:

```text
submit_count_qkv = 0
submit_count_mlp = 0
dispatch_cb_count = 0
sync_cb_count = 0
worker_run_count = 0
```

Root cause:

- `--no-cots-auto-graph-split` allowed full graph with AOT compile.
- The AOT artifact was not keyed by `BatchDescriptor.cots_route_signature`.
- The generated full-graph artifact contained prefetch hooks but not
  `cots_submit_gemm` / `cots_sync_then_uva`.
- Replay therefore looked fast because the CPU submit/sync route had been
  branch-pruned out of the captured artifact.

Fix:

- Native COTS weight graph disables AOT compile for CUDA graph capture, both
  piecewise and full graph.
- Full-graph capture preserves the decorated `BatchDescriptor` even when the
  inner model run uses runtime graph mode `NONE`.
- Route-specialized compile variants are used for native COTS CPU routes.

Validation:

- B=64 route-specialized full graph produced nonzero replay work after
  post-capture reset.
- Host-callback full graph:

```text
dispatch_cb_count = 168
sync_cb_count = 168
worker_run_count = 168
d2h_replay_bucket_bytes = 77070336
uva_replay_bucket_bytes = 53673984
```

- Wait-kernel full graph:

```text
dispatch_cb_count = 168
sync_cb_count = 0
worker_run_count = 168
d2h_replay_bucket_bytes = 77070336
uva_replay_bucket_bytes = 53673984
```

For full graph, `submit_count_qkv` / `submit_count_mlp` remain zero during
replay because the captured CUDA graph replays recorded D2H and host-function
nodes rather than calling the Python/C++ submit API again. The validity
counters are the callback, worker, and replay-byte counters.

### 2. Piecewise Graph Prefetch Serialization Was Removed

Old symptom:

- B=64 pure-prefetch graph was slower than eager in the first sync experiment.
- This indicated the problem was not only CPU submit/sync overhead.

Fix:

- `CompilationConfig.cots_splitting_ops()` now includes:

```text
vllm::wait_prefetch
vllm::start_prefetch
vllm::cots_submit_gemm
vllm::cots_sync_then_uva
```

- CUDAGraph wrapper options can skip wrapper-level offloader stream
  `sync_prev_onload()` / `join_after_forward()`.
- That skip is guarded narrowly: it only applies to an active native COTS
  weight graph with all four COTS split ops present.

Result:

- B=64 pure-prefetch graph/eager returned to parity.
- The original B=1 long-decode wait-kernel graph win was preserved.

### 3. Reproducible Matrix Runner Added

New runner:

```text
/TTC/David/Benchmarks/planner/bench_sync_mode_matrix.py
```

It crosses:

- eager host
- eager wait
- piecewise wait
- piecewise host
- full host
- full wait

against:

- B=64 short decode, decode-only, `f_cpu_store=0.30`, `output_len=4`,
  `cpu_threads=24`
- B=1 long decode, uniform, `f_cpu_store=0.05`, `output_len=128`,
  `cpu_threads=16`

The runner intentionally enables COTS counters and post-capture counter reset:

```text
VLLM_COTS_COUNTERS=1
VLLM_COTS_DUMP_COUNTERS_ON_SHUTDOWN=1
VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE=1
```

## Current Measurements

### Current Host/Wait/Full/Eager Matrix

Result root:

```text
/TTC/results/planner/sync_mode_matrix_20260602/current_host_wait_full_piecewise_eager
```

CPU-route latencies:

| scenario | eager host | eager wait | piecewise host | piecewise wait | full host | full wait |
|---|---:|---:|---:|---:|---:|---:|
| B64 short decode CPU | 1.6567 s | invalid | 1.8218 s | 1.7840 s | 1.8916 s | 1.9539 s |
| B1 long decode CPU | 2.4536 s | invalid | 2.4307 s | 2.3021 s | 2.6197 s | 2.4862 s |

Pure-prefetch latencies:

| scenario | eager host | eager wait | piecewise host | piecewise wait | full host | full wait |
|---|---:|---:|---:|---:|---:|---:|
| B64 short decode prefetch | 0.7772 s | 0.7772 s | 0.7770 s | 0.7770 s | 0.8758 s | 0.8916 s |
| B1 long decode prefetch | 3.5076 s | 3.5066 s | 3.4994 s | 3.5026 s | 3.9070 s | 3.8974 s |

Important interpretation:

- `eager + wait_kernel` is invalid by design for CPU routes:
  `wait_kernel` requires graph capture because it replaces a captured graph
  sync node.
- Piecewise wait-kernel beats piecewise host-callback for both tested CPU
  cells in this current rerun.
- Full graph is slower than piecewise for all tested COTS prefetch/CPU rows.
- The B=64 full-host no-offload baseline timed out once before engine startup,
  but the COTS prefetch and CPU rows for that same arm completed.

### B=64 CPU No-Counter Diagnostics

The matrix above is counter-enabled and only three measured iterations, so it
is not the production source of truth for small B=64 CPU differences.

No-counter diagnostic roots:

```text
/TTC/results/planner/b64_piecewise_gap_diagnostics_20260602/no_counters
/TTC/results/planner/b64_piecewise_gap_diagnostics_20260602/no_counters_iter21
```

First no-counter pass, seven measured iterations:

| mode | pure CPU decode | pure prefetch |
|---|---:|---:|
| eager host | 1.8162 s | 0.7774 s |
| piecewise wait | 1.7751 s | 0.7773 s |

Longer no-counter pass, twenty-one measured iterations:

| mode | pure CPU decode mean | stdev | min | median | max |
|---|---:|---:|---:|---:|---:|
| eager host | 1.7749 s | 0.1151 s | 1.6563 s | 1.7440 s | 2.0203 s |
| piecewise wait | 1.7906 s | 0.0932 s | 1.6483 s | 1.7692 s | 2.0001 s |

Welch comparison for the twenty-one-iteration CPU rows:

```text
graph - eager = +15.7 ms (+0.89%)
standard error = 32.3 ms
approx 95% CI = [-49.6 ms, +81.0 ms]
```

Pure prefetch remained stable and at parity:

| mode | pure prefetch mean | stdev |
|---|---:|---:|
| eager host | 0.7772 s | 0.00026 s |
| piecewise wait | 0.7773 s | 0.00045 s |

Conclusion:

- The earlier B=64 CPU eager edge is not statistically pinned down in the
  production/no-counter path.
- Treat B=64 CPU as within-noise parity, not as a reason to add per-bucket
  eager fallback by default.

### Full-Graph A/B Results After Compatibility Fix

B=64 route-specialized full graph:

```text
/TTC/results/planner/graph_sync_default_recheck_20260602/fullgraph_route_specialized_b64_f030
/TTC/results/planner/graph_sync_default_recheck_20260602/fullgraph_route_specialized_wait_b64_f030
```

| full graph sync | no offload | pure prefetch | pure CPU decode |
|---|---:|---:|---:|
| host callback | 0.1335 s | 0.8773 s | 1.8278 s |
| wait kernel | 0.1339 s | 0.9507 s | 1.7914 s |

B=1 route-specialized full graph:

```text
/TTC/results/planner/graph_sync_default_recheck_20260602/fullgraph_route_specialized_b1_out128_f005_host
/TTC/results/planner/graph_sync_default_recheck_20260602/fullgraph_route_specialized_b1_out128_f005_wait
```

| full graph sync | no offload | pure prefetch | pure CPU decode |
|---|---:|---:|---:|
| host callback | 2.0119 s | 3.8851 s | 2.6250 s |
| wait kernel | 2.0115 s | 3.9039 s | 2.4909 s |

Conclusion:

- Full graph is no longer invalid for native COTS CPU routes.
- Full graph is still not the default because valid full graph is slower than
  current piecewise/eager results in both tested regimes.

### Original B=1 Graph Win Still Holds

Old Phase 1c capture-gap gate:

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

Focused rerun:

```text
/TTC/results/planner/graph_sync_default_recheck_20260602/b1_out128_f005
```

| arm | latency | delta vs eager |
|---|---:|---:|
| native eager | 2.439471 s | baseline |
| piecewise COTS split + host callback | 2.416969 s | -22.5 ms |
| piecewise COTS split + wait kernel | 2.301858 s | -137.6 ms |

Conclusion:

- The original B=1 long-decode regime still supports wait-kernel as the best
  graph sync mode.
- The B=64 short-decode noise should not be used to globally flip graph sync
  to host-callback.

## Code Touch Points

vLLM changes:

- `vllm/compilation/decorators.py`
  - Added native COTS weight graph detection.
  - Disabled AOT compile for native COTS weight graph capture.
- `vllm/v1/worker/gpu/cudagraph_utils.py`
  - Preserved decorated `BatchDescriptor` during full-graph capture.
- `vllm/config/compilation.py`
  - Added `wait_prefetch` and `start_prefetch` to COTS split ops.
- `vllm/compilation/cuda_graph.py`
  - Added CUDAGraph wrapper options for offloader sync/join behavior.
- `vllm/compilation/backends.py`
  - Skips wrapper-level offloader drains only for active native COTS weight
    piecewise graphs with all four COTS split ops present.
- `vllm/envs.py`
  - Registered `VLLM_COTS_DUMP_COUNTERS_ON_SHUTDOWN`.
- `tests/compile/test_config.py`
  - Added tests for disabling auto COTS split while preserving full-graph CPU
    compatibility.

TTC changes:

- `David/Benchmarks/planner/bench_sync_mode_matrix.py`
  - Added reproducible host/wait/full/eager matrix runner.
- `David/Docs/phase1_findings.md`
  - Updated Phase 1 source-of-truth summary.
- `David/Docs/phase1c_findings.md`
  - Updated Phase 1c appendix summary.
- `David/Docs/graph_eager_sync_mode_investigation_report.md`
  - This final handoff report.

## Validation

Syntax and whitespace:

```text
python3 -m py_compile \
  David/Benchmarks/planner/bench_sync_mode_matrix.py \
  vllm/vllm/compilation/backends.py \
  vllm/vllm/compilation/cuda_graph.py \
  vllm/vllm/compilation/decorators.py \
  vllm/vllm/config/compilation.py \
  vllm/vllm/envs.py \
  vllm/vllm/v1/worker/gpu/cudagraph_utils.py \
  vllm/tests/compile/test_config.py

git diff --check
git -C vllm diff --check
```

All passed.

vLLM config tests:

```bash
TTC_DOCKER_WORKDIR=/TTC/vllm scripts/ttc-docker-env.sh thesis \
  'pytest tests/compile/test_config.py -q'
```

Result:

```text
44 passed
```

## Reproduction Pointers

Current matrix runner:

```bash
TTC_DOCKER_WORKDIR=/TTC/FastTTS-thesis scripts/ttc-docker-env.sh thesis \
  'python /TTC/David/Benchmarks/planner/bench_sync_mode_matrix.py \
   --exp --keep-going \
   --results-dir /TTC/results/planner/sync_mode_matrix_20260602/current_host_wait_full_piecewise_eager'
```

Direct graph wait-kernel B=64 CPU route:

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

Direct eager host-callback B=64 CPU route:

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

## What Remains

No remaining blocker from this session.

Optional future work:

1. Add Planner/Profiler support for a profile-gated eager fallback. The
   profiler should record runtime mode, graph family, and sync mode when it
   measures a dispatch cell.
2. Use Nsight Systems on the noisy B=64 CPU route only if we want to chase a
   strict graph win rather than accepting within-noise parity.
3. Run broader full-graph sweeps only if another workload suggests full graph
   could become a default candidate.
4. Return to forced-context/logprob parity work now that the graph/eager
   runtime issue is classified.
