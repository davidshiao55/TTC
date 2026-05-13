# Phase 1c Findings: Native COTS Runner And Graph-Compatible Dispatch

Date: 2026-05-13

Status: landed. Phase 1c production is the native COTS runner with
out-of-graph dispatch state, live-token row capping, fast COTS-aware
piecewise graph split, and `wait_kernel` sync.

## Outcome

Phase 1c turned the Phase 1a/1b Python-heavy COTS path into a production
runtime substrate:

- `CotsOffloadConfig.cpu_runner="native"` is the default.
- Native COTS supports `enforce_eager=False`.
- `auto_graph_split=True` is the default graph policy for native COTS:
  use piecewise CUDA graphs, split at `vllm::cots_submit_gemm` and
  `vllm::cots_sync_then_uva`, and use `wait_kernel` sync.
- `cpu_runner="python"` remains only as an eager-mode kill-switch for
  A/B diagnosis and parity checks.
- `cots_capture_sync_mode` is now only
  `Literal["host_callback", "wait_kernel"]`.
- Prototype-only ablations, fused `wait_uva_kernel`, dryrun burst helpers,
  task fire-count dumps, stale bench arms, raw results, and standalone
  probe trees were removed.

The key production result on the focused Qwen2.5-7B BF16 decode workload
(`B=1`, `input_len=8`, `output_len=128`, `f_cpu_store=0.05`,
`cpu_num_threads=16`) is:

| Arm | Seconds / generate | Delta vs native eager |
|---|---:|---:|
| `native_eager_real` | 2.5357 | baseline |
| `piecewise_cots_split_wait_kernel_real` | 2.4573 | -78.4 ms |

The broader grid (`B={1,4}`, `input_len={8,128,512}`,
`output_len={32,128}`) showed the split path won all 12/12 cells against
native eager and all 12/12 cells against legacy full capture. Short decode
cells can win by less than the 50 ms decision margin, so this is a tested
default for the Phase 1c Qwen2.5-7B surface, not a universal Planner rule.

Compact benchmark summary:
`David/Benchmarks/phase1c/results/phase1c_final_summary.json`.

## Final Architecture

The public import remains stable:

```python
from vllm.model_executor.offloader.cots import CotsOffloader
```

The public facade is `vllm/model_executor/offloader/cots.py`. The retained
implementation is split by responsibility:

- `cots_storage.py`: storage and partition-facing exports
  (`CotsLinearHandle`, prefetch pool, prefetch streamer).
- `cots_runners.py`: runtime runner-facing exports
  (`PythonCotsRunner`, `NativeCotsRunner`, slab specs).
- `cots_operators.py`: operator/UVA-facing exports
  (`CotsQKVOp`, `CotsSwiGLUMLPOp`, UVA helper).
- `cots_offloader.py`: offloader lifecycle, module patching, dispatch
  table construction, and BaseOffloader hooks.
- `cots_utils.py`: shared thread, QKV picker, and UVA helpers.
- `cots_ops.py`: native custom ops and out-of-graph dispatch registry.
- `vllm/csrc/cots/`: C++ `CotsCpuInfer`, task queue, BF16 kernels, and
  wait-kernel launchers.

Storage remains the Phase 1a/1b tensor-granular split:

- QKV is column-split with K/V-biased CPU assignment.
- MLP `gate_up` is column-split.
- MLP `down_proj` is row-split with transposed CPU storage.
- `o_proj` stays GPU-resident.
- `f_cpu_compute + f_prefetch <= f_cpu_store` is preserved per dispatch
  bucket.

Runtime has two runner choices:

- `NativeCotsRunner` is production. It owns one C++ `CotsCpuInfer`
  registry entry per offloader, installs one slab per
  `(layer_idx, bucket, op_kind)`, submits work from stream host callbacks,
  and returns CPU results through the UVA activation-return path.
- `PythonCotsRunner` is eager-only. It keeps the older
  `ThreadPoolExecutor` path so correctness and performance can be compared
  without the native substrate.

The CUDA graph boundary is controlled by two custom ops:

- `vllm::cots_submit_gemm` bundles D2H of the live activation rows into
  the slab input with the CPU-work submit callback.
- `vllm::cots_sync_then_uva` waits for the CPU work and copies the slab
  output back to the GPU result buffer through the UVA kernel.

Native custom ops take CUDA tensors and scalar ids only. Pinned CPU
addresses and slab state stay in C++ so Inductor does not materialize CPU
views inside captured graphs.

## Runtime Flow

At load/wrap time:

1. `CotsOffloader` discovers QKV, `gate_up`, and `down_proj` modules.
2. `CotsLinearHandle` installs GPU-resident and CPU-pinned weight slices.
3. The dispatch table is built from the Planner factory or uniform config.
4. Optional prefetch buffers are sized from the largest needed bucket.
5. The native runner installs C++ slabs for every active
   `(layer_idx, bucket, op_kind)`.
6. If graph mode uses `wait_kernel`, each slab gets host-mapped
   `req_slot` and `done_slot` cells.

At runtime:

1. The model runner calls `CotsOffloader.on_dispatch` outside the compiled
   graph with the active `BatchDescriptor`.
2. `on_dispatch` publishes the active bucket and live token count to the
   native registry.
3. `cots_submit_gemm` resolves the slab from out-of-graph state and submits
   the CPU task after D2H.
4. GPU GEMMs run on the GPU-resident and prefetched slices.
5. `cots_sync_then_uva` waits for the matching CPU task. In the default
   graph path, `wait_kernel` replaces the captured sync host callback.
6. The CPU output is copied into the GPU output buffer by the UVA kernel.
7. Scatter/concat logic assembles the exact unsplit output.

The live-token cap is important: graph buckets are capacity-sized, but
decode often has fewer real rows. The worker uses the live cap to avoid
CPU GEMM work for padded rows, while slab allocation and graph shape stay
bucket-based.

## Public Config

Retained public surface:

| Config/API | Status |
|---|---|
| `offload_backend="cots"` | supported |
| `CotsOffloadConfig.cpu_runner: Literal["native", "python"]` | supported |
| `dry_run` | supported diagnostic |
| `auto_graph_split` | default `True` |
| `cpu_num_threads_by_bucket` | supported Planner hook |
| `set_live_num_tokens` | supported runtime hook |
| `set_runtime_num_tokens` | legacy alias |
| `get_counters/reset_counters` | retained aggregate diagnostics |

`cots_capture_sync_mode` supports only:

- `"host_callback"`: legacy sync callback. This is the field default and
  the eager fallback.
- `"wait_kernel"`: GPU wait kernel on host-mapped completion cells.
  Native graph mode with `auto_graph_split=True` upgrades to this mode
  unless the user opts out with `--no-cots-auto-graph-split`.

Retained env-gated diagnostics:

- `VLLM_COTS_NVTX`
- `VLLM_COTS_COUNTERS`
- `VLLM_COTS_WAIT_KERNEL_DIAG`
- legacy umbrella `VLLM_COTS_DIAG`
- post-capture counter reset hook
- compile-cache debug hook

Removed from the supported surface:

- `VLLM_COTS_ABLATE_*`
- `wait_uva_kernel`
- C++ `set_ablations`
- C++ fused wait+UVA launcher
- dryrun burst pybind helper
- per-task fire-count dump plumbing

## Decisions From Experiments

### Native Runner Became The Production Substrate

The Python runner established correctness in Phase 1a/1b, but it cannot be
captured safely: `ThreadPoolExecutor.submit` and `future.result()` are host
operations with no CUDA graph replay semantics. Native COTS moved submit
and sync onto CUDA stream host callbacks and C++ slabs, making graph-mode
execution possible.

Decision: native is default; Python is eager-only kill-switch.

### Full Capture With Host Callbacks Backpressured Replay

The first graph-compatible native path put COTS submit/sync host callbacks
inside full CUDA Graph capture. It was correct but slower than native eager
because replay blocked inside captured graph nodes.

Focused full-capture timing:

| Arm | Seconds / generate | Delta vs native eager |
|---|---:|---:|
| `native_eager_real` | 2.5327 | baseline |
| `capture_host_callback_real` | 2.7197 | +187.0 ms |
| `capture_wait_kernel_real` | 2.6325 | +99.8 ms |

Decision: full capture is retained only as a regression/diagnostic
baseline, not the default.

### M2 Kernel-Counter Sync Was Rejected

The M2 direction tried to avoid sync host callbacks with a GPU-visible
counter protocol. It did not provide a clean enough correctness and
scheduling story for production: it complicated ordering, still needed
careful replay-time state management, and did not address the larger graph
structure problem exposed by full capture.

Decision: reject M2 and keep the simpler per-slab value-signal wait kernel.

### Wait Kernel Landed, But Only With The Split Graph

`wait_kernel` replaces the sync-side host callback with a small GPU kernel
that spins until the CPU worker writes `done_slot >= req_slot`. It removed
part of the full-capture cost, but full capture still did not beat eager.
It became valuable after the COTS ops were split out of captured graph
regions.

Decision: keep `host_callback` as the compatibility path; use
`wait_kernel` for the native graph default through `auto_graph_split`.

### Fast COTS Piecewise Split Is The Default

The winning structural change was to make the COTS submit and sync/UVA ops
piecewise graph split points. This leaves COTS orchestration outside
captured CUDA graph nodes while the surrounding GPU-only regions still use
piecewise graphs.

Focused split timing:

| Arm | Seconds / generate | Delta vs native eager |
|---|---:|---:|
| `native_eager_real` | 2.5441 | baseline |
| `capture_wait_kernel_real` | 2.6233 | +79.2 ms |
| `piecewise_cots_split_host_callback_real` | 2.5230 | -21.1 ms |
| `piecewise_cots_split_wait_kernel_real` | 2.4337 | -110.4 ms |

After compile-cache cleanup, the fast split still passed with normal cache
enabled:

| Arm | Seconds / generate | Delta vs native eager |
|---|---:|---:|
| `native_eager_real` | 2.5357 | baseline |
| `piecewise_cots_split_wait_kernel_real` | 2.4573 | -78.4 ms |

Greedy token/text parity matched `native_eager_real`.

Decision: `auto_graph_split=True` is the Phase 1c graph-mode default.

### Live-Token And Out-Of-Graph Dispatch Fixed Bucket Drift

Full graph replay can expose max-sized persistent buffers while the actual
decode step has fewer live rows. Earlier routing inferred dispatch from
compile-visible tensor shapes and could select the wrong bucket or
over-compute padded rows.

The fix was:

- publish active bucket and live row count outside the graph;
- resolve native task id from `(layer_idx, active_bucket, op_kind)`;
- clamp CPU GEMM to the live row count;
- keep slab allocation and CUDA graph tensor shapes bucket-based.

Decision: bucket/task routing is runtime state, not a compile-visible
scalar argument.

### Fused Wait+UVA Was Correct But Not Worth Keeping

The fused `wait_uva_kernel` prototype combined waiting and pinned-output
copy into one CUDA kernel. It preserved greedy parity, but on the split
path it was only 0.6 ms/generate faster on the mean and changed sign
between repeats.

Focused A/B:

| Arm | Mean seconds / generate | Delta vs native eager |
|---|---:|---:|
| `native_eager_real` | 2.5377 | baseline |
| `piecewise_cots_split_wait_kernel_real` | 2.4426 | -95.1 ms |
| `piecewise_cots_split_wait_uva_real` | 2.4420 | -95.7 ms |

Decision: remove `wait_uva_kernel` from config, CLI, C++ launchers,
tests, and bench arms.

### Probe Ablations Explained The Direction, Then Were Removed

The temporary ablation flags separated D2H, UVA, submit host callback, and
sync host callback costs. They showed that captured host-function
backpressure, especially sync-side blocking, was the important issue.
They were useful for choosing the wait-kernel and split-graph directions,
but unsafe to leave as supported behavior because a stray env var could
silently benchmark a non-production path.

Decision: document the conclusion, delete `VLLM_COTS_ABLATE_*`, C++
`set_ablations`, and related tests.

### Compile Cache Needed A Targeted Fallback

Fast COTS split originally exposed an AOT serialization failure for nested
standalone subgraphs. The issue was not COTS math. The fix keeps normal
compile cache enabled for serializable subgraphs and uses in-memory compiled
callables only for the unsaveable nested pieces.

Decision: keep the compile-cache debug hook; treat partial cache fallback
as a startup/cache-quality limitation, not a runtime correctness issue.

## Retained Validation

Retained Phase 1c tests cover:

- native extension load and native runner install;
- custom-op ordering dependencies;
- CUDA graph capture and replay;
- live-token row capping and bucket dispatch;
- thread policy and optional worker affinity;
- multi-engine safety;
- Python kill-switch parity;
- wait-kernel install, capture replay, parity, and worker exception paths;
- BF16 custom kernels;
- pinned storage, strided D2H, contiguous down-proj, and y-pinned views.

Retained reusable benches:

- `bench_capture_gap_qwen.py`
- `bench_capture_gap_qwen_grid.py`
- `check_capture_piecewise_parity_qwen.py`
- `bench_thread_policy_sweep.py`

Raw result trees under `David/Benchmarks/phase1c/results/` are no longer
tracked. Regenerate them from the retained benches when needed.

## Phase 2 Handoff

Phase 1c leaves a graph-compatible weight-offload substrate for attention
offload work:

- native COTS is the default substrate;
- Python runner is available only for eager diagnostics;
- graph-mode default is fast COTS split plus wait kernel;
- CPU worker row count is live-token capped;
- COTS routing state is published out of graph;
- retained diagnostics are explicit and env-gated.

Phase 2 can build CPU suffix attention and online-softmax merge on top of
this without carrying Phase 1c's rejected probe paths forward.
