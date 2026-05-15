# Phase 1c Appendix: Native Runner And Graph-Compatible Dispatch

Date: 2026-05-15 cleanup

Status: production runtime appendix. The top-level Phase 1 source of truth is
`phase1_findings.md`; this file keeps the final Phase 1c runtime details.

## Outcome

Phase 1c made COTS production-faithful:

- `CotsOffloadConfig.cpu_runner="native"` is the default.
- Native COTS supports `enforce_eager=False`.
- `auto_graph_split=True` is the default graph policy.
- COTS submit/sync custom ops are piecewise graph split points.
- `wait_kernel` is the native graph sync path.
- CPU worker row count is live-token capped.
- Active bucket and live row count are published out of graph.
- Python runner remains only as an eager diagnostic fallback.

Focused Qwen2.5-7B BF16 result:

| Arm | Seconds / generate | Delta vs native eager |
|---|---:|---:|
| native eager real | 2.5357 | baseline |
| piecewise COTS split + wait kernel | 2.4573 | -78.4 ms |

The broader grid over `B={1,4}`, `input_len={8,128,512}`, and
`output_len={32,128}` showed the split path winning all 12/12 cells against
native eager and all 12/12 cells against legacy full capture.

## Final Runtime Structure

The retained implementation is split by responsibility:

| File | Responsibility |
|---|---|
| `cots_storage.py` | storage, snapping, loader closures, prefetch pool/streamer |
| `cots_operators.py` | QKV and fused MLP forward semantics |
| `cots_offloader.py` | lifecycle, module patching, dispatch table, graph hooks |
| `cots_runners.py` | Python and native runner facades |
| `cots_ops.py` | custom ops and native out-of-graph registry |
| `cots_utils.py` | QKV picker, UVA helper, shared utilities |
| `vllm/csrc/cots/` | C++ native runner, task queue, BF16 kernels, wait kernel |

Runtime flow:

1. `CotsOffloader` discovers QKV, gate/up, and down modules.
2. `CotsLinearHandle` installs GPU-resident and CPU-pinned slices.
3. The dispatch table produces per-bucket prefetch and CPU-compute geometry.
4. The native runner installs one slab per `(layer_idx, bucket, op_kind)`.
5. Before replay/forward, COTS publishes active bucket and live token count.
6. `vllm::cots_submit_gemm` submits native CPU work.
7. GPU permanent/prefetched slice compute proceeds.
8. `vllm::cots_sync_then_uva` waits and returns CPU output through UVA.
9. Scatter/add-reduce assembles the exact full output.

## Graph Decision

Full graph capture with COTS host callbacks was correct but slower:

| Arm | Seconds / generate | Delta vs native eager |
|---|---:|---:|
| native eager real | 2.5327 | baseline |
| capture host callback real | 2.7197 | +187.0 ms |
| capture wait kernel real | 2.6325 | +99.8 ms |

The final decision is to split COTS custom ops out of captured graph regions and
use `wait_kernel` for sync. The wait kernel is useful with split graphs; full
capture remains only a diagnostic baseline.

## CPU Worker Kernels

Production native CPU work uses:

- BF16 natural GEMM for QKV.
- BF16 fused gate/up/SwiGLU into BF16 scratch for MLP.
- BF16 transposed GEMM for the down projection.

The fused MLP worker was added late in Phase 1c cleanup after CPU-only gap
diagnostics showed the old MLP sequence leaked exposed work at heavier slices.
It improves heavier exposed MLP cases but does not move the strict 5% boundary,
because the boundary cell was already mostly overlapped.

The production path intentionally keeps BF16 scratch semantics. The FP32-scratch
candidate and runtime kernel selector were removed.

## Production Cleanup

Retained:

- native runner;
- piecewise split graph;
- wait-kernel sync;
- active-bucket and live-token runtime state;
- pure-prefetch zero-CPU-compute fast path;
- control-plane `--cots-dry-run`;
- MLP 64-channel snapping;
- active-adjacent MLP prefetch slots;
- transposed down-proj CPU storage;
- env-gated diagnostics.

Removed:

- `VLLM_COTS_ABLATE_*`;
- fused `wait_uva_kernel`;
- dryrun burst helpers;
- task fire-count dumps;
- diagnostic target-kind and full-shape MLP config fields;
- FP32 MLP candidate and runtime MLP kernel selector;
- old raw result trees.

`VLLM_DISABLE_COMPILE_CACHE=1` appeared only in diagnostic graph sweeps for a
TorchInductor standalone cache issue. It is not part of the production path.

## Validation Kept

Retained tests cover:

- native extension load and native runner install;
- custom-op ordering dependencies;
- CUDA graph capture and replay;
- live-token capping and bucket dispatch;
- thread policy and optional worker affinity;
- multi-engine safety;
- Python kill-switch parity;
- wait-kernel install, replay, parity, and worker exception paths;
- BF16 custom kernels;
- pure-prefetch fast path;
- MLP snap64 geometry and BF16 scratch reference semantics.

## Phase 2 Handoff

Phase 2 should build attention offload on this runtime:

- native COTS is the default substrate;
- Python runner is diagnostic-only;
- graph-mode default is split graph plus wait kernel;
- routing is out-of-graph runtime state;
- CPU work is live-token capped;
- rejected Phase 1 probe paths are gone.
