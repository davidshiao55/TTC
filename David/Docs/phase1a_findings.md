# Phase 1a Appendix: Static Split Prototype

Date: 2026-05-15 cleanup

Status: historical appendix. The final Phase 1 production narrative lives in
`phase1_findings.md`. This file keeps only the Phase 1a facts that explain the
final COTS design.

## What Phase 1a Proved

Phase 1a established the static tensor split that still underlies production
COTS:

- vLLM weights can be split at load time with TP-style loader closures.
- QKV, MLP gate/up, and MLP down can be offloaded at tensor granularity while
  `o_proj` remains GPU-resident.
- GPU memory freed by the split is visible to vLLM's profiler and therefore
  becomes KV-cache headroom.
- The MLP path must be a block-level operator, not independent per-Linear
  operators, because gate/up, SwiGLU, and down share the matched intermediate
  slice.
- CPU results can return through an SM-issued UVA copy kernel, avoiding copy
  engine serialization with future weight prefetch.
- The split is numerically transparent within BF16 tolerance and passes
  end-to-end greedy smoke tests.

The final storage invariant came from this phase:

```text
GPU permanent slice + CPU-stored slice = original weight
```

Later phases only decide, per bucket, how much of the CPU-stored slice is
prefetched back to GPU and how much is computed on CPU.

## Production Pieces That Survived

| Phase 1a component | Final production status |
|---|---|
| TP-style weight-loader interception | kept |
| profiler-context allocation rule | kept |
| lazy module iteration to avoid peak GPU OOM | kept |
| `CotsLinearHandle` storage abstraction | kept, now in `cots_storage.py` |
| `CotsQKVOp` per-Linear operator | kept, now graph/native-aware |
| fused MLP block operator | kept, later accelerated with native BF16 MLP worker |
| SM-issued UVA activation return | kept |
| wrap-time fail-fast checks | kept |
| QKV K/V-biased picker | kept for Phase 2 attention handoff |

The original single-file prototype was later split into:

- `cots_storage.py`
- `cots_operators.py`
- `cots_offloader.py`
- `cots_runners.py`
- `cots_ops.py`
- `cots_utils.py`
- `vllm/csrc/cots/`

## What Changed Later

The Phase 1a performance story is no longer the production story.

| Phase 1a behavior | Final production behavior |
|---|---|
| Python `ThreadPoolExecutor` CPU runner | native C++ runner is default |
| eager-only COTS | graph-compatible native COTS |
| static CPU-compute-only dispatch | per-bucket three-way dispatch |
| uniform storage/compute fraction | storage is static; compute routing is bucket-specific |
| old MLP CPU worker sequence | BF16 fused gate/up/SwiGLU + transposed down worker |
| no prefetch path | layer-ahead prefetch with K=2 slot rotation |

The important Phase 1a diagnostic was that the isolated Phase 0 overlap math
was too optimistic end-to-end. The old Python runner had a visible host
orchestration tax, and real CPU work interfered with the main CUDA dispatch
thread. That finding is why the native runner moved into Phase 1c before Phase
2.

## Historical Validation

The original Qwen2.5-7B smoke at `f_cpu_store=0.09` showed:

- about `1.13 GB` of weight memory freed;
- baseline-compatible `gpu_memory_utilization=0.9`;
- identical greedy output on the smoke prompt in the retained run;
- Nsight-visible overlap between GPU slice GEMM, CPU GEMM, and UVA return;
- exactly one UVA return per fused MLP block.

Those facts remain useful, but the old latency comparisons against native
prefetch are superseded by the current-code rerun summarized in
`phase1_findings.md` and `phase1_analysis_findings.md`.

## Legacy Section Map

Older docs may still refer to Phase 1a section numbers. Use this map when
reading those references:

| Old reference | Current location |
|---|---|
| `§1.3` fused MLP block | "What Phase 1a Proved" and `phase1_findings.md` production path |
| `§1.7` memory saved | "Historical Validation" above |
| `§1.8` smoke/parity | "Historical Validation" above |
| `§1.13b` COTS vs prefetch | superseded by `phase1_analysis_findings.md` COTS-vs-native table |
| `§1.14` dry-run/free-regime gap | superseded by `phase1_analysis_findings.md` CPU-compute and prefetch-gap sections |

## Reader Guidance

Use this file only when you need the origin of a production invariant. For
current numbers, current architecture, and final Planner guidance, read
`phase1_findings.md`.
