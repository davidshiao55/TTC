# Dispatch Validation Investigation Report

Date: 2026-06-01

Follow-up to `dispatch_validation_surprise_report.md`.

Status update, 2026-06-02: this report is the root-cause note for the original
bogus graph pure-CPU measurement. The structural routing bug described here was
fixed by separating capture buckets, dispatch buckets, and COTS route
signatures. Current follow-up work has moved to graph-vs-eager performance and
wait-kernel behavior; see `graph_eager_sync_mode_investigation_report.md`.

## Executive Summary

Update after counter, parity, and compile-graph checks: the surprising
graph-mode decode-only result is invalid as a CPU-performance result. The
measured `B=64`, `f_cpu_store=0.30`, `output_len=128` pure-CPU decode slowdown
was only `1.155x`, but the measured graph path did not execute COTS CPU work.

Direct timing of one actual fused COTS MLP slab at the matching shape
(`B=64`, hidden `3584`, MLP intermediate slice `5683`, `24` threads) measured
about `13.6 ms` for one layer. Across 28 layers that is about `382 ms` of CPU
MLP work per decode step before QKV. The observed long-output graph run adds
only `0.3685 s` total over 128 generated tokens, or about `2.9 ms/token`. Those
two facts cannot both describe a correctly waited full CPU decode path.

The current conclusion is therefore:

- The pure-prefetch path is genuinely very expensive in this validation cell.
- The graph-mode decode-only pure-CPU endpoint is bogus: the compiled graph has
  no `cots_submit_gemm` / `cots_sync_then_uva` nodes for that run.
- The earlier interpretation ("CPU decode is surprisingly fast at B=64,
  f=30%") is false and does not match the Phase 0 math.
- Native weight graph mode is currently safe only when COTS routing geometry is
  uniform across captured buckets, or until routing geometry is moved behind an
  out-of-graph/per-capture structural boundary.

## What Was Checked

### Existing Controls

The prior dry-run control at `f_cpu_store=0.15` showed near-CPU hybrid and pure
CPU are essentially identical when real offloaded work is skipped:

| B | near-CPU dry-run | pure CPU dry-run | dry-run gap |
|---:|---:|---:|---:|
| 16 | 0.4818 s | 0.4818 s | about 0.08 ms |
| 64 | 0.5559 s | 0.5533 s | about 2.6 ms |

That rules out dispatch-table lookup, graph control flow, and hook overhead as
the main reason hybrid loses. The remaining gap is real prefetch work.

### Module Isolation

I ran targeted decode-only B=64, `f_cpu_store=0.30`, graph-mode isolation
sweeps with `--cots-weight-modules qkv` and `--cots-weight-modules mlp`.

Result paths:

```text
/TTC/results/planner/dispatch_module_isolation/20260601_qkv_only_b64_f030/summary.md
/TTC/results/planner/dispatch_module_isolation/20260601_mlp_only_b64_f030/summary.md
```

QKV-only:

| policy | latency | slowdown |
|---|---:|---:|
| none | 0.6287 s | 1.000x |
| pure prefetch | 0.6738 s | 1.072x |
| near-CPU hybrid `(0.2625, 0.0375)` | 0.6489 s | 1.032x |
| pure CPU | 0.6418 s | 1.021x |

MLP-only:

| policy | latency | slowdown |
|---|---:|---:|
| none | 0.6297 s | 1.000x |
| near-CPU hybrid `(0.2625, 0.0375)` | 1.0998 s | 1.747x |
| pure CPU | 0.9318 s | 1.480x |

The MLP-only pure-prefetch graph cell segfaulted during engine init. That is a
separate graph-mode bug in the diagnostic module subset path; the combined
QKV+MLP pure-prefetch path still runs and shows the same high cost.

The weight split explains why MLP dominates:

| module set | saved bytes at `f_cpu_store=0.30` |
|---|---:|
| QKV only | 0.2826 GB |
| MLP only | 3.4296 GB |
| QKV + MLP | 3.7122 GB |

So the global dispatch curve mostly measures MLP prefetch behavior.

### Longer Decode Confirmation

I repeated the combined B=64, `f_cpu_store=0.30` decode-only comparison with
`output_len=128`, `num_iters_warmup=2`, `num_iters=3`.

Result path:

```text
/TTC/results/planner/dispatch_confirmation/20260601_combined_b64_f030_out128/summary.md
```

| policy | latency | slowdown |
|---|---:|---:|
| none | 2.3812 s | 1.000x |
| pure prefetch | 21.6733 s | 9.102x |
| near-CPU hybrid `(0.2625, 0.0375)` | 3.7887 s | 1.591x |
| pure CPU | 2.7498 s | 1.155x |

This table is the suspicious result. At face value it says the CPU endpoint is
nearly free over a long decode. The consistency check below shows that this
cannot be reconciled with the profiled worker cost, so the table should now be
used as a bug-finding artifact rather than Planner evidence.

### Phase-0 Consistency Check

The Phase 0 CPU curve already predicted that `B=64`, `f=0.30` should be far
outside the free regime. The profile table reports about `222 us/MB` for MLP1
and `223 us/MB` for MLP2 at `B=64`, `f=0.30`. With Qwen2.5-7B's MLP sizes, that
implies tens of milliseconds per MLP block per layer, not a few milliseconds per
whole decode step.

I also timed the current native fused COTS MLP worker directly:

```text
B=64, hidden=3584, MLP slice K=5683, threads=24
actual CotsWeightTaskRunner fused MLP slab: 13.63 ms/layer
```

Even ignoring QKV, 28 layers gives:

```text
13.63 ms/layer * 28 layers ~= 382 ms/decode step
```

But the long graph validation measured:

```text
2.7498 s - 2.3812 s = 0.3686 s total extra
0.3686 s / 128 generated tokens ~= 2.9 ms/token
```

So the graph validation CPU endpoint is under by roughly two orders of
magnitude relative to direct worker timing.

A short `output_len=16` rerun showed the same smell:

```text
/TTC/results/planner/dispatch_counter_probe/20260601_b64_f030_out16_graph_eager/summary.md

graph none:            0.3445 s
graph pure CPU decode: 0.6831 s  (1.983x)
graph pure prefetch:   2.9742 s  (8.634x)
```

The pure-CPU extra at `output_len=16` is about `0.339 s`, almost the same total
extra as the `output_len=128` run. That points to a fixed non-decode cost
rather than per-token CPU decode work.

### Counter / Parity Root Cause

I added shutdown counter dumping for native weight runners and resettable
counter probes around CUDA graph capture. For graph-mode decode-only
`B=64`, `f_cpu_store=0.30`, pure CPU decode:

```text
/TTC/results/planner/dispatch_counter_probe/20260601_b64_f030_out16_counters/summary.md

graph none:            0.3407 s
graph pure CPU decode: 0.6913 s
graph pure prefetch:   2.9046 s
```

After `post_cudagraph_capture` reset, the pure-CPU decode shutdown counters
were:

```text
submit_count_qkv = 0
submit_count_mlp = 0
dispatch_cb_count = 0
worker_run_count = 0
d2h_* = 0
uva_record_count = 0
live_set_calls = 35
```

The live-token dispatch boundary fired, but no COTS submit/sync/custom-op work
ran during measured replay.

Token parity also failed between no-offload and graph decode-only pure CPU:

```text
/TTC/results/planner/dispatch_parity/20260601_b64_f030_graph/summary.json
num_mismatches = 64 / 64
```

The COTS arm diverged into garbage tokens, and its shutdown counters again
showed zero submit/sync/worker activity. So the fast graph result is not merely
an accounting artifact; it is also incorrect.

A no-reset graph probe showed the custom ops never fired even during capture:

```text
/TTC/results/planner/dispatch_counter_probe/20260601_b64_f030_out4_graph_noreset_counters/summary.md

graph pure CPU decode: 0.4611 s
submit_count_qkv = submit_count_mlp = worker_run_count = 0
live_set_calls = 113
```

The generated torch-compile graph explains why. For the decode-only run, the
cache file:

```text
/tmp/ttc-codex-home-1012/.cache/vllm/torch_compile_cache/1c65801a3d/rank_0_0/backbone/computation_graph.py
```

contains no `cots_submit_gemm` or `cots_sync_then_uva` nodes. The model was
traced at a bucket whose decode-only fallback was pure prefetch, so the
Python-side `n_cpu_compute > 0` branches were specialized away. Runtime
`on_dispatch` can update the active bucket and live-token count, but it cannot
restore branches that Dynamo never kept in the compiled graph.

As a control, a uniform CPU dispatch table keeps the COTS custom ops:

```text
/tmp/ttc-codex-home-1012/.cache/vllm/torch_compile_cache/4d8894c5f4/rank_0_0/backbone/computation_graph.py
```

That graph contains both `cots_submit_gemm` and `cots_sync_then_uva`, and the
runtime counters are nonzero.

### True Uniform CPU Control

To force the compiled graph to contain the CPU path, I ran uniform dispatch at
`B=64`, `f_cpu_store=0.30`, `output_len=4`:

```text
/TTC/results/planner/dispatch_counter_probe/20260601_b64_f030_out4_uniform_counters/summary.md
```

| mode | policy | latency | slowdown |
|---|---|---:|---:|
| eager | none | 0.1729 s | 1.000x |
| eager | pure prefetch | 0.8222 s | 4.755x |
| eager | pure CPU | 2.8928 s | 16.730x |
| graph | none | 0.1321 s | 1.000x |
| graph | pure prefetch | 0.9545 s | 7.225x |
| graph | pure CPU | 5.6035 s | 42.412x |

The true pure-CPU counters were also nonzero:

```text
eager pure CPU:
  submit_count_qkv = 196
  submit_count_mlp = 196
  worker_run_count = 392

graph pure CPU:
  submit_count_qkv = 3164
  submit_count_mlp = 3164
  worker_run_count = 6328
```

The graph count includes capture and many bucket variants because this probe
did not reset counters after capture. The important fact is simply that true
CPU execution is slow and counter-positive, unlike the invalid decode-only
graph endpoint.

### Controlled Graph Specialization Reproduction

On 2026-06-02 I reran a minimal B=64, `f_cpu_store=0.30`, `output_len=4`,
single-iteration graph experiment with fresh compile caches:

```text
/TTC/results/planner/bug_verify_20260602/bad_graph_decode_only/summary.md
/TTC/results/planner/bug_verify_20260602/uniform_graph/summary.md
```

The bad arm temporarily bypassed the new safety guard only for reproduction.
The final source tree does not keep that bypass.

| case | dispatch layout | pure CPU latency | pure prefetch latency |
|---|---|---:|---:|
| unsafe reproduction | decode-only, nonuniform | 0.4750 s | 1.0153 s |
| control | uniform | 5.6336 s | 0.9605 s |

The latency ordering flips exactly as the bug predicts: the nonuniform
decode-only "CPU" graph is too fast, while the uniform CPU graph is very slow.

The compile-cache structure is the decisive artifact. For the unsafe
decode-only run, the COTS CPU custom ops were absent from every generated
`computation_graph.py` under:

```text
/tmp/cots_verify_bad_graph/torch_compile_cache/
```

For the uniform CPU control, the generated graph:

```text
/tmp/cots_verify_uniform_graph/torch_compile_cache/65f1347501/rank_0_0/backbone/computation_graph.py
```

contains repeated `torch.ops.vllm.cots_submit_gemm(...)` and
`torch.ops.vllm.cots_sync_then_uva(...)` calls. This verifies that the bug is
graph specialization of Python-side COTS routing geometry, not a scheduler or
worker-thread timing artifact.

### Eager Bucket Selection Verification

I also checked whether eager mode still has graph capture buckets available for
COTS dispatch. It does not. vLLM explicitly disables CUDA graphs under
`enforce_eager=True`:

```text
cfg.cudagraph_mode = NONE
cfg.cudagraph_capture_sizes = []
```

Before the structural fix, COTS then fell back in `_resolve_capture_buckets()`
to a single synthetic bucket equal to `scheduler_config.max_num_batched_tokens`.
With the validation shape (`max_model_len=2048`) the probe showed:

```text
scheduler.max_num_batched_tokens = 2048
COTS _capture_buckets = (2048,)
bucket_for(1)    -> 2048
bucket_for(64)   -> 2048
bucket_for(512)  -> 2048
bucket_for(8192) -> 2048
```

A direct dispatch-table probe confirmed that, before this structural fix, a
distinct decode row was not reachable in eager:

```text
configured table:
  64   -> (0.30, 0.00)
  2048 -> (0.00, 0.30)

resolved _dispatch_table:
  2048 -> (0.00, 0.30)

prepare_before_forward(64) current_bucket = 2048
selected entry = (0.00, 0.30)
```

If the dispatch table only contained the `64` row, eager COTS rejected it with:

```text
ValueError: cots.dispatch_table is missing captured buckets: [2048]
```

So eager mode did not suffer from the graph branch-pruning bug, but it also did
not provide per-decode-bucket dispatch choices until COTS got a separate
dispatch-bucket source independent of CUDA graph capture sizes.

### Structural Fix Checkpoint

The COTS offloader now separates the two bucket concepts:

```text
_graph_capture_buckets  = actual vLLM CUDA graph replay shapes
_dispatch_buckets       = Planner/COTS route-selection buckets
```

When a Planner dispatch table is provided, `_dispatch_buckets` comes from the
table keys. Without an explicit table, COTS uses vLLM's graph bucket grid in
graph mode, reconstructs the would-have-been graph bucket grid in eager mode,
and adds larger fallback buckets up to `max_num_batched_tokens`.

Live eager verification:

```text
cfg.cudagraph_capture_sizes []
graph ()
dispatch (64, 2048)
current 64
selected (0.3, 0.0)
```

That fixes the eager collapse bug: `prepare_before_forward(64)` now selects
the `64` dispatch row instead of the max-token fallback.

Graph/compiled mode still cannot safely share one compiled graph across
multiple Python-visible COTS geometries. The guard now checks uniformity across
dispatch buckets, not CUDA graph buckets, and rejects nonuniform native
compiled/graph routing before capture:

```text
/TTC/results/planner/bug_fix_verify_20260602/graph_nonuniform_guard_current/summary.md
```

For the old bad B=64 decode-only graph case, the pure-CPU decode cell now fails
with:

```text
RuntimeError: CotsOffloader: native compiled/graph mode requires uniform COTS
routing geometry across dispatch buckets.
```

So graph mode no longer captures or replays the wrong dispatch plan. Full
nonuniform graph support still requires route-geometry-keyed graph variants or
fixed-shape/no-op COTS operator surfaces; until then, graph mode is deliberately
restricted to uniform COTS routing geometry.

Current verification:

```text
pytest David/Tests/phase1c -q  # 273 passed, 5 skipped
pytest David/Tests/phase1b -q  # 76 passed
```

### Module-Specific Decode Probe

I added a narrow runtime and harness path for module-specific dispatch tables:

```text
--cots-dispatch-table-by-module
--module-dispatch-policy qkv-prefetch-mlp-cpu
```

The probe keeps non-decode buckets pure prefetch, then sets the decode bucket
to:

```text
QKV: pure prefetch
MLP: pure CPU
```

Result path:

```text
/TTC/results/planner/dispatch_module_policy/20260601_qkv_prefetch_mlp_cpu_b64_f030_out128/summary.md
```

| policy | latency | slowdown |
|---|---:|---:|
| none | 2.3812 s | 1.000x |
| module-specific QKV prefetch + MLP CPU | 2.8920 s | 1.215x |

Comparison to the combined long-output run:

| policy | latency |
|---|---:|
| global pure CPU decode | 2.7498 s |
| module-specific QKV prefetch + MLP CPU | 2.8920 s |
| global near-CPU hybrid `(0.2625, 0.0375)` | 3.7887 s |

So module-specific dispatch removes a large part of the bad residual MLP
prefetch cost, but QKV prefetch still does not beat simply CPU-computing QKV
for this B=64, `f_cpu_store=0.30`, output_len=128 cell.

## Root Cause

The original cost mental model was:

```text
cost ~= max(CPU compute, H2D prefetch)
```

That model is missing the key MLP terms:

```text
prefetch cost ~= exposed H2D
              + prefetched-slice MLP1 GPU GEMM
              + activation on the prefetched intermediate
              + prefetched-slice MLP2 addmm
              + stream/order tail
```

That model is still missing the MLP prefetch terms, which explains why pure
prefetch is costly. But the pure-CPU surprise has a separate implementation
root cause:

```text
per-bucket COTS routing geometry is compile-visible
        +
torch.compile traces one geometry
        =
non-traced CPU/prefetch branches disappear from the generated graph
```

`on_dispatch` currently publishes bucket and live-token state out of graph, and
the native custom ops can resolve task ids from that state. That is not enough.
The Python-side operator still branches on `n_cpu_compute` and `n_prefetch` to
decide whether the graph contains CPU custom ops, prefetched-slice GEMMs,
scatter shapes, and prefetch hooks. If those values differ by bucket, a single
compiled graph can silently represent the wrong bucket.

I changed the runtime guard so native weight graph mode now requires uniform
COTS routing geometry across capture buckets. Nonuniform per-bucket dispatch
must use eager mode until we implement a structural graph fix.

## Why Phase 1 Missed This

Phase 1 mostly asked whether any offload fraction can be free or throughput
positive, and whether COTS pure-prefetch matches native prefetch at comparable
bytes. It did not isolate decode dispatch after keeping prefill off the CPU.

The original uniform dispatch validation mixed prefill and decode. CPU compute
in prefill is expensive enough to dominate the result. The decode-only harness
removed that contamination, but the new result went too far in the other
direction: the measured CPU decode cost is now inconsistent with the Phase 0
and native-worker timing math.

## Can The Hybrid Story Be Made True?

The invalid pure-CPU decode result no longer contradicts the thesis story. In
the true uniform control, `B=64`, `f=0.30` CPU is very slow, just as Phase 0
predicted. So the replacement claim ("just use CPU at B=64") should be
discarded.

The hybrid story is still plausible, but the current graph implementation
cannot validate per-bucket hybrid dispatch because bucket-varying geometry is
compiled away. To make the story true in the production path, one of these
runtime changes is required:

1. Compile/capture separate COTS graph variants for each routing geometry, or
   at least for each phase/bucket class.
2. Move CPU/prefetch routing geometry fully behind an out-of-graph structural
   boundary so the compiled graph always contains the correct submit/sync,
   prefetch, and scatter surfaces for the active bucket.
3. Restrict graph mode to uniform routing and use eager only for exploratory
   nonuniform dispatch until (1) or (2) lands.

Once graph validation is structurally correct, the Planner should search the
profile-backed regime again. Phase 0 says `B=64`, `f=0.30` pure CPU is far
outside the free zone, while pure prefetch is also expensive at high depth.
That is exactly the kind of regime where a calibrated hybrid policy may help,
but only after the benchmark is counter-verified and token-parity checked.

The near-term thesis-safe framing is:

1. Prefetch is expensive at high offload depth in this graph validation cell.
2. CPU decode at large `B` and large slice is expensive according to Phase 0,
   direct native-worker timing, and the true uniform control.
3. The previous `1.155x` CPU endpoint is invalid because the CPU branch was
   absent from the compiled graph.
4. The Planner should remain phase-aware and module-aware, but graph validation
   needs structural support for per-bucket routing before drawing the dispatch
   optimum.

The module-isolation numbers motivated this module-specific decode probe:

```text
QKV: pure prefetch or near-prefetch
MLP: pure CPU
```

This could not be represented by the previous runtime surface, because
`cots_dispatch_table` applied one pair to every enabled module. The investigation
added a module-keyed dispatch table, for example:

```json
{
  "qkv": {"64": [0.0, 0.30]},
  "mlp": {"64": [0.30, 0.0]}
}
```

Missing module keys fall back to the existing bucket table. The existing native
slab machinery already keys work by `(layer, bucket, op_kind)`, so this was
Python-side geometry/config plumbing, not a C++ runner rewrite. This hook is
worth keeping for the Planner, even though the first direct probe did not make
isolated decode hybrid win.

## Planner Implications

For the current checkpoint:

- Do not use the `1.155x` pure-CPU graph decode result as Planner ground truth.
- Keep endpoint optima allowed, but require native counters and token parity
  before accepting any graph-mode offload latency cell.
- Treat QKV and MLP separately in the profile model; the prefetch path clearly
  has module-specific costs.
- Add model terms for prefetched-slice GPU work, especially MLP addmm, not just
  H2D bandwidth.
- In graph mode, reject nonuniform COTS routing geometry until the compiled
  graph can represent bucket-varying CPU/prefetch branches correctly.

Recommended next experiments:

```text
1. Build a graph-safe per-bucket routing strategy: per-geometry graph variants
   or an out-of-graph structural routing boundary.
2. Repeat module-specific dispatch at smaller `f_cpu_store` values after the
   graph path is counter-verified.
3. Use Nsight only after counters confirm CPU work is actually being submitted,
   waited on, and consumed by parity-checked logits.
```

For now, the honest story is that the validation harness exposed a real
prefetch problem and a confirmed graph specialization bug. The Planner story
should wait for structurally correct per-bucket graph validation.
