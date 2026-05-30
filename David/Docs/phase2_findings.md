# Phase 2 Findings: Hybrid CPU/GPU KV Cache

Last updated: 2026-05-30

This document is the cleaned Phase 2 closeout. The detailed trial log is kept in git history and the structured measurement artifact at `David/Benchmarks/phase2/phase2_kv_measurement_cells.json`.

## Executive Summary

Phase 2 implementation is generally complete. The final system implements a transparent hybrid KV cache for decode-oriented workloads:

- GPU KV stores the shared prefix region before a block-aligned split point.
- CPU KV stores the suffix region at and after the split point.
- GPU prefix attention and CPU suffix attention are merged with online softmax state merge.
- The implementation supports both Qwen2.5-style and Llama3-style GQA shapes in the current BF16/head_dim128 envelope.
- The production path is cleaned up and committed in vLLM commit `039223087` and top-level commit `28acd3d`.

The performance result is positive but narrow. Hybrid KV can improve throughput in measured capacity-frontier cells, but it is not a universal rule that lower GPU memory or more CPU KV capacity increases throughput. The planner must use profiled cells or a model that captures both scheduler-wave capacity gain and suffix-path overhead.

The current supported claim is:

> Hybrid CPU/GPU KV can increase decode throughput when the split moves the scheduler to a larger useful running wave and the added CPU suffix attention, artifact movement, and merge overhead is smaller than that wave gain.

The current unsupported claims are:

- Lower GPU memory utilization always makes hybrid better.
- More KV capacity alone predicts a win.
- Fewer CPU suffix blocks alone predicts a win.
- Hybrid should be enabled unconditionally.

## General Cost/Benefit Statement

The general Phase 2 decision is not simply CPU KV versus GPU KV. It is a placement tradeoff:

```text
enable hybrid KV if
    saved weight-offload cost
  + enabled model-fit value
  + possible scheduler/batch-wave gain
  > CPU KV suffix cost

where CPU KV suffix cost includes
    CPU suffix attention
  + Q/K/V D2H staging
  + suffix output/LSE artifact movement or UVA exposure
  + online-softmax merge
  + host/synchronization overhead
  + any scheduler fragmentation from mixed prefix/suffix work
```

The measurements in this document isolate the third benefit term: possible batch-wave gain from freeing GPU KV capacity while model weights are already GPU-resident. That is why the current headline claim is narrow and profile-gated. The first two benefit terms, saved weight-offload cost and enabled model fit, are planner-level effects and should be evaluated in the next phase when hybrid KV is combined with weight placement.

This distinction matters for larger models. If CPU KV makes enough GPU memory available to keep more weights resident, reduce weight offload, or fit a model that otherwise cannot launch, then hybrid KV can be valuable even when the pure KV-capacity throughput comparison is near-tie or negative. Conversely, if the freed GPU memory is not converted into less weight movement, model fit, or a larger useful running wave, hybrid KV only adds suffix-path overhead.

## Runtime Contract

Phase 2 uses a block-aligned split point `x = split_blocks * block_size`.

For a token row at absolute position `pos`:

- If `pos < x`, the row is GPU-prefix only.
- If `pos >= x`, the row has a GPU prefix state over `[0, x)` and a CPU suffix state over `[x, pos]`.
- Prefix and suffix states are merged with online softmax using output plus per-head LSE.

This is a per-row, position-based rule. There is no final CPU-request/GPU-request abstraction in the runtime path.

For prefill:

- If the prompt stays before the split, no CPU suffix attention is active during prefill.
- If the prompt crosses the split, rows at positions `>= x` naturally use the CPU suffix path.
- The current split-crossing prefill path is a correctness path. It is row-expanded and not optimized as a custom FlashAttention-like CPU prefill kernel.

For decode:

- Current-token Q/K/V are produced on GPU.
- Q and K/V for suffix-active rows are staged to pinned CPU buffers.
- The native prepared CPU suffix runner reads staged buffers and CPU KV metadata.
- CPU suffix outputs and LSE remain in pinned CPU memory and are consumed by GPU merge through UVA views.
- Staging-buffer reuse is protected by CUDA events.

## Production Surface

The cleaned production surface keeps one main fast path:

- Coalesced GPU prefix attention is default-on for mixed prefix/suffix batches.
- CPU suffix attention runs through the native prepared suffix runner.
- Indexed merge/writeback is used when a precomputed full-prefix output is available with suffix row indices.
- The staged-query skip avoids a GPU `index_select` when the CPU suffix worker already has a valid staged CPU query/QKV buffer.
- Scheduler-side suffix-active admission pause was removed after negative measurement.
- Planner-side automatic hybrid selection was removed. Hybrid remains profile/config driven.

The main implementation areas are:

| Area | Files |
|---|---|
| Hybrid attention runtime | `vllm/v1/attention/backends/cots_hybrid_attention.py`, `vllm/v1/attention/backends/flash_attn.py` |
| CPU/GPU KV metadata and staging | `vllm/v1/worker/cots_hybrid_kv.py`, `vllm/v1/worker/gpu_model_runner.py` |
| Online softmax merge | `vllm/v1/attention/ops/merge_attn_states.py`, `csrc/attention/merge_attn_states.cu` |
| Scheduler cleanup | `vllm/v1/core/sched/scheduler.py` |
| Measurement helpers | `David/Benchmarks/phase2/*.py`, `phase2_kv_measurement_cells.json` |

## Diagnostic Surface

The cleanup kept observation tools, not policy variants.

Kept diagnostics:

| Knob / artifact | Purpose |
|---|---|
| `VLLM_COTS_SUFFIX_COUNTERS=1` | Suffix rows, worker thread count, D2H bytes, artifact bytes, native suffix timing. |
| `VLLM_COTS_HYBRID_CUDA_TIMING=1` | GPU prefix and merge CUDA-event timing. Perturbs throughput, so use for attribution only. |
| `VLLM_COTS_HYBRID_SUBMIT_TIMING=1` | Metadata and submit overhead timing. |
| `VLLM_COTS_DIAG`, `VLLM_COTS_COUNTERS`, `VLLM_COTS_NVTX`, wait-kernel diagnostics | Broader COTS Phase 1/2 attribution tools. |
| `VLLM_COTS_HYBRID_COALESCED_PREFIX=0` | Temporary kill switch for the central coalesced-prefix fast path. |
| `analyze_kv_policy_cells.py` | Offline summary of positive/negative policy cells. |
| `run_capacity_window_sweep.py` | Reproducible capacity-window sweeps. |
| `benchmark_prefill_spillover_e2e.py` | Split-crossing prefill behavior probe. |

Removed or intentionally not kept as runtime policy variants:

- Public `VLLM_COTS_HYBRID_EARLY_SUFFIX_SUBMIT` knob.
- Suffix-active admission pause.
- Artifact-copy return path as a default or policy variant.
- Real-suffix ablation variants.
- Indexed-merge opt-in policy knob.
- Planner-side automatic hybrid policy.

The internal early suffix submit path remains for all-suffix decode when runtime conditions are safe. The coalesced-prefix mixed-row path passes precomputed prefix output and intentionally does not use early submit.

## Validation

Focused validation after cleanup:

```text
/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_merge_attn_states.py::test_merge_attn_states_indexed \
  /TTC/vllm/tests/kernels/attention/test_cots_hybrid_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py \
  /TTC/vllm/tests/v1/core/test_scheduler.py::test_cots_hybrid_suffix_decode_can_mix_waiting_prefill -q
# 60 passed, 16 warnings
```

Small no-stats E2E smoke after cleanup:

```text
Qwen/Qwen2.5-7B-Instruct
batch=4, prompt608, total640, split608, gpu_memory_utilization=0.67
hybrid initialized CPU KV store and generated successfully
out_tok_s=155.238
```

Numerical validation has focused on kernel parity, online-softmax merge equivalence, suffix attention correctness, metadata/staging behavior, and small E2E generation smoke. Exact greedy-token parity is not treated as the correctness criterion because BF16 attention order and GPU/CPU merge order can legitimately perturb logits near decision boundaries. Forced-context and kernel-level tests are the relevant checks.

## Headline Throughput Cells

The current analyzer promotes three cells at the default `win_margin=0.01`:

| Cell | Effective capacity gain | Active CPU suffix blocks | GPU-only out tok/s | Hybrid out tok/s | Hybrid/GPU |
|---|---:|---:|---:|---:|---:|
| Llama3.1-8B eager, mem0.72, total768, split640 | 19.9% | 8 | 2278.8 | 2347.7 | 1.030x |
| Llama3.1-8B eager, mem0.80, total768, split736 | 4.3% | 2 | 3706.8 | 3769.7 | 1.017x |
| Qwen2.5-7B graph, mem0.67, total768, split672 | 6.6% | 6 | 2178.3 | 2246.7 | 1.031x |

Near-ties below the 1% promotion margin:

| Cell | Effective capacity gain | Active CPU suffix blocks | Hybrid/GPU | Decision |
|---|---:|---:|---:|---|
| Llama3.1-8B graph, mem0.80, total768, split736 | 3.4% | 2 | 1.004x | Keep profile-gated / GPU-only by default. |
| Qwen2.5-7B eager, mem0.67, total768, split704 | 9.1% | 4 | 1.006x | Keep profile-gated / GPU-only by default unless margin is relaxed. |

These are small synthetic wins. They demonstrate feasibility, not a universal policy.

## Negative Controls

Capacity gain alone fails as a rule. Important negative cells:

| Cell | Effective capacity gain | Active CPU suffix blocks | Hybrid/GPU | Interpretation |
|---|---:|---:|---:|---|
| Llama eager, mem0.72, total768, split736 | 4.3% | 2 | 0.956x | Same memory as a positive split640 cell, but split736 does not move enough scheduler capacity. |
| Llama eager, mem0.75, total768, split736 | 4.3% | 2 | 0.996x | Near miss. Same fixed split is not enough below the useful wave boundary. |
| Llama eager, mem0.80, total896, split864 | 3.7% | 2 | 0.945x | Longer generated window keeps suffix cost active for too long. |
| Llama eager, mem0.80, total1024, split960 | 6.7% | 4 | 0.938x | Capacity gain does not repay suffix-path overhead. |
| Qwen graph, mem0.67, total896, split736 | 13.3% | 10 | 0.942x | Large capacity gain loses because suffix-active interval is too large. |
| Qwen graph, mem0.67, total1024, split864 | 10.3% | 10 | 0.834x | Longer sequence strongly negative. |

The total896 and total1024 results are the key boundary for the thesis claim: longer generated buckets can erase short-window capacity-frontier wins.

## Why The Positive Cells Win

The positive cells are scheduler-wave wins. They are not smooth capacity wins.

Llama mem0.72/split640:

- GPU-only effective capacity: 9.98x.
- Hybrid effective capacity: 11.97x.
- First stats-enabled wave changed from 40 running / 472 waiting to 147 running / 365 waiting.
- Most active hybrid windows were all-suffix rows, not mixed prefix/suffix rows.
- The larger wave paid for 8 CPU suffix blocks/request in this short total768 bucket.

Llama mem0.80/split736:

- Hybrid only adds two CPU suffix blocks/request.
- The 0.80 memory point crosses a useful running-wave boundary that nearby 0.79 does not.
- The two-block suffix cost is small enough to preserve a modest throughput win.

Qwen graph mem0.67/split672:

- Graph mode reduces available GPU KV capacity enough that Qwen becomes capacity-sensitive.
- split704 has less suffix work but too little capacity gain and loses.
- split672 moves to a larger useful wave and wins despite more CPU suffix blocks.

This is the central mechanism: the planner must reason about discrete admission waves, not just continuous capacity percentages.

## Model-Specific Behavior

Llama and Qwen react differently because their KV footprints and baseline capacity differ.

Llama3.1-8B has more KV pressure in the tested settings, so hybrid can matter at both:

- mem0.72 with an earlier split640, where capacity gain is large enough.
- mem0.80 with split736, where a small two-block suffix is enough after crossing a wave.

However, fixed split736 loses at mem0.72, 0.74, 0.75, 0.78, and 0.79. The fixed split does not automatically become better at lower memory.

Qwen2.5-7B has a smaller KV footprint and higher GPU-only capacity in many settings. At mem0.80, GPU-only is already high-capacity and hybrid overhead is not repaid. At mem0.67 graph mode, graph memory pressure shifts the capacity frontier enough for split672 to win.

The model-specific conclusion is:

- Llama benefits when the split lands on a useful scheduler wave and suffix length stays short enough.
- Qwen benefits only when the runtime mode and memory budget make it genuinely capacity-limited.
- Neither model supports a monotonic lower-memory rule.

## Prefill Spillover

Split-crossing prefill was profiled separately with Qwen2.5-7B, prompt704, total705, split672, unique prompts, eager mode, prefix caching disabled.

Clean B8 profile:

| Mode | Elapsed s | Prompt tok/s | Notes |
|---|---:|---:|---|
| GPU-only | 0.513 | 10971.5 | Baseline. |
| Hybrid spillover | 0.679 | 8293.9 | 32 prefill tokens/request spill to CPU suffix. |

The hybrid spillover run logged:

- 28 hybrid decode calls.
- 131712 GPU-prefix rows.
- 6272 CPU-suffix rows.
- 45.0 MiB Q D2H.
- 12.8 MiB KV D2H.
- 45.7 MiB artifact H2D.
- About 14.7 ms CPU suffix attention.
- 24 CPU worker threads observed.

Conclusion: current split-crossing prefill is correct and natural, but not a throughput path for this prefill-heavy diagnostic. A custom CPU prefill kernel could reduce row-expanded suffix prefill overhead, but it would not remove the larger split-path costs and is deferred.

## Bottleneck Summary

There is no single universal bottleneck. The limiting term depends on the cell.

Observed overhead components:

- CPU suffix attention.
- Q/K/V D2H staging.
- Pinned CPU suffix output/LSE exposure to GPU through UVA.
- Online-softmax merge.
- Host callback and prepared suffix runner envelope.
- Scheduler wave shape and waiting/running mix.
- Mixed prefix/suffix row fraction.

Rejected or low-value optimization attempts:

| Attempt | Result |
|---|---|
| Suffix-active admission pause | Negative. It stranded waiting requests and shrank useful forward batches. Removed. |
| Coalesced CPU-first overlap | Mixed/negative. Launch order is second-order and does not rescue earlier splits. Public knob removed. |
| Explicit suffix artifact H2D copy | Negative versus UVA merge path. |
| Pushing more blocks to CPU | Often negative because suffix work grows faster than useful capacity. |
| Custom prefill kernel | Deferred. Correctness path works; current throughput claim is decode capacity-frontier. |

Useful optimizations that remain:

| Optimization | Why it stayed |
|---|---|
| Coalesced GPU prefix | Converts mixed prefix/suffix batches into one GPU prefix attention plus suffix-row merge. Central to wins. |
| Indexed merge/writeback | Removes compact gather/writeback envelope for precomputed full-prefix rows. |
| Staged-query skip | Avoids unnecessary GPU query compaction when CPU staged query/QKV is already valid. |
| Generic GQA CPU suffix kernel | Supports Qwen and Llama shapes in the current envelope. |
| Default 24 suffix threads | Best observed default in the current target environment. |

## Planner Implications

Automatic hybrid selection should be profile-gated.

A planner profile key needs at least:

- Model family and attention shape.
- Execution mode: eager vs graph.
- `gpu_memory_utilization` and realized post-load GPU KV token count.
- Prompt tokens, total tokens, and generated-length bucket.
- Split tokens or split blocks.
- Effective capacity for GPU-only and hybrid.
- Active CPU suffix blocks and observed suffix row distribution.
- Measured throughput, not just predicted capacity.
- Weight-offload settings, once Phase 1/Phase 2 interaction is reintroduced.

Recommended planner policy for now:

1. Use GPU-only by default.
2. Enable hybrid only for measured cells that clear a win margin.
3. Treat near ties as GPU-only unless repeated or required for admission capacity.
4. Do not infer from lower memory alone.
5. Do not use suffix-active runtime pausing as a policy substitute.

Weight-offload interaction is deferred. Larger non-fit-in-GPU model strategy is also deferred, because that changes the objective: CPU KV may free GPU memory for weights, not only increase request capacity.

## Artifacts

Primary artifacts:

| Artifact | Purpose |
|---|---|
| `David/Benchmarks/phase2/phase2_kv_measurement_cells.json` | Structured source for measured positive, negative, and diagnostic cells. |
| `David/Benchmarks/phase2/analyze_kv_policy_cells.py` | Prints normalized policy-cell table and promoted cells. |
| `David/Benchmarks/phase2/run_capacity_window_sweep.py` | Resumable synthetic capacity-window sweep. Defaults include split640. |
| `David/Benchmarks/phase2/benchmark_ratio_e2e.py` | Main synthetic GPU-only vs hybrid throughput benchmark. |
| `David/Benchmarks/phase2/benchmark_prefill_spillover_e2e.py` | Split-crossing prefill profile. |

Important commits:

| Repo | Commit | Meaning |
|---|---|---|
| `/TTC/vllm` | `039223087` | Cleaned Phase 2 hybrid KV runtime surface. |
| `/TTC` | `28acd3d` | Recorded Phase 2 hybrid KV boundary baseline and artifacts. |

## Current Status

Implementation status: generally done for Phase 2.

Ready to move on:

- Phase 2 closeout writing.
- Planner/profile-gated selection design.
- Phase 3 end-to-end benchmarking.
- Later weight-offload interaction policy.

Deferred engineering:

- Custom CPU prefill kernel.
- Broader model support beyond current BF16/GQA/head_dim128 envelope.
- General planner model for larger models that do not fit GPU memory.
- Deeper CPU suffix kernel optimization if future profiles require a wider throughput window.
