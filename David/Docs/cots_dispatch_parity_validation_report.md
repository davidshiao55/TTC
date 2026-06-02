# COTS Dispatch Forced-Context Parity Validation

Date: 2026-06-02

Status: validation pass after the nonuniform graph-route fix and the
graph/eager sync-mode investigation.

## Executive Summary

The forced-context parity pass supports the structural correctness of the new
COTS dispatch route implementation:

- forced outputs matched the reference continuation for every tested arm;
- every expected logits record was present;
- graph and eager COTS CPU routes both executed real native CPU work;
- CPU-route drift is in the same envelope as pure-prefetch COTS drift.

Strict free-generation token parity remains too brittle as a correctness oracle.
The important observation is that pure-prefetch COTS already introduces small
logprob drift and a few top-1 flips versus no-offload. The decode-CPU COTS route
does not introduce a separate large error beyond that split/offload baseline.

## Harness

New probe:

```text
/TTC/David/Benchmarks/planner/check_cots_forced_context_parity.py
```

Method:

1. generate a no-offload reference continuation;
2. force no-offload and COTS arms through that exact token sequence;
3. record raw logprobs/top-k sets before the forcing mask;
4. compare no-offload versus COTS, and CPU-route COTS versus pure-prefetch COTS.

The probe reuses the Phase 2 logits processor:

```text
/TTC/David/Benchmarks/phase2/phase2_forced_logits_proc.py
```

## Main B=64 Validation

Configuration:

```text
model: Qwen/Qwen2.5-7B-Instruct
dtype: bfloat16
batch: 64
prompt_tokens: 8
decode_tokens: 4
f_cpu_store: 0.30
dispatch_layout: decode-only
cpu_threads: 24
topk: 20
```

Graph result root:

```text
/TTC/results/planner/cots_forced_context_parity_20260602/b64_decode_only_graph_f030
```

| graph candidate | forced outputs | records | top1 same | forced logprob delta max / mean / p95 | top20 jaccard mean |
|---|---|---:|---:|---:|---:|
| pure prefetch | pass | 256/256 | 253/256 | 0.2184 / 0.0269 / 0.0761 | 0.9672 |
| decode CPU | pass | 256/256 | 254/256 | 0.2184 / 0.0269 / 0.0866 | 0.9672 |

Graph CPU route counters after capture reset:

```text
submit_count_qkv = 112
submit_count_mlp = 112
worker_run_count = 224
```

Direct graph CPU-vs-prefetch comparison:

```text
top1 same = 255/256
forced logprob delta max / mean / p95 = 0.0842 / 0.0143 / 0.0574
top20 jaccard mean = 0.9848
```

Eager result root:

```text
/TTC/results/planner/cots_forced_context_parity_20260602/b64_decode_only_eager_f030
```

| eager candidate | forced outputs | records | top1 same | forced logprob delta max / mean / p95 | top20 jaccard mean |
|---|---|---:|---:|---:|---:|
| pure prefetch | pass | 256/256 | 252/256 | 0.1594 / 0.0290 / 0.0968 | 0.9667 |
| decode CPU | pass | 256/256 | 250/256 | 0.1594 / 0.0290 / 0.1013 | 0.9697 |

Eager CPU route counters:

```text
submit_count_qkv = 112
submit_count_mlp = 112
worker_run_count = 224
```

Direct eager CPU-vs-prefetch comparison:

```text
top1 same = 254/256
forced logprob delta max / mean / p95 = 0.1008 / 0.0174 / 0.0670
top20 jaccard mean = 0.9815
```

## Smoke Controls

No-offload smoke:

```text
/TTC/results/planner/cots_forced_context_parity_smoke_20260602/none_graph
```

`none_graph` versus `none_eager` at B=2, prompt 4, decode 2:

```text
forced outputs: pass
records: 4/4
top1 same: 4/4
forced logprob delta max / mean / p95 = 0.0469 / 0.0359 / 0.0469
top20 jaccard mean = 1.0
```

COTS graph smoke:

```text
/TTC/results/planner/cots_forced_context_parity_smoke_20260602/cots_graph
```

B=2, prompt 4, decode 2, `f_cpu_store=0.05`, decode-only:

| candidate | forced outputs | records | top1 same | forced logprob delta max / mean / p95 |
|---|---|---:|---:|---:|
| pure prefetch graph | pass | 4/4 | 4/4 | 0.0435 / 0.0227 / 0.0435 |
| decode CPU graph | pass | 4/4 | 4/4 | 0.0774 / 0.0345 / 0.0774 |

CPU route smoke counters:

```text
submit_count_qkv = 56
submit_count_mlp = 56
worker_run_count = 112
```

## Interpretation

This pass validates the graph nonuniform dispatch fix at the correctness level
we can defend:

- Graph mode is no longer silently replaying a graph with missing COTS CPU
  custom-op work.
- Eager and graph both exercise the native CPU runner for decode-CPU dispatch.
- CPU dispatch is numerically close to pure-prefetch dispatch under the same
  COTS split geometry.
- The remaining top-1 flips also appear in pure-prefetch COTS, so they should
  be treated as COTS split numerical drift rather than a dispatch-bucket or
  route-signature bug.

If exact greedy token parity is required, the next investigation should target
COTS split numerical equivalence itself, especially row-parallel reductions and
operation ordering. That is separate from the graph nonuniform routing bug.
