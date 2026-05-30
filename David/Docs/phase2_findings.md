# Phase 2 Findings: Hybrid CPU/GPU KV First Attempt

This note records the evidence for the Phase 2 planner-split hybrid KV runtime.
It is chronological: older performance sections are kept as attribution history,
while this top section and the final completion section describe the current
supported contract. Phase 2 remains a runtime mechanism only; planner split
selection and CPU-pool sizing are intentionally deferred.

## Current Runtime Contract

Supported thesis envelope for this pass:

- Single GPU, decoder-only, full attention.
- BF16 model dtype and BF16 KV dtype.
- Generic GQA BF16 attention shapes with head size 128 and
  `num_query_heads / num_kv_heads <= 8`; validated for Qwen2.5-7B
  (`28q/4kv`) and Llama3.1-8B (`32q/8kv`).
- Static block-aligned split:
  - GPU owns global blocks below `cots_kv_split_blocks`.
  - CPU owns global blocks at and above `cots_kv_split_blocks`, with suffix-local block tables passed to the CPU runner.
- COTS hybrid KV acts as a transparent CPU suffix extension of GPU KV for this envelope:
  - GPU prefix and CPU suffix prefix-cache hits are counted as one logical prefix-cache result.
  - Full CPU suffix blocks can remain resident after finish and be reused by later requests.
  - CPU suffix eviction only chooses zero-ref cached blocks under CPU-pool pressure.
  - Live blocks are protected; allocation failure falls back to vLLM's normal preemption/recompute path.
  - `reset_prefix_cache()` resets both tiers and refuses while either tier has live references.
- Attention after the split runs GPU prefix attention, CPU suffix attention with LSE, and GPU online-softmax merge.
- CPU suffix prefill is now supported through the simple row-expanded decode-style path. Large row-expanded suffix prefill chunks that exceed decode staging capacity use the native eager overflow task; full graph capture for those overflow chunks is deferred.
- Eager and COTS piecewise graph modes are both functional for decode. Hybrid KV forces the same split-graph policy used by COTS CPU runtime work.
- Native vLLM `kv_offload`, KV transfer/connectors, non-single-GPU modes, encoder-decoder models, non-BF16 model/KV dtype, and async scheduling are rejected at startup with clear diagnostics.

Deliberately not in Phase 2: CPU-aware admission, planner split selection, dynamic promotion/eviction, GPU tail fallback, native vLLM KV-offload integration, CPU-produced QKV direct handoff from Phase 1, non-BF16/non-128-head-dim CPU suffix kernels, and graph-captured suffix-prefill overflow buckets.

## Optimizations Landed In This Pass

- Vectorized CPU KV scatter replaced per-request Python `copy_` loops.
- CPU value staging was fixed to use pinned memory. The earlier `empty_like`
  allocation silently produced non-pinned memory and disabled async K/V staging.
- Q/K/V staging now uses one shared D2H stream/event.
- GQA Q/K/V tensors are staged through one combined contiguous
  `[B, num_query_heads + 2 * num_kv_heads, 128]` D2H copy when their
  storage layout allows it.
- Suffix slot masking has a CPU-position fast path for all-prefix/all-suffix
  decode steps.
- All-active CPU suffix K/V scatter now uses a thesis-owned C++ memcpy fast
  path instead of PyTorch advanced indexing.
- CPU suffix attention now pre-converts each task's query-head group to FP32
  once and reuses it across suffix tokens, avoiding repeated BF16 upcasts in
  the dot-product loop.
- CPU suffix attention now vectorizes the final FP32-to-BF16 output store with
  AVX2 integer operations while preserving round-to-nearest-even conversion.
- CPU suffix attention now processes QK logits two suffix tokens at a time
  within each cache block, reusing the per-KV-group FP32 query-head vectors
  across both tokens. This is a small parity-safe form of the same tiling idea used in
  NEO's ISPC CPU attention path, adapted to the COTS BF16/AVX2 kernel.
- Prepared native suffix attention tasks can now scatter current-token K/V
  directly from the staged generic GQA QKV artifact before running CPU suffix
  attention. This removes the Python-side `_finish_staged_kv_cache_update` from
  the native-prepared path and is the first graph-substrate step for suffix KV
  update plus CPU attention in one stream-ordered host callback.
- The CPU suffix attention and scatter custom ops now use generic GQA naming;
  the old Qwen-named custom-op and runner entry points were removed.
- Added `benchmark_prepared_suffix_runner.py` to measure direct,
  direct-scatter, eager-prepared, eager-prepared-scatter, and CUDA-graph replay
  prepared-scatter variants at the suffix-runner boundary.
- Added a focused test for the combined-QKV staging fast path.
- Added focused parity coverage for the C++ CPU suffix K/V scatter fast path.
- Added focused coverage that the prepared native suffix runner scatters QKV
  before attention and matches the direct scatter-plus-attention reference.
- Added CUDA graph replay coverage for the prepared native suffix runner:
  replayed host callbacks consume updated CPU QKV contents through the captured
  static task pointers.

An experiment that replaced the CPU suffix kernel's per-task probability
`std::vector` with stack workspace was tested and reverted. It preserved parity
but did not improve the target workload.

A follow-up attempt to raise the stack workspace threshold from 128 to 256
tokens for the 608:160 case was also tested and reverted. B=256 hybrid
throughput stayed flat (`3122 -> 3118 out tok/s`), so heap allocation above
128 tokens is not the E2E limiter.

A streaming online-softmax CPU suffix kernel was also tested and reverted. It
passed the focused suffix and hybrid attention parity tests, but did not improve
the long-suffix target and regressed the short-suffix regime.

A token-major value-accumulation pass was tested and reverted. It improved the
isolated long-suffix CPU op, but regressed the short-suffix regime and reduced
the measured B=256, 608:160 E2E throughput.

Moving CPU block-table lookup from per-token to per-block in the suffix kernel
was also tested and reverted. It was neutral at short suffix and slightly worse
at the long-suffix stress point, so it is not worth carrying as Phase 2
complexity.

Replacing natural-log softmax with mathematically equivalent base-2 softmax
(`exp2/log2` plus scale conversion) was tested and reverted. It passed focused
parity, but the split672-relevant suffix case regressed (`B=512`, suffix96, 24
threads: `1.863 -> 1.884 ms`) and the short-suffix case also regressed
(`B=512`, suffix32, 32 threads: `0.284 -> 0.299 ms`).

A matching two-token tile in the value-accumulation pass was also tested and
reverted. It passed focused parity and improved long/split672 suffix
microbenchmarks (`B=256`, suffix160: `1.103 -> 1.059 ms`; `B=512`, suffix96:
`1.863 -> 1.839 ms`), but regressed the short-suffix case (`B=512`, suffix32:
`0.284 -> 0.310 ms`) and the integrated split672 run stayed inside the
existing QK-only band (`3456 out tok/s`). The extra hot-loop branch is not
worth carrying without a clearer E2E win.

Two additional low-level hypotheses were tested and rejected:

- Pinned versus pageable CPU tensors do not explain the suffix attention gap.
- Request-granularity CPU parallel tasks are neutral versus the existing
  `(request, kv_head)` task split.
- Explicit E2E `OMP_NUM_THREADS` tuning does not recover the gap; the default
  environment already uses enough PyTorch intra-op parallelism.
- Transparent-hugepage advice does not provide an actionable CPU KV cache
  streaming win.

## Performance Snapshot

Environment:

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Mode: vLLM V1, eager, single GPU
- Weight offload: disabled for these measurements (`f_cpu_store=0`)
- Sampling: deterministic, `temperature=0.0`, `ignore_eos=True`
- Prompts are identical token-id prompts to exercise prefix sharing.

### Short Suffix, 608:32

`prompt_len = split_tokens = 608`, `max_tokens = 32`.

| Active batch | Hybrid out tok/s | GPU-only out tok/s | Hybrid / GPU-only |
|---:|---:|---:|---:|
| 64 | 2339 | 2877 | 0.81x |
| 128 | 3445 | 4593 | 0.75x |
| 256 | 4364 | 5979 | 0.73x |

Earlier same-case hybrid throughput was about `1516 tok/s`, so the local
runtime changes are material. They do not make hybrid faster than GPU-only when
GPU-only can still use prefix caching and fit the workload.

Current-code rerun after the C++ scatter fast path:

| Active batch | Hybrid out tok/s | GPU-only out tok/s | Hybrid / GPU-only |
|---:|---:|---:|---:|
| 64 | 2319 | 2821 | 0.82x |
| 512 | 4332 | 5919 | 0.73x |
| 1024 | 4332 | 5916 | 0.73x |

The B=512/B=1024 plateau is important: hybrid does not recover the gap by
simply admitting a larger active batch for this short-suffix case. The current
path becomes CPU/UVA-bound while GPU-only remains saturated and still benefits
from prefix caching.

### Capacity-Pressure Suffix, 608:160

`prompt_len = split_tokens = 608`, `max_tokens = 160`.

| Active batch | GPU-only out tok/s | Hybrid out tok/s | Hybrid / GPU-only |
|---:|---:|---:|---:|
| 256 | 5074 | 3122 | 0.62x |
| 512 | 5164 | 3114 | 0.60x |

After the CPU suffix query-preconvert optimization, the B=256 hybrid result is
`3150 out tok/s`. This is directionally positive but not a material closure of
the gap; the architecture is still dominated by the CPU suffix/UVA path once the
suffix reaches 160 tokens.

B=256 suffix-length sweep after the retained query-preconvert kernel:

| Suffix tokens | Hybrid out tok/s | GPU-only out tok/s | Hybrid / GPU-only |
|---:|---:|---:|---:|
| 64 | 3853 | 6354 | 0.61x |
| 96 | 3639 | 6682 | 0.54x |
| 128 | 3371 | 4807 | 0.70x |
| 160 | 3150 | 5074 | 0.62x |

The hybrid curve degrades steadily as suffix grows. The S=96 comparison is especially unfavorable because GPU-only still benefits strongly from shared-prefix caching. This reinforces that Phase 2 should be treated as a capacity mechanism until the CPU suffix path changes materially.

Planner split-placement sweep for fixed B=256, prompt=608, total=768:

| Split tokens | Max CPU suffix tokens | Hybrid out tok/s | Hybrid / GPU-only 5074 |
|---:|---:|---:|---:|
| 608 | 160 | 3150 | 0.62x |
| 640 | 128 | 3760 | 0.74x |
| 672 | 96 | 3662 | 0.72x |
| 704 | 64 | 5183-5196 | 1.02x |
| 736 | 32 | 4578 | 0.90x |

This is the strongest planner signal so far. Moving `x` later caps CPU suffix work and can recover throughput, but it spends GPU KV capacity on generated tokens before `x`. The split should therefore be a Planner decision constrained by both GPU KV capacity and the measured CPU suffix overlap window, not simply "cover the shared prompt and put all decode on CPU."

Larger requested batch at the best split found so far (`split=704`, `prompt=608`, `total=768`) does not yet show an actual throughput scaling win:

| Requested batch | Hybrid out tok/s | GPU-only out tok/s | Hybrid / GPU-only |
|---:|---:|---:|---:|
| 512 | 5180-5183 | 5151 | 1.01x |
| 1024 | 5182 | 5324 | 0.97x |

Iteration logging at B=512 showed both modes running in roughly similar waves instead of admitting the full requested batch concurrently. The current split704 path is therefore close to the GPU-only eager throughput plateau, but it has not yet converted the extra CPU KV storage into larger simultaneous decode capacity in this synthetic shared-prefix benchmark.

Tight-memory integrated check at `gpu_memory_utilization=0.68`, B=1024, `prompt=608`, `total=768`:

| Mode | Split | Weight offload | Out tok/s | Takeaway |
|---|---:|---|---:|---|
| GPU-only | n/a | none | 3315 | KV-starved baseline |
| Hybrid KV | 704 | none | 3117 | CPU suffix overhead exceeds capacity gain |
| GPU-only | n/a | COTS pure-prefetch `f=0.02` | 3778 | Phase 1 recovers capacity |
| Hybrid KV | 704 | COTS pure-prefetch `f=0.02` | 3478 | hybrid suffix cost is exposed under weight prefetch |
| Hybrid KV | 736 | COTS pure-prefetch `f=0.02` | 3423 | later split reduces CPU suffix but spends too much GPU KV |

Scheduler diagnostics at B=256 explain the tradeoff:

| Split | Final out tok/s | Running reqs in log snapshot | GPU KV usage | GPU blocks | CPU blocks | CPU wait/read/attn in log snapshot |
|---:|---:|---:|---:|---:|---:|---|
| 704 | 2937 | 115 | 97.6% | 727 | 162/4681 | 155.019/0.000/9.131 ms |
| 608 | 3135 | 256 | 39.3% | 293 | 2304/4681 | 13.623/0.031/47.962 ms |

This pins down the current bottleneck more sharply than the raw throughput table: split704 keeps CPU suffix small enough to be close to GPU-only speed, but it remains GPU-KV-limited. Split608 actually converts CPU KV into admission capacity, but the CPU suffix attention cost is then exposed. Closing the gap therefore needs either a faster/better-overlapped CPU suffix path or a scheduler policy that caps active CPU-suffix work while still using CPU KV as resident capacity.

Important correction after this diagnostic: the earlier B=512/B=1024 plateau was also limited by the default vLLM running-request cap. Raising max_num_seqs to 512 lets Phase 2 exercise the resident-capacity path:

| Mode | max_num_seqs | Split | Weight offload | Out tok/s | Scheduler behavior |
|---|---:|---:|---|---:|---|
| GPU-only, B=512 | 512 | n/a | none | 3099 | about 70-80 running, GPU KV about 99% |
| Hybrid KV, B=512 | 512 | 608 | none | 3244 | 512 running, CPU KV up to 4608/4681 |
| GPU-only, B=512 | 512 | n/a | COTS pure-prefetch f=0.02 | 3568-3596 | weight offload alone frees enough KV to win |
| Hybrid KV, B=512 | 512 | 608 | COTS pure-prefetch f=0.02 | 3177 | CPU suffix overhead dominates integrated path |

This is the first clean positive Phase 2 result: hybrid KV beats matched GPU-only by about 4.7% once max_num_seqs is high enough for the CPU pool to matter. But it does not yet beat Phase 1 weight-only prefetch in the same tight-memory setting. Planner output must therefore include max_num_seqs/admission capacity, not just kv_split_blocks.

Higher GPU memory budget check at B=512, max_num_seqs=512, prompt608,
total768, no weight offload:

| gpu_memory_utilization | Mode | Split | Out tok/s | Relative to GPU-only |
|---:|---|---:|---:|---:|
| 0.68 | GPU-only | n/a | 3099 | 1.00x |
| 0.68 | Hybrid KV | 608 | 3244 | 1.05x |
| 0.68 | Hybrid KV | 672 | 3188 | 1.03x |
| 0.75 | GPU-only | n/a | 5413.844 | 1.00x |
| 0.75 | Hybrid KV | 608 | 3256.871 | 0.60x |
| 0.75 | Hybrid KV | 672 | 4765.697 | 0.88x |

This confirms the planner boundary: when the GPU budget is tight enough that
GPU-only is KV-capacity-limited, CPU KV can improve throughput. Once the GPU
has enough KV capacity to keep the shared-prefix workload saturated, hybrid KV
becomes an overhead unless a later split can nearly eliminate CPU suffix work.
At 0.75, split672 is much better than split608, but it is still about 12%
below GPU-only.

The manual FastTTS planner interface now carries this launch-time knob:
per-engine `planner_config.{generator,verifier}.max_num_seqs` is emitted as the
vLLM `max_num_seqs` override. This is deliberately only a launch-time planner
output, not a runtime CPU-suffix active cap.

The synthetic E2E split-ratio harness used for these tables is now tracked at
`David/Benchmarks/phase2/benchmark_ratio_e2e.py`. Run it from
`/TTC/FastTTS-thesis` so Python resolves the thesis vLLM fork rather than the
parent namespace package.

A minimal round-robin CPU-suffix active cap was prototyped and rejected. It kept requests resident but scheduled only 384 CPU-suffix requests per step:

| Mode | Active suffix cap | Out tok/s | Verdict |
|---|---:|---:|---|
| Hybrid KV, no weight offload | none | 3244 | baseline positive Phase 2 case |
| Hybrid KV, no weight offload | 384 | 3119 | worse |
| Hybrid KV + COTS f=0.02 | none | 3177 | integrated baseline |
| Hybrid KV + COTS f=0.02 | 384 | 3059 | worse |

The reason is simple: scheduler-level capping reduces the GPU linear batch as well as CPU suffix attention. The saved CPU attention does not repay the lost GPU throughput. Do not keep this as a runtime knob; if we want to limit exposed CPU suffix work, it needs to happen inside the hybrid attention/CPU worker pipeline without shrinking the model-forward batch, or through Planner split choices.

Integrated split sweep at B=512, max_num_seqs=512, gpu_memory_utilization=0.68, COTS pure-prefetch f=0.02:

| Split | Max CPU suffix tokens | Out tok/s | Relative to weight-only 3596 |
|---:|---:|---:|---:|
| 608 | 160 | 3177 | 0.88x |
| 640 | 128 | 3363 | 0.94x |
| 656 | 112 | 3283 | 0.91x |
| 672 | 96 | 3410-3431 | 0.95x |
| 688 | 80 | 3232 | 0.90x |
| 704 | 64 | 3305 | 0.92x |

The best integrated point remains split672. Fine-grid checks at split656 and
split688 did not displace it. This closes some of the gap versus split608 but
still does not beat weight-only prefetch. A no-weight-offload split672 run was
also only `3188` out tok/s, so the integrated gap is not caused primarily by
weight prefetch interference. The split672 log sample showed the run starting
GPU-KV-limited (`Running=246`, `Waiting=85`, GPU KV `99.8%`, CPU KV
`0/4681`), then later using only a small CPU suffix (`Running=150`, GPU KV
`83.5%`, CPU KV `373/4681`, CPU wait/read/attn `11.283/0.029/13.964 ms`). So
the current best split is not deeply exploiting CPU KV; it is mostly picking
the smallest suffix that relieves just enough GPU KV pressure without making
CPU attention dominate.

After vectorizing the CPU suffix kernel's FP32-to-BF16 output store, the same
split672 integrated candidate improved slightly:

| Setting | Out tok/s |
|---|---:|
| Before vectorized output store | 3410-3431 |
| After vectorized output store | 3448-3451 |

This is a retained optimization because it is parity-safe and improves the
isolated long-suffix CPU op (`B=256`, suffix160, 32 threads: prior about
`1.248 ms`, now `1.155 ms`). It is not a structural fix: the integrated
candidate is still below the weight-only COTS reference (`3568-3596`).

After adding the two-token QK tile to the CPU suffix kernel, focused suffix
attention improved in the cases that matter:

| Case | Before QK tile | After QK tile |
|---|---:|---:|
| B=256, suffix160, 32 threads | 1.155 ms | 1.103 ms |
| B=512, suffix32, 32 threads | 0.341 ms | 0.284 ms |
| B=512, suffix96, 24 threads | 2.167 ms | 1.863 ms |

The split672 integrated candidate with COTS pure-prefetch f=0.02 measured
`3439` and `3468` out tok/s after the QK tile. This is not a structural
breakthrough, but it does not regress the prior `3448-3451` band and one repeat
slightly exceeds it. The deterministic split160 post-split task guard remained
matched between weight-only and hybrid (`2/4` vs `2/4`, all four requests
crossed the split with 224 post-split tokens).

A same-session weight-only COTS pure-prefetch f=0.02 run at the same B=512,
prompt608, total768, `gpu_memory_utilization=0.68`, and `max_num_seqs=512`
measured `3572` out tok/s. The current post-QK hybrid best band is therefore
about `96-97%` of the matched weight-only reference, leaving a measured
`3-4%` integrated gap in this tight-memory synthetic setting.

A follow-up non-timed repeat set after adding the wait-breakdown instrumentation
confirms the residual gap is reproducible and smaller than the earlier loose
band:

| Mode | Out tok/s repeats | Mean | Range |
|---|---:|---:|---:|
| Hybrid KV split672 + COTS f=0.02 | 3468.2, 3467.5, 3447.9 | 3461.2 | 3447.9-3468.2 |
| Weight-only COTS f=0.02 | 3569.7, 3565.4, 3541.9 | 3559.0 | 3541.9-3569.7 |

The mean hybrid/weight-only ratio is `97.25%`, or a `2.75%` remaining gap. The
ranges do not overlap, so the gap is still real; it is just narrow enough that
larger architectural changes need a clear lower-bound argument before they are
worth implementing.

The split-ratio harness now exposes `--enforce-eager true|false` for
execution-mode measurements. A matched graph-capable weight-only COTS f=0.02
run using `--enforce-eager false` measured `3553` out tok/s at the same B=512,
prompt608, total768, `gpu_memory_utilization=0.68`, and `max_num_seqs=512`
setting. This falls inside the eager weight-only repeat range (`3541.9-3569.7`).
vLLM selected piecewise CUDA graphs plus the COTS `wait_kernel` sync path, but
graph memory reduced available KV cache (`15,104 -> 14,592` GPU tokens in the
run). The current residual gap should therefore not be treated as a generic
eager-mode artifact; graph work only matters for Phase 2 if it specifically
removes the Q/QKV-ready synchronization exposed by hybrid attention.

The padded free-generation task probe at the actual split672 candidate is not
a stable parity guard. A weight-only versus hybrid run produced `0/4` versus
`1/4` correct extracted answers, but the immediate weight-only versus
weight-only control produced the same two answer sets in the opposite role
(`1/4` then `0/4`). Treat this probe only as a post-split execution smoke test
for heavily padded prompts.

A tracked fixed-context parity guard now lives at
`David/Benchmarks/phase2/check_hybrid_forced_context_parity.py`. It first
records a GPU-only reference continuation, then forces GPU-only and hybrid KV
through the same tokens while comparing raw logprobs. At split672 with
COTS f=0.02:

| Probe | Post-split top-1 same | Post-split forced-token logprob delta | Post-split top20 Jaccard |
|---|---:|---:|---:|
| GPU-only vs hybrid | 383/384 | max 0.0873, mean 0.00337, p95 0.0175 | mean 0.9716 |
| GPU-only vs GPU-only control | 383/384 | max 0.0874, mean 0.00272, p95 0.0119 | mean 0.9734 |

This is the current quality guardrail for local hybrid-attention changes:
hybrid drift is inside the same-mode eager/logits-processor control envelope,
and forced outputs were identical in both runs.

A post-QK-tile split sweep around the prior integrated optimum confirms that
the Planner split recommendation does not move:

| Split | Max CPU suffix tokens | Out tok/s |
|---:|---:|---:|
| 640 | 128 | 3392 |
| 656 | 112 | 3305 |
| 672 | 96 | 3439-3468 |
| 688 | 80 | 3241 |
| 704 | 64 | 3286 |

The QK tile improves the CPU suffix primitive, but not enough to make earlier
splits with larger CPU suffix budgets better. The current best integrated
planner candidate remains split672 for this B=512, prompt608, total768,
gpu_memory_utilization=0.68, COTS f=0.02 setting. The current optimization
target is no longer a large throughput deficit; it is a narrow residual gap
against weight-only while preserving the extra resident KV capacity.

A timing-enabled post-QK-tile split672 run still points at CPU suffix attention
and QKV readiness as the exposed terms. A representative log at `Running=150`
and `3471 out tok/s` reported, over 28 layers:

| Component | ms |
|---|---:|
| GPU prefix attention | 3.902 |
| CPU suffix wait | 11.050 |
| CPU suffix attention | 8.802 |
| Q D2H copy | 1.918 |
| CPU suffix scatter | 1.516 |
| GPU merge/UVA artifact read | 1.735 |

At matched artifact sizes seen in earlier timing (`Q D2H ~=18.264 MB`), CPU
suffix attention is now roughly `6.1-7.9 ms` over 28 layers instead of the
previous representative `8.4 ms`, confirming the QK tile moved the intended
component. Merge remains measurable, but it is not the next largest term.

The runtime metrics now split aggregate CPU suffix wait into Q/QKV-ready wait
and separate K/V-ready wait. A timed split672 rerun measured `3343 out tok/s`
with timing enabled and showed the aggregate wait is almost entirely Q/QKV
readiness. At the representative `Running=150` point over 28 layers:

| Component | ms |
|---|---:|
| Total CPU suffix wait | 6.701 |
| Q/QKV-ready wait | 6.471 |
| Separate K/V-ready wait | 0.230 |
| Actual Q/QKV D2H copy timing | 0.997 |
| CPU suffix scatter | 0.905 |
| CPU suffix attention | 7.651 |
| GPU merge/UVA artifact read | 1.170 |
| GPU prefix attention | 2.469 |

Later full-batch-ish slices often reported `kv_wait_ms=0.000` and
`qkv_wait_ms ~= wait_ms`. This confirms the next synchronization target is
Q/QKV readiness from the GPU-produced canonical QKV tensor, not CPU suffix KV
scatter or a standalone K/V D2H path.

The same timed setup with combined-QKV staging disabled regressed to `3318`
out tok/s. At the comparable `Running=150` log point it reported:

| Staging mode | Out tok/s | Total wait | Q/QKV wait | K/V wait | CPU attn |
|---|---:|---:|---:|---:|---:|
| Combined QKV staging on | 3343 | 6.701 ms | 6.471 ms | 0.230 ms | 7.651 ms |
| Combined QKV staging off | 3318 | 11.825 ms | 9.926 ms | 1.899 ms | 9.982 ms |

This keeps the prior decision intact: combined-QKV staging should remain
enabled. Query-only staging reduces the individual query artifact but exposes a
separate K/V-ready wait and does not close the integrated gap.

Directly reusing Phase 1 CPU-produced QKV artifacts is also not a local win for
the current best candidate. With Qwen2.5-7B QKV dimensions
`Q=3584, K=512, V=512`, head-aligned KV-biased placement rounds the
`f_cpu_store=0.02` QKV CPU slice to zero columns:

| f_cpu_store | Q CPU heads | K CPU heads | V CPU heads |
|---:|---:|---:|---:|
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 1 | 1 |
| 0.10 | 0 | 2 | 2 |
| 0.25 | 1 | 4 | 4 |
| 0.50 | 10 | 4 | 4 |
| 1.00 | 28 | 4 | 4 |

CPU suffix attention needs all 28 query heads. Therefore a direct artifact path
from `CotsQKVOp` would not help the measured split672, f=0.02 case unless the
planner intentionally moves much more QKV compute to CPU, which would be a
larger Phase 1/Phase 2 co-design rather than a small residual-gap fix.

An E2E validation of the new CPU suffix budget at the same split did not show a
runtime-cap win:

| Setting | Out tok/s | Interpretation |
|---|---:|---|
| split672, `max_num_seqs=512` | 3410-3431 | Current integrated baseline |
| split672, `max_num_seqs=384` | 3421 | Same within noise |

This means the 28-layer CPU suffix budget is useful for planner feasibility
modeling, but lowering `max_num_seqs` at split672 does not close the current
gap. The measured run is already constrained elsewhere for enough of decode
that the cap is not binding in a helpful way.

Minimal real-FastTTS smoke with generator eager mode and the manual planner
wired through the benchmark YAML:

| Mode | Problems | Split | Generator max_num_seqs | Total time | Avg/problem |
|---|---:|---:|---:|---:|---:|
| Weight-only COTS f=0.02 | 2 | n/a | 4 | 47.91 s | 23.96 s |
| Hybrid KV + COTS f=0.02 | 2 | 672 | 4 | 51.13 s | 25.56 s |

The run validates the benchmark integration path rather than the final accuracy
claim. The hybrid engine initialized with `split_blocks=42`, a 4 GiB CPU KV
pool, and eager execution. Tokenization of the saved completions shows one of
the two problems crossed the 672-token split (`max_total=751` for hybrid), so
the path is not purely dead code. The project evaluator reported `0/2` for both
variants under `agg_strategy=last`; the selected extracted answers were the
same for both variants. This is too small and too parser-sensitive to certify
quality, but it does not show a hybrid-specific correctness regression. A real
quality claim still needs a larger MATH subset after the performance path is
worth measuring end-to-end.

An aggressive post-split FastTTS smoke was also run with `split_blocks=10`
(`split_tokens=160`), `max_model_len=512`, one beam-search iteration, and a
non-matching stop string so generation reliably crosses the split:

| Mode | Problems | Completion tokens | Generator latency | Generator tok/s | Eval |
|---|---:|---:|---:|---:|---:|
| Weight-only COTS f=0.02 | 2 | 2048 | 9.065 s | 225.9 | 0/2 |
| Hybrid KV + COTS f=0.02 | 2 | 2048 | 11.190 s | 183.0 | 0/2 |

Both problems crossed the split in all four completions. Max post-split
generation was 143 and 206 tokens for the two prompts. Hybrid runstats showed
the CPU suffix path was active:

| Counter | Value |
|---|---:|
| `hybrid_decode_calls` | 11284 |
| `hybrid_cpu_kv_blocks_max` | 60 / 4681 |
| `hybrid_preemptions` | 0 |
| `cpu_suffix_attn_ms` | 349.5 |
| `cpu_suffix_wait_ms` | 2242.0 |
| `q_d2h_bytes` | 323.5 MB |
| `kv_d2h_bytes` | 92.4 MB |
| `kv_uva_h2d_bytes` | 328.6 MB |

This is intentionally not a throughput win case (`max_num_seqs=4`, no KV
capacity pressure). It is a real-workload stress test of the post-split path.
The result matches the synthetic short-batch diagnosis: exposed cost is mostly
Q/K/V readiness and suffix artifact movement, not CPU suffix math.

A deterministic task-level split160 probe using
`David/Benchmarks/phase2/check_hybrid_task_accuracy.py` crossed the split for
all four tasks with 224 post-split generated tokens each:

| Mode | Accuracy | Crossed split |
|---|---:|---:|
| Weight-only COTS f=0.02 | 2/4 | 4/4 |
| Hybrid KV + COTS f=0.02 | 2/4 | 4/4 |

This does not certify end-to-end accuracy, but it is a useful guardrail for the
aggressive split path: no correctness drop was observed under deterministic
sampling while exercising CPU suffix attention for every request.

The same split160 FastTTS setup was rerun for one problem with the now-removed
detailed timing flags (`COTS_HYBRID_TIME_D2H=1`,
`COTS_HYBRID_TIME_GPU_PREFIX=1`). The timing flags perturbed the path, so use
this historical result for attribution rather than headline throughput:

| Metric | Total | Per hybrid layer call |
|---|---:|---:|
| `hybrid_decode_calls` | 4760 | n/a |
| `cpu_suffix_wait_ms` | 867.2 ms | 0.182 ms |
| `gpu_prefix_attn_ms` | 114.9 ms | 0.024 ms |
| `cpu_suffix_attn_ms` | 126.9 ms | 0.027 ms |
| `hybrid_merge_ms` | 83.1 ms | 0.017 ms |
| `q_d2h_copy_ms` | 49.5 ms | 0.010 ms |
| `cpu_suffix_scatter_ms` | 34.9 ms | 0.007 ms |

This pins down the small-batch post-split bottleneck. The raw D2H copy, CPU
suffix attention, scatter, and merge are all real but individually small at
B=4. The exposed term is the per-layer wait for staged QKV to become
CPU-visible. That wait is not explained by copy bandwidth; it is mostly the
eager synchronization structure around GPU-produced QKV and CPU suffix work.
The current code already stages Q before the KV-cache update and launches GPU
prefix attention before waiting, so there is no obvious local reorder left that
preserves parity. Closing this part of the gap likely needs a design change
such as graph/host-function scheduling for suffix work, or a planner policy
that avoids tiny active batches where this fixed per-layer wait is exposed.

Detailed timing on the high-batch split672 integrated candidate gives a
different bottleneck mix. With B=512, COTS weight prefetch `f=0.02`, and timing
flags enabled, throughput was `3331 out tok/s` (lower than the untimed
`3448-3451` because timing synchronizes). A representative log point near
`3457 out tok/s` and `Running=150` showed:

| Component over 28 layers | ms |
|---|---:|
| GPU prefix attention | 2.425 |
| CPU suffix wait | 6.951 |
| CPU suffix attention | 8.411 |
| Q D2H copy | about 1.0 |
| CPU suffix scatter | about 0.9 |
| GPU merge/UVA artifact read | about 1.2 |

So the planner-relevant split672 case is not the same as tiny-batch split160:
CPU suffix math again becomes the largest exposed term, while QKV-ready wait
remains significant. This supports continuing with narrow CPU suffix kernel
improvements only when they are parity-safe and E2E-positive, and avoiding more
scheduler-side active caps that shrink GPU linear batch.

Matched Nsight attribution for the best integrated candidate used an in-process
capture range (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) to avoid tracing model load.
These spans are attribution data, not replacements for the normal multiprocess
E2E throughput numbers above:

| Mode | Capture span | Kernel sum | Explicit H2D | Explicit D2H | Notable hybrid kernel |
|---|---:|---:|---:|---:|---:|
| Hybrid split672 + COTS f=0.02 | 24.45 s | 15.36 s | 103.2 GB / 4.75 s | 12.55 GB / 0.595 s | merge_attn_states 0.506 s |
| Weight-only COTS f=0.02 | 23.18 s | 21.42 s | 153.1 GB / 6.49 s | ~0 GB / 0.001 s | n/a |

This confirms two useful points. First, the hybrid path is not accidentally
bulk-reloading CPU suffix KV to GPU: explicit H2D is smaller than weight-only,
while the new traffic is D2H artifact movement. Second, the remaining best-case
capture-span delta is roughly the size of D2H artifacts plus GPU merge work,
with additional eager synchronization around that path. The next optimization
should therefore target the suffix artifact/merge pipeline, not scheduler
active caps or another split sweep.

A focused merge microbenchmark was added at
`David/Benchmarks/phase2/benchmark_merge_uva.py` to isolate this artifact path.
On the current kernel:

| Batch | Suffix output | Suffix LSE | Merge median ms/layer |
|---:|---|---|---:|
| 64 | GPU | GPU | 0.0037 |
| 64 | UVA | GPU | 0.0224 |
| 64 | GPU | UVA | 0.0063 |
| 64 | UVA | UVA | 0.0251 |
| 512 | GPU | GPU | 0.0043 |
| 512 | UVA | GPU | 0.1565 |
| 512 | GPU | UVA | 0.0325 |
| 512 | UVA | UVA | 0.1828 |

This narrows the merge-side gap: the large BF16 suffix output UVA stream
dominates; suffix LSE UVA is measurable but smaller. A CUDA-kernel experiment
that loaded LSE once per token-head through shared memory passed parity
(`test_merge_attn_states.py`, `test_cots_hybrid_attention.py`) but was neutral
on the target benchmark (`B=512` UVA output + UVA LSE stayed about `0.180 ms`),
so it was reverted.

During diagnosis, an explicit non-parity lower-bound mode was temporarily added
behind `COTS_HYBRID_DEBUG_PREFIX_ONLY_LOWER_BOUND=1`. It ran GPU prefix
attention and the Q/K/V CPU staging plus CPU KV scatter path, but bypassed CPU
suffix attention and the UVA merge. This mode has now been removed from the
runtime; the table below is retained only as historical attribution evidence.

Same-session split672 integrated A/B at B=512, max_num_seqs=512,
gpu_memory_utilization=0.68, COTS pure-prefetch f=0.02:

| Mode | Out tok/s | Meaning |
|---|---:|---|
| Real hybrid KV | 3282 | Current suffix attention + UVA merge path |
| Prefix-only lower bound | 4326 | Keeps QKV staging/scatter, removes suffix math and merge |

The real-hybrid control in this A/B is lower than the earlier `3448-3451`
band, so do not replace the headline number with this one. The important signal
is the gap: the current eager architecture has enough remaining headroom to
beat the weight-only COTS reference if suffix attention and merge artifacts are
substantially reduced. QKV staging/scatter alone is not the hard floor.

A follow-up OSRT/NVTX/CUDA trace of the same split672 integrated candidate did
not expose a clean host-scheduling target. The OSRT summary was dominated by
background timed waits, while CUDA attribution showed `cudaEventSynchronize`
time plus the expected D2H artifact copies and merge kernel work. Runtime
counters inside the trace still pointed to CPU suffix attention as the exposed
cost once suffix length becomes material. This supports using focused
multi-layer CPU suffix benchmarks for the next planner budget rather than
adding another host-side synchronization variant.

This is a stronger negative result than the 608:32 case. Iteration logging for
the B=256 GPU-only run shows vLLM initially schedules all 256 decode requests
together despite the conservative full-length "maximum concurrency" banner
(`38.60x` for 768-token requests). Prefix caching and block reuse make the
effective shared-prefix footprint much smaller than the naive
`batch * max_model_len` estimate. Later in decode, active generation count
drops as KV pressure rises, but GPU-only remains faster because the current
hybrid CPU/UVA suffix path is the limiting throughput path.

### Longer Prefix, 2048:128

`prompt_len = split_tokens = 2048`, `max_tokens = 128`, active batch 64.

| Mode | Out tok/s |
|---|---:|
| Hybrid | 2098 |
| GPU-only | 2768 |

Increasing the prefix:suffix ratio helps overlap, but it is not sufficient by
itself to beat GPU-only in this eager no-weight-offload setup.

## Bottleneck

For `B=64`, `608:32`, wall-clock instrumentation without cProfile overhead:

| Component | Calls | Total ms | Per call us |
|---|---:|---:|---:|
| QKV stage enqueue | 868 | 39.7 | 45.7 |
| KV update fast path | 868 | 2.8 | 3.2 |
| finish staged KV update | 868 | 418.7 | 482.4 |
| CPU KV scatter | 868 | 23.4 | 26.9 |
| CPU suffix attention | 868 | 66.3 | 76.3 |

For this short-suffix case, `finish staged KV update` is dominated by waiting
for staged QKV to become CPU-visible. Scatter and CPU suffix attention are
secondary. This means the short-suffix gap is mostly the cost of moving
per-layer Q/K/V artifacts to CPU, not a scalar CPU attention fallback or
missing OpenMP parallelism.

The CPU suffix kernel is on the intended AVX2/FMA path and uses
`at::parallel_for` over `(batch, kv_head)` tasks.

Decode-only Nsight on the optimized path shows that raw D2H copy engine time is
not the whole bottleneck:

| Direction | Total ns | Count | Median ns | Interpretation |
|---|---:|---:|---:|---|
| D2H | 4,536,518 | 204 | 23,009 | Combined QKV artifact copies are small and can overlap with prefix attention. |
| H2D explicit memcpy | 60,479 | 57 | ~1,000 | No bulk CPU suffix KV reload appears. |

The exposed delay is therefore the full artifact/synchronization path:
QKV readiness, CPU suffix attention, UVA merge reads, and host scheduling.

Latest parity-safe B=48 rerun (`prompt_len = split_tokens = 608`,
`max_tokens = 8`, eager, no weight offload) confirms that replacing PyTorch
`copy_(non_blocking=True)` with a raw C++ D2H primitive is not the right next
optimization:

| Measurement | Value |
|---|---:|
| GPU-only E2E | 1601 out tok/s |
| Hybrid E2E | 1276-1311 out tok/s |
| Hybrid / GPU-only | 0.80-0.82x |
| PyTorch aggregate QKV D2H microbench | 0.49 ms / 28 layers |
| Existing native dryrun D2H submit microbench | 0.96-1.04 ms / 28 layers |

Steady-state hybrid runtime stats for the same B=48 case:

| Component | Total per decode step, 28 layers |
|---|---:|
| Host wait for QKV-ready event | ~9.4-9.5 ms |
| Timed QKV D2H copy itself | ~0.53 ms |
| CPU K/V scatter into suffix cache | ~0.45-0.49 ms |
| GPU prefix attention | ~1.47-1.51 ms |
| CPU suffix attention | ~0.80-1.14 ms |
| GPU merge, including UVA artifact reads | ~0.68-0.72 ms |

The important distinction is that `wait_ms` is mostly waiting for GPU-produced
QKV to become available to the copy stream; the raw copy engine work is already
small. GPU-only also pays QKV projection time, so `wait_ms` should not be read
as purely incremental overhead. The incremental hybrid cost is the split path
after QKV readiness: CPU scatter/attention, GPU UVA merge, and per-layer eager
synchronization.

Current-code B=256, 608:160 detailed timing shows a different long-suffix
limit. Near the final decode steps, totals across 28 layers are approximately:

| Component | Total per decode step, 28 layers |
|---|---:|
| Host wait for QKV-ready event | ~12.6-13.2 ms |
| Timed QKV D2H copy itself | ~2.58 ms |
| CPU K/V scatter into suffix cache | ~2.16-2.22 ms |
| GPU prefix attention | ~5.22-5.25 ms |
| CPU suffix attention | ~58.9-59.3 ms |
| GPU merge, including UVA artifact reads | ~3.13-3.18 ms |
| Suffix read reuse wait | ~0.03 ms |

So the remaining long-suffix gap is CPU suffix attention itself. QKV copy,
scatter, and merge are still real, but they are no longer the dominant term once
the CPU suffix reaches 160 tokens at B=256.

The E2E CPU suffix time is also not a Python timing artifact. A multi-layer
microbenchmark that calls the same COTS suffix op once per layer over 28
distinct CPU KV caches reproduces the late-decode E2E cost:

| Case | One layer | 28 distinct layer caches | 28-layer per-layer |
|---|---:|---:|---:|
| B=256, suffix=160, 24 threads, pinned | 1.624 ms | 57.117 ms | 2.040 ms |
| B=256, suffix=160, 32 threads, pinned | 1.232 ms | 51.002 ms | 1.821 ms |
| B=512, suffix=32, 24 threads, pinned | 0.454 ms | 24.553 ms | 0.877 ms |

This closes the discrepancy between the old single-layer microbenchmark and the
E2E stats. Reusing one layer's cache repeatedly overstates CPU attention
performance because it gives unrealistically warm cache/TLB behavior. The real
decode path streams through many large per-layer CPU KV regions.

For the current best integrated split candidate (`split=672`, max suffix 96),
the 28-layer streaming benchmark shows a sharp CPU compute-capacity cliff at
the default 24-thread setting:

| Batch | Suffix tokens | 28-layer median ms | Median ms/layer |
|---:|---:|---:|---:|
| 64 | 96 | 8.853 | 0.316 |
| 128 | 96 | 17.368 | 0.620 |
| 150 | 96 | 19.879 | 0.710 |
| 192 | 96 | 25.522 | 0.911 |
| 256 | 96 | 34.237 | 1.223 |
| 384 | 96 | 51.480 | 1.839 |
| 448 | 96 | 63.059, unstable p90 417.772 | 2.252 |
| 480 | 96 | 286.766 | 10.242 |
| 512 | 96 | 245.822 | 8.779 |

This is now a planner-relevant constraint: CPU KV storage capacity and CPU
suffix compute capacity are not the same thing. Until an E2E run proves a higher
safe point, treat roughly `<=384` active requests at suffix length 96 as the
conservative CPU suffix budget on this machine.

Huge-page advice was tested because the multi-layer result suggested possible
TLB pressure. THP is configured as `madvise` on this machine, but it is not a
useful optimization for the current CPU KV layout:

| B=256, suffix=160, 28 layers, 24 threads | Median ms |
|---|---:|
| Pinned CPU KV cache, no advice | 57.117 |
| Pinned CPU KV cache, `MADV_HUGEPAGE` | 56.598 |
| Pageable CPU KV cache, no advice | 59.863 |
| Pageable CPU KV cache, `MADV_HUGEPAGE` | 126.170 |

The pinned result is within noise, and pageable huge-page advice is much worse
under this allocation pattern. Do not add huge-page plumbing to the Phase 2 runtime.

Isolated C++ scatter result at B=64:

| Scatter path | Median us |
|---|---:|
| PyTorch indexed assignment | 12.6 |
| COTS C++ memcpy scatter | 3.1 |

This improves the local scatter component, but it does not materially move E2E
throughput because the dominant exposed cost has shifted elsewhere.

Standalone B=512, suffix=32, prefix=608 per-layer primitive:

| Component | Median ms |
|---|---:|
| GPU prefix attention | 0.730 |
| CPU suffix attention | 0.630 |
| Full hybrid primitive | 1.023 |
| Exposed beyond prefix | 0.292 |

Thread-count sweep for this primitive showed that using all 32 logical threads
is still best among tested settings on the i9-14900KF:

| Threads | CPU suffix ms | Hybrid ms |
|---:|---:|---:|
| 4 | 1.673 | 2.219 |
| 8 | 0.863 | 1.272 |
| 12 | 1.235 | 1.631 |
| 16 | 0.920 | 1.317 |
| 24 | 0.634 | 1.026 |
| 32 | 0.473 | 0.986 |

The CPU has AVX2/FMA but no AVX512 BF16, so the current BF16-to-FP32 load path
is the right vector path for this machine.

Long-suffix thread sweep after the B=256, 608:160 E2E bottleneck was isolated:

| Threads | CPU suffix ms, before query preconvert | CPU suffix ms, after query preconvert |
|---:|---:|---:|
| 4 | 3.677 | 3.196 |
| 8 | 1.950 | 1.705 |
| 12 | 2.767 | 2.376 |
| 16 | 2.390 | 1.885 |
| 20 | 2.039 | 1.859 |
| 24 | 1.723 | 1.627 |
| 28 | 1.485 | 1.387 |
| 32 | 1.331 | 1.248 |

The optimized 32-thread result is about 6% faster for B=256, suffix=160.
For B=512, suffix=32, the same optimization moves the isolated 32-thread
suffix op from the earlier `0.473 ms` result to `0.341 ms`. This confirms the
repeated query BF16 upcast was real waste, but the long-suffix E2E gain remains
small because the full hybrid path also pays QKV readiness/staging, CPU scatter,
GPU prefix attention, UVA merge, and eager per-layer synchronization.

Rejected streaming online-softmax experiment:

| Case | Query-preconvert kernel | Streaming online-softmax |
|---|---:|---:|
| B=256, suffix=160, 32 threads | 1.248 ms | 1.256 ms |
| B=512, suffix=32, 32 threads | 0.341 ms | 0.539 ms |

The streaming version removed probability storage and a separate value pass, but
it introduced per-token accumulator rescaling. On this CPU that tradeoff is not
worth carrying.

Additional negative A/B checks after the query-preconvert kernel:

| Experiment | Result | Decision |
|---|---:|---|
| Disable combined QKV staging, B=256, 608:160 | 3150 -> 3000 out tok/s | Keep combined QKV staging enabled. |
| Disable combined QKV staging, B=512, split672 integrated | 3410-3414 -> 3387 out tok/s | Combined staging still wins in the planner candidate region. |
| Explicit H2D copy for suffix output/LSE before merge, B=512, split672 integrated | default 3431, explicit 3444 out tok/s | Too small to justify another path or violating UVA artifact design. |
| Force OMP_NUM_THREADS=32 MKL_NUM_THREADS=32, B=256, 608:160 | latest repeat: 3150 -> 3142 out tok/s | No launcher-side thread override needed. |
| Isolated suffix op, pinned vs pageable host tensors, B=256, suffix=160 | both about 1.2 ms/layer | Pinned cache memory is not the E2E limiter. |
| Isolated suffix op, contiguous vs interleaved CPU block table, B=256, suffix=160 | 1.255 -> 1.319 ms/layer | Runtime block interleaving hurts slightly, but does not explain the E2E gap. |
| Per-block block-table lookup cleanup in CPU suffix kernel | B=256, suffix=160: prior 1.248 ms/layer, cleanup 1.258; B=512, suffix=32: prior 0.341, cleanup 0.337 | Neutral/noisy, reverted. |

Rejected token-major value accumulation experiment:

| Case | Query-preconvert kernel | Token-major value pass |
|---|---:|---:|
| B=256, suffix=160, 32 threads | 1.248 ms | 1.130 ms |
| B=512, suffix=32, 32 threads | 0.341 ms | 0.364 ms |
| B=256, 608:160 E2E hybrid | 3150 out tok/s | 3023 out tok/s |

The isolated long-suffix improvement did not translate to E2E. The average
decode run includes many short and mid-size suffix steps, and the larger stack
accumulator footprint appears to hurt enough outside the final long-suffix
steps that the full run is slower.

Rejected memory-pinning and task-granularity hypotheses:

| Case | Existing behavior | Alternative |
|---|---:|---:|
| B=256, suffix=160, pinned tensors | 1.245 ms | pageable tensors: 1.264 ms |
| B=512, suffix=32, pinned tensors | 0.340 ms | pageable tensors: 0.340 ms |
| B=256, suffix=160, `(request, kv_head)` tasks | 1.245 ms | request-only tasks: 1.246 ms |
| B=512, suffix=32, `(request, kv_head)` tasks | 0.340 ms | request-only tasks: 0.341 ms |

These results rule out CPU cache pinning and ATen task count as meaningful
contributors to the current gap.

Rejected E2E CPU-thread configuration hypothesis:

| B=256, 608:160 hybrid setting | Out tok/s |
|---|---:|
| Default environment (`torch.get_num_threads() == 24`) | 3150 |
| `OMP_NUM_THREADS=32`, `MKL_NUM_THREADS=32` | 3142 |
| `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16` | 2867 |

The worker is not accidentally running CPU suffix attention with a tiny thread
pool. Increasing threads beyond the default does not help, and reducing to 16
hurts full-system throughput.

Experiment-only artifact mode at B=512, suffix=32, prefix=608:

| Artifact path | Hybrid primitive ms |
|---|---:|
| Current UVA path | 1.016 |
| Explicit H2D copy to preallocated GPU buffers | 0.993 |

Explicit copies improve this synthetic primitive by only about 2-3% and would
compete with Phase 1 weight prefetch H2D traffic. This is not enough evidence
to change the Phase 2 UVA artifact decision.

## Phase 1 Weight-Offload Interaction

Tested the same `B=64`, `608:32` hybrid-KV case with native COTS weight
offload enabled:

- `cots_f_cpu_store = 0.09`
- `cots_f_prefetch = 0.0`
- `cpu_runner = native`

Result:

| Mode | Out tok/s |
|---|---:|
| Hybrid KV only (`f_cpu_store=0`) | 2339 |
| Hybrid KV + COTS weight CPU compute (`f_cpu_store=0.09`) | 149 |

Subcomponent timing for the `f_cpu_store=0.09` run:

| Component | Calls | Total ms | Per call us |
|---|---:|---:|---:|
| QKV stage enqueue | 868 | 80.5 | 92.8 |
| KV update fast path | 868 | 8.0 | 9.2 |
| finish staged KV update | 868 | 12235.1 | 14095.7 |
| CPU KV scatter | 868 | 4168.5 | 4802.4 |
| CPU suffix attention | 868 | 94.3 | 108.6 |

This is a negative result, but it is important. The current Phase 1 QKV path
does **not** expose CPU-produced Q/K/V artifacts to Phase 2. `CotsQKVOp`
computes the CPU slice, returns it to a CUDA `y_dst` through UVA, and scatters
CPU/GPU/prefetch outputs into one canonical CUDA QKV tensor. Hybrid KV then
stages that final CUDA tensor back to CPU.

Therefore enabling Phase 1 CPU compute currently adds a CPU synchronization
dependency before QKV is CPU-visible to suffix attention. It worsens the exposed
D2H/wait bottleneck instead of removing it.

## Parity Notes

Focused tests still validate the numerical pieces:

- CPU suffix attention matches a PyTorch reference within tolerance.
- Hybrid prefix/suffix merge matches a full dense attention reference within
  tolerance.
- The C++ CPU suffix K/V scatter fast path is byte-exact against PyTorch
  indexed assignment.

Integrated greedy generation is more sensitive:

| Smoke | Result |
|---|---|
| Identical prompts, B=4, prompt=608, decode=8 | GPU-only and hybrid token outputs matched exactly. |
| Prompts with only the final prompt token perturbed, B=4, prompt=608, decode=8 | 1 of 4 requests diverged under greedy decoding. |
| GPU vs GPU control, B=4, prompt=608, split672, decode=160, top-20 logprobs | 3/4 exact; first divergence at generated position 22, before CPU suffix starts. |
| GPU vs hybrid KV, same probe, no weight offload | 3/4 exact; first divergence at generated position 22, before CPU suffix starts. Post-split shared positions: top1 288/288, chosen-token logprob delta max 0.112, p95 0.0053. |
| COTS weight-only vs hybrid+weight, same probe | 3/4 exact; first divergence at generated position 60, before CPU suffix starts. Post-split shared positions: top1 287/288, chosen-token logprob delta max 0.088, p95 0.0093. |
| Forced-context GPU vs hybrid+weight, B=4, prompt=608, split672, decode=160 | Forced outputs matched. Top1 same 638/640 overall and 383/384 post-split. Post-split forced-token logprob delta max 0.0873, mean 0.00337, p95 0.0176. |
| Forced-context GPU vs GPU control, same probe | Forced outputs matched. Top1 same 638/640 overall and 383/384 post-split. Post-split forced-token logprob delta max 0.0874, mean 0.00292, p95 0.0159. |

The newer split672 probes narrow the parity risk. Exact greedy divergence is not
specific to hybrid KV, because the GPU-vs-GPU control diverges before the CPU
suffix path is active. In the stronger forced-context probe, GPU-vs-hybrid has
the same top-1 mismatch count as GPU-vs-GPU under the exact same continuation,
including the single post-split mismatch. That makes the current hybrid
numerical drift look comparable to ordinary eager run-to-run drift for this
small probe, not like cache corruption. This is still not a proof of arbitrary
output parity; use task accuracy for planner evaluation, and keep the dense
attention/unit parity tests as the correctness backstop.

Added `David/Benchmarks/phase2/check_hybrid_task_accuracy.py` as a small
task-level probe. It pads simple math prompts to 608 tokens, runs decode across
the split672 boundary, and grades extracted answers with the existing
`accuracy_evaluation` parser/grader. Current evidence:

| Probe | Post-split coverage | Result | Interpretation |
|---|---:|---:|---|
| 12-step prompt, decode160, `ignore_eos=True` | 96 tokens/request | weight-only 0/4, hybrid 0/4 | Invalid accuracy signal; prompts were truncated before boxed answers. |
| 12-step prompt, decode288, `ignore_eos=True` | 224 tokens/request | weight-only 2/4, hybrid 2/4 | Both modes equal, but prompt is still artificial. |
| 8-step prompt, decode224, normal EOS | >=115 tokens/request | weight-only 2/4, hybrid 1/4 | Too small to call a regression by itself. |
| 8-step prompt, decode224, normal EOS, weight-only vs weight-only control | >=115 tokens/request | 2/4 vs 1/4 | Same-path control reproduces the apparent one-task drop. |

Conclusion: this probe does not show a hybrid-specific task accuracy failure,
but it also is not strong enough to certify split672. It should remain a
low-cost guardrail. Planner/default enablement still needs a real FastTTS
accuracy run on a held-out task subset.


## FastTTS Forced Post-Split Gate

Added FastTTS-side persistence for existing COTS hybrid KV counters in
`FastTTS-thesis/models/vllm_wrapper.py` under `runstats.json -> generator ->
cots_hybrid_kv`. The accumulator records max/current CPU/GPU block use,
preemptions, recomputed suffix tokens, decode calls, CPU suffix timing, wait
components, and D2H/UVA byte counts. This makes FastTTS runs usable for Phase 2
performance diagnosis instead of relying only on periodic vLLM log lines.

A normal two-iteration MATH-500 smoke with split672 initialized successfully but
did **not** exercise CPU suffix attention: generated continuations were only
about 100-166 tokens and the split was 672. That run remains only an
initialization/scheduler smoke test.

Forced post-split stress gate:

- Dataset: first 2 MATH-500 problems.
- Search: `beam_width=4`, `n=4`, `num_iterations=1`, `max_tokens=256`,
  unreachable stop string, temperature 0.7.
- Generator: Qwen2.5-7B, eager COTS weight path, `max_model_len=512`.
- Hybrid split: `split_blocks=10` => `split_tokens=160`.
- Prompt lengths for these two problems are about 77 and 140 tokens, so prefill
  remains GPU-only and every completion crosses the split during decode.

Result:

| Mode | Wall time | Avg/problem | Generator latency/problem | Completion tokens | Accuracy |
|---|---:|---:|---:|---:|---:|
| Eager weight-only | 38.53 s | 19.27 s | 4.637 s / 4.428 s | 1024 + 1024 | 0/2 |
| Eager hybrid KV | 41.78 s | 20.89 s | 5.536 s / 5.654 s | 1024 + 1024 | 0/2 |

Hybrid overhead in this forced post-split E2E gate:

- Wall-clock: +8.4% including model init/verifier overhead.
- Generator-only latency: about +23% on the measured generation calls.
- Scheduler batch stayed at 4 running requests, no queueing.
- No preemptions and no recomputed CPU suffix tokens.

Persisted hybrid counters for the generator:

| Counter | Value |
|---|---:|
| Max GPU KV blocks used | 28 |
| Max CPU KV blocks used | 60 / 4681 |
| Hybrid decode calls | 11284 |
| CPU suffix attention | 349.5 ms |
| CPU suffix wait | 2242.0 ms |
| CPU suffix read wait | 10.1 ms |
| Q D2H bytes | 323.5 MB |
| KV D2H bytes | 92.4 MB |
| UVA H2D artifact bytes | 328.6 MB |

Additional one-problem timing A/B using the now-removed detailed timing flags:

| Case | Generator latency | CPU wait | Q D2H copy | GPU prefix | CPU suffix attn | Scatter | Merge | Wait minus Q copy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| COTS weight 2%, full-QKV staging on | 5.751 s | 182.19 us/call | 10.40 us/call | 24.14 us/call | 26.67 us/call | 7.33 us/call | 17.46 us/call | 171.79 us/call |
| COTS weight 2%, full-QKV staging off | 5.818 s | 190.94 us/call | 20.60 us/call | 24.49 us/call | 23.56 us/call | 7.14 us/call | 17.13 us/call | 170.34 us/call |
| Hybrid KV only, no COTS weight split | 5.516 s | 268.22 us/call | 3.81 us/call | 20.02 us/call | 47.32 us/call | 6.85 us/call | 11.43 us/call | 264.42 us/call |

Nsight Systems queue-check on a synthetic forced post-split decode:

- Command shape: `batch=4`, prompt 144, split 160, total 416, COTS weight 2%,
  eager/in-process V1, capture range around measured generation only.
- Output: `/tmp/phase2_hybrid_queue_b4.nsys-rep`.

| Nsight metric | Value |
|---|---:|
| Captured CUDA runtime span | 6.95 s |
| GPU op union active time | 6.02 s |
| GPU-wide idle gaps | 0.93 s |
| GPU active percentage over GPU-op span | 86.6% |
| Gaps >= 100 us | 1399 gaps, 291.3 ms total |
| Gaps >= 500 us | 23 gaps, 22.0 ms total |
| `cudaEventSynchronize` API time | 1.034 s across 28804 calls |
| H2D memcpy time / volume | 2.684 s / 62.9 GB |
| D2H memcpy time / volume | 36.0 ms / 263 MB |
| Hybrid prefix FA kernels | 7140 calls, 75.3 ms total |
| Hybrid merge kernels | 7140 calls, 72.2 ms total |

Interpretation from Nsight:

- The `cudaEventSynchronize` total is close to the GPU-wide idle-gap total and
  to the internal CPU suffix wait counter. That supports the current hypothesis:
  the exposed cost is per-layer synchronization / artifact readiness, not raw
  D2H transfer.
- D2H traffic is small in both time and volume. H2D is much larger and is still
  dominated by the Phase 1 weight-prefetch path, so adding explicit H2D suffix
  reloads remains the wrong direction.
- GPU prefix attention is not a large enough overlap window in this small-batch
  forced gate: the traced prefix FA calls total only ~75 ms, while host-side
  synchronization gaps are an order larger.

Notes:

- Full-QKV staging is still the better current staging mode. Disabling it
  slightly worsens generator latency and roughly doubles measured Q copy time.
- The small 2% COTS weight path is not the main wait source. Removing it does
  not remove the per-layer wait; the residual wait after actual Q copy becomes
  noisier/worse in the one-problem timing run.
- Actual Q D2H copy time is small. The dominant `cpu_suffix_wait_ms` component
  is waiting for the staged Q/QKV artifact to become ready on the CUDA stream,
  which effectively introduces a per-layer host/GPU synchronization point.
- This also tightens the architectural constraint: suffix attention can overlap
  with GPU prefix attention, but not with QKV projection before Q exists or with
  output/MLP work after attention output is needed. On this forced gate, GPU
  prefix attention is only about 20-24 us/call, while CPU suffix attention plus
  artifact/merge work is larger, so the available overlap window is too small.

Interpretation:

- The run confirms the CPU suffix path is active under FastTTS and that the runtime
  frees both tiers cleanly at finish (`*_blocks_used` returns to 0 in the final
  sidecar snapshot).
- The measured bottleneck is orchestration/wait, not raw CPU attention math:
  accumulated CPU suffix attention is only ~0.35 s, while CPU wait is ~2.24 s.
- The default run leaves `gpu_prefix_attn_ms` at 0 because GPU timing is off by
  default. The timed one-problem A/B records GPU prefix attention at only
  ~20-24 us/call, which is too small to hide the CPU suffix/artifact path. Use
  this timed mode or Nsight evidence before making planner overlap decisions.
- This is **not** a quality certification. The one-long-step setup intentionally
  changes FastTTS search, and temperature 0.7 makes selected answers differ
  between runs. It is an integration/performance stress gate. Quality evidence
  still comes from deterministic/forced-context parity probes plus future real
  FastTTS accuracy subsets.

## Prepared Runner Graph Probe

The prepared suffix runner's custom-op boundary is CUDA-graph capturable in a
focused probe. Capturing `cots_suffix_attn_submit` + `cots_suffix_attn_sync`
works, replay works, and a focused test verifies that replayed host callbacks
read updated CPU QKV contents through the same prepared task pointers.

Short benchmark from `benchmark_prepared_suffix_runner.py` with pinned CPU
buffers, 24 CPU threads, 28 layers:

| Case | Direct | Direct + scatter | Prepared | Prepared + scatter | Graph prepared + scatter |
|---|---:|---:|---:|---:|---:|
| B=64, suffix32 | 3.133 ms | 3.461 ms | 3.516 ms | 4.170 ms | 3.873 ms |
| B=256, suffix160 | 54.259 ms | 56.019 ms | 59.936 ms | 59.740 ms | 58.542 ms |

Interpretation: graph replay removes a small amount of prepared-runner overhead
(`59.740 -> 58.542 ms` at B=256/suffix160), but the host-callback prepared
path remains slower than direct blocking scatter plus attention in isolation
(`56.019 ms`). This is still useful for full vLLM graph compatibility, but it
should not be expected to close the whole eager hybrid gap by itself.

## Planner Implications

- Hybrid KV should currently be treated as a **capacity mechanism**, not a
  same-batch speedup mechanism.
- The Planner should enable CPU suffix KV only when the target active batch or
  suffix budget would otherwise exceed GPU KV capacity.
- If GPU-only can fit the workload with prefix caching, GPU-only remains faster
  in the measured eager setup.
- The Planner must budget QKV artifact D2H, not only CPU suffix attention
  compute. Raw D2H copy time is small, but QKV readiness plus the CPU
  suffix/UVA merge path is exposed.
- Larger GPU-prefix/CPU-suffix ratios help, but do not remove the D2H artifact
  cost.
- For short suffixes, increasing active batch alone is not sufficient; B=512
  and B=1024 plateau at the same hybrid throughput because the CPU/UVA suffix
  path saturates.
- For the measured 608:160 capacity-pressure case, doubling active batch from
  256 to 512 does not improve hybrid throughput either. Do not add scheduler
  admission complexity until the CPU/UVA suffix path changes.
- Planner evaluation should use task accuracy, not only greedy token
  equality. The forced-context split672 probe puts logit drift inside the
  GPU-vs-GPU control envelope for this small case, but the planner still needs
  workload-level quality evidence.

## Next Technical Options

Low-risk next measurements:

- Use the prefix-only lower-bound mode only as a diagnostic guardrail. The first
  split672 A/B shows that suffix math plus UVA merge are large enough to explain
  the remaining integrated gap, so the next performance work should target that
  path directly rather than scheduler admission caps.
- Add/keep a deterministic post-split quality guard for any optimization that
  touches hybrid attention or artifact movement. The one-long-step FastTTS gate
  above is intentionally not a parity proof.
- Run a real FastTTS task-level accuracy check before treating split672 as a
  quality-preserving planner candidate. The small padded-prompt probe is useful
  as a guardrail, but the same-path control is too unstable for certification.
- Validate the new 28-layer CPU suffix budget against E2E runs by varying
  `max_num_seqs` and split placement around the measured cliff. This should be
  a planner constraint check, not a new runtime active-cap mechanism.

Potential implementation work, in increasing complexity:

- Integrate the captured prepared-scatter suffix task into the vLLM graph path
  experimentally. The isolated task boundary captures and replays, but the engine
  still rejects hybrid KV when CUDA graphs are enabled and the surrounding hybrid
  metadata/staging path has not been made graph-safe.
- Add planner admission logic that only enables hybrid KV when it increases
  feasible active batch under the current GPU KV budget; the planner should
  currently prefer split672-ish for the measured 608:160 tight-memory case.
- Investigate graph/host-function integration for the CPU suffix task. The
  forced post-split FastTTS gate now points directly at per-layer
  synchronization/queueing overhead, so this is more promising than another
  local CPU kernel micro-optimization.
- Add an explicit CPU-QKV artifact interface from `CotsQKVOp` to hybrid KV only
  if the host-function path cannot remove enough of the readiness wait. Without
  this interface, Phase 1 CPU QKV compute cannot make Q/K/V visible to Phase 2
  without first re-materializing them through the canonical CUDA QKV tensor.

Do not add a GPU tail fallback yet. The current scheduler/manager deliberately
does not allocate GPU suffix blocks after the split, so a tail fallback would be
a larger architecture change rather than a local optimization.


## 2026-05-19: Graph Live-Count Override and Retest

After making the graph-compatible native suffix runner functional, a live-row
override was added to avoid doing CPU suffix attention for padded CUDA graph
bucket rows. The suffix runner now accepts runtime `(num_tokens,
scatter_count)` values out-of-graph, clamps attention to live rows, scatters
only live current-token K/V rows, and writes neutral suffix state for padded
rows (`output=0`, `lse=-inf`). A focused CUDA graph test captures batch 4 and
replays with live batch 2, validating that only live rows scatter and compute.

One implementation trap appeared immediately: publishing the live counts once
per layer added a Python/pybind call on every attention layer. This was changed
to cache the last `(live_rows, scatter_rows)` pair in `CotsHybridKVStore`, so
steady-state decode republishes only when the live shape changes. Focused tests
remain green: `27 passed` across suffix runner, hybrid attention, and worker KV
tests.

End-to-end forced-context graph parity after this change remained within the
existing guardrail envelope for the small graph probe (`B=4`, prompt608,
split608, decode32, no weight offload): forced outputs matched, top-1 agreement
was `127/128`, mean forced-token logprob delta was `0.0199`, p95 was `0.0715`,
and max was `0.1223`. This is close to the prior same-mode control envelope,
but keep the forced-context probe as the correctness gate for future graph path
edits.

Performance was retested at the prior integrated candidate:

- Model: Qwen2.5-7B-Instruct BF16
- `prompt=608`, `total=768`, `split=672`, requested `B=512`
- `gpu_memory_utilization=0.68`, `max_num_seqs=512`
- COTS weight prefetch enabled with `f_cpu_store=f_prefetch=0.02`

| Mode | Execution | Out tok/s | Notes |
|---|---|---:|---|
| Weight-only COTS | eager | 3568 | matched baseline, same as previous band |
| Hybrid KV + COTS | eager | 3200 | after live-count cache; still below prior hybrid band |
| Hybrid KV + COTS | graph/piecewise | 3148 | graph memory reduced GPU KV cache to 14,336 tokens; no E2E win |

The prepared suffix runner microbenchmark does show a small local graph benefit
at the split672 suffix length (`B=512`, suffix96, 28 layers, 24 threads, pinned
inputs): eager prepared-scatter median `73.4 ms`, graph prepared-scatter median
`70.9 ms` (~3.4% faster). That local benefit does not appear in the full vLLM
run, so graph compatibility should be treated as a correctness/production path
step rather than a throughput fix by itself. The E2E limiter is still the hybrid
CPU/UVA suffix path and its surrounding integration, not merely Python launch
overhead.

The current retest is also lower than the earlier best hybrid eager band
(`~3448-3468 out tok/s`). Since the weight-only baseline reproduced cleanly,
this points to changes in the newer hybrid graph/staging path rather than
measurement drift. Before spending more time on graph machinery, the next
performance investigation should compare current eager hybrid against the
earlier post-QK path with timing counters/Nsight, focusing on Q/K/V staging,
CPU suffix wait, and merge/UVA artifact movement.

## 2026-05-19: Live Counts Moved to the Forward Dispatch Boundary

The suffix live-count override now follows the Phase 1c split-graph pattern more
closely. Instead of publishing `(live_rows, scatter_rows)` as a side effect of
per-layer hybrid decode metadata construction, `GPUModelRunner._publish_forward_dispatch()`
publishes the hybrid suffix live counts once per forward, immediately after the
weight offloader dispatch state. This keeps the graph runtime scalar state at
the same boundary used for native COTS weight offload.

The per-layer metadata builder is now side-effect free with respect to runtime
counts. `CotsHybridKVStore.on_dispatch()` computes the suffix
rows from the current token positions and split point, caches duplicate values,
and publishes `(0, 0)` for prefix-only or empty graph replays. The native suffix
runner now uses `-1` as the no-override sentinel, so `0` is a real live value:
zero-live replay skips CPU suffix attention, leaves K/V caches untouched, and
writes neutral suffix state (`output=0`, `lse=-inf`) for the captured bucket.

Validation after the refactor:

| Check | Result |
|---|---:|
| Focused suffix/hybrid/worker tests | `27 passed` |
| Forced-context graph parity | forced outputs matched; top-1 `127/128` |
| Forced token logprob delta | max `0.107889`, mean `0.0168281`, p95 `0.0665054` |
| Hybrid eager E2E, B512 prompt608 split672 total768 | `3195.149 out tok/s` |
| Hybrid graph/piecewise E2E, same config | `3167.950 out tok/s` |

This confirms the dispatch-boundary refactor is correctness-neutral and does not
move the current E2E performance band. The remaining gap is still in the hybrid
suffix data path and overlap behavior rather than live-count publication.

### Dispatch Boundary Naming Cleanup

The runner-side helper was renamed from `_publish_offloader_dispatch()` to
`_publish_forward_dispatch()` because it now publishes all out-of-graph
pre-forward runtime state, not just weight-offloader state. The method builds a
single `ForwardDispatchInfo`, calls `get_offloader().on_dispatch(...)`, and then
calls `CotsHybridKVStore.on_dispatch(...)` when hybrid KV is enabled. This keeps
weight-offload and KV-offload state as sibling subsystem hooks without putting
KV-specific positions/split logic into the weight offloader API.

Post-cleanup validation stayed green: focused hybrid KV tests were `27 passed`;
the forced-context graph parity retry matched forced outputs with top-1
`127/128` and max forced-token logprob delta `0.092062`.

## 2026-05-19: Graph Stability Stress and PIECEWISE Policy Fix

A repeat stress harness was added at
`David/Benchmarks/phase2/stress_hybrid_graph_parity.py`. It runs
`check_hybrid_forced_context_parity.py` in fresh Python processes, preserves
per-run logs/JSON, and continues after EngineCore death so graph instability is
measured as a failure rate instead of disappearing into a poisoned CUDA context.

The first short graph stress reproduced the earlier illegal-memory-access
problem: `B=4`, prompt608/split608/decode32, hybrid graph, no weight offload,
`VLLM_DISABLE_COMPILE_CACHE=1` failed `2/3` runs. With
`CUDA_LAUNCH_BLOCKING=1`, the error surfaced during
`hidden_states[logits_indices]` immediately after model forward, which indicates
a graph replay/capture hazard before sampling rather than a logits-processor
problem. The eager hybrid control passed `3/3`, confirming the math path itself
was stable.

Root cause: hybrid KV-only graph runs were still allowed to use vLLM's
`FULL_AND_PIECEWISE` mode. That can place the COTS suffix host-callback path
inside full CUDA graph capture. Weight-offload runs already avoided this because
Phase 1c auto graph split forced `PIECEWISE` whenever native COTS weight compute
was active. Hybrid KV needs the same policy even when `f_cpu_store=0`.

Fix: `_apply_cots_graph_defaults()` now treats either native COTS weight graph
work or `cots.hybrid_kv_enabled` as COTS CPU-side runtime work requiring
PIECEWISE graph mode. Hybrid KV-only graph runs force `PIECEWISE` but do not add
the weight-offload COTS split ops or switch `cots_capture_sync_mode` to
`wait_kernel`; those remain weight-offload-specific.

Validation after the policy fix:

| Check | Result |
|---|---:|
| Pre-fix graph stress, `B=4`, 3 repeats | `2/3` failed |
| Eager hybrid control, `B=4`, 3 repeats | `0/3` failed |
| Post-fix graph stress, `B=4`, 3 repeats | `0/3` failed |
| Post-fix graph stress + `CUDA_LAUNCH_BLOCKING=1`, `B=4`, 1 repeat | `0/1` failed |
| Post-fix graph stress, `B=8`, 2 repeats | `0/2` failed |
| Focused suffix/hybrid/config tests | `30 passed` |

The fixed graph logs confirm `cudagraph_mode=<CUDAGraphMode.PIECEWISE: 1>` and
show only piecewise capture. The takeaway is important for production shape:
for Phase 2, graph compatibility means split graph around attention/CPU-runtime
boundaries, not full-model CUDA graph capture.

## 2026-05-19: Unified Python COTS Runtime Dispatch

A small Python-side coordinator was introduced at
`vllm/v1/worker/cots_runtime.py` to make Phase 1 weight offload and Phase 2
hybrid KV share the same runner-side pre-forward dispatch boundary.
`GPUModelRunner._publish_forward_dispatch()` now delegates to `CotsRuntime`,
which builds one `ForwardDispatchInfo`, calls the weight offloader singleton,
and then calls the hybrid KV runtime when enabled.

This is a structural cleanup only: it does not merge KV state into
`CotsOffloader`. Weight offload still owns parameter compute/prefetch; hybrid KV
still owns suffix live/scatter counts and CPU suffix metadata. The point is to
make the production invariant explicit: all COTS CPU runtime features receive
their live/bucket dispatch state once before forward.

Validation after the refactor:

| Check | Result |
|---|---:|
| Focused suffix/hybrid/config tests | `30 passed` |
| Graph parity stress, `B=4`, 2 repeats | `0/2` failed |
| Hybrid KV + COTS weight eager, B512 prompt608 split672 total768 | `3186.834 out tok/s` |
| Hybrid KV + COTS weight graph, same config | `3169.612 out tok/s` |
| Phase 1 COTS weight-only eager, same request grid | `3810.298 out tok/s` |
| Phase 1 COTS weight-only graph, same request grid | `3862.019 out tok/s` |

No performance regression was observed versus the previous dispatch-boundary
implementation (`~3195 eager hybrid`, `~3168 graph hybrid`). The next unification
step should be C++ substrate cleanup: move `CotsSuffixAttentionInfer` sync from
its private `cudaLaunchHostFunc` sync callback to the same host-mapped
wait-kernel pattern used by Phase 1 native weight offload. That should unify
sync semantics and may reduce graph replay overhead for suffix attention.



## 2026-05-19: Suffix Attention Wait-Kernel Sync

`CotsSuffixAttentionInfer` now uses the same host-mapped request/done slot
pattern as the Phase 1 native weight runner. Each prepared suffix task can
install one `uint32_t` req/done pair. The submit host callback increments a
per-task sequence, queues the CPU suffix worker, then publishes `req=seq`; the
worker publishes `done=seq` after writing suffix output/LSE, including exception
paths so the GPU wait kernel is not left spinning. The sync custom op is now
`task_id` aware and calls `sync_or_wait_on_stream(task_id, stream)`, which uses
the wait kernel when installed and keeps the previous host-callback sync as a
fallback.

The prepared native suffix runner installs wait-kernel sync for every task at
construction time. This makes Phase 2 suffix sync match the Phase 1 graph
substrate without folding KV logic into the weight offloader.

Validation after the change:

| Check | Result |
|---|---:|
| Rebuild `_cots_C` | passed |
| Focused suffix/hybrid/KV tests | `27 passed` |
| COTS graph-policy config tests | `3 passed` |
| Graph parity stress, `B=4`, 2 repeats | `0/2` failed |
| Hybrid KV + COTS weight eager, B512 prompt608 split672 total768 | `3217.712 out tok/s` |
| Hybrid KV + COTS weight graph, same config | `3201.575 out tok/s` |
| Phase 1 COTS weight-only eager, same request grid | `3805.797 out tok/s` |
| Phase 1 COTS weight-only graph, same request grid | `3823.751 out tok/s` |

The hybrid numbers are slightly above the previous reference band (`3186.834`
eager, `3169.612` graph). The Phase 1-only controls stayed in the same band
(`3810.298` eager, `3862.019` graph previously); the graph run was about 1% lower
than the prior single run, which is within the run-to-run variance observed in
these one-shot E2E measurements. No new correctness or graph-stability issue was
observed.

Next step: use this unified substrate to remove remaining graph-only diagnostic
surface where possible, then move the remaining Phase 2 runner work toward the
same production pattern: fixed-shape graph buckets with out-of-graph live
metadata and no per-replay Python-side synchronization decisions.


## 2026-05-19: Suffix Graph/Eager Attribution Counters

Added env-gated native suffix runner attribution. `CotsSuffixAttentionInfer`
now records, when `VLLM_COTS_SUFFIX_COUNTERS=1` or `VLLM_COTS_DIAG=1`:

- submit/dispatch/worker counts
- worker queue wait and worker busy time
- captured capacity rows, live rows, padded rows, and scatter rows
- wait-kernel launch count
- optional wait-kernel immediate/lagging/spin counters when
  `VLLM_COTS_SUFFIX_WAIT_KERNEL_DIAG=1` or `VLLM_COTS_WAIT_KERNEL_DIAG=1`

The counters are drained through `CotsHybridKVStore` into `CotsHybridKVStats`
and printed by the existing log-stats path. The diagnostic env vars are also
registered in `vllm/envs.py` to avoid unknown-env warning noise. Production
runs with these env vars unset keep the counter work off the hot path except for
cold branch checks.

Validation:

| Check | Result |
|---|---:|
| Rebuild `_cots_C` | passed |
| Focused suffix/hybrid/KV tests | `27 passed` |
| COTS graph-policy config tests | `3 passed` |
| Graph parity stress, `B=4`, 1 repeat | `0/1` failed; top-1 `128/128` |

Diagnostic B512 prompt608 split672 total768 results with suffix counters and
wait-kernel diag enabled:

| Mode | out tok/s | Attribution summary |
|---|---:|---|
| Hybrid graph | `3190.071` | sampled suffix rows always `live == capacity`, `padded=0`; wait kernel lagged on every sampled launch (`28/0/28` launches/immediate/lag in active intervals) |
| Hybrid eager | `3209.068` | same: sampled suffix rows always `live == capacity`, `padded=0`; wait kernel lagged on every sampled launch |

Clean no-diagnostic B512 results after adding the counters:

| Mode | out tok/s |
|---|---:|
| Hybrid graph | `3204.651` |
| Hybrid eager | `3196.406` |

Takeaway: the earlier `~0.5%` graph loss is not stable enough to call a real
regression. With diagnostics off, graph slightly won in this pass; previous
passes had eager slightly ahead. The attribution counters did not reveal a
Graph-only padding problem: at the tested B512 point, suffix captured rows equal
live rows. The wait kernel lag appears in both eager and graph, so it is mostly
showing that CPU suffix attention is on the critical path in the current design,
not that graph replay adds a unique suffix wait.

The next measurement target should be graph split overhead outside the suffix
worker: time the GPU prefix/merge side with CUDA events or Nsight and compare
PIECEWISE graph vs eager around the attention boundary. Current suffix-native
counters are enough to rule out padded suffix rows and task-queue contention as
the main source of the tiny eager/graph variation at this shape.

## 2026-05-19: GPU Prefix/Merge CUDA Event Attribution

Added env-gated CUDA event timing around the GPU prefix FlashAttention call and
the online-softmax merge in `cots_hybrid_decode_attention`. The timing is enabled
only with `VLLM_COTS_HYBRID_CUDA_TIMING=1` or `VLLM_COTS_DIAG=1`, and the new
env var is registered in `vllm/envs.py`. This is diagnostic-only: it records
CUDA events and synchronizes them before reporting, so throughput from timing
runs should not be treated as production throughput.

Validation:

| Check | Result |
|---|---:|
| Python compile, touched files | passed |
| Focused suffix/hybrid/KV tests | `27 passed` |
| COTS graph-policy config tests | `3 passed` |
| Graph parity stress, `B=4`, 1 repeat | `0/1` failed; top-1 `127/128` |

Diagnostic B512 prompt608 split672 total768 results with suffix counters,
wait-kernel diag, and GPU prefix/merge CUDA timing enabled:

| Mode | out tok/s | Attribution summary |
|---|---:|---|
| Hybrid graph | `3179.460` | sampled active intervals: GPU prefix/merge about `4.4/1.7 ms` at full active rows and `2.3/1.0 ms` later; CPU suffix attention about `13.5-19.3 ms`; suffix rows `live == capacity`, `padded=0` |
| Hybrid eager | `3135.291` | sampled active intervals: GPU prefix/merge about `4.5/1.8 ms` at full active rows and `2.3-2.5/0.7-0.9 ms` later; CPU suffix attention about `8.1-13.5 ms`; suffix rows `live == capacity`, `padded=0` |

Clean no-diagnostic B512 results after adding the timing hooks:

| Mode | out tok/s |
|---|---:|
| Hybrid graph | `3201.639` |
| Hybrid eager | `3208.005` |

Takeaway: the timing hooks did not introduce a production regression when
disabled. In the clean run, graph was only `0.20%` below eager, which is within
the run-to-run variation seen in previous one-shot E2E measurements. The CUDA
event attribution also does not point to graph split overhead or merge as the
dominant cost at this shape. CPU suffix attention remains the large visible
component, and the wait kernel is lagging in both graph and eager because the
GPU is genuinely reaching the suffix merge before the CPU suffix task has
finished.

The next useful measurement should be Nsight-level validation of the same
boundary without explicit event synchronization: confirm the graph/eager CUDA
timeline has no unexpected CPU suffix H2D reloads, quantify graph split launch
overhead directly, and verify the wait-kernel spin region lines up with CPU
suffix work rather than Python/runtime scheduling delay.

## 2026-05-19: Nsight Boundary Attribution and Suffix Worker NVTX

Captured Nsight Systems traces without CUDA event timing to validate the hybrid
attention boundary directly. The B512 graph/eager traces used the same production
shape as the clean throughput runs: prompt608, split672, total768, batch512,
COTS weight `2%/2%`, CPU KV pool 4 GiB. The traces were captured with
`--capture-range=cudaProfilerApi` around measured generation, so Nsight stopped
the process at range end and the reports should be treated as structural traces,
not throughput runs.

Trace outputs:

| Mode | Report |
|---|---|
| B512 graph | `/tmp/phase2_b512_graph_boundary.nsys-rep`, `/tmp/phase2_b512_graph_boundary.sqlite` |
| B512 eager | `/tmp/phase2_b512_eager_boundary.nsys-rep`, `/tmp/phase2_b512_eager_boundary.sqlite` |
| B128 graph with suffix-worker NVTX | `/tmp/phase2_b128_graph_suffix_nvtx.nsys-rep`, `/tmp/phase2_b128_graph_suffix_nvtx.sqlite` |

B512 graph/eager CUDA summary:

| Metric | Graph | Eager |
|---|---:|---:|
| GPU event span | `26.324 s` | `26.226 s` |
| GPU active union | `21.175 s` | `21.668 s` |
| GPU idle union | `5.150 s` | `4.559 s` |
| Gaps >=100 us | `12386`, `4065 ms` | `11836`, `3915 ms` |
| Gaps >=500 us | `926`, `1976 ms` | `932`, `1917 ms` |
| Wait-kernel GPU time | `5542.8 ms` | `5489.9 ms` |
| FlashAttention split-kernel time | `2324.4 ms` | `2301.8 ms` |
| Merge kernel time | `514.0 ms` | `492.3 ms` |
| GEMM-family kernel time | `11770.2 ms` | `11844.1 ms` |
| CUDA graph launches | `12760`, `343.5 ms`, `26.9 us avg` | none |

B512 memcpy summary:

| Direction | Graph | Eager | Interpretation |
|---|---:|---:|---|
| H2D | `103.388 GB`, max `2.753 MB` | `103.388 GB`, max `2.753 MB` | Weight prefetch-sized transfers dominate. |
| D2H | `12.552 GB`, max `1.233 MB` | `12.552 GB`, max `1.668 MB` | Q/KV artifact staging, not suffix reload. |
| D2D | about `3.5 MB` | about `3.5 MB` | Negligible. |

No bulk CPU-suffix reload appears. A suffix KV reload at this shape would require
large H2D transfers for suffix KV blocks; the largest H2D copy in both traces is
only the `2.753 MB` weight-prefetch-sized transfer, and the total H2D volume is
identical between graph and eager. D2H volume is also identical, matching the
expected direction for Q/KV artifacts.

To close the remaining attribution gap, added env-gated C++ NVTX ranges to
`CotsSuffixAttentionInfer` under existing `VLLM_COTS_NVTX=1`:

- `cots:suffix_dispatch_cb`
- `cots:suffix_worker`
- `cots:suffix_scatter`
- `cots:suffix_attention`
- `cots:suffix_zero_live`
- `cots:suffix_sync_cb_wait` for the fallback host-callback sync path

The follow-up B128 graph trace used prompt608, split672, total704, batch128,
COTS weight `2%/2%`, CPU KV pool 1 GiB. Its Nsight totals show the CPU suffix
worker ranges line up with the GPU wait-kernel spin:

| B128 graph component | Calls | Total | Avg |
|---|---:|---:|---:|
| `cots:suffix_worker` | `896` | `248.584 ms` | `277.4 us` |
| `cots:suffix_attention` | `896` | `200.872 ms` | `224.2 us` |
| `cots:suffix_scatter` | `896` | `37.882 ms` | `42.3 us` |
| `cots:suffix_dispatch_cb` | `896` | `2.659 ms` | `3.0 us` |
| GPU wait kernel | `896` | `255.808 ms` | `285.5 us` |
| GPU merge kernel | `896` | `41.985 ms` | `46.9 us` |
| GPU split FlashAttention kernels | `2744` | `271.176 ms` | `98.8 us` |

Takeaway: the wait-kernel spin is explained by the CPU suffix worker, not by a
separate graph-only scheduling hole. The graph/eager B512 traces are also close
on the large idle-gap counts and wait-kernel totals. CUDA graph split launch API
cost is measurable (`343.5 ms` total under Nsight for B512), but it is much
smaller than wait-kernel time, GEMM time, or H2D weight-prefetch time and does
not explain a production throughput gap.

Validation after adding suffix-worker NVTX:

| Check | Result |
|---|---:|
| Rebuild `_cots_C` | passed |
| Focused suffix/hybrid/KV tests | `27 passed` |
| Clean B512 hybrid graph | `3208.977 out tok/s` |
| Clean B512 hybrid eager | `3206.248 out tok/s` |

The diagnostic NVTX ranges are useful enough to keep for now because they are
env-gated and directly support Nsight attribution. They should remain classified
as diagnostic surface: production benchmarking should keep `VLLM_COTS_NVTX`
unset.



## Transparent CPU Suffix Cache Management

Implemented the first management-layer step toward treating CPU KV as a
transparent extension of GPU KV. The COTS CPU suffix block pool now tracks CPU
blocks by request, block hash, refcount, and a simple LRU eviction order over
zero-ref cached blocks.

Current behavior:

- GPU prefix blocks still use vLLM's normal prefix-cache coordinator.
- CPU suffix blocks are cached only for full blocks at global block indices
  `>= cots_kv_split_blocks`.
- Freed hashed CPU suffix blocks remain resident and reusable by later requests.
- Allocation can evict zero-ref cached CPU suffix blocks when fresh CPU capacity
  is needed.
- `reset_prefix_cache()` now resets both tiers, but fails while either tier has
  live referenced blocks.
- `KVCacheBlocks` can carry both GPU block objects and CPU suffix block IDs, so
  the scheduler/worker interface remains the same shape as the existing hybrid
  path.

This intentionally does not reuse vLLM native `kv_offload` load/store semantics.
The native system is built around CPU-resident blocks being loaded back to GPU;
COTS Phase 2 keeps suffix KV on CPU and uses CPU attention directly, so the
useful part to mirror is the cache bookkeeping idea, not the CPU->GPU reload
path.

Historical note: this section originally landed as management-only support.
The next section adds first-pass CPU suffix prefill execution for prefix-cache
residual chunks after the split.

Pressure behavior now covered:

- KV cache config generation now applies COTS-aware capacity checks beside
  vLLM's native GPU KV check: GPU only needs the block-aligned prefix, while
  the CPU pool must fit one full max-length suffix
  `ceil((max_model_len - split_tokens) / block_size)` per layer group. This
  keeps the CPU side symmetric with the GPU cache requirement that at least one
  request must be runnable. Startup capacity reporting is also COTS-aware and
  reports GPU-prefix concurrency, CPU-suffix concurrency, and the effective
  min of the two.
- Cached zero-ref CPU suffix blocks remain resident, count as used capacity, and
  are evicted when fresh CPU capacity is needed.
- Live cached CPU suffix blocks are protected from eviction. Allocation returns
  `None` instead, preserving the existing scheduler preemption/recompute path.
- Once the live owner is freed, the same cached block can be evicted and reused
  by a new suffix sequence.

Full vLLM e2e pressure validation now uses
`David/Benchmarks/phase2/check_hybrid_cpu_pool_pressure_e2e.py`. The probe uses
a tiny two-block CPU suffix pool: one block is needed for the cached suffix hit,
and one block is needed for the live partial tail token. A one-block pool cannot
make scheduler progress for a `split + full_suffix_block + tail` prompt, so the
probe fails fast below 2 MiB.

The e2e pressure sequence is:

- Populate suffix A, leaving one full CPU suffix block cached.
- Populate a different suffix B; this must evict cached A under the two-block
  pressure shape.
- Reuse B and observe GPU+CPU hits.
- Reuse A and observe GPU-only hits, proving A's CPU suffix block was evicted.

Validation:

| Check | Result |
|---|---:|
| COTS KV config sizing/reporting tests | `4 passed` |
| COTS hybrid config derivation + unsupported-mode tests | `2 passed` |
| Hybrid KV manager tests | `11 passed` |
| Focused suffix/hybrid/worker/KV tests | `42 passed` |
| Focused capacity/scheduler hardening slice | `7 passed` |
| Eager CPU pool pressure e2e | `populate_b=64`, `b_hit=80`, `a_after_eviction=64`, passed |
| Graph CPU pool pressure e2e | `populate_b=64`, `b_hit=80`, `a_after_eviction=64`, passed |

Runtime hardening after the capacity checks: `_validate_cots_hybrid_kv_config()`
now fails fast if COTS hybrid KV is combined with native vLLM KV offloading,
vLLM KV transfer/connectors, non-single-GPU parallel modes, encoder-decoder
models, or non-BF16 model/KV dtype. The existing async-scheduling rejection is
kept. This keeps the Phase 2 runtime inside the tested thesis envelope:
single-GPU, decoder-only, BF16, COTS-owned CPU suffix KV. The scheduler unit-test
fixture now selects BF16 automatically when it constructs a COTS hybrid-KV
scheduler, so unit tests exercise the supported path instead of tripping this
production guard accidentally.


## CPU Suffix Prefill Execution

Added the first transparent execution support for CPU suffix prefix-cache hits.
The implementation is intentionally simple: suffix prefill is represented as
row-expanded decode-style CPU attention. Each scheduled token at position
`>= split_tokens` becomes one CPU suffix attention row with its own suffix-local
visible length.

Supported now:

- Prefix-cache residual prefill after the split, including the normal vLLM
  block-granular residual where `q_len` can be `1..block_size`.
- Repeated rows for the same request in one prefill chunk.
- GPU prefix attention over `[0, x)` with `causal=False`, CPU suffix attention
  over `[x, current_position]`, then online-softmax merge.
- CPU suffix K/V scatter for every suffix prefill token before the CPU suffix
  attention worker reads that token's visible suffix.
- Scheduled chunks that cross the planner split: rows before `x` run normal
  one-token GPU attention with their own `position + 1` visible length; rows at
  or after `x` run the hybrid GPU-prefix + CPU-suffix merge.

The native suffix attention kernel did not need a new multi-query kernel for
this pass. At this point it accepted a batch of independent `[28, 128]` query
rows, so prefill rows were flattened into that batch. This is now superseded by
the generic GQA kernel noted in the current contract.

Graph note: decode graph buckets remain on the existing fixed task IDs. If a
row-expanded prefill batch exceeds the decode batch staging capacity, it uses a
per-layer overflow native task and synchronous D2H staging. This keeps graph
decode stable while making CPU suffix prefill functionally correct first.

Still unsupported by design in this pass:

- CUDA graph capture/replay for row-expanded suffix prefill overflow tasks.
- At this point in the chronology, any model shape other than the then-current
  Qwen2.5-7B BF16 suffix kernel shape. This is now superseded by the generic
  GQA kernel noted in the current contract.

Validation after adding CPU suffix prefill:

| Check | Result |
|---|---:|
| Worker metadata + CUDA hybrid attention tests | `25 passed` |
| Focused suffix/hybrid/worker/KV tests | `37 passed` |
| Scheduler mixed waiting-prefill regression | `1 passed` |

## CPU Suffix Prefix-Cache E2E Validation

Added `David/Benchmarks/phase2/check_hybrid_prefix_cache_e2e.py` to validate
transparent CPU suffix prefix-cache reuse in a full vLLM engine run. The probe
runs two requests in one engine:

- Request A prompts exactly to `split_tokens`, then decodes one extra token past
  the suffix block that Request B will reuse. This is required because vLLM
  samples a generated token before that token's KV is materialized; generated
  token `n` becomes cacheable on decode step `n + 1`.
- Request B uses Request A's prompt plus one full generated CPU suffix block,
  then one uncached tail token. A successful run must report prefix-cache hits
  greater than the GPU split; for the current probe that means `80 = 64 GPU
  prefix + 16 CPU suffix` hit tokens.

The e2e probe exposed one integration bug: row-expanded suffix prefill overflow
published graph-style live-count overrides before all previously submitted
stream callbacks had drained. That could make the native suffix worker observe a
runtime scatter count larger than the prepared task's captured capacity. The
fix is to clear live-count overrides in `CotsHybridKVStore.on_dispatch()` when
`suffix_rows > max_num_reqs`; overflow tasks are populated with exact live row
and scatter counts and do not need graph bucket overrides.

Validation after the fix:

| Check | Result |
|---|---:|
| Focused Phase 2 regression suite | `38 passed` |
| Eager e2e CPU suffix prefix-cache probe | `second_delta_hits=80`, passed |
| Graph e2e CPU suffix prefix-cache probe | `second_delta_hits=80`, passed |
| Eager multi-request CPU suffix prefix-cache probe | `delta_hits=320` for 4 requests, passed |
| Graph multi-request CPU suffix prefix-cache probe | `delta_hits=320` for 4 requests, passed |

The graph probe used the existing COTS piecewise graph path (`enforce_eager=false`)
and confirmed that CPU suffix prefix-cache reuse works through the graph-enabled
runner for the tested Qwen2.5-7B BF16 paths.

The multi-request probe is `David/Benchmarks/phase2/check_hybrid_prefix_cache_multi_e2e.py`.
It populates one CPU suffix block, then submits four cache-hit prompts with
different uncached tail tokens. The aggregate cache metric is the useful signal:
`delta_queries=324`, `delta_hits=320`, which is exactly
`4 * (64 GPU prefix + 16 CPU suffix)` hit tokens.

## 2026-05-20: Runtime Completion Pass

This pass closed the remaining runtime-shape issues around the transparent CPU
suffix cache, startup diagnostics, row-expanded suffix prefill, and native
suffix-runner lifetime. The production path now keeps the diagnostic metrics and
NVTX ranges env-gated, and removes failed experimental modes from the runtime.

Implementation details that matter for correctness:

- Startup validation now rejects unsupported hybrid-KV combinations before the
  worker reaches late `NotImplementedError` paths. This includes native vLLM KV
  offload, KV transfer/connectors, non-single-GPU parallel modes,
  encoder-decoder models, non-BF16 model/KV dtype, and async scheduling.
- The CPU suffix kernel and worker validate CPU block-table bounds before
  running attention, so invalid scheduler/worker metadata fails with a clear
  diagnostic instead of reading past the CPU KV pool.
- Native suffix submitted tasks now snapshot task metadata for eager submissions
  and retain graph-submitted task descriptors for graph replay. Graph submissions
  use a no-op shared-pointer deleter; eager submissions are callback-owned.
- Eager submitted tasks originally snapshotted Q/QKV inputs inside the stream
  host callback, after stream-ordered D2H staging copies completed. That fixed
  stale staging reads but was too expensive. The follow-up leased-staging path
  now disables this snapshot for static decode buckets while keeping it for
  row-expanded overflow and diagnostic paths.
- Static staging buffers and static CPU metadata are per layer and guarded by
  reuse events. Suffix-prefill overflow clears live-count overrides and uses
  exact task sizes so a large residual chunk cannot corrupt a smaller captured
  decode bucket.

Validation after the callback-snapshot safety path:

| Check | Result |
|---|---:|
| Rebuild `_cots_C` | passed |
| Suffix attention + native runner tests | `13 passed` |
| Focused Phase 2 regression suite | `52 passed` |
| Config/capacity hardening slice | `7 passed` |
| Eager single-request prefix-cache e2e | `second_delta_hits=80`, passed |
| Graph single-request prefix-cache e2e | `second_delta_hits=80`, passed |
| Eager multi-request prefix-cache e2e | `delta_hits=320`, passed |
| Graph multi-request prefix-cache e2e | `delta_hits=320`, passed |
| Eager CPU-pool pressure e2e | `b_hit=80`, `a_after_eviction=64`, passed |
| Graph CPU-pool pressure e2e | `b_hit=80`, `a_after_eviction=64`, passed |
| Forced-context eager hybrid parity | forced outputs matched; post-split top1 `383/384`; post-split forced-logprob delta max/mean/p95 `0.0645/0.00270/0.0156` |
| Forced-context graph hybrid parity | forced outputs matched; post-split top1 `384/384`; post-split forced-logprob delta max/mean/p95 `0.1096/0.00282/0.0152` |
| Matched GPU graph control | forced outputs matched; post-split top1 `383/384`; post-split forced-logprob delta max/mean/p95 `0.0814/0.00260/0.0107` |
| Graph stress, fresh processes | `2/2` runs passed; zero engine failures |

The final parity check is back in the old small-probe envelope after moving the
eager Q/QKV snapshot from submit time to the stream callback. The broken version
was instructive: copying staged QKV at submit time is memory-safe but can read
stale CPU staging data before the D2H copy completes, causing large post-split
logit drift.

Intermediate performance evidence after the callback-snapshot safety fix was
more conservative than the earlier unsafe no-snapshot prototype. The B512,
prompt608/split608/total768, no-weight-offload,
`gpu_memory_utilization=0.68`, `max_num_seqs=512` run completed without crash but
measured `1153.984` output tok/s. The matched GPU-only reference from the same
setup was about `3116` output tok/s. The follow-up leased-staging section below
replaces this as the latest safe-path performance result.

Latest Nsight evidence remains the 2026-05-19 boundary trace. It already confirms
the important traffic invariant for this runtime: no bulk CPU suffix KV reload to
GPU appears, H2D is weight/prefetch-sized, and the new suffix artifacts move D2H
plus UVA-read by the merge path. No new Nsight trace was captured after the
callback-snapshot safety fix because the fix changes CPU-side staging lifetime,
not the intended transfer directions.

Phase 2 handoff status: the runtime mechanism is now implemented and guarded for
the tested envelope, with prefix cache and piecewise graph functionality covered
by e2e probes. The largest remaining non-planner risk is performance: the safe
native runner needs a lower-overhead way to make Q/QKV staging lifetime-stable,
or the later planner must choose splits and batch regimes where this cost is
small enough to hide.



## 2026-05-20 Follow-up: Leased Static Staging

The final safety/performance follow-up moved static decode closer to the Phase 1
activation-buffer pattern: prepared suffix tasks now use four per-layer staging
slots, and a CUDA event lease protects each slot until the GPU merge has consumed
the CPU suffix output/LSE. For static decode buckets this lets the native CPU
suffix worker read the staged CPU Q/QKV directly, so the eager callback no longer
needs to snapshot Q/QKV. Row-expanded suffix-prefill overflow remains on the
callback-snapshot path because it is exact-size/eager and not graph-captured.

Two fixes were needed to make this production-safe:

- Native suffix tasks gained an explicit `snapshot_inputs` flag. Static staged
  tasks pass `False`; overflow and diagnostic paths keep `True`.
- Static LSE staging is now per slot and per live batch size, not a sliced
  `[num_heads, max_num_reqs]` slab. The native runner assumes contiguous
  `[num_heads, batch]`, and slicing the second dimension was only safe when
  `batch == max_num_reqs`.

Phase 2 supports exactly four staging slots. An eight-slot experiment crashed in
the native dispatch callback, so the runtime keeps the validated four-slot
setting fixed instead of exposing an unstressed native lifetime shape.

Validation after leased staging:

| Check | Result |
|---|---:|
| Focused suffix/hybrid/worker/KV/scheduler regression suite | `52 passed` |
| Unsupported staging-slot override | fails fast with clear `ValueError` |
| Forced-context eager hybrid parity | forced outputs matched; post-split top1 `383/384`; post-split forced-logprob delta max/mean/p95 `0.0877/0.00238/0.00840` |
| Forced-context graph hybrid parity | forced outputs matched; post-split top1 `383/384`; post-split forced-logprob delta max/mean/p95 `0.0678/0.00249/0.0104` |
| Graph stress, fresh processes | `2/2` runs passed; zero engine failures |

Performance after leased staging, no weight offload, B512/prompt608/total768,
`gpu_memory_utilization=0.68`, `max_num_seqs=512`:

| Mode | Split | Output tok/s | Notes |
|---|---:|---:|---|
| Hybrid KV | 608 | `1181.611` | Stable, but slow. |
| Hybrid KV | 672 | `1514.532` | Better due to shorter CPU suffix, still far below GPU-only. |
| GPU-only | 672 argument ignored | `3059.735` | Same harness and memory budget; vLLM capacity-limits active requests. |

This closes the immediate buffer-lifetime safety question but does not recover
the earlier high-throughput no-snapshot number. The rejected no-snapshot path was
fast because it allowed unsafe reuse/overlap and reproduced the old B512 native
segfault. The current safe leased path still leaves a large performance gap, so
the next useful work is attribution of the remaining cost: staging-slot waits,
CPU suffix task launch/queue time, suffix attention kernel time, wait-kernel
spin time, and graph/eager boundary overhead.

## 2026-05-20 Follow-up: Hybrid/GPU Gap Attribution

After leased staging made the native suffix runner memory-safe again, the main
question was whether the remaining hybrid-vs-GPU gap came from staging lifetime
machinery, graph/eager boundaries, or the CPU suffix attention itself. The
answer from the current probes is: the gap is dominated by CPU suffix attention
work, and it scales strongly with suffix length.

Diagnostic B512, prompt608/split672/total768, no weight offload,
`gpu_memory_utilization=0.68`, `max_num_seqs=512`, counters/timing enabled:

| Signal | Observation |
|---|---:|
| End-to-end throughput with diagnostics | `1489.384` output tok/s |
| CPU suffix read wait | about `0.05 ms` per logged forward |
| QKV ready wait | `0.000 ms` in the logged forwards |
| Native CPU suffix busy time | about `14 ms` per logged forward |
| Native CPU suffix queue time | about `0.1 ms` per logged forward |
| GPU prefix attention + merge | a few ms, below CPU suffix busy time |

The microbenchmark agrees with the e2e attribution. For the representative
post-split shape `batch=126`, `suffix_seq_len=96`, `layers=28`, pinned inputs,
the native suffix runner sits around `15-16 ms` depending on thread count. This
matches the e2e CPU busy time, so the safe staging lease itself is not the large
remaining tax.

A small CPU-kernel optimization attempt was tested and rejected: the public
validation was skipped from the native runner path and the extra probability
normalization pass was folded into the V pass. The change rebuilt cleanly and the
suffix attention tests still passed, but it did not improve either the
microbenchmark or the B512 e2e throughput. The experiment was reverted so Phase
2 does not carry a failed kernel variant.

The useful split-placement sweep, no weight offload, B512/prompt608/total768,
`gpu_memory_utilization=0.68`, `max_num_seqs=512`:

| Mode | Split | Output tok/s | Relative to GPU-only |
|---|---:|---:|---:|
| GPU-only | n/a | `3059.735` | `1.00x` |
| Hybrid KV | 672 | `1514.532` best prior clean run, `1507.513` repeat | `0.49x` |
| Hybrid KV | 736 | `2197.847` | `0.72x` |
| Hybrid KV | 752 | `2553.205` | `0.83x` |
| Hybrid KV graph | 752 | `2501.676` | `0.82x` |

The split sweep shows the current implementation behaves as expected but is
still expensive: moving `x` later reduces the CPU suffix and closes most of the
gap, while graph mode does not remove the remaining cost. Even a one-block CPU
suffix remains slower than the matched GPU-only harness in this unconstrained
throughput comparison.

A follow-up apples-to-apples rerun with the old Phase 1 weight path enabled
(`cots_f_cpu_store=0.02`, `cots_f_prefetch=0.02`), B512/prompt608/split672/
total768, measured `1693.392` output tok/s on the current safe runtime. This
separates two effects: the `1181-1514` results above are Phase-2-only/no-weight
runs, but the current safe path is still far below the previous `~3200` hybrid
with-weight band. The remaining regression is therefore not just the absence of
Phase 1 weight offload; it is the loss of the old unsafe no-lease staging overlap.

One production cleanup was kept from this pass. `create_offloader()` now returns
`NoopOffloader` for Phase-2-only COTS runs where `f_cpu_store == 0` and
`f_prefetch == 0`. Hybrid KV still selects the COTS runtime from config, but
there is no Phase 1 weight dispatch bookkeeping on every forward. The split752
rerun measured `2552.263` output tok/s, so this is a semantic cleanup rather
than a material performance fix.

Current conclusion: the remaining gap is not explained by Q/QKV staging waits,
GPU merge cost, queueing, or graph boundary overhead. It is the CPU suffix
attention kernel/work granularity itself. Closing it materially requires a
larger change than cleanup: a faster CPU suffix kernel, more aggressive
per-layer/task batching, or a planner/runtime policy that chooses very late
splits and only uses CPU KV in tight-memory capacity regimes. GPU-tail fallback
and CPU-aware admission could also help, but they remain outside the Phase 2
runtime contract.

## 2026-05-20 Follow-up: Safe Overlap Attempts Toward Throughput Win

The next attempt targeted the safest recoverable overlap from the old unsafe
staging path. The native suffix submit was previously queued on the main
attention stream after GPU prefix attention was enqueued, using `prefix_out` as
the custom-op CUDA anchor. The new eager-only path submits the native suffix task
on the existing per-layer Q/QKV copy stream, waits on the Q/QKV ready event
there, and leaves the main stream to wait only at the merge point. CUDA graph
capture keeps the old main-stream path for now.

Implementation kept:

- `CotsHybridDecodeMetadata` now carries an optional `suffix_submit_stream`.
- `CotsPreparedNativeSuffixAttentionRunner.run_gqa_bf16_suffix_attention()` can
  submit on that stream and sync on the caller stream.
- The side-stream path is gated by `not torch.cuda.is_current_stream_capturing()`
  so graph behavior is unchanged.
- The submit/sync custom op now uses `query` as the CUDA device anchor instead
  of `prefix_out`, avoiding an accidental dependency on prefix attention output.

Validation for the kept change:

| Check | Result |
|---|---:|
| Focused suffix/hybrid/worker/KV regression suite | `51 passed` |
| `git diff --check` | passed |
| Rebuild after reverted C++ trial | passed |

Performance evidence, B512/prompt608/total768, `gpu_memory_utilization=0.68`,
`max_num_seqs=512`, eager:

| Mode | Split | Weight offload | Output tok/s | Interpretation |
|---|---:|---|---:|---|
| Previous safe hybrid | 672 | COTS `0.02/0.02` | `1693.392` | Apples-to-apples baseline before side-stream submit. |
| Side-stream hybrid | 672 | COTS `0.02/0.02` | `1755.664-1760.170` | About `+4%`; correct direction, not enough. |
| Side-stream hybrid | 736 | COTS `0.02/0.02` | `2700.320` | Later split reduces CPU suffix cost but still below GPU-only/weight-only references. |
| Side-stream hybrid | 752 | COTS `0.02/0.02` | `3053.355` | Roughly tied with no-weight GPU-only `3059.735`, but capacity gain is only about one block/request. |
| GPU-only | n/a | none | `3059.735` | Reference from same B512/prompt608/total768 harness. |

Thread and kernel-granularity checks did not produce a real win:

| Trial | Result |
|---|---:|
| `OMP_NUM_THREADS=32 MKL_NUM_THREADS=32`, split672 with side-stream | `1760.170` output tok/s; no material change. |
| CPU suffix microbench, pageable vs pinned, B126/seq96/layers28 | Same band; CPU KV pinning is not the large gap. |
| Hugepage-advised microbench | Only a few percent local improvement. |
| Request-granularity CPU kernel parallelization | Rejected and reverted; worsened 24-thread microbench and was neutral at 32 threads. |

A prompt-near-split probe also did not establish a throughput win. With
prompt736/split752/total768 and no weight offload, GPU-only measured
`3884.222` output tok/s while hybrid measured `3603.565` output tok/s. The CPU
suffix is small in this regime, but the extra capacity still did not repay the
hybrid attention/merge overhead in this short-decode synthetic harness.

Current conclusion: safe side-stream submission recovers only a small part of
the old unsafe-overlap number. The Phase 2 throughput claim remains unproven and
currently false for the tested B512/Qwen2.5-7B synthetic regimes. The next
material path is not more staging cleanup; it is either a substantially faster
CPU suffix attention implementation, or a planner/runtime regime where GPU-only
is strongly KV-starved and the CPU suffix length remains within the measured
small-overlap window. Any such path must be validated against forced-context
parity before being treated as a real win.



## 2026-05-20 Follow-up: Corrected Side-Stream and Selective Overflow Probe

The previous eager side-stream suffix-submit experiment produced an apparent
throughput improvement, but forced-context parity showed stale-output behavior:
post-split top-1 fell to `227/384` and forced-token logprob deltas were large.
Root cause: submit was enqueued on the D2H/query stream, while the main stream
could launch the wait-kernel sync before the submit host callback had published
the new request sequence. The fix records a per-staging-slot submit-done event
after the submit callback node and makes the main stream wait on that event
before launching suffix sync.

Corrected side-stream parity is back in the previous numerical envelope for the
split672 forced-context probe (`B=4`, prompt608, decode160, weight split
`0.02/0.02`): forced outputs matched, post-split top-1 was `383/384`, and
post-split forced-token logprob delta max/mean/p95 was
`0.0887/0.00360/0.0230`.

Capacity-frontier measurements after the correctness fix still do not make the
static all-rows split useful:

| Mode | GPU util | Total | Split | Batch | out tok/s |
|---|---:|---:|---:|---:|---:|
| GPU-only, async off | 0.68 | 768 | n/a | 512 | 3558.326 |
| Hybrid all-rows | 0.68 | 768 | 672 | 512 | 1723.442 |
| Hybrid all-rows | 0.68 | 768 | 752 | 512 | 3044.413 |
| GPU-only, async off | 0.66 | 768 | n/a | 512 | 2139.079 |
| Hybrid all-rows | 0.66 | 768 | 672 | 512 | 1205.268 |
| Hybrid all-rows | 0.66 | 768 | 752 | 512 | 1806.387 |

A minimal selective-overflow prototype was also tested: requests stay
GPU-only while full GPU KV fits; overflow requests use CPU suffix. This avoids
charging CPU suffix to every active row, but the tested version still does not
beat GPU-only:

| Mode | GPU util | Total | Split | Batch | out tok/s |
|---|---:|---:|---:|---:|---:|
| GPU-only, async off | 0.66 | 768 | n/a | 512 | 2139.079 |
| Selective hybrid | 0.66 | 768 | 672 | 512 | 1343.021 |
| Selective hybrid | 0.66 | 768 | 752 | 512 | 1922.378 |
| GPU-only, async off | 0.66 | 1024 | n/a | 512 | 1133.488 |
| Selective hybrid | 0.66 | 1024 | 896 | 512 | 942.103 |

Interpretation: the throughput claim remains unproven and false for the tested
runtime. Correcting the stream race removes the invalid speedup. Selective
overflow is directionally better than all-rows CPU suffix for near-tail splits,
but not enough. The next credible work is not another small stream tweak; it is
either a much cheaper CPU suffix path, or a scheduler design that overlaps CPU
suffix work with independent GPU work more like NEO's two-sub-batch policy.
Until then, Phase 2 should be described as a correct mechanism and negative
throughput result in these regimes, not as a win.


## 2026-05-20 Follow-up: Tight-Memory Selective Hybrid Win and Stats Fix

A later B512 probe found one real throughput-positive regime, but only after
fixing a measurement artifact. The selective-overflow policy is the relevant
mechanism and is now the default COTS hybrid KV placement: requests stay
full-GPU while full GPU KV fits; overflow requests use CPU suffix KV.

First, one invalid result was identified and fixed. With `--enable-log-stats`,
`make_cots_hybrid_kv_stats()` was resetting `cots_hybrid_full_gpu_req_ids` and
`cots_hybrid_cpu_suffix_req_ids`. That made logging mutate the request placement
policy and produced an artificial `~1030` output tok/s run at
`gpu_memory_utilization=0.65`. The stats path now preserves those sets, and
`tests/v1/core/test_hybrid_kv_cache_manager.py` covers this regression.
The obsolete `VLLM_COTS_HYBRID_SELECTIVE_FULL_GPU` diagnostic toggle was removed
after this policy became the default. A post-cleanup eager parity smoke without
that env passed at `B=4`, prompt608/split608/decode160: forced outputs matched,
top1 `640/640`, forced-token delta max/mean/p95 `0.1228/0.00418/0.0289`, and
top20 Jaccard mean `0.9787`.

Throughput evidence after removing the logging artifact, Qwen2.5-7B, B512,
prompt608/split608/total768, `max_num_seqs=512`, eager, COTS weight split
`0.02/0.02`, diagnostics off:

| Mode | GPU util | out tok/s | Interpretation |
|---|---:|---:|---|
| GPU-only, async off | 0.650 | `606.537` prior matched run | GPU-only is capacity-limited but still slightly faster than hybrid. |
| Selective hybrid | 0.650 | `568.140` confirmed rerun | Still loses; no throughput claim here. |
| GPU-only, async off | 0.648 | `164.468` | GPU-only falls into the tight-memory capacity cliff. |
| Selective hybrid | 0.648 | `412.848` | Real win: `2.51x` over matched GPU-only. |

This makes the Phase 2 capacity claim true only in a narrow tight-memory regime:
below the `0.65` point GPU-only throughput collapses much faster than the CPU
suffix cost grows. At `0.68`, `0.66`, and `0.65`, the current safe
implementation still does not beat matched GPU-only for the same
B512/prompt608/total768 synthetic workload.

Correctness/parity evidence for the winning regime is acceptable but not
bit-exact. Forced-context parity at `gpu_memory_utilization=0.648`, `batch=16`,
`max_num_seqs=512`, prompt608/split608/decode160, COTS weight split
`0.02/0.02`:

| Probe | Result |
|---|---|
| GPU-vs-GPU control | exact: top1 `2560/2560`, all deltas `0`. |
| Hybrid forced outputs | matched reference continuation for all requests. |
| Hybrid top1 agreement | `2549/2560`. |
| Hybrid forced-token logprob delta | max `0.152719`, mean `0.006098`, p95 `0.041079`. |
| Hybrid top20 Jaccard | mean `0.97548`, p95 `1.0`. |

A separate `batch=16`, `max_num_seqs=16`, `gpu_memory_utilization=0.648` parity
run produced hybrid NaNs in the captured logits; this did not reproduce when
the parity probe used `max_num_seqs=512`, which matches the B512 benchmark
capacity setting. Treat the `max_num_seqs=512` result as the relevant evidence
for this throughput claim, and keep the smaller-capacity-mismatched NaN as a
diagnostic item rather than proof of the B512 path.

Current conclusion: Phase 2 is no longer uniformly negative. It has a measured
B512 throughput win when GPU KV memory is tight enough (`0.648`) and
output-forced parity remains within the current BF16 hybrid tolerance envelope.
The claim should still be stated narrowly: selective CPU suffix KV is useful as
a capacity-extension mechanism at the GPU memory cliff, not as a general
replacement for GPU-only KV when GPU-only still has enough effective
prefix-shared capacity.


## 2026-05-20 Follow-up: Final-Aware Selective Placement and Current Tight-Memory Evidence

A later current-tree rerun narrowed the previous tight-memory conclusion and
found one real selective-placement bug. The old `412.848 out tok/s` split608
number did not reproduce and should not be used as stable evidence. More
importantly, the selective path was deciding `full GPU` from the next allocation
only. That could classify a request as full-GPU even though its full
`prompt + max_tokens` context could not fit, creating late overcommit/preemption
and mixed-batch parity drift.

The fix is intentionally small: a request is assigned to full-GPU KV only
when its final context target `min(prompt_tokens + max_tokens, max_model_len)`
can be allocated/reserved on GPU. Otherwise it uses CPU suffix KV from the first
post-split allocation. This keeps placement from changing mid-decode.

Current B512/Qwen2.5-7B/eager/COTS `0.02/0.02`, `prompt=608`, `total=768`,
`gpu_memory_utilization=0.648`, `max_num_seqs=512`, diagnostics off results:

| Mode | Split tokens | Out tok/s | Interpretation |
|---|---:|---:|---|
| GPU-only, async off | n/a | `164.468`, `164.608` | Mean `164.538`; matched tight-memory cliff reference. |
| Selective hybrid, final-aware | 608 | `377.620`, `375.865` | Mean `376.743`; stable `2.29x` win over matched GPU-only at the cliff. |
| Selective hybrid, final-aware | 672 | `133.399` | Still loses; split672 spends too much GPU capacity before CPU KV helps. |

The repeat makes the capacity claim stable again, but only for the planner-relevant
choice where the split is very close to the shared prompt. Split672 is not a
valid win point at this memory budget because effective GPU-prefix concurrency is
only about `1.40x`; there is not enough extra capacity to repay the CPU suffix
and hybrid merge cost.

Correctness/parity after the final-aware fix:

| Probe | Result |
|---|---|
| Selective batch2 before fix | forced outputs matched, but `top1=309/320`, max delta `7.222`; bad mixed-placement drift. |
| Non-selective batch2 CPU suffix | forced outputs matched, `top1=318/320`, max delta `0.106`; CPU suffix kernel itself was not the issue. |
| Selective batch2 after fix | forced outputs matched, `top1=320/320`, max delta `0.106`; mixed-placement drift fixed. |
| Selective batch16 after fix | forced outputs matched; current rerun `top1=2548/2560`, forced-token delta max/mean/p95 `0.1527/0.00660/0.0420`, top20 Jaccard mean `0.9725`. |

A second small correctness fix was also kept: GPU prefix-cache commits now use
the same request-aware GPU-token ownership as allocation. Before this, a request
selected for full-GPU KV could still commit prefix-cache hashes only up to the
hybrid split. Focused unit coverage now checks both final-context selective
placement and full-GPU selective cache commits.


### Split672 Cutoff Diagnosis

Additional final-aware selective B512 runs at the same tight-memory point fill in
the curve between split608 and split672:

| Mode | Split tokens | Effective init concurrency | Prefix-shared unique GPU blocks/request | Approx. concurrent unique requests after shared prefix | Out tok/s |
|---|---:|---:|---:|---:|---:|
| Selective hybrid | 608 | `1.55x` | 1 | 22 | `377.620`, `375.865` |
| Selective hybrid | 640 | `1.48x` | 3 | 7 | `221.084` |
| Selective hybrid | 656 | `1.44x` | 4 | 5 | `183.676` |
| Selective hybrid | 672 | `1.40x` | 5 | 4 | `133.399` |
| GPU-only reference | 768 | `1.23x` | 11 | 2 | `164.468` |

The key detail is prefix sharing. In this harness, prompts share `607` of
`608` tokens, so only `37` full 16-token blocks are shared. With the observed
`944` GPU KV tokens (`59` blocks), only `22` blocks remain for per-request
unique GPU KV after the shared prefix. Moving the split later spends those
unique blocks before CPU KV can increase decode concurrency:

```text
unique_gpu_blocks_per_req = ceil(split_tokens / 16) - 37
```

That gives `1`, `3`, `4`, and `5` unique GPU blocks/request for splits
`608`, `640`, `656`, and `672`. Split672 reduces CPU suffix work, but it also
allows only about four unique active requests after the shared prefix, so the
extra CPU KV capacity is not large enough to repay the CPU suffix/merge cost.
This is a planner constraint rather than a new runtime mechanism: for this
memory budget and prompt/decode shape, the split must be close to the shared
prefix.


### Prompt672/Split672 Fairness Probe

To check whether split672 itself is unusable or only mismatched to the
prompt608 workload, the same tight-memory B512 benchmark was rerun with the
shared prefix aligned to the split: `prompt=672`, `split=672`, `total=768`,
`decode=96`, `gpu_memory_utilization=0.648`, `max_num_seqs=512`, COTS
`0.02/0.02`, eager mode.

| Mode | Prompt | Split | Decode | Out tok/s | Result |
|---|---:|---:|---:|---:|---|
| GPU-only | 672 | n/a | 96 | `204.297`, `204.577` | mean `204.437` |
| Selective hybrid, final-aware | 672 | 672 | 96 | `211.076`, `210.400` | mean `210.738`, `+3.1%` over GPU-only |

This is the fair split672-style result: once the shared prefix actually reaches
the split point, extra CPU suffix capacity is just enough to beat the CPU suffix
and merge cost. The win is modest but repeatable in this narrow probe, and it
uses the safe final-aware selective placement path.

Forced-context parity for the same prompt672/split672 shape also stays in the
current BF16 hybrid tolerance band: forced outputs matched, `top1=1525/1536`,
forced-token logprob delta max/mean/p95 `0.1764/0.0102/0.0592`, and top20
Jaccard mean `0.9708`.

Takeaway: split672 is not intrinsically bad. It was bad for prompt608 because
the split was beyond the shared prefix and consumed too many unique GPU KV
blocks per request before CPU KV helped. For a prompt/shared-prefix length near
672, the same split becomes viable and narrowly throughput-positive.


## 2026-05-28 Follow-up: Generic GQA Naming and Llama Throughput Check

The Phase 2 CPU suffix kernel is now shape-parameterized for GQA models with
BF16 KV, head size 128, and up to eight query heads per KV head. The public
custom-op and runner names use `gqa_bf16` / `run_gqa_bf16_suffix_attention`;
the old `qwen_bf16` compatibility entry points were removed. Focused coverage
validates both Qwen2.5-7B (`28q/4kv`) and Llama3.1-8B (`32q/8kv`) shapes.

Isolated prepared-suffix runner benchmark, B256/suffix160/24 threads:

| Model shape | Mode | Median ms/layer | Synthetic layer tok/s |
|---|---|---:|---:|
| Qwen2.5-7B | direct | 2.084 | 122846 |
| Qwen2.5-7B | prepared_scatter | 2.525 | 101379 |
| Llama3.1-8B | direct | 3.854 | 66416 |
| Llama3.1-8B | prepared_scatter | 4.767 | 53706 |

Llama E2E throughput probe, `meta-llama/Llama-3.1-8B-Instruct`, B512,
prompt608/total768, `max_num_seqs=512`, eager, async scheduling disabled,
`gpu_memory_utilization=0.80`, CPU pool 12 GiB:

| Mode | Split | CPU suffix blocks/CPU-tier request | Effective capacity | Out tok/s |
|---|---:|---:|---:|---:|
| GPU-only | n/a | n/a | 30.23x | 3771.986, 3733.769 |
| Hybrid selective | 608 | 10 | 38.18x | 3234.531 |
| Hybrid selective | 672 | 6 | 34.55x | 3544.703 |
| Hybrid selective | 704 | 4 | 32.98x | 3761.551 |
| Hybrid selective | 736 | 2 | 31.54x | 3792.670 |
| Hybrid selective | 752 | 1 | 30.87x | 3804.750, 3806.259 |

At `gpu_memory_utilization=0.75`, Llama is more KV-limited than at `0.80`: GPU
KV cache drops to 13,520 tokens and GPU-only concurrency drops to 17.60x. The
hybrid result is worse at every tested split:

| Mode | Split | CPU suffix blocks/CPU-tier request | Effective capacity | Out tok/s |
|---|---:|---:|---:|---:|
| GPU-only | n/a | n/a | 17.60x | 3052.701 |
| Hybrid selective | 608 | 10 | 22.24x | 2218.756 |
| Hybrid selective | 672 | 6 | 20.12x | 2420.745 |
| Hybrid selective | 704 | 4 | 19.20x | 2423.891 |
| Hybrid selective | 736 | 2 | 18.37x | 2438.079 |
| Hybrid selective | 752 | 1 | 17.98x | 2447.293 |

At `gpu_memory_utilization=0.68`, Llama GPU-only could not allocate any KV
blocks after loading 14.99 GiB of weights, so the Qwen tight-memory setting is
not a matched Llama comparison point.

A diagnostic split752 rerun with iteration logging showed the main decode plateau
at `73` generation requests. In other words, the current scheduler/placement did
not turn the one-block CPU suffix into a larger active decode batch at this
setting; the run mostly keeps the lower GPU-limited batch size and adds hybrid
attention/suffix machinery plus a long tail. This is why the naive capacity
intuition does not show up in throughput.

Takeaway: the Llama model has higher KV-capacity pressure, but it also makes the
CPU suffix kernel much heavier. In this harness, Llama only ties/wins at `0.80`
when the split is almost at the tail and the run is near the GPU-only behavior.
At `0.75`, the current scheduler does not realize enough extra active decode
capacity to pay for even the one-block hybrid path. Treat this as a
scheduler/placement limitation plus CPU-suffix overhead, not a general proof
that CPU KV becomes less useful under tighter memory.


## 2026-05-28 Follow-up: Position-Based Hybrid KV Split

The selective request-tier placement described above is now superseded. The old
policy classified whole requests as either full-GPU or CPU-suffix and tried the
full-GPU path first when it fit. That made split752 diagnostics misleading: a
sequence that had not physically crossed the split could be treated as a
permanent full-GPU request, so `COTS CPU KV blocks` and hybrid decode calls could
remain zero even in a nominal hybrid run.

The current Phase 2 invariant is purely position-based:

```text
gpu_tokens(seq_len) = min(seq_len, split_x)
cpu_tokens(seq_len) = max(seq_len - split_x, 0)
```

There is no `full_gpu_req_ids` / `cpu_suffix_req_ids` state and no full-GPU
fallback. The scheduler still grows KV incrementally like normal vLLM decode:
CPU KV is not allocated just because `prompt_tokens + max_tokens` could exceed
the split. CPU suffix blocks appear only when the actual sequence length crosses
`split_x`. The default full-input-length admission check remains split-aware for
chunked prefill, but possible future decode length is not hard-reserved.

Focused unit coverage now checks the intended behavior:

- `len < split_x`: GPU prefix blocks only; CPU suffix blocks are absent.
- `len == split_x`: still GPU-only; CPU suffix blocks are absent.
- `len > split_x`: GPU prefix remains capped at the split and new suffix blocks
  are allocated from the CPU pool.
- A prompt/cache hit beyond the split is recovered as GPU prefix cache plus CPU
  suffix cache, not as a full-GPU request.

This means the Llama 0.75/0.80 throughput tables above should be read as
measurements of the old selective policy. They are still useful for diagnosing
why the old results were confusing, but they are not final evidence for the
position-based hybrid KV mechanism.


Corrected Llama B512 reruns with the position-based split, same
`meta-llama/Llama-3.1-8B-Instruct`, prompt608/total768, `max_num_seqs=512`,
eager, async scheduling disabled, CPU pool 12 GiB:

| GPU util | Mode | Split | Effective capacity | Out tok/s |
|---:|---|---:|---:|---:|
| 0.75 | GPU-only | n/a | 17.54x | 3031.053 |
| 0.75 | Position split hybrid | 608 | 22.16x | 1305.255 |
| 0.75 | Position split hybrid | 672 | 20.05x | 1750.865 |
| 0.75 | Position split hybrid | 704 | 19.14x | 2102.140 |
| 0.75 | Position split hybrid | 736 | 18.30x | 2501.148 |
| 0.75 | Position split hybrid | 752 | 17.91x | 2657.276 |
| 0.80 | GPU-only | n/a | 30.15x | 3748.958 |
| 0.80 | Position split hybrid | 608 | 38.08x | 1301.455 |
| 0.80 | Position split hybrid | 672 | 34.45x | 2032.684 |
| 0.80 | Position split hybrid | 704 | 32.89x | 2460.390 |
| 0.80 | Position split hybrid | 736 | 31.46x | 3138.812 |
| 0.80 | Position split hybrid | 752 | 30.79x | 3301.310 |

The activation sanity check for 0.75/split752 confirmed the intended path is now
active: the stats log reported `COTS CPU KV blocks: 53/6144` and
`COTS hybrid decode calls: 32`. That run was stats-enabled and produced
`2641.683 out tok/s`, close to the no-stats `2657.276` throughput point.

Conclusion: correcting the placement semantics resolves the diagnostic mystery
but removes the old apparent 0.80 near-tail tie. Llama still loses to matched
GPU-only in this prompt608/total768 harness. The best corrected split is 752 at
both memory points, but it remains about `12%` below GPU-only. The current
limiting factor is no longer inactive CPU KV; it is the cost of the CPU suffix
path and merge relative to the small one-block capacity benefit.


## 2026-05-28 Follow-up: Llama Performance Gap Investigation

The next investigation separated three possible causes of the Llama gap:

- CPU suffix compute itself.
- Per-step Python/metadata overhead before any request crosses the split.
- Mixed-row hybrid attention overhead once only some decode rows have suffix KV.

Stats-enabled Llama 0.80/split752 runs showed that CPU suffix compute is not the
dominant cost near the tail split. During active suffix intervals, the log showed
small accumulated CPU attention time, e.g. `COTS CPU wait/read/attn:
0.000/0.057/1.811 ms` over a two-second stats window with 32 hybrid decode
calls. The larger problem was overhead around the hybrid path.

One concrete overhead was fixed in `GPUModelRunner._build_attention_metadata`.
Before the fix, Phase 2 built COTS hybrid decode metadata once per layer on every
decode step, even when all scheduled token positions were still below `split_x`.
For Llama this meant 32 repeated Python scans through the same pre-split rows for
most of the run. The runner now precomputes whether the current step contains
any suffix positions and skips per-layer COTS metadata construction until at
least one scheduled token has `position >= split_x`.

Post-fix throughput:

| Setting | Before | After | Matched GPU-only | Outcome |
|---|---:|---:|---:|---|
| 0.80, split752, B512 | 3301.310 | 3429.256 | 3748.958 | still -8.5% |
| 0.80, split736, B512 | 3138.812 | 3250.544 | 3748.958 | still loses |
| 0.75, split752, B512 | 2657.276 | 2726.125 | 3031.053 | still -10.1% |
| 0.72, split752, B512 | n/a | 1933.718 | 2272.196 | still loses |
| 0.72, split736, B512 | n/a | 1838.233 | 2272.196 | still loses |
| 0.80, split752, B1024 | n/a | 3312.744 | 3717.366 | still loses |
| 0.80, split752, B512, graph | n/a | 3210.254 | 3748.958 | worse than eager |

The metadata short-circuit is a real improvement, especially at 0.80/split752,
but it is not enough to produce a Llama throughput win. Lowering GPU memory to
0.72 did not reveal a hidden cliff win; both split736 and split752 still trail
GPU-only. CUDA graph mode also did not help in this configuration because graph
memory reduced available KV capacity and throughput fell below the eager hybrid
run.

Current conclusion: the existing Phase 2 Llama path does not yet achieve a
robust throughput win for prompt608/total768. Near-tail splits provide only a
small capacity increase (`768 / 752 = 1.021x` for split752), so even modest
hybrid overhead erases the benefit. Wider CPU suffixes increase capacity more,
but the mixed hybrid path becomes too expensive.

The next credible optimization target is the mixed-row decode path in
`flash_attn.py`: today, a partially hybrid batch runs separate GPU prefix work
for prefix-only rows and suffix rows, then CPU suffix attention plus online
softmax merge for suffix rows. A better implementation would keep GPU prefix
attention coalesced for all rows, then run CPU suffix attention and merge only
for the suffix subset. A scheduler-level alternative is to group suffix-active
rows away from prefix-only rows, but that is a larger policy change and should
come after the lower-level mixed-row overhead is measured.


## 2026-05-28 Follow-up: Active Split GPU Path Isolation

A targeted mixed-row diagnostic was added after the metadata short-circuit. The
new counters separate the partial hybrid path into:

- suffix-active hybrid decode calls,
- prefix-only rows inside mixed batches,
- suffix rows inside mixed batches,
- GPU time and wall time for the mixed prefix-only path.

The Llama 0.80/split752 timing probe confirmed that the active split slowdown is
mostly not CPU suffix compute. In active intervals, CPU suffix attention was
usually around `1.7-2.6 ms` per stats window, while the mixed prefix-only GPU
path was much larger: about `7.8-22.6 ms` GPU time and `19.7-58.8 ms` wall time
across the same 32 layer calls. This is exactly the split-weight-style failure
mode we suspected: once the split is active, the GPU path is no longer the same
as GPU-only.

An env-gated prototype, `VLLM_COTS_HYBRID_COALESCED_PREFIX=1`, then replaced the
partial mixed path with one coalesced GPU prefix FlashAttention call over all
rows and reused those prefix states for CPU-suffix merge. This removes the extra
prefix-only FlashAttention launch and avoids recomputing GPU prefix states for
suffix rows.

Throughput result, Llama 0.80/split752, B512, eager:

| Mode | Out tok/s | Notes |
|---|---:|---|
| GPU-only | 3748.958 | matched baseline |
| Hybrid split752, metadata short-circuit | 3429.256 | current default hybrid |
| Hybrid split752, coalesced prefix prototype | 3445.185 | +0.5% over default hybrid |

The coalesced prototype is correct enough to run the benchmark, but it does not
recover the missing throughput. Timed coalesced runs still showed about
`8-10 ms` GPU prefix time per active 32-layer stats window, while CPU suffix
attention stayed around `1.6-1.8 ms`. The remaining overhead is therefore not the
second prefix-only launch alone. It is the active split GPU prefix semantics:
materializing prefix output plus LSE for online-softmax merge, capping suffix rows
at `split_x`, and the associated tensor/indexing work around that path.

Current conclusion: for Llama prompt608/total768, Phase 2 is not blocked by CPU
suffix arithmetic. It is blocked by the GPU-side cost of making the split
mergeable. A real win likely requires a lower-level attention primitive that
returns/merges prefix LSE with less overhead, or a design that avoids online
softmax merge for the common near-tail case. The env-gated coalesced-prefix path
is useful as a diagnostic upper-bound attempt, but not yet a production fix.


## 2026-05-28 Follow-up: FlashAttention LSE Microbenchmark

The active split investigation suggested that the GPU prefix path was expensive
once it had to produce mergeable online-softmax state. To isolate that, a focused
FlashAttention microbenchmark was added:

```text
David/Benchmarks/phase2/benchmark_flash_attn_lse.py
```

It uses the same paged-cache layout as vLLM FlashAttention and Llama-shaped
decode tensors: `Hq=32`, `Hkv=8`, `D=128`, BF16, block size 16. The sweep compares
normal output-only decode against output+LSE variants, with both `num_splits=0`
(normal eager heuristic) and `num_splits=1` (the original Phase 2 suffix-prefix
path).

Representative B512 results:

| Case | KV len | return LSE | out buffer | num_splits | Avg ms |
|---|---:|---:|---:|---:|---:|
| full_out_only_s0 | 768 | no | yes | 0 | 1.7606 |
| full_lse_out_s0 | 768 | yes | yes | 0 | 1.7601 |
| prefix_out_only_s0 | 752 | no | yes | 0 | 1.7254 |
| prefix_lse_out_s0 | 752 | yes | yes | 0 | 1.7250 |
| prefix_lse_alloc_s0 | 752 | yes | no | 0 | 1.7253 |
| prefix_out_only_s1 | 752 | no | yes | 1 | 1.7248 |
| prefix_lse_out_s1 | 752 | yes | yes | 1 | 1.7248 |
| prefix_lse_alloc_s1 | 752 | yes | no | 1 | 1.7253 |

The same pattern held across B64/B128/B192/B256/B512: returning LSE did not
measurably slow the raw FlashAttention kernel for these decode shapes, and
`num_splits=1` was not meaningfully worse than the normal `num_splits=0` path.

Updated conclusion: the remaining Llama gap is not explained by raw FA
`return_softmax_lse=True` overhead. The slowdown is in the composed active-split
path around the kernel: suffix Q/K/V staging, CPU/GPU handoff, UVA artifact reads,
merge, extra tensor/indexing work, and scheduler wave effects from tiny near-tail
capacity gains. The next diagnostic should therefore be a no-CPU/dummy-suffix
hybrid path that keeps the same GPU prefix+merge orchestration but replaces CPU
suffix attention/artifact movement with synthetic GPU-resident suffix states. If
that still loses, the orchestration/merge path is the issue; if it recovers, the
CPU artifact handoff is the issue.

## 2026-05-28 Follow-up: Dummy-Suffix Dry Run

A diagnostic dummy-suffix mode was added for the active split path:

```text
VLLM_COTS_HYBRID_DUMMY_SUFFIX=gpu|cpu|prefix_only
```

This is deliberately not a correctness mode. It keeps the hybrid scheduler,
position-based split metadata, and GPU prefix attention. The `gpu` variant
supplies neutral GPU-resident suffix state (`lse=-inf`) and still runs
`merge_attn_states`; the `cpu` variant fills the existing pinned CPU suffix
artifact buffers with neutral state and exposes them back through UVA; the
`prefix_only` variant writes prefix FlashAttention output directly and returns
before suffix state construction or online-softmax merge. This separates CPU
suffix arithmetic, UVA artifact consumption, mixed-row prefix work, and merge
cost.

Same-session Llama3.1-8B B512 reruns used `prompt608/total768/split752`,
`gpu_memory_utilization=0.80`, `max_num_seqs=512`, eager mode, async scheduling
disabled, and a 12 GiB CPU KV pool:

| Mode | Out tok/s | Interpretation |
|---|---:|---|
| GPU-only | 3742.012 | matched baseline for this session |
| Default hybrid split752 | 3513.954 | real CPU suffix path |
| Hybrid split752 + dummy GPU suffix | 3557.424 | no CPU suffix compute, no UVA artifact read, still merges |
| Hybrid split752 + dummy CPU/UVA suffix | 3579.364 | no CPU suffix compute, keeps pinned CPU artifact read and merge |
| Hybrid split752 + prefix-only skip-merge | 3503.148 | removes merge for suffix rows, but keeps separate mixed-row prefix path |
| Coalesced-prefix + dummy CPU/UVA suffix | 3510.162 | coalescing alone did not provide an upper-bound win |
| Coalesced-prefix + dummy GPU suffix | 3538.466 | merge remains expensive even without CPU/UVA artifact state |
| Coalesced-prefix + prefix-only skip-merge | 3662.878 | removes both mixed-row extra prefix work and merge |

The first dummy suffixes recovered only about `+1.2-1.9%` over default hybrid,
while the remaining gap to GPU-only stayed around `4.3-4.9%`. The CPU/UVA dummy
being slightly faster than the GPU dummy is within noise and argues against UVA
artifact reads as the primary bottleneck. The stronger signal is the coalesced
prefix-only run: once the mixed-row prefix path and merge are both removed, the
active split path rises to `3662.878 out tok/s`, only about `2.1%` below the
matched GPU-only run.

Updated diagnosis: the Llama split752 gap is a composition problem, not a CPU
suffix arithmetic problem. The default partial path pays for separate mixed-row
prefix work, and the merge/suffix-state path costs enough that the tiny near-tail
capacity gain (`30.15x -> 30.79x`, about `+2.1%`) cannot win. A real fix should
therefore either avoid the online-softmax merge for near-tail splits, make the
coalesced-prefix path production-quality and reduce merge/suffix-state overhead,
or have the planner choose only split points/workloads where CPU KV increases
active batch by much more than this overhead.

Coalesced-prefix plus prefix-only split sweep, same Llama3.1-8B B512 setup:

| Split | Effective capacity | Out tok/s | Relative to GPU-only |
|---:|---:|---:|---:|
| 752 | 30.79x | 3662.878 | 0.979x |
| 736 | 31.46x | 3428.306 | 0.916x |
| 704 | 32.89x | 2993.206 | 0.800x |
| 672 | 34.45x | 2781.259 | 0.743x |
| 608 | 38.08x | 2100.071 | 0.561x |

This upper-bound sweep is pessimistic for wider suffixes. Even after removing
real CPU suffix compute, CPU/UVA artifact reads, and online-softmax merge, moving
the split earlier does not uncover a throughput win. The best upper bound is
still the near-tail split752 point, and it remains slightly below GPU-only. This
means the extra nominal capacity is not converting into useful throughput in the
current Llama synthetic workload. For Llama, Phase 2 should be planner-gated
unless a workload sits at a sharper admission cliff than this harness, or unless
a future implementation removes enough active split overhead to make the near-tail
case exceed GPU-only.

## 2026-05-28 Follow-up: Wrapper Microbench and EngineCore Timing

A focused attention-wrapper microbenchmark was added:

```text
David/Benchmarks/phase2/benchmark_hybrid_wrapper_attention.py
```

It compares raw paged FlashAttention against the exact COTS hybrid wrapper in
`VLLM_COTS_HYBRID_DUMMY_SUFFIX=prefix_only` mode. It reports both CUDA-event time
and synchronized wall time so Python/tensor launch overhead is visible.

Representative B32/B64 Llama-shaped results show the attention wrapper itself is
not the reason wider splits collapse in E2E. At B64:

| Split | raw full FA wall ms | raw prefix+LSE wall ms | hybrid direct prefix-only wall ms | coalesced prefix-only wall ms |
|---:|---:|---:|---:|---:|
| 752 | 0.249 | 0.247 | 0.267 | 0.268 |
| 736 | 0.249 | 0.242 | 0.263 | 0.263 |
| 704 | 0.250 | 0.235 | 0.254 | 0.254 |
| 672 | 0.249 | 0.225 | 0.246 | 0.247 |
| 608 | 0.249 | 0.206 | 0.227 | 0.227 |

This curve moves in the expected direction: earlier splits make attention cheaper
in isolation. That is the opposite of the E2E split sweep, where earlier splits
get much slower. The residual Llama gap is therefore not inside the attention
wrapper alone.

An env-gated EngineCore timing probe was then added under
`VLLM_COTS_HYBRID_ENGINE_TIMING=1`, splitting each step into scheduler,
`execute_model` submission, model future/sample, and scheduler update. Matched
Llama B512 split752 traces:

| Mode | Out tok/s | schedule sum ms | execute_submit sum ms | sample sum ms | update sum ms | total timed step ms |
|---|---:|---:|---:|---:|---:|---:|
| GPU-only | 3733.336 | 449.110 | 2445.241 | 18835.272 | 164.652 | 21897.448 |
| Coalesced + prefix-only hybrid | 3629.283 | 804.403 | 4302.504 | 17188.939 | 166.799 | 22465.909 |

The upper-bound hybrid actually spent about `1.65 s` less in the model/sample
section, but lost that back through about `+1.86 s` in `execute_model` submission
and about `+0.36 s` in scheduler time. Binning by decode width shows this is not
only a final tiny-tail effect: in the 100-399 scheduled-token bin,
`execute_submit_ms` averaged `5.35 ms` for GPU-only and `10.23 ms` for the
upper-bound hybrid.

Updated pin-down: for this Llama harness, the reason Phase 2 does not beat
GPU-only is CPU-side active-hybrid overhead before the model future, primarily
worker submission/input-prep and secondarily scheduler bookkeeping. The attention
kernel path is already fast enough in isolation, and the unrealistic prefix-only
upper bound proves that removing CPU suffix compute/UVA/merge is insufficient
while this submit/scheduler tax remains. The next code target is therefore the
`execute_model` preparation path for active COTS hybrid KV metadata, not the CPU
suffix kernel.

## 2026-05-28 Follow-up: Active Metadata Reuse and Remaining Llama Gap

The submit-path probe was expanded with `VLLM_COTS_HYBRID_SUBMIT_TIMING=1`,
which logs per-step GPUModelRunner buckets and an attention-metadata breakdown.
For the same Llama3.1-8B B512 `prompt608/total768/split752` upper-bound run,
active suffix steps spent most of the extra submit time rebuilding identical
per-request hybrid metadata once per layer:

| Mode | Out tok/s | execute_submit sum ms | sample sum ms | active suffix attention metadata avg ms |
|---|---:|---:|---:|---:|
| GPU-only + submit timing | 3720.615 | 2677.834 | 18904.949 | n/a |
| Coalesced + prefix-only before reuse | 3596.193 | 4600.195 | 17402.939 | 5.47 |
| Coalesced + prefix-only after reuse | 3735.678 | 3595.862 | 17557.130 | 1.22 |

The fix was to build the request-level `CotsHybridDecodeMetadata` once per
attention group and reuse its common CPU block table, suffix lengths, scatter
rows, active row indices, and mixed-prefix selectors for the remaining layer
metadata objects. Only layer-local cache pointers, staging slots, output buffers,
and task ids are rebuilt per layer. The diagnostic counters confirm the intended
shape: active metadata construction now calls the heavy request path once instead
of 32 times (`kv_calls=1`), and the repeated row-filter/cache-key work disappears.

No-timing Llama controls after metadata reuse:

| Mode | Out tok/s | Relative to GPU-only timing baseline (`3733.336`) | Interpretation |
|---|---:|---:|---|
| Coalesced + prefix-only | 3817.099 | 1.022x | upper bound now clears GPU-only |
| Coalesced + dummy GPU suffix | 3740.889 | 1.002x | merge with GPU-resident neutral suffix is near tie |
| Coalesced + dummy CPU/UVA suffix | 3707.361 | 0.993x | pinned CPU artifact/UVA path costs roughly 1% |
| Coalesced + real CPU suffix | 3661.273 | 0.981x | real CPU suffix work still loses about 2% |

This changes the diagnosis: the original ``we can never win`` gap was largely a
Python/metadata issue and is fixable. After that fix, the remaining Llama loss is
inside the active suffix execution path. Timed real-suffix runs show active
`forward_launch_ms` rising from about `7.9 ms` in prefix-only to about `17.7 ms`
with the real suffix path, while metadata stays near `1.4 ms`. A one-off
`H2D-copy-before-merge` probe improved the dummy CPU/UVA control (`3775.889` out
tok/s) but slightly hurt the real CPU suffix path (`3647.390` out tok/s), so it
is not a production fix.

COTS suffix counters on the real path show the native runner is active only once
suffix rows exist. The runner uses a single queue worker, and the CPU attention
kernel uses ATen `parallel_for` over `batch * num_kv_heads`. Per reporting window,
CPU suffix attention submit time was about `1.7-1.8 ms` for 32 layer calls, while
native worker busy time ranged from about `4-10 ms` depending on live suffix rows;
queue wait was usually sub-ms, with a few small-row windows around `2-3 ms`.

An env-gated async artifact-copy probe then tested direction (2) directly:
`VLLM_COTS_HYBRID_SUFFIX_ARTIFACT_COPY=1` submits the native suffix task without
immediate stream sync, waits for the CPU suffix completion on a side stream,
copies suffix output/LSE from pinned CPU memory into GPU buffers, then makes the
merge consume the GPU buffers.

No-timing Llama B512 `prompt608/total768/split752` results:

| Mode | Out tok/s | Interpretation |
|---|---:|---|
| Coalesced + dummy CPU suffix + async artifact copy | 3735.583 | explicit GPU artifact copy removes most of the dummy CPU/UVA penalty |
| Coalesced + real CPU suffix + async artifact copy | 3590.214 | worse than real CPU suffix through UVA (`3661.273`) |

The control is useful: explicit GPU-resident artifacts help when the suffix
state is already ready. The real path regresses because the CPU suffix output is
ready late enough that the explicit H2D copy lands on the critical path. The
current wait mechanism is the wait-kernel path, not a legacy blocking sync
callback, so the remaining issue is scheduling/overlap: the suffix task and its
artifact movement are not far enough ahead of the merge.

The earlier suffix-submit diagnostic was then added under
`VLLM_COTS_HYBRID_EARLY_SUFFIX_SUBMIT=1`. It prepares and enqueues the native
CPU suffix task before the GPU prefix attention launch, then waits only at merge
time.

No-timing Llama B512 `prompt608/total768/split752` results:

| Mode | Out tok/s | Interpretation |
|---|---:|---|
| Coalesced + real CPU suffix + early submit | 3609.587 | earlier submit alone is worse than the existing UVA path |
| Coalesced + real CPU suffix + early submit + async artifact copy | 3685.634 | best real-suffix result so far, but still below GPU-only/dummy-GPU |

Lower-memory sweep, same Llama B512 `prompt608/total768` workload,
`gpu_memory_utilization=0.75`, coalesced prefix + early submit + async artifact
copy for hybrid:

| Mode / split | GPU prefix blocks | Reported effective capacity | Out tok/s | vs GPU-only |
|---|---:|---:|---:|---:|
| GPU-only | 48 | 17.54x | 3005.794 | 1.000x |
| Hybrid split752 | 47 | 17.91x | 2883.133 | 0.959x |
| Hybrid split736 | 46 | 18.30x | 2700.400 | 0.898x |
| Hybrid split704 | 44 | 19.14x | 2261.581 | 0.752x |
| Hybrid split672 | 42 | 20.05x | 1860.999 | 0.619x |
| Hybrid split640 | 40 | 21.05x | 1585.802 | 0.528x |
| Hybrid split608 | 38 | 22.16x | 1365.424 | 0.454x |

This sweep confirms the lower-memory intuition only helps if the capacity gain is
large enough relative to the active suffix overhead. For this Llama workload, the
near-tail split752 saves only one GPU KV block per request (`48 -> 47`), while
earlier splits increase capacity modestly but make the CPU suffix path much
heavier. Throughput falls monotonically as the split moves earlier.

This clarifies the tradeoff. Moving suffix submission earlier gives the CPU task
a larger overlap window, but it also moves suffix task preparation/host-callback
launch work before the GPU prefix launch. Without explicit GPU artifact buffers,
that tradeoff loses. With artifact copy, the extra overlap is useful enough to
beat the original real-suffix path (`3661.273`) and the artifact-copy-only path
(`3590.214`), but it still does not reach the GPU-only timing baseline
(`3733.336`) or dummy-GPU merge control (`3740.889`).

Next target: reduce the real suffix critical path itself. The remaining loss is
not explained by request metadata, not by GPU prefix attention, and not by UVA
artifact reads alone. The best current real path still pays native suffix compute
plus output/LSE materialization/copy. Planner-gating Llama near-tail splits should
remain the default unless a profile predicts enough capacity gain to cover that
measured suffix execution overhead.

## 2026-05-29 Follow-up: Qwen Current-Code Check

The same current-code path used for the Llama investigation was tested on
`Qwen/Qwen2.5-7B-Instruct`: coalesced prefix, early native suffix submit, and
async suffix artifact copy. Workload: B512 `prompt608/total768`, eager,
`max_num_seqs=512`.

At `gpu_memory_utilization=0.80`:

| Mode / split | Reported effective capacity | Out tok/s | vs GPU-only |
|---|---:|---:|---:|
| GPU-only | 82.96x | 5794.626 | 1.000x |
| Hybrid split752 | 84.72x | 5686.217 | 0.981x |

At `gpu_memory_utilization=0.75`:

| Mode / split | Reported effective capacity | Out tok/s | vs GPU-only |
|---|---:|---:|---:|
| GPU-only | 54.15x | 5491.545 | 1.000x |
| Hybrid split752 | 55.30x | 5295.210 | 0.964x |
| Hybrid split736 | 56.50x | 4812.990 | 0.876x |
| Hybrid split704 | 59.07x | 3927.592 | 0.715x |

Qwen has much more KV capacity than Llama in this setup because it has fewer KV
heads, so the same near-tail split only raises effective capacity by about 2.1%.
That is not enough to pay the active hybrid suffix path. Moving the split earlier
increases capacity, but throughput drops quickly as the CPU suffix grows. The
current code therefore does not produce a Qwen throughput win on this synthetic
B512 `prompt608/total768` workload.

## 2026-05-29 Follow-up: Model-Specific Tight-Memory Check

A tighter memory check used model-specific feasible settings rather than assuming
Llama and Qwen have the same GPU footprint. In these runs, Llama loaded about
`14.99 GiB` of weights, while Qwen loaded about `14.25 GiB`, leaving Qwen with
roughly `0.74 GiB` more KV headroom at the same `gpu_memory_utilization`.

Workload: B512 `prompt608/total768`, eager, `max_num_seqs=512`, current best
hybrid path (coalesced prefix + early suffix submit + async artifact copy).

Llama at `gpu_memory_utilization=0.70`:

| Mode / split | Available KV | Reported effective capacity | Out tok/s | vs GPU-only |
|---|---:|---:|---:|---:|
| GPU-only | 0.46 GiB | 4.94x | 1304.707 | 1.000x |
| Hybrid split752 | 0.46 GiB | 5.04x | 1167.173 | 0.895x |
| Hybrid split736 | 0.46 GiB | 5.15x | 1130.189 | 0.866x |

Qwen at `gpu_memory_utilization=0.70`:

| Mode / split | Available KV | Reported effective capacity | Out tok/s | vs GPU-only |
|---|---:|---:|---:|---:|
| GPU-only | 1.04 GiB | 25.33x | 4092.449 | 1.000x |
| Hybrid split752 | 1.04 GiB | 25.87x | 3874.464 | 0.947x |

Qwen at `gpu_memory_utilization=0.67`:

| Mode / split | Available KV | Reported effective capacity | Out tok/s | vs GPU-only |
|---|---:|---:|---:|---:|
| GPU-only | 0.33 GiB | 8.04x | 2159.427 | 1.000x |
| Hybrid split752 | 0.33 GiB | 8.21x | 2010.113 | 0.931x |
| Hybrid split736 | 0.33 GiB | 8.39x | 1927.641 | 0.893x |

Even in the tightest feasible settings tested, the near-tail hybrid split still
loses. The reason is the same ratio math as before: split752 saves one GPU KV
block per request, so capacity rises by only about `48/47 = 1.021x`, while the
active CPU suffix path costs more than that. Moving to split736 raises capacity
slightly more, but the second CPU suffix block already hurts throughput enough to
make it worse than split752.

## 2026-05-29 Follow-up: Heads-Per-KV Specialization Probe

This round targeted the real one-block suffix path for the Llama/Qwen split752
case. The prepared native runner already disables eager input snapshots for the
static pinned staging path, so the remaining runner overhead is mostly real
scatter + suffix attention + artifact/merge orchestration.

A small diagnostic flag was added to `benchmark_prepared_suffix_runner.py`:
`--no-snapshot-inputs`. This lets the microbenchmark match the production static
staging path instead of the pessimistic eager snapshot path.

First attempted fix: route prepared runner calls through an internal unchecked
suffix-attention entry point after `populate_task()` shape validation. This built
and passed focused tests, but did not materially change the one-block runner
microbenchmark. The public custom op still keeps full validation.

Second attempted fix: specialize the CPU suffix attention hot loop by compile-time
`heads_per_kv` and dispatch for values 1-8. This keeps the generic GQA contract
while allowing the compiler to shrink/unroll the inner per-KV-head query group.
It helped both target shapes in the exact B512/suffix16 runner probe:

| Model shape | Mode | Before median | After median | Change |
|---|---|---:|---:|---:|
| Llama3.1-8B | direct | 27.554 ms / 32 layers | 26.106 ms / 32 layers | +5.3% |
| Llama3.1-8B | prepared_scatter | 35.099 ms / 32 layers | 33.660 ms / 32 layers | +4.1% |
| Qwen2.5-7B | direct_scatter | 16.604 ms / 28 layers | 16.085 ms / 28 layers | +3.1% |
| Qwen2.5-7B | prepared_scatter | 17.897 ms / 28 layers | 16.995 ms / 28 layers | +5.0% |

Validation run after the C++ change:

```text
/opt/conda/envs/thesis/bin/python -m pytest \
  tests/kernels/attention/test_cots_suffix_attention.py \
  tests/kernels/attention/test_cots_suffix_attention_runner.py -q
# 22 passed
```

Llama E2E reruns, B512 `prompt608/total768/split752`, `gpu_memory_utilization=0.80`,
eager, async scheduling disabled, `max_num_seqs=512`:

| Mode | Extra knobs | Out tok/s |
|---|---|---:|
| GPU-only | none | 3707.061, 3739.187 |
| Hybrid | coalesced + early submit + artifact copy | 3670.804, 3680.926 |
| Hybrid | coalesced only | 3720.178, 3719.584 |

The important change is that the best post-specialization hybrid setting is now
the simpler coalesced path without early suffix submit or explicit artifact copy.
The early/copy path was useful when CPU suffix work was longer, but after the
kernel specialization it moves extra work onto the critical path.

Conclusion: this is progress, but not a robust throughput win yet. Hybrid
`split752` is now in the GPU-only noise band and sometimes slightly above one
matched GPU-only run, but the second GPU-only repeat reached `3739 out tok/s`.
The remaining planner rule should still be conservative: choose hybrid only when
profiling predicts a capacity/admission benefit larger than this measured
near-tail overhead. The next useful target is reducing the artifact/merge or
submission envelope enough to turn the current near-tie into a repeatable win.

## 2026-05-29 Follow-up: Early-Submit-Only Decomposition

This round decomposed the remaining split752 gap after the heads-per-KV CPU
kernel specialization. Workload: B512 `prompt608/total768`, eager,
`max_num_seqs=512`, async scheduling disabled, coalesced prefix enabled for all
hybrid runs.

Llama at `gpu_memory_utilization=0.80`, split752:

| Mode / knob | Out tok/s | Notes |
|---|---:|---|
| GPU-only | 3712.193, 3739.608 | same-session baseline range |
| Hybrid prefix-only dummy | 3722.876 | split GPU-prefix path is not slower |
| Hybrid dummy GPU suffix | 3728.912 | merge with GPU fake suffix is not the gap |
| Hybrid dummy CPU/UVA suffix | 3775.513 | noisy, but no real suffix runner cost |
| Hybrid real, coalesced only | 3700.659 | real suffix runner sits on critical path |
| Hybrid real, early-submit only | 3728.937, 3731.294 | recovers most real suffix cost |
| Hybrid real, artifact-copy only | 3694.624 | explicit artifact copy is harmful here |

The important correction versus the previous early/copy result is that the bad
part was the explicit artifact-copy path, not early submission itself. Without
`VLLM_COTS_HYBRID_EARLY_SUFFIX_SUBMIT`, the real suffix is submitted after the
GPU prefix returns, so even a one-block suffix remains on the critical path.
Early-submit-only overlaps that work and moves Llama split752 back into the
GPU-only noise band. It is still not a robust win: the best GPU-only repeat in
this session was `3739.608 out tok/s`, above the early-submit repeats.

Earlier splits with early-submit did not close the gap:

| Model / memory | Split | Effective capacity | Out tok/s | vs matched GPU-only |
|---|---:|---:|---:|---:|
| Llama 0.80 | 736 | 31.46x | 3495.324 | 0.935x vs 3739.608 |
| Llama 0.75 | 752 | 17.91x | 2935.956 | 0.970x vs 3028.281 |
| Llama 0.75 | 736 | 18.30x | 2768.038 | 0.914x vs 3028.281 |

This reinforces the one-block cliff: split752 only improves capacity by about
`48/47 = 1.021x`, which is too small to robustly beat GPU-only; split736 improves
capacity more, but the second CPU suffix block costs much more than it saves.

Qwen checks with early-submit-only:

| Memory | Mode / split | Effective capacity | Out tok/s | vs GPU-only |
|---|---|---:|---:|---:|
| 0.80 | GPU-only | 82.96x | 5848.190 | 1.000x |
| 0.80 | Hybrid split752 | 84.72x | 5716.989 | 0.978x |
| 0.75 | GPU-only | 54.15x | 5505.983 | 1.000x |
| 0.75 | Hybrid split752 | 55.30x | 5332.827 | 0.969x |

Qwen still loses because the available GPU KV capacity is high enough that a
near-tail one-block split only gives about a 2.1% admission gain, while the real
suffix/merge path costs more than that.

Code decision from this round: make early suffix submit the default hybrid
behavior and keep explicit artifact copy opt-in/off. The default can be disabled
with `VLLM_COTS_HYBRID_EARLY_SUFFIX_SUBMIT=0`, `false`, or `off`. This does not
create a robust throughput win by itself, but it is the cleaner production
behavior for the real suffix path because it overlaps the CPU suffix instead of
submitting it after the GPU prefix.

Validation after the Python change:

```text
/opt/conda/envs/thesis/bin/python -m py_compile \
  /TTC/vllm/vllm/envs.py \
  /TTC/vllm/vllm/v1/attention/backends/cots_hybrid_attention.py
# passed

/opt/conda/envs/thesis/bin/python -m pytest \
  tests/kernels/attention/test_cots_hybrid_attention.py \
  tests/kernels/attention/test_cots_suffix_attention_runner.py \
  tests/v1/worker/test_cots_hybrid_kv.py -q
# 37 passed
```

Current conclusion: the remaining blocker is not the split GPU-prefix kernel and
not raw CPU suffix math alone. The one-block hybrid tax is now mostly the real
suffix runner plus merge/UVA artifact envelope. To make the planner choose hybrid
for this synthetic workload, the next attempt must reduce the one-block fixed tax
below roughly the 2.1% capacity gain of split752, or find a workload/admission
point where the capacity gain is materially larger without crossing into the
second CPU suffix block.

## 2026-05-29 Follow-up: Integer Cliff and Short-Suffix Probe

This probe tested the most favorable planner scenario for a one-block split:
keep `split752`, search near GPU KV integer-admission cliffs, and shorten the
post-split region so hybrid saves one GPU KV block while doing as little real CPU
suffix work as possible. Workload stayed B512, prompt608, eager,
`max_num_seqs=512`, async scheduling disabled, coalesced prefix enabled, early
suffix submit enabled, artifact copy off.

Llama split752 integer-cliff checks:

| GPU mem | Total | Mode | Reported capacity | Out tok/s |
|---:|---:|---|---:|---:|
| 0.801 | 768 | GPU-only | 30.40x | 3792.403 |
| 0.801 | 768 | Hybrid | 31.04x | 3720.721 |
| 0.805 | 768 | GPU-only | 31.42x | 4042.524 |
| 0.805 | 768 | Hybrid | 32.09x | 3786.643 |

The integer cliff alone did not produce a win. Even when hybrid crossed the
next reported capacity boundary, the real one-block hybrid envelope cost more
than the extra admission slot repaid.

Short post-split checks at `gpu_memory_utilization=0.801`:

| Total | Mode | Reported capacity | Out tok/s | Notes |
|---:|---|---:|---:|---|
| 754 | GPU-only | 30.40x | 4093.241 | 48 GPU blocks/request |
| 754 | Hybrid | 31.04x | 4039.036 | 2 post-split decode steps |
| 753 | GPU-only | 30.40x | 4073.738 | 48 GPU blocks/request |
| 753 | Hybrid | 31.04x | 4006.775 | 1 post-split decode step |
| 753 | Hybrid prefix-only dummy | 31.04x | 4023.782 | no real suffix/merge |

Even the minimum one-token post-split case did not win. The prefix-only dummy
also stayed below GPU-only, which means some fixed hybrid path cost is present
outside raw CPU suffix math.

Strongest low-memory admission case tested:

| GPU mem | Total | Mode | Reported capacity | Out tok/s |
|---:|---:|---|---:|---:|
| 0.70 | 753 | GPU-only | 4.94x | 1399.437 |
| 0.70 | 753 | Hybrid | 5.04x | 1383.529 |

This was the best theoretical setup for a planner-visible win: one-token CPU
suffix and a reported capacity boundary around 5 concurrent requests. It still
lost by about 1.1%.

Conclusion: no throughput win has been found yet, including the integer-cliff
and shortest-suffix regimes. The remaining work should move from split search to
removing fixed hybrid overhead: metadata build/row filtering, suffix submit/sync,
UVA artifact exposure, and the merge/return path. A planner can still model
hybrid as feasible capacity, but the current implementation should not select it
for throughput on these synthetic Llama/Qwen B512 cases.


## 2026-05-29 Follow-up: Fixed-Overhead Fast Path

This round investigated why hybrid still lost in the most favorable `split752`
capacity-cliff case. A key correction: for `prompt608,total753,split752`, the
main decode never actually runs CPU suffix attention. Decode attends to the
previous length, so the last forward position is still below 752. The timing log
confirmed `suffix_scan_false` on every steady B512 decode step. This makes
`total753` a pure fixed-overhead test: hybrid gets the 47-block GPU-prefix
capacity accounting, but no real CPU suffix/merge work should be active.

Isolation runs:

| Case | Out tok/s | Observation |
|---|---:|---|
| GPU-only total753 timing | 4050.475 | baseline timing run |
| `offload_backend=cots`, no hybrid KV | 4062.798 | COTS backend selection alone is effectively no-op |
| Hybrid split752 total753 timing, before fast path | 3975.704 | no suffix active, but hybrid runtime overhead visible |
| Hybrid split768/max769 total753 timing | 4003.327 | valid no-suffix hybrid with 48-block prefix; still no win |

The COTS-backend/no-hybrid result ruled out the weight-offload backend as the
source of the fixed tax. The overhead came from hybrid-KV worker/runtime hooks
that were still active before suffix rows existed.

Implemented cleanup:

- Cache `cots_hybrid_has_suffix_positions` once during input prep and reuse it
  in slot masking and attention metadata attachment.
- Thread that cached predicate through `CotsRuntime.on_dispatch` so repeated
  no-suffix forwards skip hybrid live-count publication without rescanning
  positions.
- Carry a cached `scatter_source_indices_gpu` through common hybrid metadata so
  suffix-active layers reuse the row-index tensor instead of rebuilding it per
  layer.

Validation:

```text
/opt/conda/envs/thesis/bin/python -m py_compile \
  /TTC/vllm/vllm/v1/attention/backends/cots_hybrid_attention.py \
  /TTC/vllm/vllm/v1/worker/cots_hybrid_kv.py \
  /TTC/vllm/vllm/v1/worker/cots_runtime.py \
  /TTC/vllm/vllm/v1/worker/gpu_model_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py
# passed

/opt/conda/envs/thesis/bin/python -m pytest \
  tests/kernels/attention/test_cots_hybrid_attention.py \
  tests/kernels/attention/test_cots_suffix_attention_runner.py \
  tests/v1/worker/test_cots_hybrid_kv.py -q
# 37 passed
```

No-suffix timing improved but did not turn into a win:

| Case | Out tok/s | steady prepare med | attn metadata med | publish med |
|---|---:|---:|---:|---:|
| GPU-only total753 timing | 4050.475 | 0.390 ms | 0.085 ms | 0.004 ms |
| Hybrid total753 timing before | 3975.704 | 0.609 ms | 0.151 ms | 0.054 ms |
| Hybrid total753 after cached predicate | 3995.984 | 0.438 ms | 0.090 ms | 0.052 ms |
| Hybrid total753 after dispatch fast path | 4015.084 | 0.419 ms | 0.090 ms | 0.004 ms |

The fast path removed most of the visible Python/runtime fixed overhead. The
remaining no-suffix gap is now smaller, but still present. The fair non-timing
capacity-cliff pair after the cleanup still lost:

| Total | Mode | Out tok/s |
|---:|---|---:|
| 753 | GPU-only | 4092.799 |
| 753 | Hybrid split752 | 3998.876 |

The suffix-active Llama pair also still lost after the cleanup:

| Total | Mode | Reported capacity | Out tok/s |
|---:|---|---:|---:|
| 768 | GPU-only | 30.40x | 3843.848 |
| 768 | Hybrid split752 | 31.04x | 3775.715 |

Conclusion: this cleanup is correct and reduces fixed overhead, but it does not
create a throughput win. The next plausible implementation attempt is not more
split sweeping. It is the real suffix artifact/merge return path: avoid the
per-layer suffix-row `index_select`/`index_copy_` envelope by adding an indexed
merge/writeback path that consumes full prefix output plus suffix row indices
and writes merged rows directly into the full output. That targets the remaining
fixed cost that capacity gains have not yet overcome.


## 2026-05-29 Follow-up: Indexed Merge/Writeback Attempt

This attempt targeted the remaining real-suffix return envelope for partial
hybrid rows. The old partial/coalesced path gathered full prefix outputs into
compact suffix-row tensors, ran `merge_attn_states`, then scattered the merged
rows back into the full output with `index_copy_`. The new diagnostic path adds
a CUDA op, `merge_attn_states_indexed`, that reads full prefix rows by
`source_indices`, merges compact suffix output/LSE, and writes directly into the
full output row.

Implementation details:

- Added a CUDA indexed merge kernel in `csrc/attention/merge_attn_states.cu` and
  registered it as `torch.ops._C.merge_attn_states_indexed`.
- Added Python wrappers in `_custom_ops.py` and
  `vllm/v1/attention/ops/merge_attn_states.py`; the high-level wrapper falls
  back to the old gather/merge/scatter sequence off the supported CUDA path.
- Wired COTS hybrid decode to use it only when a precomputed full prefix output
  is present and `scatter_source_indices` is non-null.
- Kept the COTS path opt-in via `VLLM_COTS_HYBRID_INDEXED_MERGE=1` after the
  first benchmark did not show a win.

Validation:

```text
/opt/conda/envs/thesis/bin/cmake --build --preset release --target install
# rebuilt _C.abi3.so successfully

/opt/conda/envs/thesis/bin/python -m pytest \
  tests/kernels/attention/test_merge_attn_states.py::test_merge_attn_states_indexed_matches_compact \
  tests/kernels/attention/test_cots_hybrid_attention.py \
  tests/kernels/attention/test_cots_suffix_attention_runner.py \
  tests/v1/worker/test_cots_hybrid_kv.py -q
# 39 passed
```

Llama B512, prompt608, total768, split752, 0.801 GPU memory:

| Mode | Indexed merge | Out tok/s |
|---|---|---:|
| Hybrid | off / previous fast path | 3775.715 |
| Hybrid | on | 3715.162 |
| Hybrid | off, same rebuilt code | 3734.811 |

The indexed path did not close the gap. For the synchronized synthetic workload
it is also mostly not the steady-state path: after all requests cross the split
together, `scatter_source_indices` is null and COTS uses the all-suffix compact
path, not partial indexed writeback. This explains why the attempt does not
address the main Llama/Qwen B512 loss. It may still be useful for future mixed
request-length diagnostics, but it should stay opt-in until a mixed workload
shows a benefit.

Updated conclusion: the remaining throughput gap for the current synthetic cases
is in the all-suffix envelope: CPU suffix artifact exposure, UVA/merge cost, and
possibly stream synchronization around the suffix runner. The next attempt should
measure and reduce the all-suffix merge/artifact path directly, not the partial
row scatter path.


## 2026-05-29 Follow-up: All-Suffix Real-CPU Ablation

This round added a diagnostic-only real-suffix ablation knob:

```text
VLLM_COTS_HYBRID_REAL_SUFFIX_ABLATION=prefix_only|gpu|cpu
```

Unlike `VLLM_COTS_HYBRID_DUMMY_SUFFIX`, this still submits and waits for the
real native CPU suffix task. The mode then deliberately ignores or neutralizes
its result so the active all-suffix path can separate CPU submit/sync cost from
artifact/merge cost. The default remains `none`; the new env is registered along
with `VLLM_COTS_HYBRID_INDEXED_MERGE` to avoid diagnostic warning noise.

Same-session Llama3.1-8B B512, prompt608, total768, split752,
`gpu_memory_utilization=0.801`, eager, async scheduling disabled:

| Mode | Out tok/s | Interpretation |
|---|---:|---|
| GPU-only | 3793.655 | Same-session baseline. |
| Hybrid full real suffix | 3734.966 | Still loses by about 1.5%. |
| Hybrid dummy prefix-only, no CPU suffix | 3824.485 | Split GPU-prefix path itself can clear GPU-only. |
| Hybrid dummy GPU suffix, no CPU suffix | 3843.823 | Merge with GPU-resident neutral suffix also clears GPU-only. |
| Hybrid real CPU suffix, skip artifact/merge | 3733.811 | Real suffix submit/sync alone erases the win. |
| Hybrid real CPU suffix + GPU dummy merge | 3778.339 | Artifacts are not the main loss; real CPU suffix path is still too expensive. |
| Hybrid real CPU suffix + CPU dummy artifact | 3432.974 | Stress mode only: it overwrites pinned CPU buffers after compute, so it overstates artifact cost. |

The important comparison is not the CPU-dummy stress case. It is the pair
`dummy GPU suffix` versus `real CPU suffix + GPU dummy merge`: adding the real
CPU suffix runner drops throughput from `3843.823` to `3778.339` output tok/s,
roughly a `1.7%` tax. The near-tail split752 capacity gain over GPU-only is only
about `31.04 / 30.40 = 1.021x` nominal and was about `1.3%` in this same-session
throughput run, so this CPU submit/sync envelope is enough to keep hybrid below
GPU-only.

Updated diagnosis: the immediate gap is no longer the split GPU kernel, not the
online-softmax merge by itself, and not the UVA artifact read by itself. The
throughput win is available in no-real-CPU controls, but the native suffix task
submission/synchronization path consumes more than the capacity gain for a
one-block near-tail Llama split. The next attempt should therefore target the
real suffix runner envelope: reduce per-layer submit/sync overhead, increase the
overlap window without moving more work onto the critical path, or special-case
one-block suffix attention so the CPU task has less fixed overhead.

## 2026-05-29 Follow-up: Native Suffix Runner Attribution and General Kernel Attempts

This round instrumented the native suffix runner enough to split the real CPU
suffix tax into submit, dispatch callback, queue wait, scatter, and attention
work. The production path keeps counters off unless `VLLM_COTS_DIAG=1` or
`VLLM_COTS_SUFFIX_COUNTERS=1` is set.

Kept implementation changes:

- Added native suffix runner counters for submit prepare/snapshot/hostfunc,
  dispatch callback/snapshot/enqueue, worker queue wait, scatter, and attention.
- Moved `TaskQueue::Node` construction to move the queued `std::function`
  instead of copying it.
- Removed the extra CPU suffix probability-normalization pass; probabilities are
  normalized when accumulating V.
- Tuned CPU suffix attention `at::parallel_for` grain size from 1 to 8. This is
  a general scheduling-overhead reduction for all GQA shapes, not a one-block
  fast path.

Attempts that were tested and backed out:

- Row-parallel CPU KV scatter: improved the isolated scatter microbenchmark, but
  regressed Llama E2E throughput, likely from extra threadpool contention in the
  full runner.
- Online-softmax one-pass CPU suffix accumulation: correct, but slower than the
  existing two-phase logits/probs then V accumulation. It adds output-accumulator
  traffic on every token, which outweighed removing the probability buffer.
- Grain size 16: the benchmark harness wedged with the child process defunct, so
  the final code was restored to grain size 8.

Validation after the kept changes:

```text
/opt/conda/envs/thesis/bin/cmake --build --preset release --target install
# rebuilt _cots_C successfully

/opt/conda/envs/thesis/bin/python -m pytest \
  tests/kernels/attention/test_cots_suffix_attention.py \
  tests/kernels/attention/test_cots_suffix_attention_runner.py \
  tests/kernels/attention/test_cots_hybrid_attention.py \
  tests/v1/worker/test_cots_hybrid_kv.py -q
# 52 passed, 2 warnings
```

Isolated Llama3.1-8B shape, B512, suffix16, 32 layers, 24 CPU threads:

| Kernel state | direct ms | prepared ms | direct+scatter ms | prepared+scatter ms |
|---|---:|---:|---:|---:|
| Restored two-phase, grain 1 | 27.009 | 29.090 | 32.854 | 34.514 |
| Kept two-phase, grain 8 | 26.835 | 28.472 | 32.824 | 34.337 |
| Online softmax attempt | 27.483 | 28.482 | 33.407 | 34.835 |

Same-session Llama3.1-8B E2E, B512, prompt608, total768, split752,
`gpu_memory_utilization=0.801`, eager, async scheduling disabled:

| Mode / code state | Out tok/s |
|---|---:|
| GPU-only baseline | 3817.934 |
| Hybrid, kept grain 8 suffix kernel | 3742.514 |

CUDA timing plus suffix counters confirm the remaining bottleneck. In active
hybrid intervals, GPU prefix attention was only a few milliseconds across the 32
layers, while native suffix worker busy time was often comparable or larger.
Representative intervals showed GPU prefix/merge around `1.6-4.7/0.2-0.9 ms`
while suffix worker busy/scatter/attention was around
`3.4-8.0/0.1-1.7/2.3-5.9 ms`. The diagnostic run itself is slower because CUDA
timing synchronizes events, but the attribution is clear: real CPU suffix work is
on the critical path for this near-tail one-block Llama split.

Updated conclusion: the general changes narrowed the suffix runner a little, but
not enough to create a throughput win. The no-real-CPU controls still show that
the split GPU path can beat GPU-only; the real CPU suffix worker remains the tax
that exceeds the small one-block capacity gain. The next promising direction is
a more structural CPU suffix attention optimization or a planner policy that
uses hybrid only where the capacity gain is larger than this measured real-CPU
suffix tax. A one-block-only fast path may help the current synthetic case, but
it would be a narrow optimization rather than evidence that the general hybrid
mechanism wins.


## 2026-05-29 Follow-up: Same-Block K/V Prefetch in CPU Suffix Kernel

This round tested a minimal general CPU suffix-kernel change: while processing a
suffix token, prefetch the next same-block K and V BF16 head vectors. This keeps
the current GQA online-softmax structure and does not add a one-block special
case or request tiling.

Code decision: keep the prefetch. It improved both Llama and Qwen isolated
suffix timings and passed the focused Phase 2 regression suite.

Validation:

```text
cmake --build --preset release --target install
# rebuilt _cots_C successfully

/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 38 passed, 2 warnings
```

Isolated suffix runner, B256, suffix256, 24 CPU threads:

| Model shape | Mode | Before median ms | Prefetch median ms | Change |
|---|---|---:|---:|---:|
| Llama3-8B | direct | 166.509 | 158.462 | 0.952x |
| Llama3-8B | direct+scatter | 171.295 | 162.536 | 0.949x |
| Llama3-8B | prepared | 193.958 | 184.436 | 0.951x |
| Llama3-8B | prepared+scatter | 201.209 | 192.010 | 0.954x |
| Qwen2.5-7B | direct | 82.540 | 76.061 | 0.922x |
| Qwen2.5-7B | direct+scatter | 86.481 | 78.209 | 0.904x |
| Qwen2.5-7B | prepared | 91.882 | 88.221 | 0.960x |
| Qwen2.5-7B | prepared+scatter | 94.639 | 90.285 | 0.954x |

E2E Llama, B512, prompt608, total768, split752, eager, async scheduling
disabled, CPU pool 12 GiB:

| GPU memory | Mode | Effective capacity | Out tok/s |
|---:|---|---:|---:|
| 0.80 | GPU-only | 30.15x | 3714.555, 3730.229 |
| 0.80 | Hybrid | 30.79x | 3724.744, 3716.241 |
| 0.75 | GPU-only | 17.54x | 3037.171 |
| 0.75 | Hybrid | 17.91x | 2883.314 |

Interpretation: the prefetch is a real local kernel improvement, but it still
does not create a robust throughput win. At `gpu_memory_utilization=0.80`, the
best split752 case is now essentially tied with GPU-only: one hybrid repeat is
slightly above one GPU repeat, but the two-run means are `3720.5` hybrid versus
`3722.4` GPU-only. At `0.75`, hybrid still loses decisively because the one-block
capacity gain is too small relative to the fixed real-suffix/merge envelope.
Planner policy should still treat this as a measured near-tie, not a repeatable
throughput win.


## 2026-05-29 Follow-up: Persistent Sharding Attempt and Snapshot Attribution

This round tested a general runner-side optimization idea: replace the prepared
suffix runner's per-layer `at::parallel_for` entry with a COTS-owned persistent
static shard executor over the same `(sequence, KV-head)` task space. This was
not a one-block special case and still used the generic GQA suffix kernel.

Code decision: do **not** keep the persistent sharding change. It passed the
focused correctness suite when forced with `VLLM_COTS_SUFFIX_NUM_THREADS=4`, but
it did not improve the isolated prepared path and regressed Qwen. The attempted
executor was reverted; the installed extension was rebuilt after the revert.

Validation after revert:

```text
cmake --build --preset release --target install
# rebuilt _cots_C successfully

/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 38 passed, 2 warnings
```

Isolated suffix runner, B256, suffix256, 24 CPU threads, with the persistent
sharding attempt active:

| Model shape | Mode | Median ms | Interpretation |
|---|---|---:|---|
| Llama3-8B | direct | 158.836 | unchanged vs kept prefetch path |
| Llama3-8B | direct+scatter | 167.271 | unchanged |
| Llama3-8B | prepared | 185.616 | no meaningful win vs prior `184.436` |
| Llama3-8B | prepared+scatter | 192.533 | no meaningful win vs prior `192.010` |
| Qwen2.5-7B | direct | 76.348 | unchanged |
| Qwen2.5-7B | direct+scatter | 78.585 | unchanged |
| Qwen2.5-7B | prepared | 94.796 | worse than prior `88.221` |
| Qwen2.5-7B | prepared+scatter | 99.684 | worse than prior `90.285` |

Conclusion from the failed attempt: the remaining prepared-path gap is not
mainly PyTorch's per-call parallel runtime entry. Replacing it with our own
persistent static sharding adds enough wakeup/function-call overhead to cancel
or reverse any benefit.

The useful attribution came from the suffix runner counters. With default eager
input snapshots enabled, Llama prepared+scatter spent about `11.7 ms/run` in the
CUDA host callback snapshot copy at B256/suffix256/32 layers:

| Counter group | ms/run |
|---|---:|
| wall median | 177.469 |
| dispatch callback snapshot | 11.703 |
| worker busy | 168.430 |
| worker scatter | 3.949 |
| worker attention | 163.782 |

With snapshots disabled, the isolated prepared runner nearly collapses to the
direct CPU work:

| Model shape | Direct+scatter median ms | Prepared+scatter no-snapshot median ms | Gap |
|---|---:|---:|---:|
| Llama3-8B | 162.544 | 165.025 | +2.481 ms / 32 layers |
| Qwen2.5-7B | 78.412 | 79.819 | +1.407 ms / 28 layers |

A no-snapshot Llama counter run reported wall median `165.392 ms/run`, worker
busy `163.714 ms/run`, worker attention `158.273 ms/run`, worker scatter
`4.756 ms/run`, dispatch callback total only `0.086 ms/run`, and dispatch
snapshot only `0.003 ms/run`.

Important interpretation: eager snapshots explain the large isolated prepared
benchmark gap, but the production static-staging decode path already disables
snapshots when pinned staging slots are protected by reuse events. Therefore this
finding is diagnostic rather than a new throughput fix. It says the prepared
runner envelope is already down to roughly `0.05-0.08 ms/layer` after protected
staging, and the remaining E2E near-tail loss is the real CPU suffix work plus
merge/submit timing being slightly larger than the one-block capacity gain. The
next general attempt should not be another thread-launch substrate; it should
either reduce the actual suffix attention/scatter work, improve overlap in the
E2E layer schedule, or move the planner toward cases with a larger measured
capacity/admission gain.



## 2026-05-29 Follow-up: Fused One-Block Scatter + Suffix Attention

This round targeted the remaining short-suffix runner cost by folding the CPU KV
scatter into the suffix attention worker for the common near-tail case. The
generic all-suffix version was tested first, but it regressed long-suffix
prepared runs. The retained implementation therefore gates fusion to
`max_suffix_blocks <= 1`; longer suffixes keep the existing standalone scatter
followed by the generic GQA suffix attention kernel.

Code decision: keep the gated fused scatter path. It is not a separate
one-block attention kernel: it uses the same generic GQA suffix attention loop,
but copies the newly generated K/V row into the CPU cache inside the worker
before the attention group reads from the cache. This removes the standalone
scatter call for the hot one-block throughput path without adding per-token
staged-K/V reads in the long-suffix loop.

Validation:

```text
cmake --build --preset release --target install
# rebuilt _cots_C successfully

/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 38 passed, 2 warnings
```

Isolated near-tail suffix runner, B512, suffix16, 24 CPU threads, snapshots
disabled:

| Model shape | Mode | Median ms |
|---|---|---:|
| Llama3.1-8B | direct | 24.540 |
| Llama3.1-8B | direct+scatter | 30.466 |
| Llama3.1-8B | prepared | 25.837 |
| Llama3.1-8B | prepared+scatter | 26.911 |
| Qwen2.5-7B | direct | 12.816 |
| Qwen2.5-7B | direct+scatter | 15.182 |
| Qwen2.5-7B | prepared | 14.178 |
| Qwen2.5-7B | prepared+scatter | 14.300 |

The important local result is that `prepared+scatter` is now close to
`prepared` for the one-block shape, especially on Qwen. For the long-suffix
Llama B256/suffix256 shape, the fused path was not kept; the standalone
prepared+scatter rerun after the gate was `165.847 ms`, back in the previous
no-snapshot envelope.

E2E Llama, B512, prompt608, total768, split752, `gpu_memory_utilization=0.80`,
eager, async scheduling disabled, CPU pool 12 GiB:

| Mode | Effective capacity | Out tok/s |
|---|---:|---:|
| GPU-only | 30.15x | 3697.721, 3737.507 |
| Hybrid | 30.79x | 3713.310, 3699.936 |

Interpretation: the fused one-block scatter path improves the isolated runner,
but it still does not produce a robust E2E throughput win. One same-session
hybrid run beat one GPU-only run by about 0.4%, but the repeat flipped the
ordering. The two-run means are `3717.6` GPU-only versus `3706.6` hybrid. The
current evidence is therefore near-tie/no-repeatable-win, not a planner-worthy
hybrid advantage.



## 2026-05-29 Follow-up: Coalesced Prefix Default and First Llama Win

This round revisited the E2E gap after the fused scatter change. The key finding
was that the current throughput runs were not using the coalesced-prefix hybrid
path by default. In mixed decode steps, that meant prefix-only rows and suffix
rows were handled through separate prefix attention paths. Enabling coalesced
prefix lets one GPU prefix attention cover all rows, then merges CPU suffix only
for the suffix-active rows.

Code decision: make `VLLM_COTS_HYBRID_COALESCED_PREFIX` default-on in
`vllm/envs.py`, while preserving `VLLM_COTS_HYBRID_COALESCED_PREFIX=0` as the
diagnostic kill switch. Artifact-copy and indexed-merge remained off; both
regressed this workload.

Validation:

```text
/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_hybrid_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 52 passed, 2 warnings
```

Llama3.1-8B E2E, B512, prompt608, total768, `gpu_memory_utilization=0.80`,
eager, async scheduling disabled, CPU pool 12 GiB:

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only | n/a | 30.15x | 3740.354 |
| Hybrid, coalesced prefix | 736 | 31.46x | 3756.531, 3750.388 |
| Hybrid, coalesced default sanity | 736 | 31.46x | 3764.667 |

This is the first current-code Llama E2E throughput win against the same
GPU-memory setting. The best default-on hybrid run is `1.0065x` the fresh
GPU-only repeat; the two explicit coalesced-prefix runs are `1.0043x` and
`1.0027x`. The win is small, but it is no longer just the split752 near-tie.
The planner-relevant condition is that one block of GPU KV savings was still too
small; the two-block split736 capacity increase was large enough to offset the
remaining split-attention overhead after coalescing.

Negative controls from the same round:

| Hybrid variant | Split | Out tok/s | Interpretation |
|---|---:|---:|---|
| Coalesced prefix | 752 | 3729.749 | one-block capacity gain still too small |
| Coalesced + artifact copy | 752 | 3672.895 | explicit artifact copy hurts |
| Coalesced + indexed merge | 752 | 3710.828 | indexed merge hurts this all/mixed-row case |
| Coalesced + dummy GPU suffix | 752 | 3722.355 | no-real-CPU headroom is not enough at one block |
| Coalesced prefix | 801 mem, split752 | 3740.070 | still below GPU-only 3793.228 at that cliff |

Updated conclusion: the throughput win is achievable, but only after both
conditions hold: the split GPU path must use coalesced prefix by default, and the
planner must choose a split with enough integer/admission gain to pay the
remaining LSE/merge/native-suffix overhead. For this Llama synthetic case that
means split736 at 0.80, not the earlier split752 near-tail setting.



## 2026-05-29 Follow-up: Robustness Checks After Coalesced Default

After the first Llama split736 win, this round checked whether the same current
code/defaults generalize across model geometry and lower-memory settings. All
throughput runs below kept `VLLM_COTS_SUFFIX_NUM_THREADS=24`, eager execution,
async scheduling disabled, B512, prompt608, total768, and CPU pool 12 GiB.

Qwen2.5-7B at `gpu_memory_utilization=0.80`:

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only | n/a | 82.96x | 5837.726 |
| Hybrid | 736 | 86.57x | 5728.208 |
| Hybrid | 752 | 84.72x | 5722.396 |

Qwen still loses at 0.80. The reason is not CPU kernel raw speed; it is that
Qwen already has very high GPU KV capacity because of its smaller KV footprint,
so the extra capacity from moving one or two suffix blocks to CPU does not repay
the split-attention overhead.

Qwen2.5-7B at `gpu_memory_utilization=0.67`:

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only | n/a | 8.04x | 2167.625 |
| Hybrid | 752 | 8.21x | 2065.000 |

Even under tight Qwen KV memory, the one-block capacity increase is only about
`1.02x`, while the hybrid path costs more than that.

Llama3.1-8B at `gpu_memory_utilization=0.75`:

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only | n/a | 17.54x | 3041.334 |
| Hybrid | 736 | 18.30x | 2975.872 |
| Hybrid | 752 | 17.91x | 2929.964 |

The Llama win at 0.80 does not extend to 0.75. At this lower memory point,
capacity improves, but the scheduler/throughput dynamics do not repay the extra
hybrid work. Split736 is still better than split752 here, but both lose.

One stability caveat: a diagnostic rerun of Llama split736 at 0.80 without
`VLLM_COTS_SUFFIX_NUM_THREADS=24` failed during engine initialization inside
`cudaHostAlloc` from `install_wait_kernel_sync_for_task`. This happened after
many rapid benchmark launches, so it should not be overinterpreted as a thread
policy result. The verified win condition remains explicit about the 24-thread
suffix setting.

Updated planner conclusion: hybrid should not be globally enabled. The current
measured win region is narrow but real: Llama3.1-8B, 0.80 GPU memory,
split736/two CPU suffix blocks, coalesced-prefix default, 24 suffix threads. For
Qwen and for lower-memory Llama 0.75, GPU-only remains faster under the same
synthetic workload.


## 2026-05-29 Follow-up: Default Suffix Thread Policy

The previous Llama split736 win was still fragile because the verified runs set
`VLLM_COTS_SUFFIX_NUM_THREADS=24`. With the env var absent, the native suffix
runner previously returned `0` from the thread-policy helper and therefore did
not force the CPU attention worker thread count.

Code decision: make the native suffix runner default to
`min(std::thread::hardware_concurrency(), 24)` threads when
`VLLM_COTS_SUFFIX_NUM_THREADS` is unset or empty. Explicit invalid values or
`0` still mean "do not override PyTorch threads" for diagnostics.

Validation:

```text
cmake --build --preset release --target install
# rebuilt _cots_C successfully

/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_hybrid_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 52 passed, 2 warnings
```

A small no-env counter probe with `VLLM_COTS_SUFFIX_COUNTERS=1` reported
`COTS suffix worker threads req/obs: 24/24` once CPU suffix rows became active,
confirming that the default policy is actually used by the native worker.

Same-session no-env Llama3.1-8B E2E, B512, prompt608, total768,
`gpu_memory_utilization=0.80`, eager, async scheduling disabled:

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only | n/a | 30.15x | 3732.583 |
| Hybrid | 736 | 31.46x | 3732.934 |

Interpretation: the explicit `VLLM_COTS_SUFFIX_NUM_THREADS=24` launch condition
is no longer required. The current no-env pair is effectively a tie with a tiny
hybrid edge (`1.00009x`), smaller than the earlier split736 repeats, so the
planner conclusion stays conservative: hybrid is measurable only in a narrow
Llama 0.80/split736 region and should remain profile-gated. The earlier Qwen
and lower-memory Llama checks still lose under this synthetic workload.


## 2026-05-29 Follow-up: Longer Context and Two-Block Fusion

This round tested two possible ways to turn the narrow Llama near-tie into a
clearer throughput win. First, we tried longer total context with later splits;
second, we expanded the fused scatter path from one suffix block to two suffix
blocks, matching the current Llama split736 win candidate.

Longer-context Llama3.1-8B checks, B512, eager, async scheduling disabled,
`gpu_memory_utilization=0.80`, no explicit suffix-thread env:

| Prompt / total | Mode | Split | Effective capacity | Out tok/s |
|---|---|---:|---:|---:|
| 896 / 1024 | GPU-only | n/a | 22.61x | 3629.406 |
| 896 / 1024 | Hybrid | 896 | 25.84x | 2589.174 |
| 896 / 1024 | Hybrid | 960 | 24.12x | 3467.995 |
| 896 / 1024 | Hybrid | 992 | 23.34x | 3417.246 |
| 960 / 1024 | GPU-only | n/a | 22.61x | 3887.210 |
| 960 / 1024 | Hybrid | 960 | 24.12x | 3139.507 |

The longer-context result is negative. Split896 has enough capacity gain, but
128-token CPU suffix attention is too expensive. Split960 reduces CPU suffix to
64 tokens and gets much closer, but still loses. Split992 is the two-block
analogue of split736, but at total1024 the capacity gain is too small to repay
hybrid overhead. Aligning the prompt to split960 also loses because the short
decode length exposes hybrid overhead more than resident-capacity gain.

Implementation change kept: set `kFusedScatterMaxSuffixBlocks = 2`. This keeps
the same generic GQA suffix loop, but skips the standalone scatter call for
small suffixes up to two blocks. It directly targets split736 and similar
planner candidates without adding a separate one-block-only kernel.

Validation after the two-block gate and pooled wait-slot change:

```text
cmake --build --preset release --target install
# rebuilt _cots_C successfully

/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_hybrid_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 52 passed, 2 warnings
```

Llama3.1-8B split736 current-code result, B512, prompt608, total768,
`gpu_memory_utilization=0.80`:

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only | n/a | 30.15x | 3721.399 |
| Hybrid, two-block fused scatter | 736 | 31.46x | 3787.940, 3781.640 |

This is now a clearer repeatable win than the earlier near-tie: the two hybrid
repeats are about `1.016x` to `1.018x` over the same-session GPU-only run.

Robustness checks remain conservative:

| Model / memory | Mode | Split | Effective capacity | Out tok/s | Interpretation |
|---|---|---:|---:|---:|---|
| Qwen2.5-7B / 0.80 | Hybrid | 736 | 86.57x | 5745.603 | Slightly better than prior split736, still below GPU-only 5837.726. |
| Llama3.1-8B / 0.75 | Hybrid | 736 | 18.30x | 3000.362 | Better than prior split736, still below GPU-only 3041.334. |
| Llama3.1-8B / 0.80, total1024 | Hybrid | 992 | 23.34x | 3417.246 | Two-block fusion is not enough for the longer-context split992 case. |

A reliability fix landed in the same round: suffix wait-kernel sync now uses one
mapped host slot pool instead of two `cudaHostAllocMapped` calls per prepared
task. At B512, the previous path could perform tens of thousands of tiny mapped
host allocations during engine initialization and repeatedly hit a segfault in
`cudaHostAlloc` after many benchmark launches. After pooling, the same
total1024/split992 launch that had just crashed initialized and completed.

Updated planner conclusion: hybrid is now convincingly positive for the
Llama3.1-8B 0.80/split736 synthetic point, but it is still not globally
profitable. The planner should treat the two-block fused path as the preferred
small-suffix candidate, then rely on measured profile/admission tables rather
than enabling hybrid for all models, memory budgets, or longer contexts.


## 2026-05-29 Follow-up: Qwen Tight-Memory Split Sweep

After the two-block fused scatter change made Llama split736 clearly positive,
this round revisited Qwen2.5-7B under the tighter `gpu_memory_utilization=0.67`
setting. The hypothesis was that Qwen might need a slightly earlier split than
Llama because its GPU-only KV capacity is small at this memory budget, while its
CPU suffix attention is cheaper than Llama's.

Workload: Qwen2.5-7B, B512, prompt608, total768, eager, async scheduling
disabled, no explicit suffix-thread env.

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only | n/a | 8.04x | 2156.910, 2161.736 |
| Hybrid | 736 | 8.39x | 2102.544 |
| Hybrid | 720 | 8.58x | 2122.198 |
| Hybrid | 704 | 8.77x | 2154.900, 2160.484 |

Split704 is the best current Qwen point, but it is still a noise-level tie, not
a planner-worthy win. Its two-run mean is `2157.692` versus GPU-only mean
`2159.323`. Split736 has too little capacity gain, while split720 and split704
show that adding capacity helps until the larger CPU suffix cost eats it.

A four-block fused scatter gate was tested specifically for split704, because
that split has four CPU suffix blocks and was almost tied. It passed the suffix
runner tests, but the target throughput regressed to `2142.811 out tok/s`. The
experiment was reverted; the retained gate remains
`kFusedScatterMaxSuffixBlocks = 2`. This is consistent with the earlier
all-suffix fusion attempt: folding scatter into the attention loop helps very
short suffixes, but starts hurting once the suffix is longer.

Test coverage was strengthened with a prepared native runner case that scatters
QKV into a true two-block suffix (`max_suffix_blocks=2`) for both Qwen and Llama
GQA shapes. Targeted validation after reverting the four-block experiment:

```text
/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py -q
# 9 passed, 2 warnings
```

Full focused Phase 2 validation with the added coverage:

```text
/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_hybrid_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 54 passed, 2 warnings
```

Updated conclusion: the current implementation has a real Llama win, but Qwen
still does not have a robust positive point in this synthetic B512/prompt608/
total768 sweep. The Qwen 0.67 result is useful for the planner because it marks
the boundary: extra CPU KV capacity can bring hybrid to parity, but the current
CPU suffix/merge envelope is still a little too expensive for a reliable Qwen
throughput win.

## 2026-05-29 Follow-up: Unchecked Native Scatter Fallback

The next Qwen throughput attempt targeted the remaining split704 gap rather than
changing the split policy. Counter runs showed that split704 uses four CPU suffix
blocks, so it does not take the retained two-block fused scatter path. In this
fallback path the prepared native runner was still calling the public checked
scatter wrapper on every layer. That wrapper revalidated tensor shapes, dtypes,
block IDs, and block offsets before the actual K/V copy, even though the runner
had already built the metadata.

Change: add an internal `gqa_bf16_scatter_suffix_kv_unchecked_at` helper and use
it only from the prepared native worker's standalone scatter fallback. The
public Python-visible `gqa_bf16_scatter_suffix_kv` entry point remains checked.

Target workload: Qwen2.5-7B, B512, prompt608, total768,
`gpu_memory_utilization=0.67`, split704, eager, async scheduling disabled, no
explicit suffix-thread env.

| Mode | Effective capacity | Out tok/s |
|---|---:|---:|
| GPU-only | 8.04x | 2159.264, 2156.436 |
| Hybrid split704 | 8.77x | 2166.316, 2163.732 |

The paired means are `2157.850` GPU-only versus `2165.024` hybrid, or about
`1.0033x` for hybrid. This is still a very small win, but it is the first
same-session Qwen tight-memory result that is consistently above GPU-only rather
than a tie/slight loss.

Diagnostic caveat: the counter run itself perturbs throughput (`2142.667 out
tok/s` with counters enabled before this optimization), but it usefully showed
that the residual Qwen cost was on the native suffix worker path rather than
submit/callback overhead. The standalone scatter slice was smaller than CPU
suffix attention, so this optimization was expected to be incremental, not a
large step-function improvement.

Updated conclusion: we now have two narrow positive points: Llama3.1-8B at
0.80/split736 and Qwen2.5-7B at 0.67/split704. Both are small synthetic wins,
not enough to enable hybrid unconditionally. The planner should still require
per-model/per-memory profiling, but hybrid is no longer universally losing in
our measured throughput path.

## 2026-05-29 Follow-up: Small-Suffix Two-Pass CPU Attention

After unchecked standalone scatter, the residual Qwen split704 cost was still
inside the native suffix worker, with CPU suffix attention dominating submit,
callback, queue, and scatter time. A quick thread-policy probe did not improve
things: Qwen split704 at `gpu_memory_utilization=0.67` produced `2161.656 out
tok/s` with `VLLM_COTS_SUFFIX_NUM_THREADS=16` and `2101.280 out tok/s` with
`VLLM_COTS_SUFFIX_NUM_THREADS=32`, so the retained default cap of 24 threads
remains the best observed policy among these checks.

The next general kernel attempt was a small-suffix two-pass attention path for
`seq_len <= 128`. The existing online-softmax loop is robust, but for short CPU
suffixes it does two exponentials per token per query head and rescales the
partial output vector every token. The new path computes logits/max first, then
runs a softmax/value pass with one exponential per token and no per-token output
rescale. Longer suffixes keep the existing online path.

Implementation notes:

- The threshold is `kTwoPassMaxSeqLen = 128` in the GQA CPU suffix kernel.
- Scratch logits use a fixed thread-local buffer sized for `kMaxHeadsPerKV` so
  the hot path avoids per-task heap allocation. A first version with a large
  template-dependent stack array triggered a GCC 11 internal compiler error at
  `-O3`; the thread-local fixed buffer rebuilt cleanly.
- Exact-threshold correctness coverage was added with `seq_lens=[128, 128]` for
  both Qwen and Llama GQA shapes.

Qwen2.5-7B, B512, prompt608, total768, `gpu_memory_utilization=0.67`, split704,
eager, async scheduling disabled, no explicit suffix-thread env:

| Mode | Effective capacity | Out tok/s |
|---|---:|---:|
| GPU-only | 8.04x | 2162.435 |
| Hybrid split704, two-pass | 8.77x | 2180.878, 2168.010 |

The two hybrid runs average `2174.444 out tok/s`, about `1.0055x` over the fresh
GPU-only run. This is still modest, but it is a clearer margin than the
unchecked-scatter-only result.

Llama3.1-8B, B512, prompt608, total768, `gpu_memory_utilization=0.80`, split736:

| Mode | Effective capacity | Out tok/s |
|---|---:|---:|
| GPU-only | 30.15x | 3742.022 |
| Hybrid split736, two-pass | 31.46x | 3777.188 |

Llama remains positive at about `1.0094x` over the fresh GPU-only baseline,
though this single hybrid run is slightly below the earlier best two-block
fusion pair (`3787.940`, `3781.640`). The important point is that the general
small-suffix two-pass path improves Qwen without destroying the Llama win.

Focused validation after the change:

```text
/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_hybrid_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 58 passed, 2 warnings
```

Updated conclusion: hybrid now has small but repeatable positive synthetic
points for both Qwen tight-memory and Llama mid-memory settings. The win is not
large enough to justify unconditional hybrid admission, but the planner now has
credible positive profile points rather than only parity/near-losses.

## 2026-05-29 Follow-up: Two-Pass Boundary Checks

The small-suffix two-pass CPU attention path improves the native suffix cost, but
it does not make every previously losing split profitable. Two paired boundary
checks are important for planner policy.

Llama3.1-8B, B512, prompt608, total768, split736,
`gpu_memory_utilization=0.75`:

| Mode | Effective capacity | Out tok/s |
|---|---:|---:|
| GPU-only | 17.54x | 3043.805 |
| Hybrid split736, two-pass | 18.30x | 2985.842 |

This point remains negative. Even though hybrid increases effective capacity by
about 4.3%, the lower-memory Llama run still does not gain enough batching
benefit to cover the hybrid suffix path overhead.

Qwen2.5-7B, B512, prompt608, total768, split736,
`gpu_memory_utilization=0.80`:

| Mode | Effective capacity | Out tok/s |
|---|---:|---:|
| GPU-only | 82.96x | 5829.575 |
| Hybrid split736, two-pass | 86.57x | 5758.882 |

This point also remains negative. The two-pass path narrows the earlier Qwen
0.80 gap, but the capacity gain at split736 is still too small for a throughput
win.

Planner implication: hybrid should be selected only from measured positive
profile cells, not from a monotonic "less GPU memory always helps hybrid" rule.
The current positive cells are Qwen 0.67/split704 and Llama 0.80/split736; Qwen
0.80/split736 and Llama 0.75/split736 should remain GPU-only unless a later
optimization changes the measured balance.

## 2026-05-29 Follow-up: Graph-Mode and Rejected Return-Path Attempts

After the small-suffix two-pass kernel produced narrow eager-mode wins, this
round checked whether the remaining gap could be reduced by restoring older
kernel tiling, changing the suffix artifact return path, or moving the same
cells to CUDA graph mode.

First, the old two-token QK tile was restored inside the current generic
small-suffix two-pass path as an experiment. It rebuilt and passed focused
suffix correctness (`28 passed, 2 warnings`), but the focused prepared runner
regressed in both model shapes, so the edit was reverted:

| Shape | Case | Before | With QK pair tile | Decision |
|---|---|---:|---:|---|
| Qwen2.5-7B | B512, suffix64, prepared+scatter | 57.396 ms | 60.842 ms | Revert |
| Llama3.1-8B | B512, suffix32, prepared+scatter | 64.650 ms | 66.395 ms | Revert |

This is different from the earlier pre-two-pass QK tile result. In the current
two-pass structure, token pairing increases register pressure and does not help
the hot short-suffix path.

Second, explicit suffix artifact H2D copy was tested with
`VLLM_COTS_HYBRID_SUFFIX_ARTIFACT_COPY=1`. This avoids UVA reads of the CPU
suffix output/LSE during merge, but the extra copy and synchronization cost more
than they save:

| Model / setting | Hybrid default | Hybrid with artifact copy | Decision |
|---|---:|---:|---|
| Qwen2.5-7B, 0.67, split704, eager | 2180.878, 2168.010 | 2126.835 | Keep UVA merge path |
| Llama3.1-8B, 0.80, split736, eager | 3777.188 | 3728.901 | Keep UVA merge path |

Third, graph mode was evaluated on the two positive eager cells. Llama remains
slightly positive, but Qwen does not:

| Model / setting | Mode | Effective capacity | Out tok/s |
|---|---|---:|---:|
| Llama3.1-8B, 0.80, split736, graph | GPU-only | 29.96x | 3753.398 |
| Llama3.1-8B, 0.80, split736, graph | Hybrid | 30.98x | 3766.721 |
| Qwen2.5-7B, 0.67, split704, graph | GPU-only | 7.62x | 2192.026 |
| Qwen2.5-7B, 0.67, split704, graph | Hybrid | 7.75x | 2005.585 |

Graph memory changes the capacity math: both modes lose some KV capacity to CUDA
graph pools, and the COTS graph path defaults to PIECEWISE capture. For Llama,
the hybrid capacity increase still beats the suffix overhead by a small margin.
For Qwen, graph-mode hybrid overhead is too large and the eager Qwen win should
not be treated as graph-robust.

A benchmark-only flag, `--cots-auto-graph-split {auto,true,false}`, was added to
`benchmark_ratio_e2e.py` so graph policy can be A/B tested without changing
runtime defaults. Disabling the automatic COTS graph split for the Qwen hybrid
cell failed during engine initialization in legacy FULL+PIECEWISE capture with
`cudaErrorInvalidValue` from `_synchronize_static_metadata_reuse()` while
profiling CUDA graph memory. For now, the automatic PIECEWISE policy is the only
validated COTS hybrid graph policy in this workload.

Updated conclusion: the measured throughput win is real but narrower in graph
mode. Current validated positive cells are Llama3.1-8B 0.80/split736 in eager and
graph mode, and Qwen2.5-7B 0.67/split704 in eager mode only. Artifact-copy and
current two-pass QK pairing should not be kept. The planner/profiler must record
execution mode as part of the profile key; eager Qwen data cannot be reused for
graph-mode admission.

## 2026-05-29 Follow-up: Qwen Graph Split Recovery

The first Qwen graph-mode check used the eager-optimal split704 and lost badly:
GPU-only was `2192.026 out tok/s`, while hybrid split704 was only
`2005.585 out tok/s`. The graph logs showed why this was not directly comparable
to the eager split704 result: CUDA graph memory reduced hybrid GPU KV capacity to
`5456` tokens, so split704 gave only `7.75x` effective capacity versus GPU-only
`7.62x`. The eager split704 point had `8.77x` effective capacity, so graph mode
had erased most of the planner benefit.

A small graph-mode split sweep recovered the Qwen win by moving the split earlier:

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only graph | n/a | 7.62x | 2192.026, 2193.239 |
| Hybrid graph | 704 | 7.75x | 2005.585 |
| Hybrid graph | 688 | 7.93x | 2087.088 |
| Hybrid graph | 672 | 8.12x | 2242.814, 2251.883 |

This is an important correction to the previous graph-mode conclusion. Qwen graph
mode is not inherently unable to win; the eager-optimal split was wrong once CUDA
graph memory reduced available GPU KV. With split672, the hybrid graph runs beat
the fresh GPU-only graph baseline by about `2.3-2.7%`.

Updated planner implication: split selection must be keyed by execution mode and
actual graph-memory-adjusted GPU KV capacity. For Qwen2.5-7B at
`gpu_memory_utilization=0.67`, the current positive graph cell is split672, not
split704. The broader rule is not "graph mode disables Qwen hybrid"; it is
"graph mode changes the effective capacity curve, so the profiler must resweep
splits under the graph policy it will actually run." The automatic PIECEWISE COTS
graph policy remains required; disabling it still fails in legacy full capture.

## 2026-05-29 Follow-up: Llama 0.75 Split Resweep

After Qwen graph split recovery, the lower-memory Llama loss was rechecked to see
whether the earlier negative result was simply a bad split. Same-session eager
runs used Llama3.1-8B, B512, prompt608, total768,
`gpu_memory_utilization=0.75`, async scheduling disabled, no explicit suffix
artifact copy, default 24 suffix threads:

| Mode | Split | Effective capacity | Out tok/s | vs fresh GPU-only |
|---|---:|---:|---:|---:|
| GPU-only eager | n/a | 17.54x | 3019.741 | 1.000x |
| Hybrid eager | 752 | 17.91x | 2938.010 | 0.973x |
| Hybrid eager | 736 | 18.30x | 3006.745 | 0.996x |
| Hybrid eager | 720 | 18.71x | 2946.139 | 0.976x |

Split736 is still the best lower-memory Llama point, but it remains just below
the fresh GPU-only baseline. Moving the split later gives too little capacity
gain; moving it earlier gives slightly more capacity but raises suffix overhead
faster than admission capacity improves.

The same lower-memory point was then checked under graph mode:

| Mode | Split | Effective capacity | Out tok/s | vs GPU-only graph |
|---|---:|---:|---:|---:|
| GPU-only graph | n/a | 17.35x | 3102.410 | 1.000x |
| Hybrid graph | 736 | 17.83x | 2885.644 | 0.930x |

Graph mode does not rescue this Llama 0.75 case. The hybrid graph run has only a
small capacity gain after graph-memory accounting, while it still pays the
PIECEWISE split path and real CPU suffix return path. This reinforces the planner
rule: lower GPU memory is not monotonic evidence for hybrid. The selected split
must produce enough measured effective-capacity gain, under the exact execution
mode, to exceed the suffix path tax. For current code, Llama 0.75 should remain
GPU-only, while the validated positive Llama cell remains the mid-memory 0.80
split736 case.

## 2026-05-29 Follow-up: Qwen Graph Win Repeat Validation

A same-session current-code repeat was run for the strongest positive cell found
so far: Qwen2.5-7B, B512, prompt608, total768,
`gpu_memory_utilization=0.67`, graph mode, async scheduling disabled. The split
is the graph-reswept split672, not the eager-optimal split704.

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only graph | n/a | 7.62x | 2179.022 |
| GPU-only graph | n/a | 7.62x | 2177.591 |
| Hybrid graph | 672 | 8.12x | 2249.883 |
| Hybrid graph | 672 | 8.12x | 2243.443 |

Same-session means: GPU-only `2178.306 out tok/s`, hybrid `2246.663 out tok/s`,
or `1.031x` hybrid/GPU-only. Combining these two fresh repeats with the previous
Qwen graph split672 evidence gives GPU-only mean `2185.469` and hybrid mean
`2247.006`, or `1.028x` hybrid/GPU-only across four runs per side.

This is the cleanest current Phase 2 throughput-win cell. It is larger and more
repeatable than the narrow Llama 0.80/split736 edge. The mechanism is also clear:
CUDA graph memory changes the effective capacity curve, and split672 raises Qwen
hybrid from `7.62x` GPU-only capacity to `8.12x` while the Qwen CPU suffix path is
cheap enough for that extra admission capacity to dominate. Planner/profiler
selection should therefore include this cell as a hybrid-positive graph-mode
profile entry, while still rejecting Qwen graph split704 and Llama 0.75.

## 2026-05-29 Follow-up: Profile-Gated Planner Hook

Historical note: this planner hook was removed on 2026-05-30 and is retained here only as investigation history.

The repeatable Qwen graph win had been turned into an executable planner
selection rule rather than only a prose result. `FastTTS-thesis/planner.py` keeps
manual planner behavior unchanged, but adds an optional `phase2_kv_profile` block
that can hold inline measured cells or load them from a JSON artifact. The
selector only enables COTS hybrid KV when all of these are true:

- the measured cell matches role, model, dtype, GPU-memory utilization,
  `enforce_eager`/graph mode, `max_model_len`, and any configured
  `max_num_seqs`;
- the measured hybrid output-token throughput is at least `1 + win_margin` times
  the matched GPU-only baseline;
- the engine plan did not already provide an explicit manual KV plan.

The measured-cell artifact lives at
`David/Benchmarks/phase2/phase2_kv_measurement_cells.json`. It includes the current
positive and negative synthetic cells, including Qwen graph 0.67/split672,
Qwen eager 0.67/split704, Llama 0.80/split736, and the rejected Llama 0.75 and
Qwen graph split704 cells. With the default `win_margin=0.01`, the robust Qwen
graph split672 cell is selected. After the current-code Llama eager repeat below,
Llama 0.80/split736 also clears the default margin; narrower sub-1% cells still
require an explicitly lower margin and remain less planner-worthy.

Focused validation for this planner hook:

```text
/opt/conda/envs/thesis/bin/python -m py_compile \
  /TTC/FastTTS-thesis/planner.py \
  /TTC/David/Benchmarks/phase2/test_phase2_manual_planner.py
# passed

/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/David/Benchmarks/phase2/test_phase2_manual_planner.py -q
# 6 passed

/opt/conda/envs/thesis/bin/python -m json.tool \
  /TTC/David/Benchmarks/phase2/phase2_kv_measurement_cells.json
# valid JSON
```

This closes the immediate planner concern raised during the investigation: the
planner will not blindly choose hybrid just because CPU KV exists. It now has a
profile-gated path that selects hybrid for the measured positive graph-mode Qwen
cell and leaves measured negative cells GPU-only.

## 2026-05-29 Follow-up: Planner Dry-Run Tightening

Historical note: this tightening applied to the removed profile-gated hook.

The profile-gated planner hook was tightened after dry-running it against the
validated Qwen graph cell. Three benchmark details are now part of the
measured profile contract:

- `async_scheduling=false` is a matched profile field. A config that explicitly
  enables async scheduling will not consume the cell.
- `disable_hybrid_kv_cache_manager=true` is carried as a measured
  `engine_overrides` entry. The selector applies it when absent, and rejects the
  cell if the launch config explicitly conflicts.
- Workload shape is now part of the profile key. The Qwen-positive artifact cell
  records `batch=512`, `prompt_tokens=608`, `total_tokens=768`,
  `suffix_tokens=160`, and `prompt_mode=shared`; the selector only consumes that
  cell when the planner config provides the same workload descriptor.

Dry-run resolved generator vLLM kwargs for the validated Qwen graph cell now
match the benchmark-critical launch shape:

```json
{
  "async_scheduling": false,
  "cots_f_cpu_store": 0.0,
  "cots_f_prefetch": 0.0,
  "cots_kv_cpu_pool_bytes": 12884901888,
  "cots_kv_h2d_mode": "uva",
  "cots_kv_split_blocks": 42,
  "disable_hybrid_kv_cache_manager": true,
  "disable_log_stats": true,
  "dtype": "bfloat16",
  "enforce_eager": false,
  "gpu_memory_utilization": 0.67,
  "max_model_len": 768,
  "max_num_seqs": 512,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "offload_backend": "cots",
  "trust_remote_code": true
}
```

Additional focused validation:

```text
/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/David/Benchmarks/phase2/test_phase2_manual_planner.py -q
# 10 passed
```

This makes the current positive cell planner-ready in the practical sense: the
profile selector emits the same COTS KV geometry and execution-mode knobs used by
the repeat-validated benchmark, and measured negative or runtime-mismatched cells
remain GPU-only.


## 2026-05-29 Follow-up: Llama Eager Profile Promotion

Historical note: this promotion applied to the removed profile-gated hook; the cell remains useful measurement data.

The Llama3.1-8B eager split736 cell was rerun because the existing profile
artifact recorded it as a narrow `1.009x` win, just below the default planner
margin. Same-session current-code repeats used B512, prompt608, total768,
`gpu_memory_utilization=0.80`, eager mode, async scheduling disabled,
`max_num_seqs=512`, CPU pool 12 GiB, no explicit suffix-thread env:

| Mode | Split | Effective capacity | Out tok/s |
|---|---:|---:|---:|
| GPU-only eager | n/a | 30.15x | 3711.019, 3702.521 |
| Hybrid eager | 736 | 31.46x | 3776.895, 3762.405 |

The fresh means are `3706.770 out tok/s` for GPU-only and `3769.650 out tok/s`
for hybrid, or `1.017x` hybrid/GPU-only. The profile artifact now stores this
current-code repeat under `llama-eager-0.80-split736-current-repeat`, so the
default `win_margin=0.01` selector can choose both the robust Qwen graph cell and
this Llama eager cell when the exact workload/runtime key matches. Llama 0.75 and
Qwen graph split704 remain negative control cells.

Focused planner coverage was extended with a default-margin Llama eager selection
test; a workload mismatch test remains in place so this profile still cannot leak
to a different prompt/batch shape.

Focused implementation validation after this profile promotion still passes the
Phase 2 hybrid attention/suffix/KV suite:

```text
/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_hybrid_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention.py \
  /TTC/vllm/tests/kernels/attention/test_cots_suffix_attention_runner.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 58 passed, 2 warnings
```


## 2026-05-30 Follow-up: Longer-Sequence Policy Probe

A first post-cleanup policy probe tested whether increasing the generation
length makes the existing hybrid wins more general. The hypothesis was that a
longer total sequence should increase the value of each GPU KV token saved by
the split, especially when the split stays near the tail. The result was
negative for both model families in the tested cells.

Llama3.1-8B, B512, prompt608, total1024, `gpu_memory_utilization=0.80`, eager,
async scheduling disabled, CPU pool 12 GiB:

| Mode | Split | Effective capacity | CPU suffix tokens at end | Out tok/s | vs GPU-only |
|---|---:|---:|---:|---:|---:|
| GPU-only | n/a | 22.61x | n/a | 2617.170 | 1.000x |
| Hybrid | 864 | 26.80x | 160 | 2310.760 | 0.883x |
| Hybrid | 928 | 24.95x | 96 | 2415.867 | 0.923x |
| Hybrid | 960 | 24.12x | 64 | 2455.014 | 0.938x |
| Hybrid | 1008 | 22.97x | 16 | 2437.183 | 0.931x |

The total768 Llama split736 cell was narrowly positive, but the total1024 sweep
did not generalize it. Even split960 has a larger capacity gain than the
total768 split736 case, but doubling the suffix length from 32 to 64 tokens and
running a longer active hybrid interval erases the benefit. The split1008
one-block control is also negative: exact split1024/no-suffix hybrid is rejected
at initialization because COTS hybrid KV requires at least one CPU suffix block,
but activating only the final 16 tokens still loses about 7% to GPU-only.

Qwen2.5-7B, B512, prompt608, total1024, `gpu_memory_utilization=0.67`, graph
mode, async scheduling disabled, CPU pool 12 GiB:

| Mode | Split | Effective capacity | CPU suffix tokens at end | Out tok/s | vs GPU-only |
|---|---:|---:|---:|---:|---:|
| GPU-only graph | n/a | 5.72x | n/a | 1154.997 | 1.000x |
| Hybrid graph | 928 | 5.88x | 96 | 964.014 | 0.835x |
| Hybrid graph | 864 | 6.31x | 160 | 963.753 | 0.835x |

For Qwen graph mode, increasing `max_model_len` to 1024 moves the run into a
very memory-limited regime, but COTS graph mode also has fewer GPU KV tokens
than the GPU-only graph baseline (`5456` vs `5856`). Split928 therefore gives
only a tiny effective-capacity increase, and split864 gives about a 10% capacity
increase but pays too much active suffix/merge overhead.

Policy implication: longer total length is not a sufficient monotonic predictor
for hybrid. The useful predictor has to include at least execution mode,
graph-memory-adjusted GPU KV tokens, split-derived effective capacity gain, and
end-of-run CPU suffix length. Current positive cells stay narrow: Qwen graph
total768/split672 and Llama eager total768/split736. The split1008 Llama control
suggests a fixed active-hybrid tax remains even when CPU suffix work is only one
block, so the next attempt should measure and reduce the active transition path
rather than assume a later split is cheap enough.


## 2026-05-30 Follow-up: One-Block Active-Suffix Tax Breakdown

The Llama total1024/split1008 one-block control was rerun with
`VLLM_COTS_HYBRID_CUDA_TIMING=1`, `VLLM_COTS_SUFFIX_COUNTERS=1`,
`VLLM_LOG_STATS_INTERVAL=5`, and vLLM stats enabled. This run is diagnostic, not
a throughput baseline, because CUDA timing instrumentation lowered end-to-end
throughput to `2151.767 out tok/s`.

The active windows showed the minimum suffix case is not primarily CPU suffix
compute bound. Representative 5-second stats windows over 32 layer calls:

| Window shape | Mixed prefix gpu/wall ms | CPU suffix attn ms | Native busy ms | UVA artifacts MB |
|---|---:|---:|---:|---:|
| 96 suffix rows | 5.858 / 17.530 | 1.779 | 2.037 | 0.799 |
| 192 suffix rows | 5.710 / 18.508 | 1.819 | 3.226 | 1.597 |
| 192 suffix rows, large prefix rows | 35.688 / 95.073 | 3.968 | 7.518 | 1.597 |
| 64 suffix rows, large prefix rows | 19.333 / 52.852 | 3.972 | 4.925 | 0.532 |
| 32 suffix rows | 5.941 / 16.988 | 1.705 | 0.382 | 0.266 |

Submit/dispatch overhead was small in comparison: suffix submit
prep/snapshot/launch was typically about `0.13-0.16 ms` total per stats window,
and dispatch callback/snapshot/enqueue was usually below `0.15 ms`. Q/K/V D2H
and UVA artifacts were also tiny in bandwidth terms.

This strengthens the current bottleneck diagnosis: even the one-block suffix
case pays a mixed active-prefix path whose wall time is tens of milliseconds per
stats window, while CPU suffix attention is usually only a few milliseconds. The
next throughput attempt should therefore target the active mixed prefix/LSE path
or the scheduling pattern that creates large mixed prefix-row work, not another
CPU suffix micro-optimization.

## 2026-05-30 Follow-up: Coalesced Prefix and Wrapper-Tax Isolation

The Llama total1024/split1008 one-block control was rerun as a matched A/B for
the partial-hybrid prefix strategy:

| Mode | Coalesced prefix | Out tok/s | vs matched GPU-only |
|---|---:|---:|---:|
| GPU-only | n/a | 2629.646 | 1.000x |
| Hybrid split1008 | on | 2429.151 | 0.924x |
| Hybrid split1008 | off | 2384.422 | 0.907x |

This rules out the coalesced-prefix path as the source of the loss. It helps by
about 1.9% relative to the older split prefix/suffix-row path, but the one-block
hybrid case still loses about 7.6% to GPU-only while gaining only about 1.6%
effective KV capacity.

The raw FlashAttention LSE microbench also ruled out LSE materialization as the
main tax. For the Llama shape (`B=512`, `Hq=32`, `Hkv=8`, `D=128`,
`seq=1024`, `split=1008`), prefix out-only was `2.2892 ms`, while prefix+LSE
was `2.2898 ms`.

The no-op-suffix wrapper microbench is more explanatory. It replaces CPU suffix
attention with a no-op and therefore measures the remaining mergeable hybrid
prefix/control path:

| Batch | Split | Active suffix rows | Raw full FA ms | Coalesced no-op hybrid ms | Capacity gain | Kernel-path tax |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1008 | 64 | 2.328 | 2.437 | 1.6% | 4.7% |
| 512 | 960 | 64 | 2.325 | 2.331 | 6.7% | 0.3% |
| 512 | 1008 | 256 | 2.324 | 2.657 | 1.6% | 14.3% |
| 512 | 960 | 256 | 2.331 | 2.552 | 6.7% | 9.5% |

So the split960 no-op path can in principle beat the raw attention kernel when
only a small fraction of rows are suffix-active, but it loses once the active
suffix fraction rises. A stats-enabled split960 e2e run showed the real
scheduler repeatedly enters the latter regime: representative 5-second windows
had 288, 576, 608, 768, and 1024 suffix rows over 32 layer calls, plus occasional
large mixed-prefix windows such as `43040` prefix rows with `576` suffix rows and
`57760` prefix rows with `416` suffix rows. The instrumented run produced
`2086.899 out tok/s`, so it is diagnostic rather than a throughput baseline.

Updated policy implication: the planner cannot model hybrid as just
`capacity_gain(split) - cpu_suffix_cost(split)`. It also needs an active-suffix
mix term: the fraction and clustering of rows that have crossed the split while
other rows remain prefix-only. Hybrid wins are plausible only when the saved GPU
KV capacity is larger than the mergeable-prefix wrapper tax for the expected
active-suffix mix. A useful next implementation attempt should reduce that tax
for general mixed batches, or change scheduling so suffix-active rows are less
interleaved with large prefix-only work.

## 2026-05-30 Follow-up: Merge Artifacts and Scheduler-Budget Probe

A focused merge benchmark was extended to parameterize the Llama shape and to
compare direct UVA reads against explicit H2D artifact copy plus GPU-resident
merge. For `Hq=32`, `D=128`, BF16 suffix output plus FP32 LSE:

| Suffix rows | Artifact MB | UVA output+LSE merge ms | Copy artifacts + GPU merge ms | Delta |
|---:|---:|---:|---:|---:|
| 256 | 2.130 | 0.105 | 0.103 | -0.002 |
| 512 | 4.260 | 0.205 | 0.193 | -0.011 |
| 1024 | 8.520 | 0.406 | 0.374 | -0.032 |

GPU-resident merge itself is nearly free (`0.003-0.007 ms` across these rows),
but explicit copy still has to move the same artifact bytes over PCIe. The copy
path is only marginally faster than UVA and does not explain or close the e2e
gap. This supports leaving the deprecated explicit-copy runtime path deleted.

A small implementation cleanup was made in `cots_hybrid_decode_attention`: when
the coalesced-prefix path already provides precomputed prefix output/LSE, the
suffix merge path now skips unused GPU prefix metadata (`gpu_block_table`
`index_select`, `torch.arange` query lengths, and `torch.full` prefix lengths).
Focused tests still pass:

```text
/opt/conda/envs/thesis/bin/python -m pytest \
  /TTC/vllm/tests/kernels/attention/test_cots_hybrid_attention.py \
  /TTC/vllm/tests/v1/worker/test_cots_hybrid_kv.py -q
# 27 passed, 2 warnings
```

The cleanup is correct but not a throughput unlock. The wrapper microbench moved
only slightly for the important split960 Llama shape:

| Split | Active suffix rows | Before coalesced no-op ms | After coalesced no-op ms |
|---:|---:|---:|---:|
| 960 | 64 | 2.331 | 2.322 |
| 960 | 256 | 2.552 | 2.548 |

Finally, a coarse scheduler-budget probe tested whether limiting chunked-prefill
work reduces the large mixed-prefix windows seen in the stats run. The knob helps
only a little and is not a general win:

| max_num_batched_tokens | GPU-only out tok/s | Hybrid split960 out tok/s | Hybrid/GPU |
|---:|---:|---:|---:|
| default 8192 | 2617.170 | 2455.014 | 0.938x |
| 2048 | 2623.088 | 2475.680 | 0.944x |
| 1024 | 2600.561 | 2430.475 | 0.935x |

The 2048-token budget slightly improves hybrid, but 1024 slows it more than it
helps. Coarse global prefill throttling is therefore too blunt. The policy lever
that remains plausible is suffix-active-aware scheduling: avoid interleaving
large prefix-only prefill chunks with suffix-active rows when the expected
active-suffix mix makes the mergeable-prefix tax larger than the capacity gain.

## 2026-05-30 Follow-up: Planner Policy Hook Removed

The profile-gated planner hook described above has been removed from
`FastTTS-thesis/planner.py`. The current evidence shows real positive cells, but
it does not yet define a general planner policy: Qwen and Llama react differently
to memory pressure, graph mode changes the effective capacity curve, and Phase 3
will also need to account for interactions with weight offload.

Current planner behavior is intentionally manual again:

- `planner_config.generator.kv` / `planner_config.verifier.kv` can still force a
  diagnostic hybrid KV split and CPU pool.
- `phase2_kv_profile` / `kv_profile` blocks are ignored for now.
- The JSON cell artifact is retained as measurement data, not as launch-time
  policy input.

This keeps the codebase honest before the next round of policy work: throughput
measurements can guide experiments, but the planner will not automatically choose
hybrid until the combined KV/weight-offload policy is modeled and validated.
