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
- Qwen2.5-7B BF16 attention shape: 28 query heads, 4 KV heads, head size 128.
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

Deliberately not in Phase 2: CPU-aware admission, planner split selection, dynamic promotion/eviction, GPU tail fallback, native vLLM KV-offload integration, CPU-produced QKV direct handoff from Phase 1, model-general CPU suffix kernels, and graph-captured suffix-prefill overflow buckets.

## Optimizations Landed In This Pass

- Vectorized CPU KV scatter replaced per-request Python `copy_` loops.
- CPU value staging was fixed to use pinned memory. The earlier `empty_like`
  allocation silently produced non-pinned memory and disabled async K/V staging.
- Q/K/V staging now uses one shared D2H stream/event.
- Qwen-style Q/K/V tensors are staged through one combined contiguous
  `[B, 36, 128]` D2H copy when their storage layout allows it.
- Suffix slot masking has a CPU-position fast path for all-prefix/all-suffix
  decode steps.
- All-active CPU suffix K/V scatter now uses a thesis-owned C++ memcpy fast
  path instead of PyTorch advanced indexing.
- CPU suffix attention now pre-converts each task's 7 BF16 query heads to FP32
  once and reuses them across suffix tokens, avoiding repeated BF16 upcasts in
  the dot-product loop.
- CPU suffix attention now vectorizes the final FP32-to-BF16 output store with
  AVX2 integer operations while preserving round-to-nearest-even conversion.
- CPU suffix attention now processes QK logits two suffix tokens at a time
  within each cache block, reusing the 7 FP32 query-head vectors across both
  tokens. This is a small parity-safe form of the same tiling idea used in
  NEO's ISPC CPU attention path, adapted to the COTS BF16/AVX2 kernel.
- Prepared native suffix attention tasks can now scatter current-token K/V
  directly from the staged `[B, 36, 128]` QKV artifact before running CPU suffix
  attention. This removes the Python-side `_finish_staged_kv_cache_update` from
  the native-prepared path and is the first graph-substrate step for suffix KV
  update plus CPU attention in one stream-ordered host callback.
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
this pass. It already accepts a batch of independent `[28, 128]` query rows, so
prefill rows are flattened into that batch.

Graph note: decode graph buckets remain on the existing fixed task IDs. If a
row-expanded prefill batch exceeds the decode batch staging capacity, it uses a
per-layer overflow native task and synchronous D2H staging. This keeps graph
decode stable while making CPU suffix prefill functionally correct first.

Still unsupported by design in this pass:

- CUDA graph capture/replay for row-expanded suffix prefill overflow tasks.
- Any model shape other than the existing Qwen2.5-7B BF16 suffix kernel shape.

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
- `CotsPreparedNativeSuffixAttentionRunner.run_qwen_bf16_suffix_attention()` can
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
