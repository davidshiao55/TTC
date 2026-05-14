# Phase 1 Analysis: Free Regime And KV-Throughput

Date: 2026-05-14

Status: free-regime sweep, COTS-vs-native prefetch comparison, COTS
prefetch-gap decomposition, push-free-zone follow-up probes, and adjusted
KV-throughput focused probe are complete on current Phase 1c code. The old
full KV-throughput grid is no longer recommended because the free-regime sweep
shows CPU-heavy variants are dominated and prefetch-only is not close to free.

## Purpose

This analysis answers two Phase 1 questions on the current Phase 1c production
COTS substrate:

1. **Pure-prefetch equivalence:** does COTS pure-prefetch match native vLLM
   prefetch at matched offloaded bytes, in both graph and eager mode?
2. **Prefetch-free gap:** why does pure prefetch fail the simple overlap math,
   and is the gap from exposed H2D copies or from COTS overhead?
3. **Free-zone expansion:** which knobs or implementation directions can move
   pure prefetch into the 5% latency band?
4. **Free regime:** for each COTS variant, how much weight can be offloaded
   while staying within 5% of no-offload latency at the same workload?
5. **KV-throughput crossover:** can the extra GPU KV-cache capacity unlocked by
   weight offload raise offline throughput above no-offload at
   `gpu_memory_utilization=0.75`?

The scope is vLLM-isolated, using `Qwen/Qwen2.5-7B-Instruct` BF16. FastTTS
end-to-end validation should use only the winning vLLM points after this pass.

## How To Run

Run from `/TTC/FastTTS-thesis` in the `thesis` conda environment so Python
resolves the editable vLLM install rather than `/TTC/vllm` as a namespace
package.

```bash
cd /TTC/FastTTS-thesis

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_vs_native_prefetch.py \
  --exp --smoke

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_prefetch_gap.py \
  --exp --keep-going --batch-sizes 64 --modes graph \
  --f-values 0.005 0.01 0.02 0.0357 --repeat 3

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_free_regime.py \
  --exp

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_kv_throughput.py \
  --exp
```

Smoke check before the full free-regime sweep:

```bash
/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_free_regime.py \
  --exp --smoke
```

Focused free-regime rerun, using the shape supported by the completed sweep:

```bash
/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_free_regime.py \
  --exp --focused-grid --batch-sizes 1 64
```

Adjusted KV-throughput focused probe:

```bash
/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_kv_throughput.py \
  --exp --keep-going --focused-grid \
  --workloads short medium long \
  --prefetch-f-values 0.02 0.05 0.09 \
  --collab-f-values 0.09 \
  --collab-ratios 0.9 \
  --repeat 1
```

The harnesses write `summary.json` and `summary.md` under
`/TTC/results/phase1_analysis/.../<timestamp>/`. The Markdown files contain the
tables that should be copied back into this findings document after the fresh
run is complete.

## Method

Pure-prefetch equivalence check:

- `vllm bench latency`
- `input_len=8`, `output_len=128`, `max_model_len=2048`
- `gpu_memory_utilization=0.75`
- modes: graph mode and diagnostic `--enforce-eager`
- COTS pure-prefetch: `f_cpu_store=f_prefetch=n_layers/28`
- native prefetch: `prefetch_defer`, `N=1`, `K in {1,2}`, matched layer counts
  `(1,2,4,7,14)` with group sizes `(28,14,7,4,2)`
- optional stock shipped-native arm: pass `--native-backends prefetch
  prefetch_defer`

Prefetch-gap decomposition:

- `vllm bench latency`
- `input_len=8`, `output_len=128`, `max_model_len=2048`, `batch_size=64`
- `gpu_memory_utilization=0.75`
- graph mode production path, with one eager-mode diagnostic at `f=0.02`
- arms: no offload, COTS pure-prefetch, COTS pure-prefetch with
  `--cots-dry-run`
- the original run used the old copy-disabled dry-run semantics: hooks, slot
  bookkeeping, events, waits, graph behavior, and split GPU compute remained
  active, while H2D prefetch copies were skipped
- after 2026-05-14, public `--cots-dry-run` means control-plane dry-run and
  additionally skips prefetched-slice GPU compute contribution
- theory uses only current Phase 1 COTS target tensors:
  `qkv_proj`, `gate_up_proj`, and `down_proj`

Push-free-zone follow-up probes:

- Same harness as prefetch-gap decomposition
- Higher latency batches: `B=128` and `B=256`
- Graph policy diagnostic: `--no-cots-auto-graph-split`
- QKV placement diagnostic: `--no-cots-kv-biased`
- Tiny fractions under the best observed setting: `f ∈ {0.001,0.0025,0.005}`

Free-regime sweep:

- `vllm bench latency`
- `input_len=8`, `output_len=128`, `max_model_len=2048`
- `batch_size ∈ {1,4,16,64}`
- `gpu_memory_utilization=0.75`
- 3 fresh-process repeats per cell
- Arms: no offload, COTS prefetch-only, COTS CPU-only, COTS 50/50 collaborative
- `f_cpu_store ∈ {0.005,0.01,0.02,0.0357,0.05,0.0714,0.09,0.15}`

KV-throughput sweep:

- `vllm bench throughput`
- random dataset, `--disable-detokenize`
- `num_prompts=512`, `max_num_seqs=256`, `max_num_batched_tokens=8192`
- `gpu_memory_utilization=0.75`
- workloads: `(8,128)`, `(32,512)`, `(32,1024)`
- Arms: no offload; prefetch-only at `f ∈ {0.02,0.05,0.09,0.15,0.30}`;
  collaborative at `f ∈ {0.05,0.09,0.15,0.30}` with
  `f_prefetch/f_cpu_store ∈ {0.75,0.90}`, plus the `f=0.30`, ratio `0.50`
  Phase 1b anchor.

Classification rules:

- Free latency cell: mean latency `<= 1.05x` no-offload and CV `<= 3%`.
- Throughput win: mean output-token throughput `>= 1.05x` no-offload and
  CV `<= 3%`.
- Throughput tie: within ±5% of no-offload, subject to the same CV gate.

## Result Tables

Fresh COTS-vs-native-prefetch source of truth:
`/TTC/results/phase1_analysis/cots_vs_native_prefetch/20260514T050345Z/summary.md`.

### COTS Pure-Prefetch vs Native Prefetch

This rerun compares COTS pure-prefetch
`f_cpu_store=f_prefetch=n_layers/28` against native `prefetch_defer` at matched
offloaded layer counts. Native `K in {1,2}` was swept; the table below reports
the faster native K for each row. The depth labels map to COTS fractions:
`01L=0.0357`, `02L=0.0714`, `04L=0.1429`, `07L=0.25`, `14L=0.50`.

| mode | B | 01L | 02L | 04L | 07L | 14L |
|---|---:|---:|---:|---:|---:|---:|
| graph | 1 | 1.022 | 0.963 | 0.963 | 0.939 | 0.911 |
| graph | 64 | 1.157 | 1.085 | 1.029 | 1.003 | 0.994 |
| eager | 1 | 0.965 | 0.947 | 0.959 | 0.946 | 0.945 |
| eager | 64 | 1.026 | 0.962 | 0.959 | 0.945 | 0.944 |

Values are `COTS latency / best native latency`; `<1` means COTS is faster.
The current-code answer is: COTS pure-prefetch matches native prefetch in the
sense that both paths are same-order and usually within about 5% once the
offload depth is not tiny. The main exception is graph-mode `B=64` at shallow
offload: native is faster by `15.7%` at `01L` and `8.5%` at `02L`. At deeper
offload, COTS is equal or faster, especially in eager mode.

Native `K=2` is consistently the better native setting as depth grows,
especially at `B=64`; it nearly closes the graph-mode high-batch gap at `07L`
and `14L`. However, neither native nor COTS pure-prefetch is close to
no-offload at these whole-layer fractions. Even the smallest `01L` graph-mode
cell slows no-offload by `1.50x` at `B=1` for COTS and `1.23x` at `B=64` for
best native. So this experiment is an equivalence check, not evidence for a
free prefetch regime.

Fresh prefetch-gap source of truth:
`/TTC/results/phase1_analysis/prefetch_gap/20260514T_graph_b64_decompose/summary.md`.

Eager diagnostic source:
`/TTC/results/phase1_analysis/prefetch_gap/20260514T_eager_b64_f002_decompose/summary.md`.

Corrected control-plane dry-run source:
`/TTC/results/phase1_analysis/prefetch_gap/20260514T_new_dryrun_diagnose_b64_b128/summary.md`.

### COTS Prefetch Free-Gap Decomposition

The simple overlap calculation is optimistic. For `B=64`, no-offload latency
is `2.3724 s`, or `0.662 ms` per generated token per layer. The current COTS
Phase 1 target tensors contain `440,401,920` bytes per layer. At an assumed
`28 GB/s` H2D bandwidth, copying the whole target slice would take
`15.73 ms/layer`, giving an ideal free fraction of `0.042`. That calculation
assumes the entire layer time is usable overlap and that COTS adds zero
non-copy overhead.

The dry-run experiment shows those assumptions do not hold:

| f_prefetch | real slowdown | dry slowdown | real extra ms/(tok*layer) | dry extra ms/(tok*layer) | exposed copy/wait ms/(tok*layer) | exposed/theory |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0050 | 1.140 | 1.109 | 0.093 | 0.072 | 0.020 | 0.258 |
| 0.0100 | 1.157 | 1.132 | 0.104 | 0.088 | 0.016 | 0.105 |
| 0.0200 | 1.173 | 1.125 | 0.114 | 0.082 | 0.032 | 0.101 |
| 0.0357 | 1.419 | 1.035 | 0.277 | 0.023 | 0.254 | 0.453 |

At the practically interesting low fractions, the gap is mostly not H2D tail.
For `f=0.02`, only `0.032 ms/(token*layer)` is exposed copy/wait cost, about
`10%` of the serial copy time. The larger term is the dry-run cost:
`0.082 ms/(token*layer)`, or a `12.5%` slowdown even when real H2D copies are
disabled. Eager mode shows the same pattern at `f=0.02`: real slowdown
`1.119`, dry-run slowdown `1.106`, and only `0.010 ms/(token*layer)` of
real-minus-dry exposed copy/wait. So the low-fraction gap is not just a CUDA
Graph artifact.

At `f=0.0357`, the bottleneck changes. Dry-run is almost free (`1.035x`), but
real-minus-dry exposes `0.254 ms/(token*layer)`, about `45%` of the serial copy
time. This is still better than no overlap, but one-layer-ahead prefetch does
not leave enough usable slack once fixed layer work, graph boundaries, events,
and wait placement are accounted for.

The likely implementation gap is therefore two-part:

- For tiny prefetch fractions (`f <= 0.02`), reduce the non-copy COTS path:
  hooks, stream/event traffic, slot bookkeeping, graph split side effects, and
  shape/layout changes. This is the dominant term and is theoretically
  removable.
- For one-layer-equivalent prefetch (`f ~= 0.0357`), reduce exposed copy tail:
  coalesce the per-layer H2D copies, start prefetch farther ahead when memory
  allows, or use a lighter persistent copy schedule with fewer waits.

The current implementation is not at the mathematical overlap bound. The math
does not prove pure prefetch is impossible; it proves that the available margin
is small. A realistic target is making `f=0.005-0.01` free first. Making
`f=0.02` free requires both a much cheaper dry path and slightly less exposed
copy tail. Reaching the ideal `f~=0.04` boundary would require deeper/better
prefetch scheduling, not just faster bookkeeping.

After correcting `--cots-dry-run` to a control-plane dry-run, the attribution
is cleaner. The public dry-run now preserves wrappers, bucket/slot logic, graph
dependencies, and hook structure, but skips active offloaded work from both
COTS paths: no CPU GEMM/UVA contribution, no prefetch H2D, and no prefetched
GPU-slice compute contribution. With that semantic, a focused one-repeat graph
probe gives:

| B | f_prefetch | real slowdown | control dry slowdown | real extra ms/(tok*layer) | control extra ms/(tok*layer) | active offloaded-work ms/(tok*layer) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.0010 | 1.113 | 1.094 | 0.074 | 0.062 | 0.013 |
| 64 | 0.0050 | 1.138 | 1.085 | 0.091 | 0.056 | 0.035 |
| 64 | 0.0100 | 1.162 | 1.078 | 0.107 | 0.051 | 0.056 |
| 64 | 0.0200 | 1.187 | 1.068 | 0.123 | 0.045 | 0.078 |
| 64 | 0.0357 | 1.421 | 0.990 | 0.277 | -0.007 | 0.283 |
| 128 | 0.0010 | 1.075 | 1.039 | 0.058 | 0.030 | 0.028 |
| 128 | 0.0050 | 1.079 | 1.038 | 0.061 | 0.029 | 0.032 |
| 128 | 0.0100 | 1.105 | 1.037 | 0.081 | 0.028 | 0.053 |
| 128 | 0.0200 | 1.124 | 1.036 | 0.095 | 0.028 | 0.068 |
| 128 | 0.0357 | 1.309 | 0.990 | 0.238 | -0.008 | 0.246 |

This corrected run changes the diagnosis:

- The old large dry-run gap was partly an artifact of the previous semantic,
  because that path still executed prefetched-slice GPU math. The corrected
  dry-run at `B=128` is only `3.6-3.9%` above no-offload for `f <= 0.02`.
- At `B=64`, the control-plane floor alone is still `6.8-9.4%`, so no nonzero
  prefetch fraction can pass a 5% free-regime gate at that batch without
  reducing fixed COTS control/graph overhead.
- At `B=128`, the control-plane floor is inside the 5% gate, but it leaves only
  about one percentage point of slack. Even `f=0.001` adds enough active
  offloaded-work cost to end at `1.075x`, outside the free zone.
- The active work is not purely exposed serial H2D. For low fractions the
  active increment is far below the serial-copy bound, so overlap is hiding
  much of the byte movement. The remaining cost is the whole active split path:
  extra small GEMMs, QKV scatter/combine, MLP split compute, event/wait timing,
  and any residual exposed copy tail.
- The `f=0.0357` rows show the second bottleneck: once the prefetched slice is
  whole-layer sized, active work dominates even though the control dry-run is
  effectively free within one-repeat noise.

So the free-zone gap is not theoretically impossible from PCIe math alone.
The current implementation is missing the mathematical overlap bound because
turning on any nonzero prefetch fraction moves the layer onto a heavier split
execution path, and the 5% latency budget is too small to absorb that fixed
plus active overhead. At `B=128`, the practical target is now concrete: shave
roughly `2.5-3%` latency from the active pure-prefetch path to make
`f=0.001-0.005` free; making `f=0.02` free requires a larger rewrite of the
split GPU path.

Push-free-zone sources:

- `/TTC/results/phase1_analysis/prefetch_gap/20260514T_push_freezone_b128/summary.md`
- `/TTC/results/phase1_analysis/prefetch_gap/20260514T_push_freezone_b128_no_auto_graph_split/summary.md`
- `/TTC/results/phase1_analysis/prefetch_gap/20260514T_push_freezone_b128_no_kv_biased/summary.md`
- `/TTC/results/phase1_analysis/prefetch_gap/20260514T_push_freezone_b256/summary.md`
- `/TTC/results/phase1_analysis/prefetch_gap/20260514T_push_freezone_b128_no_auto_tinyf/summary.md`

### Pushing The Prefetch Free Zone

The current COTS pure-prefetch real path pays for more than copies. The old
copy-disabled dry-run used for the probes below still ran the split GPU math
shape:

- QKV does permanent-slice `F.linear`, prefetched-slice `F.linear`, then
  `index_copy_` scatter back into canonical `[Q|K|V]` order.
- MLP does the permanent MLP block, then a separate prefetched MLP block.
  The prefetched gate/up slice is currently two `F.linear` calls because the
  slot layout stores fixed-max gate and up regions separately.
- Layer hooks still issue `wait_prefetch` / `start_prefetch`, and the streamer
  still records/waits events and updates slot metadata.

That explains why tiny fractions are not byte-linear: once any nonzero
prefetch slice exists, COTS moves onto the split path.

Observed push-free-zone probes:

| probe | f | slowdown | dry slowdown | note |
|---|---:|---:|---:|---|
| B=64 default graph | 0.005 | 1.140 | 1.109 | original decomposition |
| B=128 default graph | 0.005 | 1.082 | 1.052 | larger batch helps |
| B=128 no auto graph split | 0.005 | 1.068 | 1.059 | best `f=0.005` probe |
| B=128 no KV bias | 0.005 | 1.132 | 1.103 | worse; reject |
| B=256 default graph | 0.005 | 1.105 | 1.070 | too much GPU split overhead |
| B=128 no auto graph split | 0.001 | 1.067 | 1.072 | 3 repeats; still not free |

The best observed tuning knob is `B=128` plus `--no-cots-auto-graph-split`,
but even `f=0.001` remains outside the 5% gate. This is the practical current
floor: tuning alone cannot push the production COTS pure-prefetch path into a
meaningful free zone at high batch. Larger batch helps until about B=128, then
B=256 regresses, likely because the extra small GEMMs/scatter are now competing
with already-saturated GPU compute. Disabling KV-biased QKV placement is a
clear negative result.

Most promising implementation directions:

- Keep only one steady-state public dry-run semantic. This is now implemented:
  `--cots-dry-run` means "control-plane dry run": preserve COTS wrappers,
  bucket/slot logic, and graph dependencies, but skip active offloaded work
  from both paths. That means no CPU GEMM, no CPU-path data movement/UVA result
  contribution, no prefetch H2D, and no prefetched-slice GPU compute
  contribution. Output can be numerically invalid, but shapes remain valid.
  More detailed splits should be temporary benchmark-only diagnostics, not
  production API.
- Add a pure-prefetch GPU fast path that reduces split-math overhead. First
  target: MLP. Make the prefetched gate/up slot active-adjacent for the current
  bucket so the prefetched MLP1 can be one `[gate|up]` GEMM plus the normal
  fused activation instead of two separate `F.linear` calls and explicit
  `F.silu(pref_gate) * pref_up`.
- Add a QKV fast combine for contiguous cases. The current path always uses
  three-way scatter. For placement patterns that are prefix/suffix contiguous,
  concatenate or write directly into a preallocated output with narrower copies
  instead of generic `index_copy_`.
- Skip CPU-runner-only allocations and anchors for pure-prefetch configurations
  where every bucket has `n_cpu_compute == 0`. This probably helps memory/KV
  more than latency, but it removes needless capture state from the pure
  prefetch control path.
- Coalesce H2D copies after the dry-path floor is lower. At small `f`, exposed
  copy/wait is not the dominant term. At `f~=0.0357`, it is dominant, so a
  second-stage optimization should pack per-layer prefetch slices and/or start
  prefetch farther ahead when the extra slot memory is acceptable.
- Add the next temporary diagnostic before changing kernels: a benchmark-only
  split between "copy-only/no prefetched GPU math" and "prefetched GPU math/no
  H2D" for pure prefetch. The corrected public dry-run already separates
  control-plane cost from active offloaded work; the next split should separate
  byte movement from extra split GPU math inside that active term.

Recommended order: first add the active-work diagnostic split, then optimize
the MLP prefetch subpath, then optimize QKV combine/scatter, then revisit copy
coalescing/deeper prefetch. The corrected data says making `f=0.005` free at
`B=128` requires shaving roughly `2.5-3%` latency from the active path; making
`f=0.02` free requires a larger rewrite of the split GPU compute path.

Implementation note, 2026-05-14: after switching `--cots-dry-run` to the
control-plane semantic above, a tiny graph-mode smoke passed at
`B=1,input/output=8/16,f=0.01`: no-offload `0.2537 s`, real prefetch
`0.2731 s`, control dry-run `0.2559 s`. A CPU-only dry-run smoke at
`f_cpu_store=0.01,f_prefetch=0.0` also passed at `0.2565 s`. Source:
`/TTC/results/phase1_analysis/prefetch_gap/20260514T_dryrun_semantic_smoke/`.
The older decomposition tables above used the previous "copy-disabled but
split-GPU-compute still active" dry-run behavior and should be read with that
historical meaning.

Stepwise push-free-zone update, 2026-05-14:

- Added benchmark-only diagnostic flags:
  `--cots-diagnostic-skip-prefetch-h2d` and
  `--cots-diagnostic-skip-prefetch-compute`. These keep the public
  `--cots-dry-run` simple while splitting active pure-prefetch cost into
  copy/wait and prefetched-GPU-compute probes.
- Diagnostic smoke passed at
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_active_split_smoke/`.
- Pre-optimization B=128 attribution source:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_active_split_b128_default/summary.md`.
- MLP fast-path source:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_active_split_b128_mlp_fastpath/summary.md`.
- QKV fast-combine diagnostic source:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_active_split_b128_mlp_qkv_fastpath/summary.md`.
- Active-adjacent MLP source:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_active_split_b128_active_adjacent/summary.md`.
- Pure-prefetch control-cleanup source:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_active_split_b128_pure_prefetch_cleanup/summary.md`.
- Rejected col-prefetch-source coalescing probe:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_col_prefetch_src_f0p005/summary.md`.
- Rejected MLP channel-alignment probe:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_mlp_align8_f0p005/summary.md`.
- MLP prefetched-down `addmm_` probe/source:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_mlp_addmm_f0p005/summary.md`.
- Empty-COTS backend control:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_empty_cots_control/summary.md`.
- Deep gap-budget diagnostic:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_gap_budget_f0p005/summary.md`.
- MLP split-size/grid diagnostic:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_mlp_shape_grid/summary.md`.
- Isolated MLP snapping grid:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_mlp_only_snap_grid/summary.md`
  and preserved first-pass full grid
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_mlp_only_snap_grid/summary_full_grid.md`.
- Isolated QKV head-aligned grid:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_qkv_only_head_grid/summary.md`.
- Implemented 64-channel MLP snapping confirmation:
  `/TTC/results/phase1_analysis/prefetch_gap/20260514T_b128_snap64_impl_f0p005/summary.md`.

At `B=128`, the active split showed both copy and compute matter, but the first
real win was MLP compute. The kept MLP change stores the active prefetched
gate/up slices adjacent in the slot as `[gate_active|up_active]`, runs MLP1 as
one GEMM, then reuses the normal fused `SiluAndMul` activation. Layer-0
prefetch is no longer max-filled in post-init; it is lazily filled by the
pre-forward bucket check, so the fast path also works when
`f_cpu_store != f_prefetch`.

| version | f | real slowdown | dry slowdown | copy-control ms/(tok*layer) | compute-control ms/(tok*layer) | real-control ms/(tok*layer) |
|---|---:|---:|---:|---:|---:|---:|
| before optimization | 0.005 | 1.102 | 1.065 | 0.013 | 0.018 | 0.028 |
| MLP fast path kept | 0.005 | 1.083 | 1.063 | 0.009 | 0.004 | 0.015 |
| MLP + QKV diagnostic | 0.005 | 1.084 | 1.054 | 0.013 | 0.008 | 0.022 |
| active-adjacent MLP | 0.005 | 1.081 | 1.055 | 0.009 | 0.008 | 0.020 |
| + pure-prefetch cleanup | 0.005 | 1.072 | 1.046 | 0.012 | 0.011 | 0.020 |

The MLP fast path is a keeper: at `f=0.005`, compute-control dropped from
`0.018` to `0.004 ms/(tok*layer)` in the first uniform-slot implementation.
The active-adjacent version is the correct general form because it preserves
the one-GEMM MLP path even when storage and prefetch fractions differ. The QKV
fast-combine attempt was not kept: it reduced dry/control cost but increased
active cost enough that real latency stayed flat/slightly worse. A future QKV
optimization should avoid replacing one generic `index_copy_` with multiple
slice/cat kernels unless Nsight shows the new kernel mix is actually cheaper.

The pure-prefetch cleanup skips CPU-runner-only activation buffers, slab
anchors, wait-kernel setup, and runtime runner dispatch when every bucket has
`n_cpu_compute == 0`. It lowered the `B=128,f=0.005` slowdown from `1.081x`
to `1.072x` and the dry/control floor from `1.055x` to `1.046x`, but did not
reduce the active copy/math term. This was still outside the 5% gate, so a
deeper gap-budget run was needed before choosing another optimization target.

A first H2D coalescing probe tried adding a pinned active-adjacent CPU source
for col/gate-up handles, so uniform buckets could copy `[gate_active|up_active]`
with one H2D instead of two. It was not kept: at `B=128,f=0.005`, the
diagnostic copy-control term moved only from `0.012` to `0.009
ms/(tok*layer)`, while real latency did not improve (`1.094x` in the probe
versus `1.072x` in the kept cleanup run) and the approach adds extra pinned
CPU storage. The negative result points away from launch-count-only MLP copy
coalescing and toward QKV/scatter or deeper overlap diagnostics.

A channel-alignment probe snapped MLP col/row split sizes to multiples of 8 to
avoid awkward BF16 GEMM dimensions. It was not kept: `B=128,f=0.005` measured
`1.086x` real and `1.053x` dry, worse than the kept cleanup run. The issue is
therefore not just Tensor-Core-unfriendly `half - n_cpu` sizes.

The prefetched-down projection now uses in-place `addmm_` when the permanent
GPU output exists, replacing `pref_silu.matmul(dn_pref)` plus a separate
`out_gpu.add_(pref_out)`. This is a small active-path cleanup, not a large
free-zone mover: in the broadened B=128 run, `f=0.005` was `1.071x` real and
`1.046x` dry, with real-control `0.019 ms/(tok*layer)`. `f=0.001` and
`f=0.02` remained effectively unchanged (`1.066x` and `1.114x`). The main
reason to keep it is removing the prefetched-output temporary/add kernel
without a clear regression.

An empty-COTS control (`f_cpu_store=f_prefetch=0`) measured only `1.008x` at
`B=128`, so the dry-run floor is not mostly caused by selecting the COTS
backend or auto graph-split path. The remaining dry floor appears to come from
the MLP split/replacement path itself: smaller permanent GEMMs plus extra
operator structure even when active prefetch work is disabled.

Deep gap-budget diagnostic, `B=128,f=0.005`, graph mode, 3 fresh-process
repeats:

| arm | slowdown | extra ms/(tok*layer) | CV |
|---|---:|---:|---:|
| none | 1.000 | 0.000 | 0.12% |
| empty COTS (`f_cpu_store=0`) | 1.005 | 0.004 | 0.63% |
| dry, no prefetch control | 1.064 | 0.048 | 0.36% |
| dry, normal prefetch control | 1.062 | 0.047 | 0.19% |
| CPU-path dry (`f_prefetch=0`) | 1.063 | 0.048 | 0.33% |
| copy-only | 1.078 | 0.059 | 0.44% |
| compute-only | 1.070 | 0.053 | 0.36% |
| real | 1.086 | 0.065 | 0.31% |

Gap budget from that run:

- Empty COTS/backend/auto-graph cost is only `0.004 ms/(tok*layer)`
  (`~0.5%` latency).
- The storage-split permanent-GPU path with prefetch control disabled is
  `0.048 ms/(tok*layer)`, so the split/replacement floor beyond empty COTS is
  about `0.045 ms/(tok*layer)` (`~5.9%` latency).
- Normal dry-run is not slower than the no-prefetch-control dry arm within
  noise (`-0.002 ms/(tok*layer)` delta), so prefetch hooks/events/slot repair
  are not the remaining free-zone blocker at this `f`.
- CPU-path dry is also indistinguishable from the no-prefetch-control dry arm
  (`-0.001 ms/(tok*layer)` delta), so native runner dispatch setup is not the
  blocker either.
- Active offloaded work adds another `0.018 ms/(tok*layer)` (`~2.4%`
  latency). Copy-only adds `0.012`, compute-only adds `0.006`; together they
  explain the real-minus-dry gap.

Conclusion: for the near-free regime, the main remaining gap is not QKV,
prefetch control, CPU-runner control, or MLP copy launch count. It is the MLP
storage-split permanent path itself. Even if active prefetch work became free,
the measured dry floor would still be `1.062x`, outside the 5% target. The
next useful optimization must reduce the permanent MLP split/replacement
floor, likely by restoring more of vLLM's native MLP execution path or by
fusing permanent+prefetched MLP work rather than running a separate custom
split block.

Follow-up shape diagnostic: keeping MLP GPU Parameters at full native shapes
and zeroing the offloaded channels on GPU lowered the dry arm to `1.008x`
(`0.006 ms/(tok*layer)`). This sacrifices MLP memory savings, so it is only a
diagnostic, but it proves the custom COTS block itself is not the core dry
floor. The bad case is the awkward permanent MLP GEMM shape from the
`f=0.005` split (`~95` channels per MLP half offloaded).

A small split-size grid then tested planner-achievable MLP channel counts:

| f | approx MLP half channels offloaded | real slowdown | dry slowdown | note |
|---:|---:|---:|---:|---|
| 0.00338 | 64 | 1.057 | 1.022 | single repeat |
| 0.00500 | 95 | 1.096 | 1.071 | single repeat, awkward shape |
| 0.00676 | 128 | 1.042 | 1.014 | 3 repeats, CV real 0.27% |
| 0.01014 | 192 | 1.046 | 1.001 | 3 repeats, CV real 0.19% |

An isolated MLP-only confirmation then separated MLP shape from QKV shape.
The decisive cells used three fresh-process repeats at `B=128`:

| MLP half channels offloaded | f | real slowdown | dry slowdown | CV real | note |
|---:|---:|---:|---:|---:|---|
| 64 | 0.00338 | 1.032 | 0.992 | 0.31% | good |
| 96 | 0.00507 | 1.105 | 1.073 | 0.15% | bad shape cliff |
| 128 | 0.00676 | 1.033 | 0.998 | 0.29% | good |
| 192 | 0.01014 | 1.036 | 0.992 | 0.05% | good |

The preserved one-pass full MLP grid also tested `32`, `160`, and `256`
channels per half. `32`, `160`, and `256` were all much better than the
`96`-channel cliff, but `160` was close to the 5% gate in one repeat
(`1.047x`). The production rule is therefore not arbitrary fine-grain; it is
MLP channel counts snapped to multiples of `64` per half, with no hard `128`
minimum. `128` remains worth revisiting as a planner preference when memory
budget permits, because it saved twice the MLP memory of `64` with essentially
the same observed latency. This is a kernel-shape constraint, not a
memory-model constraint: the raw requested `~95` channel split is worse than
both the smaller `64` and the larger `128` splits.

An isolated QKV-only grid checked whether the existing head-aligned picker is
already efficient. For Qwen2.5-7B, one KV-head pair is `256` fused-QKV rows
(`K128+V128`, `f=256/4608=0.05556`). Three-repeat results:

| QKV rows offloaded | KV-head pairs | f | real slowdown | dry slowdown | CV real | note |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 1 | 0.05556 | 1.033 | 0.989 | 0.81% | good |
| 512 | 2 | 0.11111 | 1.036 | 0.988 | 0.83% | good |
| 768 | 3 | 0.16667 | 1.037 | 0.985 | 0.57% | good |
| 1024 | 4 | 0.22222 | 1.045 | 0.987 | 0.67% | still within 5% |

The head-aligned QKV grid has no dry-floor cliff. The current QKV snapping is
therefore fine-grained enough for this model: do not split below a head/KV
pair unless a future attention-aware planner has a specific reason. In the
ideal continuous math, more granularity weakly improves the planner optimum.
On real GPU kernels, finer-than-kernel-friendly granularity can make the
chosen point worse. A practical grid is "fine enough" once the memory step is
small relative to the planner's memory/KV-admission resolution and adjacent
grid points differ by less than benchmark noise. For this setup, `64` MLP
channels per half and one QKV KV-head pair are already small memory steps
while avoiding observed kernel-shape cliffs.

Implementation note: COTS now snaps MLP gate/up half channels and matched down
input channels to the nearest multiple of `64`, with halfway cases rounded up
and no hard `128` minimum. A focused confirmation reran the formerly bad
`f=0.005` pure-prefetch case at `B=128`; because QKV still snaps to zero at
this tiny fraction, the effective placement is the `64`-channel MLP point.
Three fresh-process repeats measured `1.036x` real and `1.003x` dry, bringing
the old `~95`-channel bad-shape case back inside the 5% free-regime gate.

Fresh free-regime source of truth:
`/TTC/results/phase1_analysis/free_regime/20260513T211352Z_full/summary.md`.

### Free-Regime Maximum `f_cpu_store`

| strategy | B=1 | B=4 | B=16 | B=64 |
|---|---:|---:|---:|---:|
| `cots_collab_50` | — | — | — | — |
| `cots_cpu_only` | — | — | — | — |
| `cots_prefetch_only` | 0.0050 | — | — | — |

Baseline no-offload latencies were stable: `2.0093 s` at `B=1`, `2.0524 s`
at `B=4`, `2.0855 s` at `B=16`, and `2.3946 s` at `B=64`. Only
prefetch-only `f_cpu_store=0.005` at `B=1` met the strict 5% free-regime gate.
The same arm already loses at larger batches, with slowdown `1.126x` at
`B=64`.

### Best Observed Slowdown By `f_cpu_store`

| B | f=0.005 | f=0.010 | f=0.020 | f=0.0357 | f=0.050 | f=0.0714 | f=0.090 | f=0.150 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.047 | 1.064 | 1.062 | 1.134 | 1.210 | 1.538 | 1.923 | 3.006 |
| 4 | 1.323 | 1.317 | 1.324 | 1.173 | 1.424 | 1.745 | 2.194 | 3.565 |
| 16 | 1.342 | 1.350 | 1.383 | 1.471 | 1.926 | 2.592 | 3.300 | 5.261 |
| 64 | 1.126 | 1.152 | 1.167 | 1.420 | 1.795 | 2.362 | 2.982 | 4.627 |

At `B=64`, the best arm for every `f_cpu_store` is prefetch-only. CPU-only
and 50/50 collaborative dispatch are dominated for throughput planning, because
CPU matmul cost scales badly with batch.

### Shape Check For Future Free-Regime Reruns

The completed sweep supports the reduced-grid intuition:

- CPU-only is closest to free at `B=1` for every tested `f_cpu_store`.
  Even at `f=0.005`, it slows down by `1.067x` at `B=1`, versus `1.328x`
  at `B=4`, `1.360x` at `B=16`, and `1.224x` at `B=64`.
- Prefetch-only improves with batch once the fraction is large enough for
  communication cost to dominate fixed overhead. For `f >= 0.0357`, the best
  prefetch-only batch is `B=64`.
- At tiny fractions (`f <= 0.02`), prefetch-only is still best at `B=1`, which
  likely reflects fixed COTS bookkeeping/buffer overhead rather than useful KV
  capacity.
- The 50/50 collaborative arm behaves like the CPU path, not the prefetch path:
  its best batch is `B=1` for every tested fraction.

So future free-regime reruns can be much cheaper: run CPU-only and 50/50
collaborative at low batch, and run prefetch-only at high batch, with only
`none` baselines at the selected batch sizes.

### KV-Throughput Implication

A throughput win needs:

```text
effective_KV_concurrency_gain > 1.05 * offload_decode_slowdown
```

The latency sweep's `B=64` prefetch-only break-even requirements are therefore:

| f_cpu_store | required concurrency gain for >5% throughput win |
|---:|---:|
| 0.005 | 1.183x |
| 0.010 | 1.210x |
| 0.020 | 1.225x |
| 0.0357 | 1.491x |
| 0.050 | 1.884x |
| 0.0714 | 2.480x |
| 0.090 | 3.131x |
| 0.150 | 4.859x |

Representative `max_model_len=2048` free-regime logs show much smaller KV-token
gains. At `B=64`, no-offload had `43,616` GPU KV tokens. Prefetch-only
increased that to `53,024` at `f=0.05` (`1.216x`) and `60,768` at `f=0.09`
(`1.393x`). Those capacity gains are far below the measured break-even
requirements, so a throughput win is unlikely unless the offline benchmark hits
a hard admission cliff where no-offload leaves the GPU under-filled and offload
crosses the cliff.

This means the KV-throughput sweep should be narrowed before spending full run
time:

- Keep no-offload and prefetch-only. Use CPU-only only as a negative control if
  needed.
- Treat collaborative ratios as diagnostics, preferably prefetch-heavy
  (`0.75` or `0.90`) rather than the dominated 50/50 split.
- Run a cheap KV-capacity probe first for the throughput configuration, because
  throughput uses different `max_model_len`, `max_num_seqs`, and graph-capture
  settings than the latency sweep.
- Prioritize near-boundary long workloads. Short `(8,128)` is mainly a negative
  control; the capacity increase is too small to overcome the measured decode
  penalty unless scheduling effects are unexpectedly nonlinear.

### Adjusted KV-Throughput Focused Probe

Fresh adjusted source of truth:
`/TTC/results/phase1_analysis/kv_throughput/20260514T_adjusted_focused_probe/summary.md`.

The adjusted probe used `vllm bench throughput`, random fixed lengths,
`num_prompts=512`, `max_num_seqs=256`, `max_num_batched_tokens=8192`,
`gpu_memory_utilization=0.75`, and `max_model_len=input_len+output_len+1`.
The `+1` is required because vLLM throughput asserts that `max_model_len` is
strictly greater than request length. CPU-only was excluded. The focused grid
kept prefetch-only plus one long-workload prefetch-heavy collaborative
diagnostic.

| workload | arm | output tok/s | throughput gain | KV tokens | KV gain | verdict |
|---|---|---:|---:|---:|---:|---|
| short `(8,128)` | none | 8011.65 | 1.000 | 43,616 | 1.000 | baseline |
| short `(8,128)` | prefetch `f=0.02` | 7045.38 | 0.879 | 46,672 | 1.070 | lose |
| medium `(32,512)` | none | 4186.85 | 1.000 | 43,616 | 1.000 | baseline |
| medium `(32,512)` | prefetch `f=0.02` | 3819.35 | 0.912 | 46,672 | 1.070 | lose |
| medium `(32,512)` | prefetch `f=0.05` | — | — | — | — | failed startup/shutdown |
| medium `(32,512)` | prefetch `f=0.09` | 2576.79 | 0.615 | 60,768 | 1.393 | lose |
| long `(32,1024)` | none | 2819.46 | 1.000 | 43,616 | 1.000 | baseline |
| long `(32,1024)` | prefetch `f=0.02` | 2614.33 | 0.927 | 46,672 | 1.070 | lose |
| long `(32,1024)` | prefetch `f=0.05` | 2107.54 | 0.747 | 53,024 | 1.216 | lose |
| long `(32,1024)` | prefetch `f=0.09` | 1545.15 | 0.548 | 60,768 | 1.393 | lose |
| long `(32,1024)` | collab `f=0.09,r=0.90` | 844.25 | 0.299 | 62,640 | 1.436 | lose |

No cell reached the tie band, let alone the `>5%` win threshold. The best
offload point was long prefetch-only `f=0.02`, but it still lost `7.3%`
throughput while increasing KV capacity only `7.0%`. Increasing the fraction
does increase KV capacity, but throughput falls faster than capacity rises:
long `f=0.05` gained `21.6%` KV tokens while losing `25.3%` throughput, and
long `f=0.09` gained `39.3%` KV tokens while losing `45.2%` throughput.

The short workload is confirmed as a negative control: no-offload already has
`302.89x` max concurrency for 136-token requests, above `max_num_seqs=256`, so
extra KV capacity cannot improve admission. Medium and long are KV-pressure
workloads, but the added capacity is still too small relative to the observed
offload cost. The prefetch-heavy collaborative diagnostic is much worse than
pure prefetch, so collaborative variants should remain out of the throughput
search unless a later implementation sharply reduces CPU-side decode cost.

Because every completed offload cell was below `0.95x` no-offload throughput,
no repeat-3 confirmation was run. Future repeat budget should go only to a new
candidate that first reaches the tie band in a one-repeat probe.

### Lower-Memory KV-Throughput Probe

Source:
`/TTC/results/phase1_analysis/kv_throughput/20260514T_gpu068_focused_probe/summary.md`.

This follow-up repeated the focused prefetch-only probe with
`gpu_memory_utilization=0.68` to test whether the `0.75` run was already near a
throughput-saturated point. The lower memory budget sharply reduces no-offload
KV capacity from `43,616` tokens at `0.75` to `12,624` tokens at `0.68`, so this
is a more favorable setting for the "offload weights to buy KV concurrency"
hypothesis.

| workload | arm | output tok/s | throughput gain | KV tokens | KV gain | verdict |
|---|---|---:|---:|---:|---:|---|
| short `(8,128)` | none | 4779.11 | 1.000 | 12,624 | 1.000 | baseline |
| short `(8,128)` | prefetch `f=0.02` | 4450.24 | 0.931 | 15,680 | 1.242 | lose |
| medium `(32,512)` | none | 1970.01 | 1.000 | 12,624 | 1.000 | baseline |
| medium `(32,512)` | prefetch `f=0.02` | 2007.60 | 1.019 | 15,680 | 1.242 | tie |
| medium `(32,512)` | prefetch `f=0.05` | 1820.80 | 0.924 | 22,048 | 1.747 | lose |
| medium `(32,512)` | prefetch `f=0.09` | 1482.37 | 0.752 | 29,776 | 2.359 | lose |
| long `(32,1024)` | none | 1153.11 | 1.000 | 12,624 | 1.000 | baseline |
| long `(32,1024)` | prefetch `f=0.02` | 1181.37 | 1.025 | 15,680 | 1.242 | tie |
| long `(32,1024)` | prefetch `f=0.05` | 1057.30 | 0.917 | 22,048 | 1.747 | lose |

The lower-memory probe shows the expected direction: once no-offload is truly
KV-starved, tiny prefetch can recover enough concurrency to tie or slightly
beat no-offload. It still does not reach the `>5%` win threshold. Larger
fractions continue to lose despite much larger KV pools. Long `f=0.09` was
terminated early after startup showed `29,776` KV tokens (`2.359x` no-offload)
but the live output rate was only around `650-775` tok/s, far below the
`1211` tok/s needed for a `1.05x` win over no-offload.

This suggests the KV-throughput opportunity is not mathematically impossible:
under tighter memory, small prefetch can become throughput-neutral or mildly
positive. But the practical winning window is narrow. The offload fraction must
be large enough to lift admission out of a KV cliff and small enough that the
per-token weight-movement penalty remains nearly hidden.

## Smoke Status

The tiny current-code smoke was run on 2026-05-13.

- Baseline no-offload passed: `B=1`, `input_len=8`, `output_len=16`,
  mean latency `0.2633-0.2644 s` across the smoke invocations.
- COTS CPU-only passed in production graph mode at `f_cpu_store=0.01`,
  `f_prefetch=0.0`: mean latency `0.2909 s`.
- COTS prefetch-only at `f_cpu_store=f_prefetch=0.01` initially failed during
  engine startup in production graph mode with
  `slot owner mismatch on mlp.gate_up_proj slot 1` during vLLM profile-run.
  The root cause was trace-time only: Dynamo runs the fake
  `start_prefetch`/`wait_prefetch` implementations, so Python slot-owner
  metadata is stale while tracing even though the real prefetch custom ops run
  when the compiled/captured graph executes.
- The fix keeps eager/runtime slot validation intact but skips that metadata
  assertion while `torch.compiler.is_compiling()`. Verified on 2026-05-13:
  the prefetch-only graph-mode harness cell now passes at mean latency
  `0.2857 s` for `B=1`, `input_len=8`, `output_len=16`, one measured
  iteration.
- The same COTS prefetch-only smoke passed with diagnostic `--enforce-eager`
  at mean latency `0.3263 s`, confirming the tensor split itself was not the
  blocker.

The graph-mode prefetch startup blocker is resolved; remaining prefetch-only
or collaborative sweep failures should be treated as fresh current-code
signals, not inherited from this trace-time assertion issue.

The COTS-vs-native-prefetch smoke was run on 2026-05-14 after clearing an
orphaned `VLLM::EngineCore` process from the interrupted throughput run. Smoke
scope: `01L`, `B=1`, graph and eager mode, one measured iteration.

| mode | none | COTS pure-prefetch | native `prefetch_defer` K=1 | COTS/native |
|---|---:|---:|---:|---:|
| graph | 2.0207 s | 3.0494 s | 2.9601 s | 1.030 |
| eager | 2.0746 s | 2.6633 s | 2.8048 s | 0.950 |

This smoke validates the harness and both mode flags. It is too small for a
final current-code claim; the full rerun should use the generated
`summary.md` as the source of truth.

## Interpretation Frame

The expected mechanism is simple and falsifiable:

- Weight offload helps throughput only if the additional KV-cache capacity
  increases effective concurrency enough to exceed the offload path cost.
- Prefetch-only cost is mostly byte-linear and token-count-insensitive.
- CPU-compute cost is token-count-sensitive and should be excluded from the
  throughput search unless the fresh free-regime sweep shows it is competitive.
- Collaborative dispatch is worth keeping in the search because Phase 1b showed
  the best split moves toward prefetch-heavy as batch grows.

Do not import the older Phase 1a Python-runner free-regime numbers as final
evidence here. Phase 1c changed the runtime substrate, graph behavior, and CPU
worker path enough that the current-code measurements are the only valid
answers for this analysis.
