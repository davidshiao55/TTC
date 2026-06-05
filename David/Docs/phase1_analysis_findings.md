# Phase 1 Analysis Appendix: Free Regime And KV Throughput

Date: 2026-05-15 cleanup

Status: final experiment appendix. The production-path narrative lives in
`phase1_findings.md`; this file keeps the current result tables and commands.

## Scope

All measurements here use the current Phase 1c production COTS path on
`Qwen/Qwen2.5-7B-Instruct`, BF16, vLLM-isolated. Historical Phase 1a/1b
numbers were used only to choose probes; final claims come from the fresh
Phase 1c runs listed below.

Classification:

- latency free: mean latency `<= 1.05x` no-offload and CV `<= 3%`;
- throughput win: output-token throughput `>= 1.05x` no-offload;
- tie: within ±5%.

## Commands

Run from `/TTC/FastTTS-thesis` in the `thesis` environment:

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
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_cpu_gap.py \
  --exp --batch-sizes 1 --modes graph \
  --thread-counts 4 8 16 \
  --f-values 0.0068 0.0135 0.02 0.0357 0.05

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_free_regime.py \
  --exp

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_phase_free_regime.py \
  --exp --keep-going \
  --workloads decode8x128:8:128 prefill128x1:128:1 mixed128x128:128:128 \
  --batch-sizes 1 16 64 \
  --f-values 0.005 0.01 0.02 0.0357 0.05 \
  --only-strategies none cots_cpu_all cots_prefetch_all \
    cots_decode_cpu_prefill_prefetch \
  --repeat 3

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_phase_free_regime.py \
  --oracle --exp --keep-going \
  --oracle-decode-buckets 1 16 64 \
  --oracle-prefill-buckets 128 512 2048 \
  --f-values 0.005 0.01 0.02 0.0357 0.05 \
  --oracle-split-ratios 0 0.25 0.5 0.75 1 \
  --oracle-validate-e2e \
  --workloads decode8x128:8:128 prefill128x1:128:1 mixed128x128:128:128 \
  --batch-sizes 1 16 64 \
  --repeat 3

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_kv_throughput.py \
  --exp --focused-grid --only-arms none cots_prefetch_only --repeat 1

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_wo_offload_e2e.py \
  --exp --depths 01L 07L 14L --batches 1 16 64

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_wo_offload_e2e.py \
  --exp --batches 1 16 --input-len 8 --output-len 128 \
  --f-cpu-store 0.01 --f-prefetch 0.0 --num-iters-warmup 2 --num-iters 3

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase0/bench_wo_offload_tradeoff.py \
  --model qwen7b --threads 24 --num-tokens 1 16 64 \
  --slice-fracs 0.02 0.0357 0.05 0.10 \
  --output-json /TTC/results/phase1_analysis/wo_cpu_compute/20260531_qwen7b_t24_smallf.json
```

For the low-memory KV crossover probe, add:

```bash
--gpu-memory-utilization 0.68
```

## COTS Pure-Prefetch vs Native Prefetch

Source:
`/TTC/results/phase1_analysis/cots_vs_native_prefetch/20260514T050345Z/summary.md`

Values are `COTS latency / best native latency`; `<1` means COTS is faster.

| mode | B | 01L | 02L | 04L | 07L | 14L |
|---|---:|---:|---:|---:|---:|---:|
| graph | 1 | 1.022 | 0.963 | 0.963 | 0.939 | 0.911 |
| graph | 64 | 1.157 | 1.085 | 1.029 | 1.003 | 0.994 |
| eager | 1 | 0.965 | 0.947 | 0.959 | 0.946 | 0.945 |
| eager | 64 | 1.026 | 0.962 | 0.959 | 0.945 | 0.944 |

Conclusion: COTS pure-prefetch and native prefetch are same-order at matched
bytes. COTS is equal or faster in most eager cells and deeper graph cells. The
remaining gap is graph-mode shallow offload at high batch.

## WO Inclusion E2E Check

Source:
`/TTC/results/phase1_analysis/wo_offload_e2e/20260531_graph_qwen7b_decode/summary.md`

This isolates the policy question "should WO participate in weight offload for
simplicity?" by running the same graph-mode `prefetch_defer` E2E workload with
the same offloaded layer placement, changing only whether `self_attn.o_proj` is
included in `--offload-params`. Workload: Qwen2.5-7B, BF16,
`input_len=8`, `output_len=128`.

| depth | B | no WO s | with WO s | delta | extra WO GiB |
|---|---:|---:|---:|---:|---:|
| `01L` | 1 | 2.8111 | 2.9473 | +4.84% | 0.024 |
| `01L` | 16 | 2.7063 | 2.8441 | +5.09% | 0.024 |
| `01L` | 64 | 2.7824 | 2.9212 | +4.99% | 0.024 |
| `07L` | 1 | 18.2534 | 19.2084 | +5.23% | 0.167 |

Conclusion: adding WO is not negligible even in the prefetch-only E2E path. The
latency tax is consistently around 5% while the memory saved is tiny at 7B.
Keep `o_proj` GPU-resident in Phase 1/2; do not simplify the Planner/runtime
strategy by making WO part of the uniform WQKV/MLP weight offload set.

The COTS CPU-compute/communication path is the more important rejection test
because it includes the activation round trip after attention merge. This
one-quantum WO snap data is the failure mode that motivated the later
production two-quantum WO snap.

Current post-cleanup sources:
`/TTC/results/phase1_analysis/cots_wo_offload_e2e/20260601_current_graph_qwen7b_f001_b1_b16/summary.md`
and
`/TTC/results/phase1_analysis/cots_wo_offload_e2e/20260601_current_graph_qwen7b_f005_b1_b16/summary.md`

| COTS f | B | no WO s | with WO s | delta | WO rows | extra WO CPU GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0% | 1 | 2.1432 | 2.1357 | -0.35% | 0 | 0.000 |
| 1.0% | 16 | 2.2831 | 2.2850 | +0.08% | 0 | 0.000 |
| 5.0% | 1 | 2.2681 | 2.4322 | +7.24% | 128 | 0.024 |
| 5.0% | 16 | 3.6944 | 4.9228 | +33.25% | 128 | 0.024 |

Regression check against the earlier no-WO COTS metrics passed: current no-WO
at `f=1%` is +0.47% for B=1 and -0.34% for B=16 versus the
2026-05-31 recorded cells. Current no-WO at `f=5%`, B=1 is +0.23% versus the
recorded smoke.

Historical source (pre-2026-06-01 head-aligned WO snap cleanup):
`/TTC/results/phase1_analysis/cots_wo_offload_e2e/20260531_graph_qwen7b_b1_f001/summary.md`
and
`/TTC/results/phase1_analysis/cots_wo_offload_e2e/20260531_graph_qwen7b_b1_f005_smoke/summary.md`

| COTS f | B | no WO s | with WO s | delta | extra WO CPU GiB |
|---:|---:|---:|---:|---:|---:|
| 1.0% | 1 | 2.1331 | 2.2481 | +5.39% | 0.007 |
| 1.0% | 16 | 2.2909 | 2.4236 | +5.79% | 0.007 |
| 5.0% | 1 | 2.2628 | 2.4358 | +7.64% | 0.033 |

Conclusion at the time: the real COTS WO path was not a harmless
simplification with a one-quantum WO snap. The later production snap ablation
kept the transparent all-module story but raised WO to a two-QKVO-quantum dense
output snap. That suppresses WO in the low-fraction cells that exposed the
fixed sync/activation-return cost, while still allowing WO to contribute at
larger `f_cpu_store`.

2026-06-05 production update: default COTS weight placement is now
`qkv,mlp,wo`, with one uniform dispatch table and fixed floor snapping. WO uses
`WO_QKVO_GRANULARITY_MULTIPLIER=2`; this removed the low-fraction WO cliff
observed here, and the remaining high-fraction all-module gap was small enough
for the simpler thesis story (`+5.7%` around `f=0.15`, `+1.7%` around
`f=0.18` versus QKV+MLP at similar bytes).

Hybrid-KV compatibility was also checked after WO landed. The combined
`qkv,mlp,wo` + hybrid KV eager path initialized and generated successfully
(`wo_ops=28`), and a short graph-mode smoke captured/replayed with the expected
piecewise graph + wait-kernel policy. A forced-context parity probe had no
forced-output failures, but WO added extra numeric drift versus the no-WO
hybrid control (`30/32` top-1 positions matched with WO, versus `32/32`
without WO). This confirms WO is mechanically compatible with hybrid KV;
production still relies on regular correctness tests for the combined path.
Sources:
`/TTC/results/phase2/wo_hybrid_compat_20260601_summary.json` and
`/TTC/results/phase2/no_wo_hybrid_compat_20260601_summary.json`.

The older post-merge primitive remains useful as a lower-level communication
diagnostic with explicit thread control:

Source:
`/TTC/results/phase1_analysis/wo_cpu_compute/20260531_qwen7b_t24_smallf.json`

| WO f | max delta / layer | max delta / decode step | WO memory saved |
|---:|---:|---:|---:|
| 2.0% | +0.123 ms | +3.44 ms | 14 MB |
| 3.57% | +0.232 ms | +6.49 ms | 26 MB |
| 5.0% | +0.307 ms | +8.59 ms | 36 MB |
| 10.0% | +0.569 ms | +15.93 ms | 72 MB |

Conclusion: WO CPU compute adds measurable decode-step latency when tiny WO
slices are allowed. The production fix is not a module-specific planner policy;
it is a coarser WO snap inside the uniform all-module offload rule.

## Prefetch Free-Zone Diagnosis

The simple overlap bound is optimistic because any nonzero prefetch fraction
moves the layer onto a split execution path. After correcting dry-run semantics,
the dominant low-fraction cost was not serial H2D; it was active split-path
work plus fixed control/graph overhead.

Important production fixes from this investigation:

- public `--cots-dry-run` is now a control-plane dry-run: no CPU GEMM, no H2D
  prefetch, and no prefetched-slice GPU compute contribution;
- MLP gate/up/down counts snap to 64-channel multiples;
- pure-prefetch skips CPU-runner-only allocations and wait setup;
- layer-0 fill is lazy and active-bucket aware;
- MLP prefetched gate/up slots are active-adjacent.

Focused post-fix confirmation:

| cell | result |
|---|---:|
| B=128, pure prefetch `f=0.005`, real | 1.036x |
| B=128, pure prefetch `f=0.005`, dry | 1.003x |

This demonstrates the implementation can make tiny high-batch prefetch free
when the bad MLP shape is avoided. It is not yet evidence for a broad
prefetch-free regime.

## Free-Regime Sweep

2026-06-04 Planner caveat: the table below is a decode-heavy E2E sweep, not a
phase-aware free-zone result. It used `input_len=8`, `output_len=128`, so the
prefill bucket contributed little to whole-request latency. Do not use it to
claim that one COTS route is free for every inference phase. The redo harness
is `David/Benchmarks/phase1_analysis/bench_cots_phase_free_regime.py`: fixed-arm
mode measures prefill-heavy, decode-heavy, and mixed shapes separately, while
`--oracle` mode sweeps candidate bucket rows, emits
`oracle_dispatch_tables.json`, and can validate the composed empirical-oracle
table E2E. Treat these oracle tables as Planner targets, not final Planner
output.

Source:
`/TTC/results/phase1_analysis/free_regime/20260513T211352Z_full/summary.md`

Workload: `input_len=8`, `output_len=128`, `max_model_len=2048`,
`gpu_memory_utilization=0.75`, three fresh-process repeats.

### Max Free `f_cpu_store`

| strategy | B=1 | B=4 | B=16 | B=64 |
|---|---:|---:|---:|---:|
| `cots_prefetch_only` | 0.0050 | n/a | n/a | n/a |
| `cots_cpu_only` | n/a | n/a | n/a | n/a |
| `cots_collab_50` | n/a | n/a | n/a | n/a |

### Best Observed Slowdown By `f_cpu_store`

| B | f=0.005 | f=0.010 | f=0.020 | f=0.0357 | f=0.050 | f=0.0714 | f=0.090 | f=0.150 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.047 | 1.064 | 1.062 | 1.134 | 1.210 | 1.538 | 1.923 | 3.006 |
| 4 | 1.323 | 1.317 | 1.324 | 1.173 | 1.424 | 1.745 | 2.194 | 3.565 |
| 16 | 1.342 | 1.350 | 1.383 | 1.471 | 1.926 | 2.592 | 3.300 | 5.261 |
| 64 | 1.126 | 1.152 | 1.167 | 1.420 | 1.795 | 2.362 | 2.982 | 4.627 |

Interpretation:

- CPU-only is closest to free at low batch.
- Prefetch-only becomes the best high-batch arm at larger fractions, but still
  usually loses the strict 5% gate.
- The 50/50 collaborative arm behaves like the CPU path and is dominated for
  throughput planning.

## CPU-Compute Gap

Main sources:

- `/TTC/results/phase1_analysis/cpu_gap/20260514T_b1_t16_grid/summary.md`
- `/TTC/results/phase1_analysis/cpu_gap/20260514T_b1_t4_boundary_3rep/summary.md`
- `/TTC/results/phase1_analysis/cpu_gap/20260515T_b1_t24_probe/summary.md`
- `/TTC/results/phase1_analysis/mlp_kernel/20260515T_fused_candidates.json`

Current `B=1` graph-mode boundary:

| requested f | actual target-byte f | qkv rows | MLP half channels | best t | best slowdown | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0.0200 | 0.0188 | 0 | 384 | 4 | 1.0501 | boundary |
| 0.0270 | 0.0250 | 0 | 512 | 4 | 1.049 | free |
| 0.0280 | 0.0292 | 256 | 512 | 4 | 1.116 | lose |
| 0.0500 | 0.0510 | 256 | 960 | 24 | 1.152 | lose |

The first QKV pair is the practical cliff. Counter diagnostics showed the added
QKV worker time is almost fully exposed because the WQKV overlap window is
short. MLP-only CPU work overlaps much better, and forced MLP-only probes moved
the free boundary only slightly, from about `512` to about `576` MLP half
channels.

The final fused BF16 MLP worker improves heavier exposed MLP cases, but does
not move the strict free boundary. Production keeps only the BF16-scratch path.

## KV-Throughput Crossover

A throughput win needs the effective KV concurrency/admission gain to exceed
the offload decode slowdown by at least the 5% win margin:

```text
effective_KV_concurrency_gain > 1.05 * offload_decode_slowdown
```

### `gpu_memory_utilization=0.75`

Source:
`/TTC/results/phase1_analysis/kv_throughput/20260515T_prefetch_update_gpu075/summary.md`

| workload | arm | output tok/s | gain | KV tokens | KV gain | verdict |
|---|---|---:|---:|---:|---:|---|
| short `(8,128)` | none | 8028.31 | 1.000 | 43,616 | 1.000 | baseline |
| short `(8,128)` | prefetch `f=0.02` | 7786.41 | 0.970 | 47,824 | 1.096 | tie |
| medium `(32,512)` | none | 4185.00 | 1.000 | 43,616 | 1.000 | baseline |
| medium `(32,512)` | prefetch `f=0.02` | 4233.02 | 1.011 | 47,824 | 1.096 | tie |
| medium `(32,512)` | prefetch `f=0.05` | 3398.54 | 0.812 | 52,640 | 1.207 | lose |
| medium `(32,512)` | prefetch `f=0.09` | 2432.56 | 0.581 | 60,144 | 1.379 | lose |
| long `(32,1024)` | none | 2824.10 | 1.000 | 43,616 | 1.000 | baseline |
| long `(32,1024)` | prefetch `f=0.02` | 2818.04 | 0.998 | 47,824 | 1.096 | tie |
| long `(32,1024)` | prefetch `f=0.05` | 2158.87 | 0.764 | 53,904 | 1.236 | lose |
| long `(32,1024)` | prefetch `f=0.09` | 1640.10 | 0.581 | 61,424 | 1.408 | lose |

### `gpu_memory_utilization=0.68`

Source:
`/TTC/results/phase1_analysis/kv_throughput/20260515T_prefetch_update_gpu068/summary.md`

| workload | arm | output tok/s | gain | KV tokens | KV gain | verdict |
|---|---|---:|---:|---:|---:|---|
| short `(8,128)` | none | 4817.65 | 1.000 | 12,624 | 1.000 | baseline |
| short `(8,128)` | prefetch `f=0.02` | 5380.57 | 1.117 | 16,848 | 1.335 | win |
| medium `(32,512)` | none | 1958.74 | 1.000 | 12,624 | 1.000 | baseline |
| medium `(32,512)` | prefetch `f=0.02` | 2270.57 | 1.159 | 16,848 | 1.335 | win |
| medium `(32,512)` | prefetch `f=0.05` | 1871.40 | 0.955 | 22,912 | 1.815 | tie |
| medium `(32,512)` | prefetch `f=0.09` | 1517.13 | 0.775 | 30,432 | 2.411 | lose |
| long `(32,1024)` | none | 1154.54 | 1.000 | 12,624 | 1.000 | baseline |
| long `(32,1024)` | prefetch `f=0.02` | 1336.44 | 1.158 | 16,848 | 1.335 | win |
| long `(32,1024)` | prefetch `f=0.05` | 1078.80 | 0.934 | 22,912 | 1.815 | lose |
| long `(32,1024)` | prefetch `f=0.09` | 895.22 | 0.775 | 30,432 | 2.411 | lose |

Conclusion: KV-throughput wins are possible, but only in a narrow window. At
`0.75`, `f=0.02` is neutral. At `0.68`, no-offload is sufficiently KV-starved
that the same tiny prefetch fraction wins. Larger fractions lose despite larger
KV pools.

## Current Planner Guidance

- For latency-free CPU compute, use MLP-first module-specific routing and avoid
  QKV CPU compute.
- For high-throughput offline serving, search prefetch-only first.
- Include tiny prefetch fractions (`f~0.02`) when no-offload is KV-starved.
- Do not spend throughput budget on CPU-only or 50/50 collaborative arms unless
  a low-batch latency target specifically needs them.
- Confirm any apparent KV-throughput win with repeat-3 before using it as a
  thesis headline.
