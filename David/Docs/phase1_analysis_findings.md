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
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_kv_throughput.py \
  --exp --focused-grid --only-arms none cots_prefetch_only --repeat 1
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
