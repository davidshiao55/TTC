# Phase 1b Appendix: Three-Way Dispatch And Layer-Ahead Prefetch

Date: 2026-05-15 cleanup

Status: historical appendix. The final production path is summarized in
`phase1_findings.md`. This file preserves the Phase 1b design decisions that
still matter for the production COTS implementation and Planner model.

## What Phase 1b Added

Phase 1b turned the static Phase 1a CPU-compute split into a three-way dispatch:

```text
GPU permanent slice + GPU-prefetched slice + CPU-computed slice
```

The production per-bucket invariant is:

```text
f_cpu_compute + f_prefetch = f_cpu_store
```

Early Phase 1b notes allowed `<=`, but the final Planner-facing abstraction uses
equality. Storage is static at engine load; dispatch is per bucket. Runtime
snapping floors the CPU-compute side to legal module geometry and assigns the
remaining CPU-stored rows to prefetch.

## Production Pieces That Survived

| Phase 1b component | Final production status |
|---|---|
| per-bucket `n_prefetch` / `n_cpu_compute` geometry | kept |
| index-set disjointness | kept |
| three-way QKV scatter | kept |
| MLP add-reduce across permanent/prefetch/CPU paths | kept |
| layer-ahead pre-compute schedule | kept |
| K=2 slot rotation | kept |
| shape-group-shared prefetch pool | kept |
| active-bucket dispatch | kept |
| `prepare_before_forward` boundary hook | kept |

The schedule remains:

```text
wait_prefetch(i) -> start_prefetch(i + 1) -> forward(i)
```

This makes wraparound ordinary: layer `N-1` starts layer `0` for the next
iteration before layer `N-1` computes. COTS does not use the native
PrefetchOffloader's deferred-wraparound op.

## What Changed Later

| Phase 1b behavior | Final production behavior |
|---|---|
| post-init max-fill for layer-0 prefetch | lazy active-bucket layer-0 fill |
| fixed-max MLP prefetch slot layout | active-adjacent `[gate_active | up_active]` |
| row-prefetch duplicate buffer | down-proj CPU storage is transposed directly |
| Python runner under the dispatch path | native C++ runner |
| old collaborative 50/50 examples as headline | final docs emphasize Planner-chosen split |

The active-adjacent MLP layout is especially important. It lets prefetched
gate/up run as one `[gate|up]` GEMM even when `f_prefetch < f_cpu_store`, which
is the normal collaborative case.

The transposed down-proj storage is the cleanup version of the Phase 1b
row-prefetch fix. Phase 1b originally found that partial down-prefetch narrowed
a strided view and lost PCIe bandwidth. The final path stores the CPU down
slice as `(n_cpu, out_dim)`, so both prefetch and CPU compute use contiguous
row prefixes.

## Planner Evidence From Phase 1b

The Phase 1b matched-depth benchmark at `f_cpu_store=0.30` showed why the
Planner needs per-bucket dispatch:

| B | CPU-only | prefetch-only | 50/50 collaborative | verdict |
|---:|---:|---:|---:|---|
| 1 | 14.61 s | 19.83 s | 12.04 s | collaborative wins |
| 4 | 19.73 s | 19.90 s | 13.94 s | collaborative wins |
| 16 | 50.61 s | 20.06 s | 28.42 s | pure prefetch wins |
| 64 | 181.71 s | 20.01 s | 95.03 s | pure prefetch wins |

The follow-up split sweep showed the 50/50 conclusion was not universal. At
`B=16`, a prefetch-heavy split (`f_prefetch=0.25` at `f_cpu_store=0.30`) beat
pure prefetch by about `14.5%`; at `B=64`, pure prefetch remained best.

Final Planner guidance:

- small batch can use collaborative dispatch when CPU work is hidden;
- medium batch needs a prefetch-heavy split;
- high batch should usually choose pure prefetch;
- CPU-only is a low-batch diagnostic or niche path, not a high-throughput path.

## COTS vs Native Prefetch

The original Phase 1b native-prefetch comparison established the tensor
granularity hypothesis: at matched bytes, spreading slices across all layers can
match or beat whole-layer native prefetch. The current-code rerun lives in
`phase1_analysis_findings.md` and is the source of truth for final numbers.

## Legacy Section Map

Older docs may still refer to Phase 1b section numbers. Use this map when
reading those references:

| Old reference | Current location |
|---|---|
| `§1b.1-1b.3` architecture and prefetch schedule | "What Phase 1b Added" and "Production Pieces That Survived" |
| `§1b.7` row-prefetch fix | "What Changed Later" |
| `§1b.8` COTS vs native prefetch | `phase1_analysis_findings.md` COTS pure-prefetch vs native prefetch |
| `§1b.9-1b.12` collaborative/planner evidence | "Planner Evidence From Phase 1b" |
| `§1b.13` active-bucket dispatch | "Production Pieces That Survived" |

## Reader Guidance

Use this file for the origin of the three-way dispatch and prefetch schedule.
Use `phase1_findings.md` for the final production architecture and current
performance claims.
