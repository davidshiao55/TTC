# Phase 1 Findings: Production COTS Weight Offload

Date: 2026-05-15

Status: source of truth for Phase 1 after cleanup. This document describes the
final production COTS path, then summarizes the experiments that shaped it.
The older milestone files are now appendices:

- `phase1a_findings.md`: static split prototype facts that survived.
- `phase1b_findings.md`: three-way dispatch and layer-ahead prefetch appendix.
- `phase1c_findings.md`: native runner and graph-mode implementation appendix.
- `phase1_analysis_findings.md`: final free-regime and KV-throughput tables.

`phase1_cleanup_triage.md` was merged into this document and removed.

## Freeze Record

Phase 1 freeze pass was run on 2026-05-15.

Baseline commits:

| Repo | Commit | Note |
|---|---:|---|
| `/TTC/vllm` | `da4894d6e` | final Phase 1 implementation |
| `/TTC` | `7729c73` | Phase 1 docs restructure parent; this freeze note advances it |

The freeze pass caught one teardown bug before the final baseline was recorded:
`NativeCotsRunner.close()` could unregister the pybind infer while
stream-ordered COTS callback/UVA work was still unwinding. The fix synchronizes
the current CUDA stream before draining the CPU queue and unregistering the
runner. This is in `/TTC/vllm` commit `da4894d6e`.

Verification run:

| Check | Result |
|---|---:|
| `python -m py_compile` on Phase 1 analysis benchmark scripts | pass |
| `python -m py_compile vllm/model_executor/offloader/cots_runners.py` | pass |
| `David/Tests/phase1_analysis/test_kv_throughput_log_parser.py` and `test_cots_vs_native_prefetch.py` | 6 passed |
| `vllm/tests/model_executor/test_cots_prefetch_trace.py` plus COTS graph-split config tests | 4 passed |
| `phase1b/test_three_way_scatter.py`, `phase1b/test_layer_ahead_smoke.py`, `phase1c/test_parity_with_python_runner.py`, `phase1c/test_stage2_substrate.py` | 54 passed |
| free-regime smoke, `B=1`, `output_len=16`, `f=0.01` | completed |
| Qwen2.5-7B COTS generation smoke, `f_cpu_store=0.09`, eager | completed |

Free-regime smoke output:

| Arm | Mean latency |
|---|---:|
| `none` | 0.2658 s |
| `cots_cpu_only_f0p01` | 0.2831 s |
| `cots_prefetch_only_f0p01` | 0.2782 s |

Artifacts: `/TTC/results/phase1_freeze/free_regime_smoke_20260515/`.

Generation smoke output was semantically sane for prompt
`The capital of France is`, producing: `Paris. Which of the following
statements is`.

## Executive Summary

Phase 1 lands a graph-compatible collaborative tensor-split weight offloader
for Qwen/Qwen2.5-7B-Instruct BF16 on the RTX 4090 target.

The production path is:

- Store a static CPU slice of QKV, MLP gate/up, and MLP down weights.
- Leave `o_proj` GPU-resident.
- Dispatch each bucket across three paths:
  GPU-resident compute, layer-ahead prefetched GPU compute, and CPU compute.
- Use native C++ CPU tasks, not the old Python `ThreadPoolExecutor`, as the
  production runner.
- Use piecewise CUDA graphs with COTS submit/sync split points and the
  wait-kernel sync path.
- Use MLP 64-channel snapping, active-adjacent MLP prefetch slots, transposed
  down-proj CPU storage, and a BF16 fused gate/up/SwiGLU CPU worker path.

Final outcome:

- COTS pure-prefetch now matches native vLLM prefetch at matched bytes and is
  usually within about 5% once the offload depth is non-trivial.
- The strict latency free regime is narrow: prefetch-only is only free at tiny
  fractions in selected cells, and CPU-only is free only for small MLP-only
  slices at `B=1`.
- KV-throughput wins are possible, but only when no-offload is truly KV-starved.
  At `gpu_memory_utilization=0.68`, prefetch `f=0.02` reaches `1.16x`
  output-token throughput on medium/long workloads; at `0.75` the same path is
  neutral, not a win.
- Larger offload fractions still lose throughput because the recurring per-token
  split-path cost rises faster than the useful KV admission gain.

## Production Path

### Public Surface

The public import remains:

```python
from vllm.model_executor.offloader.cots import CotsOffloader
```

Production configuration:

| Surface | Production status |
|---|---|
| `offload_backend="cots"` | supported |
| `CotsOffloadConfig.cpu_runner="native"` | default |
| `cpu_runner="python"` | eager-only diagnostic kill switch |
| `auto_graph_split=True` | default for native COTS |
| `cots_capture_sync_mode="wait_kernel"` | native graph default after auto-upgrade |
| `cpu_num_threads_by_bucket` | Planner hook |
| `dry_run` / `--cots-dry-run` | control-plane diagnostic |
| `VLLM_COTS_NVTX`, `VLLM_COTS_COUNTERS`, `VLLM_COTS_WAIT_KERNEL_DIAG` | env-gated diagnostics |

Removed or rejected from the supported surface:

- `VLLM_COTS_ABLATE_*`
- fused `wait_uva_kernel`
- dryrun burst helpers
- diagnostic target-kind and full-shape MLP CLI flags
- raw counter JSON dump hooks
- runtime MLP kernel selector and FP32-scratch MLP kernel candidate

### Storage And Placement

COTS offloads only the Phase 1 target tensors:

| Module | Split axis | Production behavior |
|---|---|---|
| `qkv_proj` | output columns | K/V-biased, head-aligned CPU slice |
| `gate_up_proj` | output columns | matched gate/up half-channel slice, snapped to 64 |
| `down_proj` | input columns | matched MLP intermediate slice, snapped to 64 |
| `o_proj` | none | fully GPU-resident |

The storage split is TP-style at load time: each wrapped parameter's
`param.data` is replaced with the GPU-resident slice before weight loading, and
the loader closure writes the CPU slice into pinned memory. This keeps GPU
allocations inside vLLM's model memory profiler, so freed bytes become KV-cache
headroom instead of invisible extra allocation.

MLP granularity is the final production granularity:

- Gate/up half channels and down input channels snap to multiples of `64`.
- The snap applies to both `f_cpu_store` and bucket-level `f_prefetch`.
- There is no hard `128` minimum.
- The reason is empirical: arbitrary narrow MLP shapes such as ~96 channels hit
  bad GEMM shapes; 64-channel multiples avoided the cliff while keeping enough
  placement flexibility.

QKV keeps the K/V-biased head-aligned picker because partial K/V heads are the
wrong abstraction for the Phase 2 attention handoff. The cost is coarser
granularity: the first QKV CPU-compute slice is a practical latency cliff.
Planner-facing implication: keep QKV CPU compute at zero in the free regime and
spend small CPU-compute budgets on MLP first.

### Runtime Dispatch

For each bucket and each offloaded module:

```text
f_cpu_compute + f_prefetch <= f_cpu_store
```

Runtime geometry is computed once from the dispatch table:

- `gpu_indices`: permanently GPU-resident slice.
- `prefetch_indices_by_bucket`: CPU-stored slice copied back to GPU for this
  bucket.
- `cpu_compute_indices_by_bucket`: CPU-stored slice computed on CPU for this
  bucket.

The operator uses active-bucket state published outside the graph, not slot
history, to decide the computation shape. Slot metadata only proves that bytes
are available.

### Prefetch Path

Layer-ahead prefetch is the production prefetch schedule:

```text
wait_prefetch(layer i)
start_prefetch(layer i+1)
forward(layer i)
```

Properties:

- K=2 slot rotation is the minimum safe slot count for overlapping layer `i+1`
  H2D with layer `i` compute.
- Slots are allocated per unique shape, not per layer, so the pool is shared
  across all layers with the same kind/shape.
- Wraparound is implicit: layer `N-1` starts layer `0` for the next iteration.
- The old COTS-specific deferred-wraparound machinery is not used.
- Layer-0 priming is lazy and active-bucket aware. We do not post-init max-fill
  all MLP slots into a non-active layout.

The final MLP prefetch slot layout is active-adjacent:

```text
[gate_active | up_active]
```

This is load-bearing. It allows the prefetched MLP gate/up path to run as one
`[gate|up]` GEMM even when `f_prefetch < f_cpu_store`.

The down-proj CPU storage is transposed:

```text
w_cpu: (n_cpu, out_dim)
```

Both prefetch and CPU compute narrow on dim 0, so the row-prefetch H2D is
contiguous and the CPU down kernel can consume the same layout directly. This
replaces the earlier pinned duplicate buffer and removes the strided-H2D
pitfall from Phase 1b.

Pure-prefetch is a real fast path. When every bucket has
`n_cpu_compute == 0`, COTS skips CPU-runner slabs, pinned activation buffers,
native active-dispatch updates, and wait-kernel setup for CPU work.

### CPU Compute Path

The native runner is production. It installs one C++ task slab per
`(layer_idx, bucket, op_kind)` and submits work from stream host callbacks.
CPU results return through the SM-issued UVA copy path, not a copy-engine D2H
or H2D activation copy.

Production CPU kernels:

| Task kind | CPU kernel path |
|---|---|
| QKV | BF16 natural row-major GEMM |
| MLP gate/up/SwiGLU/down | fused BF16 gate/up/SwiGLU into BF16 scratch, then BF16 transposed down GEMM |
| Down projection primitive | BF16 transposed GEMM on `(K, N)` row-major weight |

The MLP worker intentionally rounds gate/up accumulators and the SwiGLU scratch
through BF16. That mirrors the production scratch semantics and is the reference
used by the retained tests.

The FP32-scratch candidate was rejected. It was useful as a microbench probe,
but it changed intermediate precision and did not justify a runtime selector.

### Graph Path

The default graph policy is piecewise graph split at the two COTS custom ops:

- `vllm::cots_submit_gemm`
- `vllm::cots_sync_then_uva`

The surrounding GPU-only regions still use piecewise CUDA graphs. COTS
orchestration stays outside captured graph nodes, avoiding replay-time host
callback backpressure. `wait_kernel` replaces the sync-side callback by waiting
on host-mapped completion slots from a tiny GPU kernel.

Live-token capping is required. Captured graph tensors are bucket-sized, but
decode often has fewer real rows. The native worker clamps CPU GEMM to the live
row count published outside the graph.

`VLLM_DISABLE_COMPILE_CACHE=1` was only used for diagnostic graph sweeps around
a TorchInductor standalone compile-cache failure. It is not a production COTS
requirement.

## Key Production Results

### Phase 1c Graph Runtime

Qwen2.5-7B BF16, `B=1`, `input_len=8`, `output_len=128`,
`f_cpu_store=0.05`, `cpu_num_threads=16`:

| Arm | Seconds / generate | Delta vs native eager |
|---|---:|---:|
| native eager real | 2.5357 | baseline |
| piecewise COTS split + wait kernel | 2.4573 | -78.4 ms |

The broader grid over `B={1,4}`, `input_len={8,128,512}`, and
`output_len={32,128}` showed the split path winning all 12/12 cells against
native eager and all 12/12 cells against legacy full capture.

### COTS Pure-Prefetch vs Native Prefetch

Current-code source:
`/TTC/results/phase1_analysis/cots_vs_native_prefetch/20260514T050345Z/summary.md`

Values are `COTS latency / best native latency`; `<1` means COTS is faster.

| mode | B | 01L | 02L | 04L | 07L | 14L |
|---|---:|---:|---:|---:|---:|---:|
| graph | 1 | 1.022 | 0.963 | 0.963 | 0.939 | 0.911 |
| graph | 64 | 1.157 | 1.085 | 1.029 | 1.003 | 0.994 |
| eager | 1 | 0.965 | 0.947 | 0.959 | 0.946 | 0.945 |
| eager | 64 | 1.026 | 0.962 | 0.959 | 0.945 | 0.944 |

Conclusion: COTS pure-prefetch matches native prefetch same-order and is usually
within about 5% once the depth is not tiny. The main remaining miss is
graph-mode shallow offload at `B=64`; deeper offload is equal or faster.

### Strict Free Regime

Free means mean latency `<= 1.05x` no-offload and CV `<= 3%`.

Current full sweep source:
`/TTC/results/phase1_analysis/free_regime/20260513T211352Z_full/summary.md`

| strategy | B=1 | B=4 | B=16 | B=64 |
|---|---:|---:|---:|---:|
| `cots_prefetch_only` max free `f_cpu_store` | 0.0050 | n/a | n/a | n/a |
| `cots_cpu_only` max free `f_cpu_store` | n/a | n/a | n/a | n/a |
| `cots_collab_50` max free `f_cpu_store` | n/a | n/a | n/a | n/a |

The production optimizations after that sweep pushed the best high-batch
prefetch micro-cell: at `B=128`, pure prefetch `f=0.005` measured `1.036x`
real and `1.003x` dry once MLP snapping and active-adjacent slots landed.
That is useful evidence for the implementation, but not yet a broad free-zone
claim.

### CPU-Compute Boundary

Current CPU-only finding at `B=1`, graph mode, `input_len=8`,
`output_len=128`, `gpu_memory_utilization=0.75`:

| requested f | actual target-byte f | qkv rows | MLP half channels | best t | best slowdown | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0.0200 | 0.0188 | 0 | 384 | 4 | 1.0501 | boundary |
| 0.0270 | 0.0250 | 0 | 512 | 4 | 1.049 | free |
| 0.0280 | 0.0292 | 256 | 512 | 4 | 1.116 | lose |
| 0.0500 | 0.0510 | 256 | 960 | 24 | 1.152 | lose |

Planner rule for current code:

- CPU-only free-regime dispatch should be module-specific.
- Keep QKV CPU compute at zero in the free zone.
- Spend small CPU-compute budgets on MLP first, up to roughly `512-576` MLP
  half channels at `B=1` if a 5% latency tax is acceptable.
- Treat CPU-only as a diagnostic or low-batch path, not a throughput-search
  candidate at large batch.

The fused BF16 MLP worker improves exposed heavy MLP cases, for example
`f=0.05,t=4` from about `1.167x` slowdown to about `1.115x`, but it does not
move the current 5% boundary because the boundary cell was already mostly
overlapped.

### KV-Throughput Crossover

Updated source after prefetch-path cleanup:

- `gpu_memory_utilization=0.75`:
  `/TTC/results/phase1_analysis/kv_throughput/20260515T_prefetch_update_gpu075/summary.md`
- `gpu_memory_utilization=0.68`:
  `/TTC/results/phase1_analysis/kv_throughput/20260515T_prefetch_update_gpu068/summary.md`

At `gpu_memory_utilization=0.75`, the optimized prefetch path recovers
low-fraction losses into ties, but does not produce a `>5%` throughput win:

| workload | arm | output tok/s | gain | KV tokens | KV gain | verdict |
|---|---|---:|---:|---:|---:|---|
| short `(8,128)` | none | 8028.31 | 1.000 | 43,616 | 1.000 | baseline |
| short `(8,128)` | prefetch `f=0.02` | 7786.41 | 0.970 | 47,824 | 1.096 | tie |
| medium `(32,512)` | none | 4185.00 | 1.000 | 43,616 | 1.000 | baseline |
| medium `(32,512)` | prefetch `f=0.02` | 4233.02 | 1.011 | 47,824 | 1.096 | tie |
| long `(32,1024)` | none | 2824.10 | 1.000 | 43,616 | 1.000 | baseline |
| long `(32,1024)` | prefetch `f=0.02` | 2818.04 | 0.998 | 47,824 | 1.096 | tie |

At `gpu_memory_utilization=0.68`, no-offload is KV-starved enough for small
prefetch to win:

| workload | arm | output tok/s | gain | KV tokens | KV gain | verdict |
|---|---|---:|---:|---:|---:|---|
| short `(8,128)` | none | 4817.65 | 1.000 | 12,624 | 1.000 | baseline |
| short `(8,128)` | prefetch `f=0.02` | 5380.57 | 1.117 | 16,848 | 1.335 | win |
| medium `(32,512)` | none | 1958.74 | 1.000 | 12,624 | 1.000 | baseline |
| medium `(32,512)` | prefetch `f=0.02` | 2270.57 | 1.159 | 16,848 | 1.335 | win |
| long `(32,1024)` | none | 1154.54 | 1.000 | 12,624 | 1.000 | baseline |
| long `(32,1024)` | prefetch `f=0.02` | 1336.44 | 1.158 | 16,848 | 1.335 | win |

This is the cleanest final Phase 1 answer to the KV-throughput question:
offload for more KV throughput is not mathematically impossible. It can win
when the no-offload run is admission-limited and the offload fraction is tiny
enough that the per-token split-path cost stays mostly hidden. The window is
narrow: `f=0.05` and `f=0.09` still lose or only tie even with much larger KV
capacity.

## Decisions That Matter

### Native Runner Replaced The Python Prototype

The Python runner was useful for Phase 1a/1b correctness but is not production.
It cannot be captured safely and it exaggerates host orchestration and
oneDNN-vs-main-thread interference. Native C++ slabs plus stream host callbacks
are the production substrate.

### Split Graph Beat Full Capture

Full CUDA graph capture with host callbacks was correct but slower than native
eager because replay backpressured inside captured callback nodes. The winning
path keeps COTS submit/sync outside captured regions while graphing the
surrounding GPU work.

### MLP Fast Path Needed Active-Adjacent Prefetch

The earlier post-init max-fill/fixed-max MLP slot layout made the prefetched
gate and up slices non-adjacent when `f_prefetch < f_cpu_store`. That forced two
separate prefetched MLP1 GEMMs. The final path fills layer-0 lazily for the
active bucket and stores prefetched gate/up as `[gate_active | up_active]`, so
the common path is one `[gate|up]` GEMM.

### Row Prefetch Needed Transposed Storage

The collaborative path originally hit pitched H2D for partial down-proj slices.
The fix was to store the CPU down slice as `(n_cpu, out_dim)` so both prefetch
and CPU compute narrow contiguous rows. This also simplified production by
removing the separate row-prefetch duplicate buffer.

### MLP 64-Channel Snapping Was Worth Keeping

The prefetch free-zone gap was not only PCIe copy tail; tiny arbitrary split
shapes moved COTS onto slower GPU/CPU kernels. Snapping MLP gate/up/down to a
64-channel grid avoided the bad shape while preserving useful placement
resolution. At Qwen2.5-7B, 64 channels is about `0.34%` of the MLP intermediate
dimension per gate/up half.

### QKV CPU Compute Is The Free-Zone Cliff

The first QKV CPU-compute slice is a 256-row K/V head pair. It adds little
absolute worker time but most of it is exposed because WQKV's GPU overlap
window is short. Current Planner guidance is to avoid QKV CPU compute in the
free regime and revisit only if a future WQKV CPU path gets below the overlap
window or gains a longer schedule window.

### Larger Offload Fractions Still Lose KV Throughput

KV benefit is a concurrency/admission benefit that applies every iteration, but
it only helps while no-offload is admission-limited. Weight offload adds a
recurring per-layer split-path cost. A throughput win requires the effective
admission gain to exceed that cost by at least the 5% win margin. Current code
only satisfies that at tiny prefetch fractions under tighter memory.

## Validation

Retained validation covers:

- loader and TP-style split parity;
- QKV picker and MLP snap geometry;
- MLP matched-index invariant;
- UVA activation return;
- three-way scatter/add-reduce;
- prefetch slot rotation and owner/available-row checks;
- active-bucket dispatch and layer-0 repair;
- native runner install, teardown, and multi-engine safety;
- CUDA graph capture/replay and wait-kernel sync;
- pure-prefetch zero-CPU-compute fast path;
- BF16 custom kernels and production BF16 scratch semantics;
- KV-throughput log parsing.

Core reusable benchmark harnesses:

| Harness | Purpose |
|---|---|
| `David/Benchmarks/phase1_analysis/bench_cots_vs_native_prefetch.py` | COTS pure-prefetch vs native prefetch |
| `David/Benchmarks/phase1_analysis/bench_cots_prefetch_gap.py` | prefetch gap decomposition |
| `David/Benchmarks/phase1_analysis/bench_cots_free_regime.py` | latency free-regime sweep |
| `David/Benchmarks/phase1_analysis/bench_cots_cpu_gap.py` | CPU-only gap to math bound |
| `David/Benchmarks/phase1_analysis/bench_cots_kv_throughput.py` | KV-capacity throughput sweep |

## Reproduce The Final Probes

Run from `/TTC/FastTTS-thesis` in the `thesis` environment:

```bash
cd /TTC/FastTTS-thesis

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_vs_native_prefetch.py \
  --exp --smoke

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_free_regime.py \
  --exp --focused-grid --batch-sizes 1 64

/opt/conda/envs/thesis/bin/python \
  /TTC/David/Benchmarks/phase1_analysis/bench_cots_kv_throughput.py \
  --exp --focused-grid --only-arms none cots_prefetch_only --repeat 1
```

For the low-memory KV crossover probe, add:

```bash
--gpu-memory-utilization 0.68
```

## Phase 2 Handoff

Phase 2 can build CPU suffix attention on this substrate without carrying the
rejected Phase 1 probe paths forward:

- native COTS is the only production runtime substrate;
- Python runner is a diagnostic fallback only;
- graph-mode default is piecewise COTS split plus wait kernel;
- CPU worker row count is live-token capped;
- COTS routing is runtime state published outside graphs;
- pure-prefetch and CPU-compute paths both have clean zero-work fast paths;
- diagnostics are explicit and env-gated.

The open Planner work is to emit module-specific dispatch rather than uniform
fractions when needed: MLP-first CPU compute at low batch, prefetch-heavy or
pure-prefetch at high batch, and QKV CPU compute only when a future profile
proves the first head pair is no longer exposed.
