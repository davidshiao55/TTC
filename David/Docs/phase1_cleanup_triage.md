# Phase 1 COTS Cleanup Triage

This is the structural cleanup map after the Phase 1 free-zone / KV-throughput
investigation. The main cleanup principle is:

- Keep execution-path changes that improved or clarified production behavior.
- Keep benchmark artifacts under `David/` because they encode the experimental
  record.
- Remove or hide production CLI diagnostics that were only useful for
  attribution once their findings are documented.

## Production vLLM Changes To Keep

| change | files | keep reason |
|---|---|---|
| MLP 64-channel snapping | `vllm/model_executor/offloader/cots_storage.py` | This is the main result. It removes the bad `~96`-channel MLP shape and moved `B=128,f=0.005` pure prefetch to `1.036x` real / `1.003x` dry. |
| Active-adjacent MLP prefetch slot layout | `cots_storage.py`, `cots_operators.py`, `cots_offloader.py` | Required so prefetched gate/up can run as one `[gate|up]` GEMM even when `f_cpu_store != f_prefetch`. |
| Lazy layer-0 prefetch fill | `cots_offloader.py`, `cots_storage.py` | Replaces post-init max-fill, which forced non-active-adjacent col slots and blocked the MLP fast path. |
| Pure-prefetch CPU-path cleanup | `cots_offloader.py`, `cots_operators.py` | Skips CPU runner slabs, pinned activation buffers, native active-dispatch updates, and wait-kernel setup when `n_cpu_compute == 0`. This is not just diagnostic; it is the correct pure-prefetch fast path. |
| Corrected COTS dry-run semantics | `offload.py`, `arg_utils.py`, `cots_offloader.py`, `cots_operators.py`, `cots_storage.py` | Public `--cots-dry-run` now means "keep wrappers/control flow, skip active CPU and prefetch work." This remains useful as a regression/control-plane diagnostic. |
| Prefetched-down `addmm_` cleanup | `cots_operators.py` | Small but clean: removes an intermediate prefetched MLP2 output/add in the common path. |

## Benchmark / Documentation Artifacts To Keep

| artifact | keep reason |
|---|---|
| `David/Benchmarks/phase1_analysis/bench_cots_free_regime.py` | Reusable latency free-zone sweep. |
| `David/Benchmarks/phase1_analysis/bench_cots_kv_throughput.py` | Reusable KV-capacity throughput sweep with log parsing. |
| `David/Benchmarks/phase1_analysis/bench_cots_vs_native_prefetch.py` | Current-code rerun of the Phase 1b native-vs-COTS prefetch comparison. |
| `David/Benchmarks/phase1_analysis/bench_cots_prefetch_gap.py` | Useful diagnostic harness, but it depends on some temporary production diagnostics listed below. Keep under `David/`; do not treat as production API. |
| `David/Tests/phase1_analysis/test_kv_throughput_log_parser.py` | Cheap parser/command regression coverage for the throughput harness. |
| `David/Tests/phase1_analysis/test_cots_vs_native_prefetch.py` | Cheap command/focused-grid coverage for the benchmark harnesses. |
| `David/Docs/phase1_analysis_findings.md` | Source-of-truth narrative and tables for the Phase 1 analysis. |

## Diagnostics Removed From Production

These were important for attribution, but they are overexposed in the vLLM
runtime surface now that the conclusion is documented. They have been removed
from the production COTS config/CLI path.

| diagnostic | previous files | status |
|---|---|---|
| `--cots-diagnostic-skip-prefetch-h2d` | `offload.py`, `arg_utils.py`, `cots_offloader.py`, `cots_storage.py` | Removed. |
| `--cots-diagnostic-skip-prefetch-compute` | `offload.py`, `arg_utils.py`, `cots_operators.py` | Removed. |
| `--cots-diagnostic-disable-prefetch-control` | `offload.py`, `arg_utils.py`, `cots_offloader.py` | Removed. |
| `--cots-diagnostic-keep-mlp-gpu-full-shape` | `offload.py`, `arg_utils.py`, `cots_storage.py`, tests | Removed. |
| `--cots-diagnostic-target-kind` | `offload.py`, `arg_utils.py`, `cots_offloader.py`, tests | Removed. |
| `CotsPrefetchBufferPool.zero_()` | `cots_storage.py` | Removed. |

## Immediate File Hygiene

| artifact | recommendation |
|---|---|
| `David/Benchmarks/phase1_analysis/__pycache__/` | Delete whenever generated. It is local bytecode and should not be tracked. |

## Tests To Keep / Adjust

| test | recommendation |
|---|---|
| `test_mlp_counts_snap_to_64_channel_grid` | Keep. This is production behavior. |
| Pure-prefetch zero-CPU-compute tests | Keep. They guard the pure-prefetch fast path. |
| Dry-run semantic test | Keep if public `--cots-dry-run` remains. |
| `test_diagnostic_target_kind_filters_handles` | Removed with `diagnostic_target_kind`. |
| `test_dry_run_mlp_full_shape_diagnostic_keeps_gpu_weight_shapes` | Removed with the full-shape diagnostic. |

## Probably Overengineering

- Exposing five benchmark-only diagnostics as first-class CLI/config fields.
  They made experimentation easy, but they make the production COTS surface look
  much more complicated than it needs to be.
- Full-shape MLP diagnostic support in the loader. It is valuable evidence but
  intentionally violates the memory-saving objective.
- Duplicating snapping/theory math inside benchmark scripts. It is acceptable
  for reproducibility, but the cleaner long-term path is for COTS/Planner to
  report effective snapped counts directly in benchmark metadata.
- Continuing to tune prefetch micro-optimizations after snap64. The next useful
  work is Planner-facing explicit snapped counts and KV-throughput analysis, not
  another 1-2% hot-path tweak.

## Suggested Cleanup Order

1. Keep all current benchmark/docs artifacts until the Phase 1 analysis is
   finalized.
2. For the production vLLM path, keep snap64, active-adjacent MLP, lazy layer-0
   fill, pure-prefetch fast path, dry-run, and `addmm_`.
3. Add effective snapped placement reporting for benchmarks and Planner work:
   per module, report requested fraction, snapped count, effective fraction,
   and bytes.
