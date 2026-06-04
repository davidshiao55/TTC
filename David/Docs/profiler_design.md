# Profiler Design

The Profiler characterizes hardware and model behavior offline, producing cached tables that the Planner consumes at engine startup.

- **Timescale**: offline, run once per `(hardware, model, dtype)` tuple
- **Inputs**: GPU, CPU, PCIe capabilities; model architecture
- **Outputs**: profile tables (cached to disk)

The Planner is only as good as the profile it consumes — the Profiler's job is to produce accurate, well-structured data for it.

---

## 1. Profile Schema

Four tables, keyed as noted:

| Table | Key | Value | Purpose |
|---|---|---|---|
| `gpu_layer_timing` | `(sub_module, num_tokens_bucket)` | GPU time (ms) | Per-bucket GPU compute budget — determines how much CPU work can hide |
| `cpu_gemm_curve` | `sub_module → {axis, times: (batch_size, slice_frac) → ms}` | CPU GEMM time (ms) | Feasibility of a given `f_cpu` slice, per split axis |
| `pcie_h2d_bw` | `transfer_bytes` | Effective H2D BW (GB/s) | Prefetch transfer modeling |
| `cpu_attn_curve` | `(batch_size, suffix_context_len)` | CPU attention time (ms) | Attention-offload feasibility; batch-size sweet spot |

All values are stored per `(hardware, model, dtype)` tuple. Importantly, profile tables are **indexed by hardware quantities** (batch sizes, `num_tokens` buckets, transfer bytes) — not by workload parameters like `n`. Workload dependence enters the pipeline at the Planner stage.

### 1.1 `gpu_layer_timing`

Per-sub-module GPU compute time at each `num_tokens` bucket that vLLM's `CudagraphDispatcher` will capture. Sub-modules: `WQKV`, `WO`, `MLP1`, `MLP2`. `num_tokens` buckets match vLLM's `cudagraph_capture_sizes` — both the uniform-decode set (`{1, 2, 4, 8, …, max_num_seqs}`) and the mixed prefill-decode set (spans up to `max_num_batched_tokens`).

Used by the Planner to compute the GPU idle budget in each bucket (memory-BW-bound at small `num_tokens`, compute-bound at large).

### 1.2 `cpu_gemm_curve`

CPU matmul time as a function of `(batch_size, slice_frac)` for each sub-module shape. Must be measured with `F.linear` (oneDNN BF16 path), **never** `torch.mm` — the BF16 `torch.mm` path on non-AMX CPUs falls back to a scalar loop and is 100–250× slower. This is the single most important implementation detail in the Profiler (see `phase0_findings.md §0.3.2`).

**Per-sub-module axis envelope.** Each sub-module has exactly one split axis fixed by the design (see `weight_offload_design.md §Per-Sub-Module Split Axis`): col for WQKV, MLP1, and WO (Alt A); row for MLP2. The axis is stored once per sub-module in the table's envelope, not as an extra key dimension:

```
cpu_gemm_curve[sub_module] = {
    "axis":  "col" | "row",
    "times": { batch_size: { slice_frac: ms } }
}
```

Col and row produce different CPU GEMM shapes at the same `slice_frac` (col: `[N, in_dim] × [in_dim, out_dim · f]`; row: `[N, in_dim · f] × [in_dim · f, out_dim]`), so the stored timing is only meaningful together with the axis label. The Planner reads the axis from the envelope and uses the matching shape. `slice_frac` supersedes the earlier `slice_cols` key and is uniformly interpreted as "fraction of the sliced dim on CPU."

Covers enough `slice_frac` points to interpolate dispatch decisions (10–20 points per sub-module).

### 1.3 `pcie_h2d_bw`

Effective PCIe H2D bandwidth as a function of transfer size, using pinned memory. Small transfers (activation-scale, ~KB) have materially lower effective BW than large transfers (weight-slice-scale, ~MB) due to launch overhead. The Planner uses the right curve for the right purpose:

- Weight prefetch → MB-scale entry
- CPU-compute activation round-trip → KB-scale entry

### 1.4 `cpu_attn_curve`

CPU attention kernel latency as a 2D grid over `(batch_size, suffix_context_len)`. Used only when attention offloading is enabled. Extends the `num_tokens`-bucket abstraction with a context dimension because attention cost scales with KV length, not just batch.

### 1.5 `cots_snap`

Weight offload profiles also include a compact runtime-realization section:

```json
{
  "cots_snap": {
    "schema_version": 1,
    "snap_model": "cots_snap_v1",
    "storage_by_store_fraction": {
      "0.15": {
        "cpu_weight_bytes": 1849700000,
        "gpu_buffer_bytes": 224400000
      }
    }
  }
}
```

This is not a new planner decision. It records how the vLLM COTS runtime
realized requested storage fractions after tensor snapping. vLLM owns the snap
rules because they depend on model handles and runtime layout: QKV head groups,
MLP 64-channel granularity, optional WO rows, and buffer slot shapes. The
Profiler captures the realized bytes and gives the Planner exact resource facts
for the calibrated storage grid. The Planner may fall back to linear
coefficients when a profile lacks this section, but validation should use
`cots_snap` whenever it is available.

At runtime, vLLM emits the same schema as a `[CotsOffloader] cots_snap:` JSON
log line after weight loading. The profiler can ingest that line directly when
building the planner-facing `weight_dispatch_profile.json`.

---

## 2. Methodology

- **Microbenchmarks, not end-to-end.** Each table is measured in isolation with the workload shape of interest. End-to-end timing mixes effects and complicates attribution.
- **Warm up before every measurement** (multiple iterations, discard first N).
- **Report median over ≥10 runs** (small N sufficient at this grid density; noise bounds reported alongside).
- **Use pinned memory** for all CPU ↔ GPU transfers.
- **Match production path exactly**: same dtype, same kernel dispatch, same memory layout. The profile is only valid if the measured path is the path the Planner's output actually invokes.

The profile is expensive to run but cheap to consume. Run once per `(HW, model, dtype)`; consume at every engine startup.

---

## 3. Caching

Profile results are cached to disk at:

```
/TTC/David/Benchmarks/profile_cache/{hardware_id}_{model_id}_{dtype}.json
```

Exact path to be confirmed in the implementation. The cache key is the tuple `(hardware_id, model_id, dtype)`; any change invalidates the cache.

**Cache invalidation triggers** (the Profiler re-runs automatically when it detects):

- A new GPU/CPU (detected via `nvidia-smi` / `lscpu`)
- A new model (detected via `config.json` hash)
- A dtype change
- A vLLM version that changes captured bucket set

On cache hit, the Planner consumes the JSON directly. On cache miss, the Profiler runs (takes minutes to tens of minutes depending on grid density) before engine startup proceeds.

---

## 4. Relation to `phase0_findings.md`

`phase0_findings.md` is the **first iteration** of the Profiler's output: RTX 4090 + Qwen2.5-7B + BF16. It validates the schema above and establishes the baseline numbers the thesis reports. Future profile runs (e.g., 14B, different hardware) re-use the methodology documented here and write additional cache entries — the numbers change, the structure does not.

---

## 5. Minimum Viable Profile

For bootstrap: the Profiler can operate with a reduced schema where only `gpu_layer_timing` and `cpu_gemm_curve` are populated. The Planner degrades gracefully:

- No `pcie_h2d_bw` → assume a single value (22 GB/s for PCIe 4.0 x16); lose transfer-size realism
- No `cpu_attn_curve` → Planner cannot enable attention offloading; `KV_cpu_bytes` effectively 0

This lets Phase 1a proceed without the full profile.

---

## 6. Non-Goals

- **End-to-end latency curves.** The Profiler measures mechanisms, not policies; the Planner composes them into end-to-end estimates. Measuring end-to-end curves at profile time would require sweeping the Planner's search space, inverting the pipeline.
- **Online adaptation.** The Profiler does not react to runtime conditions. Drift handling is the Scheduler's concern (with re-profile triggered only on detected environment change).
- **Predicting FastTTS throughput directly.** That's the Planner's job — the Profiler supplies inputs only.
- **Workload-specific profiling.** Profile tables are keyed by hardware quantities (bucket, batch, transfer size), not by workload parameters (`n`, beam width). Workload enters only at the Planner.

---

## References

- `phase0_findings.md` — first-iteration output; concrete numbers for RTX 4090 + Qwen2.5-7B BF16
- `planner_design.md` — consumer of profile tables; documents how each table feeds into Planner variables
- `vllm_benchmarking_findings.md` — empirical findings that informed the schema (KV pressure, offloader PCIe stall)
- `vllm/docs/design/cuda_graphs.md` — source for `num_tokens` bucket set that `gpu_layer_timing` must cover
