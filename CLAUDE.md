# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master thesis project on **Test-Time Compute (TTC)** optimization for single consumer GPU-CPU systems (target: NVIDIA RTX 4090, 24 GB VRAM). The thesis extends FastTTS (a test-time search framework) with hybrid CPU-GPU offloading to overcome GPU memory limitations.

### Core Problem
FastTTS assumes model weights stay GPU-resident, which limits max model size to VRAM and bounds batch size/throughput by leftover GPU memory. This project introduces hybrid CPU-GPU offloading to break these constraints.

### System Architecture

Three components on three timescales (see `David/Docs/thesis_proposal.md`):

- **Profiler** (offline) — measures HW/model behavior; produces cached tables. See `David/Docs/profiler_design.md`.
- **Planner** (load-time) — at engine launch, solves for placement + per-`BatchDescriptor` dispatch table from profile + budgets + workload target. **Primary contribution.** See `David/Docs/planner_design.md`.
- **Scheduler** (runtime) — executes the plan: tier-aware KV admission, KV migration, dispatch lookup. See `David/Docs/scheduler_design.md`.

Three fundamental partitions the components orchestrate:

1. **Weight: 3-way compute split**. Every matmul sub-module (WQKV, WO, MLP1, MLP2) partitions across GPU compute / prefetch-to-GPU / CPU compute paths so GPU, CPU, and PCIe all contribute concurrently. Storage is static (Planner sets `f_cpu_store_m` at load time); compute dispatch is per-`BatchDescriptor` (Planner emits `(f_cpu_compute, f_prefetch_compute)` table, runtime is table lookup). Invariant: `f_cpu_compute + f_prefetch_compute ≤ f_cpu_store`. WQKV uses K/V-biased column assignment (KV-head groups on CPU slice first).
2. **KV: 2-way pool**. `KV_gpu_bytes` (shared prefix) + `KV_cpu_bytes` (per-beam suffix) per model. Prefix-on-GPU / suffix-on-CPU is a mechanism invariant, not a tunable knob. Prefix attention (GPU) and suffix attention (CPU) run concurrently, merged via online softmax (`merge_attn_states`).
3. **PCIe H2D: 100% weight prefetch**. No KV prefetch — weight prefetch strictly dominates at every scale. KV spill-to-CPU uses idle PCIe D2H.

The ~40× PCIe:GPU ratio (BF16, small-batch decode) is the fundamental constraint — pure prefetch cannot hide latency for BF16.

## Repository Structure

| Directory | Purpose |
|---|---|
| `FastTTS-AE/` | Original FastTTS baseline (untouched). Uses vllm 0.9.2 from PyPI. |
| `FastTTS-thesis/` | Modified FastTTS for thesis work. No version pins — uses vllm fork. |
| `vllm/` | Fork of vLLM (latest main). **Primary modification target.** Thesis changes on `thesis` branch. |
| `David/` | Working notes and experiments for the thesis. |
| `Offloading_Frameworks/` | Reference implementations |
| `Papers/` | Reference Papers |

## Development Environment

Docker-based on `nvidia/cuda:12.4.1-devel-ubuntu22.04` with Miniconda:

```bash
# Build the image
docker build -t davidshiao55_ttc_env .

# Run container (mounts repo at /TTC, model weights at /models)
./docker_run.sh

# First-time env setup (inside container)
./setup_env.sh
```

Model weights are stored on the host at `/home/davidshiao55/models/huggingface` and mounted into the container at `/models`. `HF_HOME=/models/huggingface` is set automatically.

### Conda Environments

Two environments are created by `setup_env.sh`:

| Environment | FastTTS | vLLM | Use for |
|---|---|---|---|
| `baseline` | `FastTTS-AE/` | 0.9.2 from PyPI | Reproducing original paper results |
| `thesis` | `FastTTS-thesis/` | `/TTC/vllm` fork | All thesis experiments |

```bash
conda activate baseline   # original FastTTS + vllm 0.9.2
conda activate thesis     # modified FastTTS + vllm fork
```

### vLLM Fork Branches

```
main    → unmodified upstream (reference / sanity check)
thesis  → thesis modifications (offloading work)
```

Switch branches to toggle between unmodified and modified vllm within the `thesis` conda env — editable install picks up changes immediately for Python-only modifications. CUDA kernel changes require a rebuild (see Development Workflow below).

### Three Experimental Combinations

All three can be run from the `thesis` env using feature flags:

```bash
conda activate thesis
python run_all_experiments.py --offload=none   # FastTTS-thesis + unmodified vllm (combo 2)
python run_all_experiments.py --offload=full   # FastTTS-thesis + thesis vllm (combo 3)

conda activate baseline
python run_all_experiments.py                  # original FastTTS + vllm 0.9.2 (combo 1)
```

### Development Workflow

**One-time setup** (inside container):
```bash
./setup_env.sh
```
This creates both envs. For the `thesis` env it:
1. Does a fast `VLLM_USE_PRECOMPILED=1` pip install to resolve all Python dependencies
2. Generates `CMakeUserPresets.json` via `tools/generate_cmake_presets.py` (auto-detects nvcc, Python, CPU cores)
3. Runs a full `cmake --build ... --target install` to compile CUDA kernels with ccache

**Python-only changes** (`.py` files in `vllm/` or `FastTTS-thesis/`):
- Nothing to do — editable installs pick up changes immediately.

**C++/CUDA changes** (`csrc/`):
```bash
conda activate thesis
./rebuild_vllm.sh   # cmake incremental build, ccache-backed — only recompiles changed files
```

**Running experiments:**
```bash
# Baseline
conda activate baseline
cd /TTC/FastTTS-AE && python run_all_experiments.py --exp --plot --dir /TTC/results/baseline

# Thesis
conda activate thesis
cd /TTC/FastTTS-thesis && python run_all_experiments.py --exp --plot --dir /TTC/results/thesis
```

> **Python path gotcha**: Never run `python` from `/TTC` when using the `thesis` env.
> Python adds `''` (CWD) to `sys.path`, so `/TTC/vllm/` is found as a namespace package named
> `vllm` — before the editable-install finder returns the real package at `/TTC/vllm/vllm/`.
> Always `cd /TTC/FastTTS-thesis` (or any directory without a `vllm` subdirectory) first.

### Profiling with Nsight Systems

`nsys` is installed at `/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys` and symlinked to `/usr/local/bin/nsys`, so it's on PATH in any shell or conda env. Use it whenever the question is "did this actually overlap?", "which engine ran this memcpy?", or "where is the GPU stalled?".

Annotate phases of interest with NVTX so they're easy to spot in the timeline:

```python
import torch.cuda.nvtx as nvtx
nvtx.range_push("prefetch_layer_5")
# ... work ...
nvtx.range_pop()
```

Common invocations (from any directory):

```bash
# CUDA + NVTX trace, output report next to the script
nsys profile -o trace.nsys-rep --trace=cuda,nvtx --force-overwrite=true \
    python my_benchmark.py

# Per-memcpy timeline (rows tagged Pinned↔Device, with Stream ID, duration, throughput).
# Useful for engine-attribution questions and overlap verification.
nsys stats trace.nsys-rep --report gputrace | head -80

# Aggregate kernel/memcpy summaries
nsys stats trace.nsys-rep --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum

# Open the timeline GUI on the host machine (forwards via X11 / XQuartz / etc.)
nsys-ui trace.nsys-rep   # if a GUI is available
```

`.nsys-rep` files can be opened in the Nsight Systems GUI on the host (the repo is bind-mounted, so no copy needed). For dense traces, use small focused probes (e.g., `David/Benchmarks/phase0/probe_engines.py`) rather than the full bench trace — easier to visually inspect a single behavior.

**Profiling vLLM with nsys**: vLLM's V1 engine spawns a worker subprocess, so naive `nsys profile python script.py` only captures the parent and misses all CUDA activity. Per [vLLM's profiling docs](https://docs.vllm.ai/en/stable/contributing/profiling/), set `VLLM_WORKER_MULTIPROC_METHOD=spawn` and pass `--trace-fork-before-exec=true --cuda-graph-trace=node` to nsys so it follows into the engine subprocess:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile \
    -o trace --trace=cuda,nvtx,osrt \
    --trace-fork-before-exec=true --cuda-graph-trace=node \
    --force-overwrite=true \
    python my_vllm_script.py
```

## FastTTS Architecture

**Two-model system**: A generator LLM produces candidate solutions step-by-step; a verifier (PRM) scores each step to guide search.

- `fasttts.py` / `core.py` — Main `FastTTS` class. Orchestrates generator + verifier through search.
- `config.py` — `FastTTSConfig` (model configs, vLLM engine params) and `SearchConfig` (search-specific params like beam width, n, approach).
- `models/vllm_wrapper.py` — Wraps vLLM engines for generator and verifier.
- `models/generator_engine.py`, `verifier_engine.py` — Engine implementations.
- `search/` — Search strategies: `beam_search`, `dvts`, `best_of_n`, `dynamic_branching`, `vg_search`.

Default models: `Qwen/Qwen2.5-Math-1.5B-Instruct` (generator), `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` (verifier).

Results go to `/TTC/results/`. Figures: `main_results_combined.pdf`, `latency_combined.pdf`, `acc.pdf`.

## Key vLLM Files for Offloading Work

These are the existing building blocks in vLLM relevant to the thesis:

- **Weight offloading**: `vllm/model_executor/offloader/prefetch.py` (PrefetchOffloader), `uva.py` (UVA offloader)
- **Cascade attention**: `vllm/v1/attention/backends/flash_attn.py` (line ~1038, `cascade_attention()`)
- **Attention merge**: `vllm/v1/attention/ops/merge_attn_states.py` (online softmax merge)
- **CPU attention**: `vllm/v1/attention/backends/cpu_attn.py`
- **KV cache offload**: `vllm/v1/kv_offload/` (CPU-GPU KV cache management)
- **Fused QKV projection**: `vllm/model_executor/layers/linear.py` (`QKVParallelLinear`)
- **FFN layers**: `MergedColumnParallelLinear` (gate_up), `RowParallelLinear` (down) in same file

## Implementation Phases

See `David/Docs/implementation_roadmap.md` for the full plan. Summary:

0. **Pre-Implementation Benchmarking** — Validate CPU GEMM throughput, GPU layer timing, PCIe bandwidth, CPU attention latency. Gates all subsequent phases.
1. **Resident Hybrid Weight Split (all sub-modules)** — Unified column-parallel split for WQKV, WO, MLP1, MLP2. WQKV uses K/V-biased column assignment. Python `CpuComputeDispatcher` prototype + `enforce_eager=True`. Planner emits `f_cpu_store` only (single-entry dispatch). Frees ~1.2 GB (7B) at the phase0-observed ~9% split.
2. **Attention Offloading** — CPU suffix attention + online softmax merge. Requires CPU attention kernel with per-head LSE. Two-pool KV allocated by Planner (`KV_gpu_bytes`, `KV_cpu_bytes`).
3. **Tensor-Granularity Prefetch** — Per-sub-module three-way dispatch (permanent/prefetch/cpu) with sub-layer pipeline. Planner gains `f_prefetch_compute` axis per `BatchDescriptor`. Enables 14B+.
4. **End-to-End Benchmarking** — RTX 4090: 7B / 14B across all configurations.
5. **CUDA Graph Integration** — Port KTransformers `CPUInfer` + `cudaLaunchHostFunc` pattern. Swap `CpuComputeDispatcher` internals; forward pass unchanged.

### Engineering Gaps (not yet in vLLM)
- Column-parallel weight split + CPU matmul + partial result concat at tensor granularity
- CPU attention kernel returning per-head LSE values
- `cudaLaunchHostFunc` glue for CUDA Graph + CPU task co-scheduling (Phase 5)

## Key Technical Constraints

- Target BF16 weights (no quantization) — PCIe:GPU ratio ~40× makes pure prefetch infeasible.
- Placement fractions are Planner outputs, not universal constants. `f_cpu_store` is per-model (load-time); `f_cpu_compute` is per-`BatchDescriptor` (per-bucket). `~9%` is the observed optimum at B=1 decode on RTX 4090 + Qwen2.5-7B BF16; see `David/Docs/phase0_findings.md`.
- Column-parallel weight splits are mathematically exact — verify with unit tests.
- All PCIe H2D bandwidth goes to weight prefetch; no KV prefetch (see `David/Docs/pcie_bandwidth_allocation_design.md`).
- CUDA Graph compatibility via `cudaLaunchHostFunc` is deferred to Phase 5; prototype with `enforce_eager=True`. `CpuComputeDispatcher` abstraction ensures the retrofit is localized.

## Design Documents

Detailed analysis lives in `David/Docs/`:

| Document | Scope |
|---|---|
| `thesis_proposal.md` | Full proposal: problem, system architecture, offloading strategy |
| `profiler_design.md` | Profile schema, methodology, caching |
| `planner_design.md` | Inputs/outputs, constraints, objective, solution method (primary contribution) |
| `scheduler_design.md` | Tier-aware admission, KV migration, dispatch lookup |
| `implementation_roadmap.md` | Phased implementation plan, phase-to-component mapping |
| `weight_offload_design.md` | Storage-vs-compute separation, tensor granularity, buffer sizing |
| `attention_offload_design.md` | Two-pool KV, CPU suffix attention, batch size tradeoff |
| `pcie_bandwidth_allocation_design.md` | Why all PCIe goes to weight prefetch |
| `phase0_findings.md` | First-iteration Profiler output on RTX 4090 + Qwen2.5-7B |
