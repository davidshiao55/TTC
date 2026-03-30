# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master thesis project on **Test-Time Compute (TTC)** optimization for single consumer GPU-CPU systems (target: NVIDIA RTX 4090, 24 GB VRAM). The thesis extends FastTTS (a test-time search framework) with **weight offloading** and **attention offloading** to overcome GPU memory limitations.

### Core Problem
FastTTS assumes model weights stay GPU-resident, which limits max model size to VRAM and bounds batch size/throughput by leftover GPU memory. This project introduces hybrid CPU-GPU offloading to break these constraints.

### Proposed Approach (Three-Pronged Offloading)
1. **Attention Offloading**: Shared prefix KV cache + attention on GPU; per-beam suffix KV cache + attention on CPU. Merged via online softmax (exact, no approximation). Batch size is the control variable for CPU attention bottleneck.
2. **Hybrid Weight Computation**: Column-parallel split — f_cpu≈9% computed on CPU in parallel with GPU. At 9%, CPU compute hides within GPU idle time (GPU is memory-BW-bound), so the split is effectively free. Universally applicable. Saves ~1.2 GB for 7B.
3. **PCIe Weight Prefetch**: Three-way split per sub-module: `W = [W_gpu_permanent | W_gpu_prefetched | W_cpu]`. All PCIe H2D bandwidth dedicated to weight prefetch (no KV prefetch). Small models: replaces f_gpu to free GPU memory. Large models: replaces f_cpu to reduce latency.

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

0. **CUDA Graph Integration** — `cudaLaunchHostFunc` callbacks to embed CPU task submission in CUDA Graphs (KTransformers approach). Done first so all subsequent phases are CUDA-Graph-compatible from the start rather than needing a retrofit.
1. **Resident Hybrid** — f_cpu≈9% column-parallel split on all layers. Zero latency cost; frees ~1.2 GB (7B).
2. **Attention Offloading** — Suffix KV → CPU; CPU attention kernel returning per-head LSE; online softmax merge via `merge_attn_states.py`.
3. **Tensor-Granularity Offloading** — Per-sub-module three-way split (permanent/prefetch/cpu) with sub-layer pipeline. Enables 14B+ models.
4. **Benchmarking** — RTX 4090: 7B / 14B / 32B across all configurations.

### Engineering Gaps (not yet in vLLM)
- `cudaLaunchHostFunc` glue for CUDA Graph + CPU task co-scheduling
- Column-parallel weight split + CPU matmul + partial result concat at tensor granularity
- CPU attention kernel returning per-head LSE values

## Key Technical Constraints

- Target BF16 weights (no quantization) — PCIe:GPU ratio ~40× makes pure prefetch infeasible.
- Optimal resident `f_cpu` ≈ 9% — universal across all model sizes; CPU compute hides within GPU idle time (GPU is memory-BW-bound). Beyond ~10%, latency increases sharply.
- Column-parallel weight splits are mathematically exact — verify with unit tests.
- All PCIe H2D bandwidth goes to weight prefetch; no KV prefetch (see `David/Docs/pcie_bandwidth_allocation_design.md`).
- CUDA Graph compatibility via `cudaLaunchHostFunc` is implemented first (Phase 0); all offloading phases build on top of it.

## Design Documents

Detailed analysis lives in `David/Docs/`:

| Document | Scope |
|---|---|
| `thesis_proposal.md` | Full proposal: problem, approach, analysis, roadmap |
| `weight_offload_design.md` | Granularity comparison, three-way split, sub-layer pipeline, buffer sizing |
| `attention_offload_design.md` | KV topology split, CPU suffix attention, batch size tradeoff |
| `pcie_bandwidth_allocation_design.md` | Why all PCIe goes to weight prefetch |
