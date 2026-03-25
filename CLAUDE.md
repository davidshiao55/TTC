# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master thesis project on **Test-Time Compute (TTC)** optimization for single consumer GPU-CPU systems (target: NVIDIA RTX 4090, 24 GB VRAM). The thesis extends FastTTS (a test-time search framework) with **weight offloading** and **attention offloading** to overcome GPU memory limitations.

### Core Problem
FastTTS assumes model weights stay GPU-resident, which limits max model size to VRAM and bounds batch size/throughput by leftover GPU memory. This project introduces hybrid CPU-GPU offloading to break these constraints.

### Proposed Approach (Three-Pronged Offloading)
1. **Attention Offloading**: Shared prefix KV cache + attention on GPU; per-beam suffix KV cache + attention on CPU. Merged via online softmax (exact, no approximation).
2. **Weight Prefetch Offloading**: Stream weights layer-by-layer from CPU to GPU via PCIe with double-buffering.
3. **Hybrid Weight Computation**: Split weight matrices — compute ~90% on GPU (prefetched), ~10% on CPU (in-place). This solves the PCIe bottleneck that makes pure prefetch catastrophically slow for BF16.

## Repository Structure

| Directory | Purpose |
|---|---|
| `FastTTS-AE/` | FastTTS artifact evaluation codebase (the baseline). Contains the paper (`FastTTS.pdf`). |
| `vllm/` | Fork of vLLM — the inference engine FastTTS is built on. **Primary modification target.** |
| `David/` | Working notes and experiments for the thesis. |
| `Offloading_Frameworks/` | Reference implementations: FlexLLMGen, llama.cpp, NEO, PowerInfer. |
| `Offloading_Papers/` | Related papers: Doppeladler, FlexGen, NEO, PowerInfer, TwinPilots. |

## Development Environment

Docker-based on `nvidia/cuda:12.1.1-devel-ubuntu22.04`:

```bash
# Build the image
docker build -t davidshiao55_ttc_env .

# Run container with GPU access, mounting repo at /TTC
./docker_run.sh
```

### FastTTS Setup (inside container or conda)

```bash
cd FastTTS-AE
conda env create -f environment.yml && conda activate FastTTS
pip install -e .
cd modified-skywork-o1-prm-inference && pip install -e . && cd ..
```

### vLLM Setup

```bash
cd vllm
pip install -e .
```

## FastTTS Architecture

**Two-model system**: A generator LLM produces candidate solutions step-by-step; a verifier (PRM) scores each step to guide search.

- `fasttts.py` / `core.py` — Main `FastTTS` class. Orchestrates generator + verifier through search.
- `config.py` — `FastTTSConfig` (model configs, vLLM engine params) and `SearchConfig` (search-specific params like beam width, n, approach).
- `models/vllm_wrapper.py` — Wraps vLLM engines for generator and verifier.
- `models/generator_engine.py`, `verifier_engine.py` — Engine implementations.
- `search/` — Search strategies: `beam_search`, `dvts`, `best_of_n`, `dynamic_branching`, `vg_search`.

Default models: `Qwen/Qwen2.5-Math-1.5B-Instruct` (generator), `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` (verifier).

### Running FastTTS Experiments

```bash
cd FastTTS-AE

# Run all experiments (3 model combos x 2 datasets x 2 methods x 7 n-values)
python run_all_experiments.py --exp

# Generate plots from results
python run_all_experiments.py --plot

# Both
python run_all_experiments.py --exp --plot --dir /path/to/results
```

Results go to `benchmarks/benchmark_results/`. Figures: `main_results_combined.pdf`, `latency_combined.pdf`, `acc.pdf`.

## Key vLLM Files for Offloading Work

These are the existing building blocks in vLLM relevant to the thesis:

- **Weight offloading**: `vllm/model_executor/offloader/prefetch.py` (PrefetchOffloader), `uva.py` (UVA offloader)
- **Cascade attention**: `vllm/v1/attention/backends/flash_attn.py` (line ~1038, `cascade_attention()`)
- **Attention merge**: `vllm/v1/attention/ops/merge_attn_states.py` (online softmax merge)
- **CPU attention**: `vllm/v1/attention/backends/cpu_attn.py`
- **KV cache offload**: `vllm/v1/kv_offload/` (CPU-GPU KV cache management)
- **Fused QKV projection**: `vllm/model_executor/layers/linear.py` (`QKVParallelLinear`)
- **FFN layers**: `MergedColumnParallelLinear` (gate_up), `RowParallelLinear` (down) in same file

## Implementation Phases (from analysis doc)

1. **Hybrid FFN Weight Computation** — Column-parallel weight split for FFN, CPU-side matmul, partial result transfer + concat. Highest impact: frees ~12 GB GPU memory at ~10% latency cost.
2. **CPU Attention with LSE Support** — Modify CPU attention kernel to return per-head LSE values.
3. **Hybrid Attention Backend** — Extend `cascade_attention()` for CPU suffix dispatch.
4. **CPU-Resident KV Cache** — Suffix KV blocks on CPU without GPU round-trip.
5. **Scheduler Integration** — Coordinate weight placement, KV placement, and CPU thread pool.
6. (Optional) **Unfused WK/WV** — Only if memory is still the bottleneck after phases 1-5.

## Key Technical Constraints

- Target BF16 weights (no quantization) — this makes pure PCIe prefetch infeasible (46x slower than GPU compute per layer).
- Optimal CPU weight fraction (`f_cpu`) is ~10% for L=32 models — balances GPU, CPU, and PCIe utilization.
- Activation transfers between CPU and GPU are tiny (<300 KB/layer) and use opposite PCIe direction from weight prefetch (full duplex).
- Column-parallel weight splits produce mathematically identical results — verify with unit tests.
