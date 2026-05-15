# AGENTS.md

Project-level guidance for Codex, Claude Code, and other coding agents.
`CLAUDE.md` imports this file with `@AGENTS.md`; keep shared guidance here
instead of duplicating it.

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
docker build -t davidshiao55_ttc_env .
./docker_run.sh
./setup_env.sh
```

Model weights are stored on the host at `/home/davidshiao55/models/huggingface`
and mounted at `/models`; `HF_HOME=/models/huggingface` is set automatically.

| Environment | FastTTS | vLLM | Use for |
|---|---|---|---|
| `baseline` | `FastTTS-AE/` | 0.9.2 from PyPI | Reproducing original paper results |
| `thesis` | `FastTTS-thesis/` | `/TTC/vllm` fork | All thesis experiments |

`vllm/` branches: `main` is the upstream reference; `thesis` contains thesis
offloading work. Editable installs pick up Python-only changes immediately.
For C++/CUDA changes under `vllm/csrc/`, run:

```bash
conda activate thesis
/TTC/rebuild_vllm.sh
```

Run experiments from the package directories, not from `/TTC`:

```bash
conda activate baseline
cd /TTC/FastTTS-AE && python run_all_experiments.py --exp --plot --dir /TTC/results/baseline

conda activate thesis
cd /TTC/FastTTS-thesis && python run_all_experiments.py --exp --plot --dir /TTC/results/thesis
python run_all_experiments.py --offload=none
python run_all_experiments.py --offload=full
```

> **Python path gotcha**: Never run `python` from `/TTC` when using the
> `thesis` env. Python may resolve `/TTC/vllm/` as a namespace package before
> the editable install resolves `/TTC/vllm/vllm/`.

### Profiling with Nsight Systems

Use `nsys` for overlap, memcpy engine attribution, and GPU-stall questions.
Annotate interesting phases with NVTX. Common commands:

```bash
nsys profile -o trace.nsys-rep --trace=cuda,nvtx --force-overwrite=true \
    python my_benchmark.py
nsys stats trace.nsys-rep --report gputrace | head -80
nsys stats trace.nsys-rep --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum
```

For vLLM V1 worker subprocess profiling:

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

When working from inside `vllm/`, also read `vllm/AGENTS.md`. For local TTC
thesis experiments, use the conda workflow above. For upstream vLLM PR work,
use the upstream `uv` workflow in `vllm/AGENTS.md`.

## Implementation Phases

See `David/Docs/implementation_roadmap.md` for the full plan.

- Phase 0: pre-implementation benchmarks complete.
- Phase 1a/1b/1c: collaborative weight offload complete.
- Phase 2: attention offloading, CPU suffix attention, and online softmax merge.
- Phase 3: end-to-end RTX 4090 benchmarking for 7B and 14B configurations.

### Engineering Gaps (not yet in vLLM)
- ~~Column-parallel weight split + CPU matmul + partial result concat at tensor granularity~~ [LANDED in Phase 1a/1b]
- ~~`cudaLaunchHostFunc` glue for CUDA Graph + CPU task co-scheduling~~ [LANDED in Phase 1c]
- CPU attention kernel returning per-head LSE values (Phase 2)

## Key Technical Constraints

- Target BF16 weights (no quantization) — PCIe:GPU ratio ~40× makes pure prefetch infeasible.
- Placement fractions are Planner outputs, not universal constants. `f_cpu_store` is per-model (load-time); `f_cpu_compute` is per-`BatchDescriptor` (per-bucket). `~9%` is the observed optimum at B=1 decode on RTX 4090 + Qwen2.5-7B BF16; see `David/Docs/phase0_findings.md`.
- Column-parallel weight splits are mathematically exact — verify with unit tests.
- All PCIe H2D bandwidth goes to weight prefetch; no KV prefetch (see `David/Docs/pcie_bandwidth_allocation_design.md`).
- CUDA Graph compatibility lands via `cudaLaunchHostFunc` (Phase 1c). Native runner is the post-Phase-1c default (`CotsOffloadConfig.cpu_runner = "native"`); supports `enforce_eager=False`. Python runner kept as kill-switch under `enforce_eager=True` for A/B diagnostics; deprecation one quarter post-Phase-1c.

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
| `phase1_findings.md` | Phase 1 source of truth: final production COTS path + current results |
| `phase1a_findings.md` | Appendix: static split prototype facts that survived |
| `phase1b_findings.md` | Appendix: three-way dispatch and layer-ahead prefetch decisions |
| `phase1c_findings.md` | Appendix: native CPU runner and graph-compatible runtime |
| `phase1_analysis_findings.md` | Appendix: free-regime and KV-throughput result tables |
