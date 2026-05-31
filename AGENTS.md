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
scripts/ttc-docker-run.sh
docker exec -it davidshiao55_ttc bash
cd /TTC && scripts/setup-env.sh
```

Model weights are stored on the host at `/home/davidshiao55/models/huggingface`
and mounted at `/models`; `HF_HOME=/models/huggingface` is set automatically.

### Codex / Agent Workflow

Prefer **host edits + Docker execution**:

- Run Codex from the host repo at `/home/davidshiao55/TTC`, not from inside
  the container. This keeps `apply_patch` reliable and avoids Docker namespace
  ownership issues.
- Run project commands inside Docker via `scripts/ttc-docker-env.sh`. The
  helper uses the existing `davidshiao55_ttc` container, activates the requested
  conda env, runs as the host UID/GID, and sets writable cache env vars for
  PyTorch/vLLM.
- Keep source edits on the host side. Use Docker for builds, tests, profiling,
  and experiments.
- For vLLM commits, run Git from Docker via `scripts/ttc-docker-env.sh` so the
  installed hooks use the container's `thesis` env. Host-side commits can fail
  because the hook path points at `/opt/conda/envs/thesis` inside Docker.

Common commands:

```bash
# Start a fresh container from the repo root.
scripts/ttc-docker-run.sh

# Run a thesis command in the container.
scripts/ttc-docker-env.sh thesis 'cd /tmp && python -c "import vllm; print(vllm.__version__)"'

# Run vLLM tests from the correct package directory.
TTC_DOCKER_WORKDIR=/TTC/vllm scripts/ttc-docker-env.sh thesis \
  'pytest tests/v1/worker/test_cots_hybrid_kv.py -q'

# Commit vLLM changes with the container's hook environment.
TTC_DOCKER_WORKDIR=/TTC/vllm scripts/ttc-docker-env.sh thesis \
  'git commit -m "Describe vLLM change"'

# Open an interactive shell with the thesis env activated.
scripts/ttc-docker-env.sh thesis
```

Keep operational shell entrypoints in `scripts/`. Do not add root-level
compatibility wrappers for new scripts.

| Environment | FastTTS | vLLM | Use for |
|---|---|---|---|
| `baseline` | `FastTTS-AE/` | 0.9.2 from PyPI | Reproducing original paper results |
| `thesis` | `FastTTS-thesis/` | `/TTC/vllm` fork | All thesis experiments |

`vllm/` branches: `main` is the upstream reference; `thesis` contains thesis
offloading work. Editable installs pick up Python-only changes immediately.
For C++/CUDA changes under `vllm/csrc/`, run:

```bash
conda activate thesis
/TTC/scripts/rebuild-vllm.sh
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

> **CMake path gotcha**: For C++/CUDA rebuilds, make sure
> `conda activate thesis` has actually taken effect before `/TTC/scripts/rebuild-vllm.sh`.
> The script calls `cmake` from `PATH`; without the thesis env it may pick up
> `/usr/bin/cmake`, which is too old for current vLLM presets.

> **pytest fork gotcha**: Some vLLM config tests rely on `pytest-forked` for
> CUDA-process isolation. If `pytest.mark.forked` is reported as unknown, a
> full-file run such as `tests/compile/test_config.py` may retain GPU memory
> across cases; install `pytest-forked` or rerun affected node IDs in fresh
> pytest processes.

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

These are the production COTS surfaces plus upstream reference points most
relevant to thesis work:

- **COTS config and graph policy**: `vllm/config/offload.py`,
  `vllm/config/vllm.py`, `vllm/engine/arg_utils.py`, and
  `tests/compile/test_config.py`.
- **Weight offloading runtime**: `vllm/model_executor/offloader/cots_offloader.py`,
  `cots_runners.py`, `cots_ops.py`, `cots_storage.py`, and C++ pieces under
  `csrc/cots/` including `cots_weight_task_runner.{h,cpp}`, `task_queue.*`,
  and `cots_wait_done_kernel.cu`.
- **Hybrid KV / suffix attention**: `vllm/v1/worker/cots_hybrid_kv.py`,
  `vllm/v1/worker/gpu_model_runner.py` integration points,
  `vllm/v1/attention/backends/cots_hybrid_attention.py`,
  `cots_suffix_attention_ops.py`, and C++ pieces under `csrc/cots/` including
  `cots_suffix_attention_task_runner.{h,cpp}` and `cots_common.h`.
- **COTS tests**: Phase 1 tests under `David/Tests/phase1a/` and
  `David/Tests/phase1c/`; Phase 2 kernel and worker tests under
  `vllm/tests/kernels/attention/test_cots_*` and
  `vllm/tests/v1/worker/test_cots_hybrid_kv.py`.
- **Upstream attention references**: `vllm/v1/attention/backends/flash_attn.py`
  for cascade attention, `vllm/v1/attention/ops/merge_attn_states.py` for
  online softmax merge, and `vllm/v1/attention/backends/cpu_attn.py` for the
  upstream CPU attention backend.
- **Upstream weight-offload references**: `vllm/model_executor/offloader/prefetch.py`
  and `uva.py`.
- **Layer split entry points**: `vllm/model_executor/layers/linear.py`
  (`QKVParallelLinear`, `MergedColumnParallelLinear`, `RowParallelLinear`).

When working from inside `vllm/`, also read `vllm/AGENTS.md`. For local TTC
thesis experiments, use the conda workflow above. For upstream vLLM PR work,
use the upstream `uv` workflow in `vllm/AGENTS.md`.

## Implementation Phases

See `David/Docs/implementation_roadmap.md` for the full plan.

- Phase 0: pre-implementation benchmarks complete.
- Phase 1a/1b/1c: collaborative weight offload complete.
- Phase 2: hybrid CPU/GPU KV implementation generally complete; throughput policy remains profile-gated.
- Phase 3: end-to-end RTX 4090 benchmarking for 7B and 14B configurations.

### Landed Thesis Pieces and Remaining Policy Work

- ~~Column-parallel weight split + CPU matmul + partial result concat at tensor granularity~~ [LANDED in Phase 1a/1b]
- ~~`cudaLaunchHostFunc` glue for CUDA Graph + CPU task co-scheduling~~ [LANDED in Phase 1c]
- ~~CPU suffix attention kernel returning output plus per-head LSE values~~ [LANDED in Phase 2]
- Phase 2 implementation is generally done: CPU suffix attention, GPU prefix
  attention, and online-softmax merge work for the current Qwen and Llama GQA
  envelopes.
- Remaining Phase 2 policy work belongs in the Planner: choose hybrid KV only
  when profiled CPU KV suffix cost is smaller than saved weight-offload cost,
  enabled model fit, or useful scheduler-wave/batch gain. Do not assume lower
  GPU memory or more CPU KV capacity automatically improves throughput.

## Key Technical Constraints

- Target BF16 weights (no quantization) — PCIe:GPU ratio ~40× makes pure prefetch infeasible.
- Placement fractions are Planner outputs, not universal constants. `f_cpu_store` is per-model (load-time); `f_cpu_compute` is per-`BatchDescriptor` (per-bucket). `~9%` is the observed optimum at B=1 decode on RTX 4090 + Qwen2.5-7B BF16; see `David/Docs/phase0_findings.md`.
- Column-parallel weight splits are mathematically exact — verify with unit tests.
- All PCIe H2D bandwidth goes to weight prefetch; no KV prefetch (see `David/Docs/pcie_bandwidth_allocation_design.md`).
- CUDA Graph compatibility is landed for the native COTS runner. Phase 1 weight offload uses piecewise graphs with COTS weight submit/sync split points and `weight_capture_sync_mode="wait_kernel"` in graph mode. Phase 2 hybrid KV uses normal piecewise attention boundaries and does not add Phase 1 weight split points unless weight offload is also active. The Python weight runner remains an eager-only A/B kill switch, not the production path.
- COTS cleanup convention: use the current names only (`weight_capture_sync_mode`, `CotsWeightTaskRunner`, `CotsSuffixAttentionTaskRunner`, `NativeCotsSuffixAttentionRunner`). Do not add deprecated compatibility aliases unless explicitly requested.

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
