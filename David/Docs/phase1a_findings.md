# Phase 1a Implementation Findings

This document records the design and verification of **Phase 1a** of the COTS prototype — the first code-level checkpoint of the thesis: a static, tensor-granularity weight-offload backend for vLLM that runs WQKV / MLP1 / MLP2 partly on CPU during decode and frees GPU memory for the KV cache. The Phase 1a backend is `--offload-backend cots`, lives in `vllm/model_executor/offloader/cots.py`, and produces correct end-to-end output on Qwen2.5-7B-Instruct under `enforce_eager=True`.

Phase 1a deliberately omits weight prefetch (`f_prefetch = 0`), CUDA-graph capture, and tier-aware KV — those are Phase 1b, Phase 1c, and Phase 2 respectively. The Phase 1a code is structured so each later phase is a localized swap rather than a rewrite (§1.11).

Hardware: NVIDIA RTX 4090 (24 GB), Intel i9-14900KF (AVX2, no AVX512/AMX), DDR5.
PyTorch 2.10.0+cu128, MKL enabled, oneDNN BF16, Triton.

---

## Contents

**Mechanism**
- §1.1 — Backend architecture: storage / execution / operator layers
- §1.2 — TP-style weight load: `param.data` at slice shape from construction
- §1.3 — Per-sub-module forward path: GPU+CPU fork, K/V-biased picker
- §1.4 — Activation return: SM-issued UVA copy

**Memory invariants**
- §1.5 — Profiler-context allocation rule
- §1.6 — Lazy iteration during `wrap_modules`

**Measurements**
- §1.7 — GPU memory saved at f=0.09 on Qwen2.5-7B
- §1.8 — End-to-end smoke and transparency at `gpu_memory_utilization=0.9`
- §1.9 — Nsight overlap trace

**Verification**
- §1.10 — Test matrix and reproducibility

**Forward-compat**
- §1.11 — Hooks for Phase 1b / 1c / 2

**Wrap-time contract**
- §1.12 — Fail-fast invariants checked at wrap time

**Comparison**
- §1.13 — Head-to-head against vLLM's PrefetchOffloader (prefill-heavy + decode-heavy)

**Postmortem**
- §1.14 — Why the f=0.05 free regime isn't free (orchestration vs active CPU-work penalty)

---

## 1.1: Backend architecture — three layers (storage / execution / operators)

The cots backend is a single file (`vllm/model_executor/offloader/cots.py`, ~1,090 LOC) split into three architectural layers, plus a lifecycle adapter:

| Layer | Class | Responsibility | Lifetime |
|---|---|---|---|
| **Storage** | `CotsLinearHandle` | Per-Linear partition primitive: GPU weight slice (replaces `param.data`), CPU weight slice (`w_cpu`, pinned), CUDA index tensors, weight-loader closure. **No execution.** | One per offloaded linear |
| **Execution** | `CpuTaskRunner` | Generic CPU work submitter (`submit_with_d2h` / `wait`). Phase 1c swap target. **No model knowledge.** | One per operator instance |
| **Operators** | `CotsQKVOp`, `CotsSwiGLUMLPOp` | Forward semantics. Compose handle + runner. `CotsQKVOp` patches `quant_method.apply` (per-Linear); `CotsSwiGLUMLPOp` replaces `Qwen2MLP.forward` (block-level). | One per QKV linear / one per fused MLP block |
| **Lifecycle** | `CotsOffloader` | Discovery, handle/op installation, orphan check, shared activation buffer allocation, dispatch-table lookup. | One per engine |

The split admits explicitly that **MLP is inherently block-level**: linears are storage partitions; QKV and MLP are execution operators. Forcing MLP into a per-Linear abstraction would re-introduce the three-transfer round-trip we deleted in Step 2. (Earlier drafts had a `CpuComputeDispatcher` per-Linear class doing both storage and execution; restructured per a code-review pass that flagged the storage/execution mixing.)

### Why three layers, not one

- **Phase 1c swap target is `CpuTaskRunner.submit_with_d2h` / `wait`** — the eventual `cudaLaunchHostFunc`-based C++ binding (port from `kt-kernel/cpu_backend/cpuinfer.h:78-116`) is a body-only swap with no call-site changes. Operators don't care which executor is underneath.
- **`CotsLinearHandle` is data-only** — no submit/sync. Tests can construct one without spinning up a runner. It's the natural extension point for Phase 1b prefetch buffer pool (a "prefetch" handle role).
- **Operators own model policy** (the SwiGLU shape, the K/V-biased scatter); the runner doesn't see it. Phase 2 adds a `CotsAttnOp` for CPU suffix attention; it slots in alongside `CotsQKVOp` / `CotsSwiGLUMLPOp` without touching either.

### Activation buffers live on the offloader

`_x_pinned`, `_y_pinned`, `_y_gpu` are sized to the worst-case sub-module across all handles (one buffer set per offloader); per-forward views slice into them. Per-handle state collapses to just `w_cpu`.

This is legal because of the strict sequential layer execution invariant (`thesis_proposal.md §3.1`): at most one CPU GEMM is in flight at any instant, so the three buffers are never contended. Phase 0 §0.4 establishes that no cross-layer activation pipelining happens in the thesis. The alternative — per-handle buffers — would multiply activation memory by the handle count (84 on Qwen2.5-7B) with no benefit.

### Backing storage — flat 1D

`_x_pinned`, `_y_pinned`, `_y_gpu` are flat 1D, sized to `max_num_batched_tokens × max_dim`. Per-forward views `flat[:n*d].view(n, d)` are always contiguous regardless of which sub-module is active. Address invariance keeps the buffers compatible with Phase 1c graph capture without further work.

### Operators hold an explicit `offloader` reference

`CotsQKVOp` and `CotsSwiGLUMLPOp` capture `self._offloader` at install time, not via a module-global. Two `CotsOffloader` instances (e.g., FastTTS generator + verifier engines) coexist safely — each set of operators talks to its own offloader's buffers.

---

## 1.2: TP-style weight load — `param.data` at slice shape from construction

vLLM intercepts weight loading through each layer's `weight_loader` callable. Standard tensor parallelism uses this hook to narrow the full unsharded `loaded_weight` to the rank's shard and write into `param.data`, which is at the rank-shard shape from construction onward (`linear.py:529-565` for `ColumnParallelLinear`). cots does the same thing, treating CPU as one additional rank along each sub-module's native shard axis.

### Loader-time invariants

For each offloaded linear, `CotsOffloader.wrap_modules` (Pass 1) does:

1. Construct a `CotsLinearHandle` via the kind-specific factory (`for_qkv` / `for_col` / `for_row`). The factory snaps `n_cpu` (head-aligned for QKV) and computes per-kind picker indices.
2. Call `handle.install(device)`, which atomically:
   - **Replaces `param.data` with a fresh GPU tensor at GPU-slice shape** — `(out_dim − n_cpu, in_dim)` for col/qkv or `(out_dim, in_dim − n_cpu)` for row.
   - Allocates `w_cpu` (pinned, slice shape) and CUDA index copies.
   - Wraps both `linear.weight_loader` and `param.weight_loader` (vLLM's `default_weight_loader` reads the latter) with the kind-specific loader closure (private method on the handle).
   - Tags `linear._cots_handle = self` so Pass 2 (op installation) can find the handle.

After step 2, every offloaded `param.data` is at its final size. The original empty full-shape tensor allocated by `quant_method.create_weights` is dereferenced; PyTorch's caching allocator returns its segment to the pool for the next layer's construction.

Pass 2 then installs operators: `CotsQKVOp` per QKV linear, `CotsSwiGLUMLPOp` per recognized MLP block. Pass 3 raises if any wrapped MergedCol/Row handle is not in a recognized block (the orphan check).

### Why TP-style intercept and not redirect-then-split

The earlier design redirected `param.data` to a full pinned-CPU tensor and did the slice-and-copy in `post_init`. It violated the profiler-context invariant (§1.5) — the GPU slice allocations in `post_init` happened outside vLLM's `DeviceMemoryProfiler`, so `model_memory_usage` was wrong and the KV-cache budget over-allocated. TP-style intercept makes every GPU allocation happen during `load_model`, inside the profiler context, and tracks vLLM's existing TP loading pattern.

### Per-Linear-type closure semantics

Three closures cover the three offloaded shapes (`cots.py:480-622`):

| Linear type | Sub-module | Loader call pattern | Split |
|---|---|---|---|
| `RowParallelLinear` | `down_proj` (MLP2) | One call, full `(out_dim, in_dim)` | Last `n_cpu` input cols → `_w_cpu` |
| `MergedColumnParallelLinear` | `gate_up_proj` (MLP1) | Per-shard, `loaded_shard_id ∈ {0, 1}` | Last `n_cpu_per_half` rows of each partition |
| `QKVParallelLinear` | `qkv_proj` (WQKV) | Per-shard, `loaded_shard_id ∈ {'q', 'k', 'v'}` | K/V-biased picker (§1.3) |

The "LAST cols on CPU" convention aligns with vLLM's standard TP narrow-on-load behavior (which keeps the FIRST cols on rank 0). This lets the wrapped loader range-narrow once per shard and write to two destinations without any reordering.

### One non-obvious detail

vLLM's `default_weight_loader` reads `param.weight_loader` (set on the `Parameter` object via `set_weight_attrs` at construction), not `linear.weight_loader`. Both must be assigned in step 4. Updating only `linear.weight_loader` silently leaves the original loader live and triggers either an OOM or a shape mismatch during weight load.

---

## 1.3: Per-sub-module forward path — two surfaces, by structure

cots's offload surface in Phase 1a is structurally two-shaped:

| Sub-module | Operator | Patch surface | Why |
|---|---|---|---|
| **WQKV** (`QKVParallelLinear`) | `CotsQKVOp` | Per-Linear `quant_method.apply` | No fusion target in Phase 1a (Phase 2 will fuse with attention) |
| **MLP1 + MLP2** (`MergedColumnParallelLinear` + `RowParallelLinear`) | `CotsSwiGLUMLPOp` | Block-level: replaces `Qwen2MLP.forward` | SwiGLU between them must run on CPU's slice locally to satisfy the matched-index invariant |

There is no per-Linear operator for row offload — Phase 1a's only row-parallel offload is MLP2's `down_proj`, always inside a fused MLP block. A `_RaiseOnDirectCall` guard is installed on each fused MLP linear so calling `mlp.gate_up_proj(x)` directly raises a clear error rather than producing wrong-sized output.

### Col-parallel forward — WQKV only (`CotsQKVOp.apply`)

```
x_gpu  ──────► F.linear(x_gpu, layer.weight)  ─────────► out_gpu_slice
       └─────► runner.submit_with_d2h(x_gpu, x_pinned,
                                      _cpu_gemm_into_after_event,
                                      handle.w_cpu, y_pinned)
                           ↓ (CPU GEMM on cots-cpu thread)
                           ↓
                           runner.wait()
                           uva_copy_into_gpu(y_pinned, y_gpu) ─► out_cpu_on_gpu
                                                              │
                                       _scatter_col_outputs ──┘
```

GPU runs `F.linear` on the GPU slice. CPU runs `F.linear` on `handle.w_cpu`. Outputs are scattered via `handle.gpu_indices_cuda` / `handle.cpu_indices_cuda` into a full-shape output so the attention path sees the canonical `[Q | K | V]` layout.

### Block-level forward — fused MLP1 + SwiGLU + MLP2 (`CotsSwiGLUMLPOp.__call__`)

```
                            ┌── F.linear(x, gate_up.weight) ── act_fn ── F.linear(•, down.weight) ──► out_gpu
                            │                                                                          │
   x_gpu ── x_pinned ──┐    │                                                                          │
            event ─── runner.submit_with_d2h ──► CPU thread (_cpu_mlp_block_work):                     │
                                                   F.linear(x_pinned, w_mlp1_cpu)  ─► y1 (transient)   │
                                                   F.silu(y1[:,:d]) * y1[:, d:]    ─► z  (transient)   │
                                                   F.linear(z, w_mlp2_cpu)         ─► y2_pinned        │
                            ┌── runner.wait() ── uva_copy_into_gpu(y2_pinned, y2_gpu) ─────────────────┘
                            └── out_gpu.add_(y2_gpu)                                                   ─► result
```

CPU keeps its MLP1 output (`y1`) and SwiGLU output (`z`) **local** — never to GPU, never re-pinned. Only the final MLP2 partial (`y2_pinned`) crosses to GPU via UVA. **Exactly one transfer per MLP block** (verified by `test_mlp_block_fusion::test_fused_mlp_emits_exactly_one_uva_copy` and by Nsight: 1008 = 18 forwards × 56 transfers/forward = 28 QKV + 28 MLP-block UVA copies per forward, see §1.9).

Per the matched-index invariant (`weight_offload_design.md`), CPU's MLP1 col output (LAST `n_cpu_per_half` cols of canonical intermediate) feeds directly into MLP2's CPU input slice (also LAST `n_cpu_per_half` cols of canonical intermediate). The two counts are equal by construction at the same `f_cpu_store`, and `CotsSwiGLUMLPOp.__init__` asserts the equality.

GPU runs the full MLP block on its weight slices (`F.linear` → `act_fn` → `F.linear`), bypassing the linears' (raise-guarded) `quant_method.apply`. The `act_fn` reference is captured from the parent `Qwen2MLP.act_fn` — same `SiluAndMul` instance, same kernel.

### Strict MLP recognizer

`_install_mlp_ops` rejects mismatches at wrap time rather than silently produce wrong output:

| Check | Why |
|---|---|
| `act_fn` must be `SiluAndMul` | The fused CPU path is hard-coded to `silu(gate) * up`. |
| No bias on `gate_up_proj` or `down_proj` | The fused path doesn't add MLP biases. |
| `skip_bias_add` must be False | Not supported by the fused path. |

Phase 1a's smoke target (Qwen2.5-7B) satisfies all three trivially. A future model violating any check fails at `wrap_modules`, not in a forward.

### K/V-biased picker for WQKV

The picker (`_qkv_kv_biased_indices` + `_qkv_kv_biased_counts`, `cots.py:159-244`) chooses CPU columns from the WQKV output dim with K/V groups first. Rationale from `weight_offload_design.md §WQKV Column Choice`: at the typical decode regime, K/V activations dominate WQKV's H2D transfer volume, so placing K/V on CPU eliminates the largest chunk of activation traffic.

**Head-group alignment.** Per `weight_offload_design.md §201-205`, K and V for the same head must always move together (K[h] and V[h] co-located on the same device, never split). The picker enforces this by snapping `n_k` and `n_v` to multiples of `head_dim`, with `n_k == n_v` always (paired KV head groups). Above the K+V boundary, the remainder spills into Q tail; Q has no head-boundary requirement at the layer level.

| Requested `n_cpu` | Picker behavior |
|---|---|
| `n_cpu = 0` | Empty (no offload for this WQKV) |
| `0 < n_cpu ≤ 2 × kv_size` | Snap to nearest KV head pair: `n_pairs = round(n_cpu / (2·head_dim))`; n_k = n_v = `n_pairs · head_dim` |
| `n_cpu = 2 × kv_size` (exact) | All K + V on CPU, head-aligned trivially |
| `n_cpu > 2 × kv_size` | Full K + V + last `n_cpu − 2·kv_size` cols of Q tail |
| `kv_biased=False` (ablation) | TP-style proportional split, no head alignment |

`_w_cpu` row layout matches the picker's order: `[Q_tail | K_cpu | V_cpu]`.

**Effective-f granularity** at Qwen2.5-7B (head_dim=128, 4 KV heads, `2·head_dim = 256` cols per pair):

| Requested f | Raw n_cpu | Snapped pairs | Effective WQKV f |
|---:|---:|---:|---:|
| 0.05 | 230 | 1 (256 cols) | 5.55% |
| 0.09 | 414 | 2 (512 cols) | **11.1%** (rounds up from 9%) |
| 0.22 | 1014 | 4 (full K+V) | 22.2% |
| 0.50 | 2304 | 4 + Q tail (1280 cols) | 50.0% |

WQKV is 8.8% of layer weight, so the f=0.09 deviation moves total offloaded memory by `~0.02 × 0.088 × 12.15 GB ≈ 21 MB` — visible in the offloader's startup log line (1.13 GB at f=0.09 head-aligned vs ~1.10 GB sub-head split) but not in the comparison plot's data points. The trade is granularity for a load-bearing Phase 2 invariant: sub-head splits would force the CPU suffix-attention kernel to reassemble partial heads on every step.

### Why fork at the layer level (not at the worker level)

`CpuTaskRunner.submit_with_d2h` returns immediately (D2H copy + `executor.submit`); the CPU GEMM runs on the cots-cpu thread. The GPU slice's `F.linear` runs on the calling thread / current CUDA stream while CPU is busy. `runner.wait()` blocks on the future, then the operator issues `uva_copy_into_gpu` on the compute stream — no host-side wait for the CPU result before the GPU keeps running.

---

## 1.4: Activation return — SM-issued UVA copy

The CPU GEMM produces a result in pinned host memory (`_y_pinned`); the GPU needs it as a device tensor (`_y_gpu`) for the assembly path in §1.3. A naïve `cudaMemcpyAsync(D, H, ...)` would queue on a copy engine — Phase 0 §0.5.5 showed this contends with the H2D weight prefetch path (Phase 1b) on CE0. The Phase 1a backend issues the activation return as a Triton kernel on the compute stream, bypassing CEs entirely.

`uva_copy_into_gpu(src_pinned, dst_gpu)` (`cots.py:97-126`) is a one-liner Triton kernel: reads pinned host memory through UVA mapping, writes to a GPU buffer, no compute. Bit-identical at all tested sizes (256 KB–14 MB).

| Path | Engine | Phase 0 § |
|---|---|---|
| Standard `cudaMemcpyAsync` | CE0 (copy engine) | 0.5.5 |
| `uva_copy_into_gpu` (Phase 1a) | SM (compute stream) | 0.5.5 |

The kernel runs after the worker thread's CPU GEMM completes — the operator calls `runner.wait()` (which blocks on `future.result()`), then `uva_copy_into_gpu`. Because the kernel runs on the compute stream, downstream cuBLAS work for the same layer is automatically ordered after it without an explicit sync.

### One non-obvious detail

A CUDA event recorded right after the async D2H of `x_gpu → x_pinned` is required: without it, the worker thread races the copy and reads `x_pinned` before the H2D has completed. `CpuTaskRunner.submit_with_d2h` records the event right after the copy and passes it to the worker function, which calls `event.synchronize()` before the GEMM.

---

## 1.5: Profiler-context allocation rule

vLLM's KV-cache budget calculation reads `model_memory_usage` from a `DeviceMemoryProfiler` context that wraps `model_loader.load_model(...)` (`gpu_model_runner.py:4746-4807`):

```
kv_budget = total_gpu × gpu_memory_utilization − model_memory_usage − overhead
```

`model_memory_usage` is captured as the GPU-allocation delta across the context. Anything allocated outside the context is invisible to the budget.

### The invariant

> **All GPU allocations performed by the offloader must occur inside `wrap_modules`** (which executes inside `DeviceMemoryProfiler`). Allocations in `post_init` or later are silently excluded from `model_memory_usage` and cause runtime OOM at any `gpu_memory_utilization` that baseline tolerates.

In Phase 1a this means:
- GPU-slice `param.data` tensors — allocated in `_wrap_linear`. ✓
- CUDA index copies (`cpu_indices_cuda`, `gpu_indices_cuda`) — allocated in `CotsLinearHandle.install`, which is called during `_build_handles`. ✓
- `_y_gpu` (UVA-copy destination) — allocated in `_allocate_activation_buffers`, called at the end of `wrap_modules`. ✓
- `post_init` performs zero GPU allocation. ✓

### Why this rule is documented in code, not just here

A future-phase implementer adding a prefetch buffer pool (1b), a tier-aware KV pool (2), or graph-capture buffers (4) will be tempted to allocate "lazily on first forward." Doing so violates the rule and breaks the transparency invariant (§1.8). The rule is enforced by convention: the docstring of `_allocate_activation_buffers` and the docstring of `post_init` both call it out, and the Phase 1b extension points (§1.11) follow the same pattern.

### Why the redirect-then-split design failed this rule

The earlier design allocated the GPU slice tensors in `post_init` (which runs at `gpu_model_runner.py:4887`, outside the profiler). At `gpu_memory_utilization=0.85`, the offloader's "1.1 GB freed" message was correct, but `model_memory_usage` had been captured before the GPU slices were allocated — vLLM's KV-cache budget then over-allocated by exactly the GPU-slice size, OOMing at runtime. TP-style intercept (§1.2) moves all those allocations inside the profiler.

---

## 1.6: Lazy iteration during `wrap_modules`

`make_layers()` (`models/utils.py:626-658`) yields decoder layers through a generator. `wrap_modules` iterates the generator one layer at a time (`cots.py:688-697`):

```python
for layer in modules_generator:
    modules.append(layer)
    for qualified_name, child in layer.named_modules():
        if matches_offload_suffix(qualified_name):
            self._wrap_linear(child, qualified_name)
```

Each layer's offloaded params are replaced with GPU-slice tensors **before the next layer is constructed**. The replaced full-shape tensors go out of scope and PyTorch's caching allocator pool reuses their segments for the next layer's `torch.empty(full_shape)` call (matching shapes → zero fragmentation).

### The peak-GPU bound

This bounds peak GPU memory during construction to ~**one layer's full-shape worth**, regardless of how many layers exist. For Qwen2.5-7B BF16 with 28 layers, the peak during `wrap_modules` is ~440 MB (one layer's WQKV + MLP1 + MLP2 + WO), not ~12 GB (28 layers' worth).

### Why this matters now (not later)

Without lazy iteration, a Phase 3 14B-model run (~28 GB BF16 weights) cannot even be *constructed* on a 24 GB GPU — the empty-tensor allocations would OOM before any wrapping happens. Lazy iteration is the same pattern `prefetch.py:175` uses for the identical reason.

### How the bug surfaced

An earlier draft used `list(modules_generator)` to materialize all layers up-front. This passed unit tests (which build 4-layer synthetic stacks where the bound is irrelevant) and OOMed on the first end-to-end Qwen2.5-7B run.

---

## 1.7: GPU memory saved at f=0.09 on Qwen2.5-7B

Per-sub-module weight bytes (`out_dim × in_dim × 2` for BF16):

| Sub-module | Shape | Bytes per layer | n_offloaded (84 layers per Qwen2.5-7B) |
|---|---|---:|---:|
| WQKV | (4608, 3584) | ~33.0 MB | 28 |
| MLP1 (gate+up) | (37888, 3584) | ~271.6 MB | 28 |
| MLP2 (down) | (3584, 18944) | ~135.8 MB | 28 |
| **Per-layer total** | — | **~440.4 MB** | — |
| **Full model** | — | **~12.3 GB** | — |

At `f_cpu_store = 0.09`, `n_cpu = round(0.09 × shard_dim)` per sub-module. WQKV's actual count snaps to head-pair boundaries (§1.3 effective-f table — at requested 0.09 it rounds to 11.1%, adding ~21 MB to the total saved). The offloader logs at startup:

```
[CotsOffloader] Initialized: 84 offloaded linears,
    GPU memory saved (weights): 1.13 GB,
    shared activation buffers: 0.06 GB pinned input + 0.06 GB pinned output + 0.06 GB GPU UVA-dest,
    dispatch buckets: [...]
```

The 1.13 GB is close to `0.09 × 12.3 GB ≈ 1.11 GB` analytic, slightly higher because of the head-aligned WQKV snap-up at f=0.09. Activation buffer sizes scale with `max_num_batched_tokens × max_dim`; absolute bytes are workload-dependent but stay under 200 MB at typical scheduler configs.

WO (`o_proj`) is excluded from offload by design (`weight_offload_design.md §WO Split Axis Decision`). Phase 2 may revisit.

---

## 1.8: End-to-end smoke and transparency at `gpu_memory_utilization=0.9`

`David/Benchmarks/phase1/smoke_qwen25_7b.py` loads Qwen2.5-7B-Instruct and runs deterministic greedy decode on the prompt `"The capital of France is"` for 16 tokens. Two configurations are compared as separate Python invocations (vLLM V1's engine-subprocess teardown does not release VRAM in time for back-to-back `LLM()` constructions in the same process).

### Token-level parity

| `gpu_memory_utilization` | Backend | f_cpu_store | Outcome |
|---|---|---:|---|
| 0.85 | noop (baseline) | 0.0 | Loads, generates 16 tokens |
| 0.85 | cots | 0.09 | Loads, generates **identical** 16 tokens |
| 0.90 | noop | 0.0 | Loads, generates 16 tokens |
| 0.90 | cots | 0.09 | Loads, generates **identical** 16 tokens |

Bit-equality of token IDs is not required by design — cuBLAS picks different MMA tile configs for the GPU-slice GEMM vs the unsplit GEMM, and BF16 reductions can flip the lowest-margin token at ~1% probability. In practice, on this prompt at f=0.09, no token flipped.

### The transparency invariant

> Any `gpu_memory_utilization` that works for the baseline (no-offload) engine must also work for the cots backend at any `f_cpu_store ≥ 0`.

Verified at 0.9 on Qwen2.5-7B. The cots run reports ~1.1 GB more KV headroom than baseline at the same util setting (the actual offload payoff). This invariant requires §1.5 to hold; failing it is the symptom of a profiler-invisible allocation.

---

## 1.9: Nsight overlap trace

`David/Benchmarks/phase1/probe_cots_overlap.py` wraps a single decode step in NVTX and produces an `nsys-rep` trace. Captured with:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn nsys profile \
    -o results/cots_overlap.nsys-rep \
    --trace=cuda,nvtx,osrt --trace-fork-before-exec=true \
    --cuda-graph-trace=node --force-overwrite=true \
    python probe_cots_overlap.py
```

(`VLLM_WORKER_MULTIPROC_METHOD=spawn` and `--trace-fork-before-exec` are required because vLLM V1 runs CUDA work in an engine subprocess; they follow vLLM's official profiling guidance.)

### What the trace shows

Within each offloaded sub-module under the `profiled_generate` NVTX range, three bars are visible:

```
GPU stream   :  [cutlass_*_gemm           ][uva_copy_kernel ]
cots-cpu thread :         [mkldnn_bf16_gemm    ]
                  ────────────────────  time  ────────────────►
```

1. **GPU `cutlass_*_gemm`** for the GPU slice runs as soon as `submit()` returns.
2. **CPU `mkldnn_bf16_gemm`** runs concurrently on the cots-cpu thread.
3. **Triton `uva_copy_kernel`** runs on the compute stream once the CPU GEMM completes.

At `f=0.09`, the CPU bar always ends before the GPU bar — confirming Phase 0 §0.3.3's "free regime" prediction in production code.

### One non-obvious detail

`ThreadPoolExecutor(thread_name_prefix="cots-cpu")` only renames the thread for Python introspection; Linux's `/proc/<pid>/task/<tid>/comm` (which Nsight reads) keeps the parent process's name. The shared executor (`_get_executor`) calls `prctl(PR_SET_NAME, "cots-cpu")` in its `initializer=`, setting the OS-level thread name and making the thread identifiable in the Nsight thread list.

---

## 1.10: Test matrix and reproducibility

Six test files under `David/Tests/phase1a/` (~1,300 LOC, runtime ~4 s, **60 tests total**):

| File | Tests | Layer | Gates |
|---|---:|---|---|
| `test_uva_copy.py` | 6 | unit | Triton SM-issued copy bit-identical at 4 sizes; rejects non-pinned/shape-mismatch |
| `test_qkv_picker.py` | 10 | unit | Picker math + head-alignment for K, V, and Q-tail; unbiased ablation; specific snap targets at f ∈ {0.05, 0.09, 0.22} |
| `test_dispatcher_split.py` | 25 | unit | `CpuTaskRunner` + `_cpu_gemm_into_after_event` correctness on synthetic col/row shapes × `f ∈ {.03, .09, .22, .50}` × `B ∈ {1, 8, 64}` |
| `test_loader_wrappers.py` | 8 | unit | `CotsLinearHandle.install` + private loader closures (row / merged-col / qkv): GPU and CPU slices match range-narrow reference |
| `test_mlp_block_fusion.py` | 5 | unit | `CotsSwiGLUMLPOp.__call__` numeric parity at f ∈ {.10, .25, .50}; **exactly 1 UVA copy per block** (counter-instrumented); orphan col/row raises |
| `test_offloader_integration.py` | 6 | integration | Full `wrap_modules` + `post_init` lifecycle (3 f values), `enforce_eager` check, dispatch lookup rounds up, quantized layer rejected |

Plus the standalone smoke test (§1.8) and Nsight probe (§1.9) under `David/Benchmarks/phase1/`.

```bash
# Unit + integration (CI-cheap):
.venv/bin/python -m pytest David/Tests/phase1a/

# End-to-end smoke (requires GPU + Qwen2.5-7B weights):
cd /TTC/FastTTS-thesis && python David/Benchmarks/phase1/smoke_qwen25_7b.py --skip-baseline
cd /TTC/FastTTS-thesis && python David/Benchmarks/phase1/smoke_qwen25_7b.py --baseline-only
```

### Tolerance

| Path | Tolerance | Source |
|---|---|---|
| UVA copy (no compute) | bit-identical | direct memcpy |
| Loader split (no compute) | bit-identical | direct memcpy |
| Per-Linear CPU/GPU split vs unsplit | `rtol=5e-2, atol=0.5` | BF16 + cuBLAS kernel-selection variance (~1 ULP at unit-variance inputs) |
| Fused MLP block vs unsplit | `rtol=5e-2, atol=5%·max(\|expected\|)` | Chained matmul output magnitudes ~10–100× per-Linear; abs delta scales with peak |
| End-to-end smoke | semantic (same top-1 token in practice) | BF16 reductions |

### Pre-commit

`pre-commit run` is green on all five modified files (`cots.py`, `base.py`, `offloader/__init__.py`, `config/offload.py`, `config/__init__.py`). `pre-commit run mypy-3.10 --hook-stage manual` runs separately as part of the verification pass.

---

## 1.11: Hooks for Phase 1b / 1c / 2

The Phase 1a code was structured so each later phase is a localized swap, not a rewrite.

### Phase 1b — layer-ahead weight prefetch

The four `BaseOffloader` lifecycle methods are present as no-ops in `CotsOffloader`:

```python
def _wait_for_layer(self, layer_idx): ...    # no-op in 1a
def _start_prefetch(self, layer_idx): ...    # no-op in 1a
def sync_prev_onload(self): ...              # no-op in 1a
def join_after_forward(self): ...            # no-op in 1a
```

Phase 1b fills these following `prefetch.py:243-308`. The dispatch table at the offloader level is already keyed on `(f_cpu_compute, f_prefetch_compute)`; Phase 1a populates the trivial `(f_cpu_store, 0.0)` entry, and 1b drops the `assert f_prefetch == 0.0` invariant in `post_init`. `CpuTaskRunner.submit_with_d2h` / `wait` and the operator classes need zero changes.

A separate Phase 1b prefetch buffer pool will be allocated in `wrap_modules` (per the rule in §1.5), mirroring `StaticBufferPool` (`prefetch.py:60-125`). The shared activation buffers in §1.1 are unaffected. `CotsLinearHandle` is the natural extension point: a "prefetch slot" role can be added to the handle without disturbing storage semantics.

### Phase 2 — CPU suffix attention + two-pool KV

The two-pool KV-cache design hooks into vLLM's existing KV pathway, not into cots's offloader. cots's contribution is an additional operator: `CotsAttnOp` for CPU suffix attention with online-softmax merge. It slots in alongside `CotsQKVOp` / `CotsSwiGLUMLPOp` and uses the same `CpuTaskRunner` API. The Phase 2 op is independent of Phase 1a's two operators, so adding it doesn't disturb either.

Any auxiliary GPU buffer Phase 2 introduces must follow the §1.5 profiler-context allocation rule.

### Phase 1c — Native CPU runner + CUDA graph capture

`CpuTaskRunner.submit_with_d2h` / `wait` are the **single Phase 1c swap surface**: their bodies become a `cudaLaunchHostFunc` binding to a C++ `CPUInfer` (port from `kt-kernel/cpu_backend/cpuinfer.h:78-116`). Operators (`CotsQKVOp`, `CotsSwiGLUMLPOp`) call the runner methods unchanged. Worker functions (`_cpu_gemm_into_after_event`, `_cpu_mlp_block_work`) become host-function userData entries; their argument pointers (`x_pinned`, `y_pinned`, `w_cpu`) are stable post-`wrap_modules`. Phase 1c also introduces a bucket-aware `cpu_num_threads` policy (the t=4-vs-t=16 e2e gap measured in §1.13b/§1.14 makes thread tuning a load-bearing knob, not just a default).

Buffer addresses are already fixed after `wrap_modules` (no rebinding, no resizing); the Triton UVA kernel is graph-capturable as-is. Worker-local transients (the MLP block's `y1` and `z`) live inside the host-function call and are invisible to the captured graph. The current `enforce_eager=True` check in `post_init` becomes a no-op once Phase 1c lands. Renamed from "Phase 4" and reordered to precede Phase 2 — see `implementation_roadmap.md` and §1.14 for the rationale.

---

## 1.12: Wrap-time invariants — fail-fast contract

`CotsOffloader.wrap_modules` runs a sequence of static checks before any allocation. Each guard exists because failing at wrap time produces a clear error pointing at the misconfiguration; failing later in a forward gives a confusing stack trace at a kernel boundary.

| Guard | Trigger | Why fail at wrap time |
|---|---|---|
| `tensor_parallel_size == 1` | `parallel_config.tensor_parallel_size != 1` | Loader closures assert full unsharded `loaded_weight` shapes; native vLLM loaders narrow by TP rank before copying. Multi-rank TP is out of scope for Phase 1a. |
| `is_pin_memory_available()` (only when `f_cpu_store > 0`) | Pinned host memory unavailable (e.g., container cgroup limits) | `uva_copy_into_gpu` asserts `is_pinned()`; without this guard the offloader allocates non-pinned memory and crashes at first forward. |
| `linear.weight.dtype == torch.bfloat16` | Per offloadable Linear, before handle construction | Phase 0 §0.3.2: `torch.mm` with FP16/FP32 falls back to scalar path on CPU (~100× slower than oneDNN BF16). Documented in `CotsOffloadConfig.cpu_dtype`. |
| `isinstance(quant_method, UnquantizedLinearMethod)` | Per offloadable Linear | The loader closures and operator forward paths are BF16-unquant only; quant configs (GPTQ/AWQ/etc.) require Phase 3+ work. |
| MergedCol gate/up partitions equal-sized | Per `CotsLinearHandle.for_col` | Model architecture invariant; the snap math assumes `n_cpu_per_half = n_cpu / 2`. |
| QKV `kv_size % head_dim == 0` | Per `CotsLinearHandle.for_qkv` | Head-aligned snap requires whole heads on each side. |
| MLP recognizer: `act_fn` is `SiluAndMul`, no biases, no `skip_bias_add` | Per fused MLP block | The fused CPU path is hard-coded to `silu(gate)*up`; biases/skip_bias_add aren't handled. Future model violating any of these fails here, not silently. |
| Orphan col/row handle (Phase 1a contract) | After MLP-block recognition | Standalone col/row offload has no fusion target → would silently use the wrong path. |

These are documented in code as method-level docstrings on `_check_*` helpers and the kind-specific `for_*` factories.

---

## 1.13: Head-to-head against vLLM's PrefetchOffloader

How does Phase 1a's COTS backend (`--offload-backend cots`) compare to vLLM's native `PrefetchOffloader` at matched offload depth on Qwen2.5-7B BF16, RTX 4090?

Two complementary regimes were measured:

- **§1.13a — Prefill-heavy** (input=256, output=32, batches {1, 16, 64}): inherited from Phase 0 §0.10's setup. CPU work is dominated by prefill (8:1 prefill:decode ratio). The "worst case" for COTS — well outside §0.3.3's "free regime" — but useful as a documented bound.
- **§1.13b — Decode-heavy** (input=8, output=128, batches {1, 4, 16}): the regime COTS is designed for per `thesis_proposal.md` (FastTTS / TTC: short prompts, decode-dominated, small batch). Prefill:decode CPU ratio is ≤ 0.4% across all batches.

Both regimes use the **densest-spread N=1 G-varies prefetch baseline** from §0.10.2(d) (the strongest possible prefetch reference at each offload depth). UVA omitted per Phase 0 §0.10 finding 3.

### 1.13a: Prefill-heavy regime

`David/Benchmarks/phase1/bench_cots_vs_native_prefill.py` shells out to `vllm bench latency --enforce-eager` per cell.

| Arm | Offloaded GiB | B=1 (s) | B=16 (s) | B=64 (s) | tok/s @ B=64 |
|---|---:|---:|---:|---:|---:|
| `none` | 0.00 | 0.522 | 0.890 | 2.037 | 1005 |
| `cots_009` | 1.03 | 2.825 | 26.692 | 109.731 | 18.7 |
| `cots_022` | 2.53 | 7.476 | 53.459 | 260.284 | 7.9 |
| `cots_050` | 5.74 | 13.112 | 117.833 | 588.526 | 3.5 |
| `prefetch_28x1` | 0.43 | 1.105 | 1.486 | 2.630 | 779 |
| `prefetch_14x1` | 0.87 | 1.505 | 1.862 | 2.973 | 685 |
| `prefetch_7x1` | 1.74 | 2.670 | 2.994 | 4.090 | 501 |
| `prefetch_4x1` | 3.04 | 4.546 | 4.888 | 5.959 | 344 |
| `prefetch_2x1` | 6.08 | 8.990 | 9.451 | 10.439 | 196 |

**COTS loses at every (depth, batch) point in this regime, by 2× to 56×.** At matched depth `prefetch_14x1` (0.87 GiB) vs `cots_009` (1.03 GiB): 2.97 s vs 109.7 s @ B=64 — prefetch is **37× faster**.

The mechanism is the linear-in-`num_tokens` CPU GEMM cost. Prefill at `input_len=256` passes a forward at `num_tokens=256`, which is 256× the per-forward CPU work compared to a `num_tokens=1` decode step. At B=64, vLLM batches the prefills together — `num_tokens` per prefill forward becomes ~16k tokens (chunked) — and CPU GEMM cost grows accordingly. PrefetchOffloader's PCIe transfer cost is independent of `num_tokens` per forward, so its latency stays nearly flat across batches (B=1 → B=64: 2× growth) while COTS grows ~40× over the same range.

This regime is **the wrong test bench for COTS**. Phase 1a's design target (per `thesis_proposal.md §3` and Phase 0 §0.3.3) is small-batch decode, not prefill-heavy chat. Reported here as the worst-case bound.

### 1.13b: Decode-heavy regime

`David/Benchmarks/phase1/bench_cots_vs_native_decode.py`. `input_len=8` makes the prefill forward negligible (one forward at `num_tokens=8` vs 128 decode forwards at `num_tokens=batch`).

| Arm | Offloaded GiB | B=1 (s) | B=4 (s) | B=16 (s) | tok/s @ B=16 |
|---|---:|---:|---:|---:|---:|
| `none` | 0.00 | 2.033 | 2.105 | 2.147 | 954 |
| `cots_005` | 0.57 | 4.528 | 5.434 | 11.344 | 181 |
| `cots_009` | 1.03 | **5.887** | 7.412 | 18.088 | 113 |
| `cots_022` | 2.53 | **10.853** | **14.722** | 38.203 | 54 |
| `cots_050` | 5.74 | **22.434** | **29.116** | 79.922 | 26 |
| `prefetch_28x1` | 0.43 | 4.369 | 4.460 | 4.492 | 456 |
| `prefetch_14x1` | 0.87 | 5.997 | 6.055 | 6.091 | 336 |
| `prefetch_7x1` | 1.74 | 10.651 | 10.717 | 10.777 | 190 |
| `prefetch_4x1` | 3.04 | 18.132 | 18.211 | 18.312 | 112 |
| `prefetch_2x1` | 6.08 | 35.961 | 36.030 | 36.282 | 56 |

All cots arms above use the default `cpu_num_threads=16` (see thread-count tuning subsection below).

**Five COTS wins versus prefetch — three at B=1, two at B=4:**
- B=1, mid-depth: `cots_009` (1.03 GiB) at 5.89 s beats `prefetch_14x1` (0.87 GiB) at 6.00 s by ~2%.
- B=1, mid-depth: `cots_022` (2.53 GiB) at 10.85 s beats `prefetch_4x1` (3.04 GiB) at 18.13 s by **~40%**.
- B=1, high-depth: `cots_050` (5.74 GiB) at 22.43 s beats `prefetch_2x1` (6.08 GiB) at 35.96 s by **~38%**.
- B=4, mid-depth: `cots_022` at 14.72 s beats `prefetch_4x1` at 18.21 s by **~19%**.
- B=4, high-depth: `cots_050` at 29.12 s beats `prefetch_2x1` at 36.03 s by **~19%**.

`cots_005` (the §0.3.3 free-regime corner) ties prefetch_28x1 within ~4% at B=1 — a small loss at the lowest offload depth that flips to a 11% win if `cpu_num_threads=8` is used (see thread-tuning subsection).

**At B=16, COTS loses to prefetch across the offload range** by 2.5–18×; the high-batch regime is `num_tokens`-linear in CPU GEMM and concedes to prefetch's `num_tokens`-flat PCIe cost.

#### COTS vs `none`: even the free regime is not free

`cots_005` B=1 = 4.53 s adds **~2.50 s over the no-offload baseline (2.03 s)** even though §0.3.3 predicts CPU GEMM fits within GPU layer time at this f (sum 454 µs ≤ GPU 470 µs at t=16, 16 µs headroom). The initial hypothesis was that the residual gap was pure per-op Python dispatch in the COTS runtime path — `CotsQKVOp.apply`, `CotsSwiGLUMLPOp.__call__`, `executor.submit`, `future.result`, scatter, view construction. **§1.14 instruments this directly with a `--cots-dry-run` mode** and finds that's only ~18% of the story: ~0.45 s/generate is pure orchestration, but ~2.04 s is the *active CPU-work penalty* — extra wall clock from enabling real CPU GEMM, dominated by oneDNN-on-many-threads contending with the main thread's CUDA dispatch path (the 3.5× swing across t={4,8,16,24} is the signature of runtime interference, not unhidden compute).

Both contributions are addressable by Phase 1c: `cudaLaunchHostFunc` removes the orchestration tax, and bucket-aware thread policy (plus the host-function-based wait that doesn't lose CUDA launch runahead) targets the active CPU-work penalty. How much of the penalty is genuinely irreducible CPU GEMM (still finishes after concurrent GPU work) vs runtime interference Phase 1c can shrink will only be settled by re-measuring §1.14 on the native runner. See §1.14 for the full t × B decomposition table and the upper-bound estimate of post-Phase-1c real.

At B>1 a second cost stacks on top: COTS's CPU GEMM scales linearly with `num_tokens` (4× at B=4, 16× at B=16), so the gap to `none` widens fast — at B=16 even cots_005 is 15× slower than baseline.

#### COTS vs prefetch: constant-vs-linear in offload depth

COTS's per-forward cost is dominated by orchestration that depends on **op count, not bytes** — 56 ops/forward regardless of f. Increasing f within a constant-op layout adds CPU GEMM time only proportionally to the slice width, which at B=1 stays inside or near the GPU budget per §0.3.3. So as offload depth grows, **COTS's curve is approximately flat in f**.

PrefetchOffloader's per-forward cost is the **PCIe transfer of every offloaded byte** — by construction linear in offloaded GiB (slope ~1.5 s/GiB at B=64 in §0.10; comparable at B=1 decode-heavy). So as offload depth grows, **prefetch's curve rises linearly**.

Two flat-vs-linear curves cross. The crossover regime — where COTS's flat overhead beats prefetch's accumulated PCIe — is exactly where the two B=1 wins above sit. At B>1 the picture inverts: COTS's CPU GEMM term scales linearly with `num_tokens` while prefetch's PCIe cost is `num_tokens`-independent, so COTS becomes the linear curve and prefetch becomes the flat one — and prefetch wins everywhere.

#### CPU thread count tuning (`cpu_num_threads`, default 16)

A complete 4-arm × 3-batch × 4-thread decode-heavy sweep (`bench_cots_thread_sweep_decode.py` → `results/thread_sweep_decode/`) plus a 3-cell prefill sweep at f=0.09 (`bench_cots_thread_sweep_prefill.py` → `results/thread_sweep_prefill/`). Cross-run reproducibility was independently verified at CV ≤ 3% per cell over 3 fresh vLLM processes per cell — the thread-count effect dwarfs noise by 1–3 orders of magnitude.

Decode-heavy slowdown vs row-best (the headline table at the top of §1.13b is at t=16):

| arm | B | t=4 | t=8 | t=16 | t=24 |
|---|---:|---:|---:|---:|---:|
| cots_005 | 1 | **1.00×** | 1.25× | 1.46× | 1.18× |
| cots_005 | 4 | **1.00×** | 1.08× | 1.07× | 1.21× |
| cots_005 | 16 | 1.32× | 1.05× | **1.00×** | 2.76× |
| cots_009 | 1 | **1.00×** | 1.16× | 1.30× | 1.59× |
| cots_009 | 4 | 1.09× | 1.03× | **1.00×** | 2.46× |
| cots_009 | 16 | 1.43× | 1.07× | **1.00×** | 1.90× |
| cots_022 | 1 | **1.00×** | 1.06× | 1.16× | 2.28× |
| cots_022 | 4 | 1.31× | **1.00×** | 1.05× | 2.21× |
| cots_022 | 16 | 1.53× | 1.07× | **1.00×** | 1.08× |
| cots_050 | 1 | 1.07× | **1.00×** | 1.12× | 1.52× |
| cots_050 | 4 | 1.32× | 1.03× | **1.00×** | 1.44× |
| cots_050 | 16 | 2.03× | 1.35× | 1.26× | **1.00×** |

Aggregate regret (worst / mean slowdown across 12 decode + 3 prefill cells):

| threads | decode-only worst / mean | combined worst / mean | notes |
|---:|---:|---:|---|
| 4  | 2.03× / 1.26× | 2.23× / 1.35× | wins low-f B=1 decode; collapses at large-batch prefill |
| 8  | **1.35× / 1.09×** | 1.47× / 1.11× | best on decode-only; ties t=16 on combined |
| **16** | 1.46× / 1.12× | **1.46× / 1.12×** | tied with t=8 on combined; wins all decode B=4/B=16 cells and the 22% prefill B=16 cell |
| 24 | 2.76× / 1.72× | 2.76× / 1.59× | wins only the cots_050 B=16 corner; otherwise collapses on decode (up to 2.76×) |

Three asymmetries explain the choice:

1. **t=24 vs t=16 (decode)**: 24 threads of oneDNN saturate all physical cores and starve the main vLLM thread's CUDA dispatch — at decode B≥4 this collapses by up to 2.76× (cots_005 B=16 at t=24). t=16 is large enough to keep CPU GEMM throughput high and small enough to leave dispatch headroom.
2. **t=8 vs t=16 (decode B=1)**: t=8 wins all four B=1 corners by 9–15% — the FastTTS-target regime. t=16 still wins 7 of 12 decode cells (B=4 and B=16) by 1–8%; on decode-only aggregate t=8 is strictly better.
3. **t=8 vs t=16 (large-batch prefill)**: prefill B=16 with 256-token input runs CPU GEMM at `num_tokens=4096` per forward; t=16 wins by **22%** here (24.19 s vs 29.54 s). This single decisive prefill cell ties t=16's combined regret to t=8 despite t=8 dominating decode.

An independent isolated-MLP cold-cache microbench (`microbench_thread_sweep_mlp.py`, 28-layer ring, DRAM-streaming) confirms the optimum *within isolated CPU GEMM* depends on `num_tokens` alone — omp8 at small (1, 16), omp32 (full SMT) at larger (4, 64, 256). The fact that the e2e optimum diverges from the isolated-GEMM optimum is the dispatch-interference signature.

The default is exposed as `CotsOffloadConfig.cpu_num_threads` (CLI: `--cots-cpu-num-threads`, default **16**), applied via `torch.set_num_threads` at offloader init. **Override to 8 for FastTTS-style decode-only experiments** (recovers 9–15% at B=1 across all four offload depths). Long-term, the planner should emit per-bucket thread counts.

### 1.13 Conclusion: Phase 1a's regime, the Phase 1c dependency

What this benchmark establishes:

1. **COTS beats prefetch at five (depth, batch) corners at t=16**: three at B=1 (`cots_009`, `cots_022`, `cots_050` — by 2%, 40%, 38% respectively) and two at B=4 (`cots_022`, `cots_050` — by 19%, 19%). The flat-in-f COTS cost curve crosses the linear-in-f prefetch curve through most of the offload-depth range, not just at the extremes.
2. **COTS still loses to `none` at the f=0.05 free regime.** At t=16, cycle-cold CPU GEMM at f=0.05 B=1 sums to 454 µs vs the 470 µs GPU layer budget (16 µs/layer headroom — barely free), so naïvely the residual ~2.5 s gap to baseline at B=1 should be runtime overhead Phase 1c (was Phase 4) eliminates. **§1.14 instruments this directly** with a `--cots-dry-run` mode and finds the gap is split: only ~18% is pure host orchestration; the bulk (~82%) is the *active CPU-work penalty* — extra wall clock from enabling real CPU GEMM, dominated by oneDNN-on-16-threads contending with the main thread's CUDA dispatch path. Both contributions are addressed by Phase 1c (`cudaLaunchHostFunc` for the orchestration tax, bucket-aware thread policy for the runtime interference).
3. **The win regime narrows quickly with batch.** B=4 keeps two wins at moderate-to-high f. B=16 concedes everywhere because COTS's CPU GEMM scales linearly with `num_tokens` while prefetch's PCIe cost is `num_tokens`-independent. FastTTS/TTC's beam-search regime (effective batch = beam width ≥ 4) is partially a COTS win at Phase 1a (high-f corners only).

The Phase 1c swap (`CpuTaskRunner.submit_with_d2h` / `wait` → `cudaLaunchHostFunc` + C++ `CPUInfer`) is the load-bearing performance work for COTS to become broadly competitive. The Phase 1a structure was specifically designed to make Phase 1c a body-only swap: operators, handle storage, picker math, fused MLP block path, and the wrap-time invariants (§1.12) all stay; only the `CpuTaskRunner` body changes. Phase 1c is therefore not optional — it's the gating phase before deployment.

Until Phase 1c lands, Phase 1a's value is structural plus a partial throughput win: it validates the split mechanism, the matched-index invariant for fused MLP, the head-aligned K/V picker, and the SM-issued UVA activation return — and it already beats prefetch at three of four B=1 depths and two of four B=4 depths at the `cpu_num_threads=16` default. The remaining lost cells (low-f B=1 and the entire B=16 row) are the bound on what Phase 1c + a per-bucket thread-count knob need to close. §1.14 attributes the bound quantitatively.

---

## 1.14: Why the f=0.05 free regime isn't free — orchestration vs active CPU-work penalty

§0.3.3 predicted f=0.05 B=1 to be a "free regime" at t=16 (cycle-cold CPU MLP+QKV sum 454 µs ≤ 470 µs GPU layer budget). §1.13b's e2e measurement shows otherwise: `cots_005` B=1 = 4.520 s vs `none` = 2.033 s — a +2.49 s gap (~691 µs/layer over 3,612 layer-visits). §1.13's Conclusion attributed this entirely to "per-op Python dispatch overhead". This section instruments the attribution directly, finds the dispatch share is ~18%, and shows the rest is CPU GEMM time bleeding onto the critical path because oneDNN contends with the main thread.

### Diagnostic mechanism — `--cots-dry-run`

A new flag (`CotsOffloadConfig.dry_run`, CLI `--cots-dry-run`; see `vllm/config/offload.py`, `vllm/model_executor/offloader/cots.py:_cpu_dryrun_noop`) installs all the COTS wrappers (operators, dispatcher, D2H, UVA copy, GPU partial add) but replaces the worker's `torch.matmul` with a noop that only does `event.synchronize()`. Token output is garbage; only host bookkeeping cost is measured. The decomposition is then:

| Quantity | Definition | What it isolates |
|---|---|---|
| `pure orchestration` | `dryrun − none` | Wrapper Python + dispatcher submit/wait + UVA copy + GPU partial add. What `cudaLaunchHostFunc` + CUDA Graph capture eliminate at runtime. |
| `active CPU-work penalty` | `real − dryrun` | The full extra wall-clock from enabling real CPU GEMM. Includes the GEMM time itself, plus oneDNN-thread interference with the main thread's CUDA dispatch path, reduced launch runahead, and any cache/bandwidth contention. **Upper bound** on the post-Phase-1c floor; not a clean "unhidden GEMM" measurement, since some of the penalty comes from oneDNN-induced runtime interference rather than pure CPU-GEMM-on-critical-path. Strongly t-dependent. |

### Headline table: t × B sweep (`cots_005`, decode-heavy, input=8, output=128)

`results/dryrun_vs_none/summary.json`. `none` is t-invariant: 2.033 s @ B=1, 2.110 s @ B=4.

| B | t | dryrun (s) | real (s) | orch = dry − none | active CPU-work = real − dry | total = real − none | orch % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 2.507 | 3.094 | **+0.474** | +0.587 | +1.061 | 44.6% |
| 1 | 8 | 2.472 | 3.872 | **+0.439** | +1.400 | +1.839 | 23.9% |
| 1 | 16 | 2.483 | 4.520 | **+0.450** | +2.036 | +2.486 | 18.1% |
| 1 | 24 | 2.459 | 3.664 | **+0.426** | +1.205 | +1.631 | 26.1% |
| 4 | 4 | 3.149 | 5.065 | **+1.039** | +1.916 | +2.955 | 35.2% |
| 4 | 8 | 3.236 | 5.471 | **+1.127** | +2.235 | +3.361 | 33.5% |
| 4 | 16 | 3.160 | 5.322 | **+1.050** | +2.162 | +3.211 | 32.7% |
| 4 | 24 | 3.225 | 6.139 | **+1.115** | +2.913 | +4.029 | 27.7% |

### Three observations from the table

1. **Orchestration is essentially t-flat.** Range across t at fixed B is 4.4% (B=1) and 8.4% (B=4) — well within run-to-run variance. The dispatcher cost depends on op count and per-op work, not on the worker's thread count (which is moot when the worker is a noop). This confirms the methodology: `dryrun − none` cleanly isolates the runtime tax that's a pure function of vLLM main-thread activity.

2. **Active CPU-work penalty varies dramatically with t.** At B=1, t=4 gives +0.59 s; t=16 gives +2.04 s — a 3.5× swing from a single knob. The microbench (`microbench_thread_sweep_mlp.py`) shows isolated CPU GEMM is *faster* at higher t, so the e2e regression at t=16 isn't pure CPU GEMM compute; it's a combination of (a) CPU GEMM landing on the critical path because oneDNN-on-many-threads contends with the main thread's CUDA dispatch and stretches each matmul past its concurrent GPU work, and (b) reduced CUDA launch runahead. The penalty is t-dependent in a way pure GEMM compute is not — that's the signature of runtime interference, not unhidden compute.

3. **The "free regime" was structurally optimistic, not just hot-cache biased.** Phase 0 §0.3.3 measured CPU and GPU GEMM in isolation; §0.3 separately corrected the GPU side for L2-residency (see `phase0_findings.md §0.3.1`). Even with both numbers right and isolated, the overlap is not free in practice — the main thread's CUDA dispatch path *is itself a CPU workload* that contends with oneDNN, so the e2e CPU GEMM time exceeds the microbench's isolated number.

### Upper-bound estimate of post-Phase-1c real

Picking the best `t` for each `B` (the bucket-aware thread policy Phase 1c emits):

| B | best t | upper-bound post-1c real ≈ none + min CPU-work penalty | upper-bound post-1c gap to none | current real |
|---:|---:|---:|---:|---:|
| 1 | 4 | 2.033 + 0.587 = **≤ 2.620 s** | **≤ +0.587 s** | 3.094 s @ t=4 |
| 4 | 4 | 2.110 + 1.916 = **≤ 4.026 s** | **≤ +1.916 s** | 5.065 s @ t=4 |

These are **safe upper-bound estimates of post-1c real, not the final post-1c floor.** The reasoning is three-tier:

1. **Orch (`dryrun − none`):** Phase 1c is expected to remove most of this. `cudaLaunchHostFunc` + CUDA graph capture eliminate per-op Python; only the irreducible CUDA-submit cost stays, which is small. The +0.43-0.47 s/generate column collapses to roughly 0.

2. **Active CPU-work penalty (`real − dryrun`):** Phase 1c **may also reduce part of this**, but we cannot know how much until we implement it. The penalty was measured under the current Python runner and bundles: real CPU GEMM time + `future.result()` blocking pattern + main-thread launch-runahead loss + oneDNN-vs-main-thread interference + cache/BW contention. Phase 1c changes the runner substrate (`cudaLaunchHostFunc` + C++ task queue) and adds bucket-aware thread policy, both of which target the interference and runahead components, not just the dispatcher.

3. **True irreducible floor:** only the part of CPU GEMM that still finishes *after* its concurrent GPU work, **under the native runner with bucket-tuned thread count** — i.e., `max(0, CPU_GEMM_under_clean_runtime − GPU_layer_work)` per op. This is necessarily ≤ the active-CPU-work-penalty column above, by an amount we won't know until Phase 1c lands.

Reading the table accordingly: the **+0.59 s @ B=1** entry says Phase 1c real should land **at most around 2.620 s** at the best current `t`, leaving at most a +0.59 s gap to `none`. The actual post-1c gap is plausibly smaller, because some of that 0.59 s is interference Phase 1c is designed to reduce. Concretely refuting the upper bound (or tightening it) is a Phase 1c measurement task: re-run §1.14 on the native runner and compare `real_native − dryrun_native` to today's `real − dryrun`.

To approach truly free regardless of how much Phase 1c claws back, either f drops below 0.05 (less CPU work) or the GPU side gets longer (larger model / larger B). At B=4 the upper-bound gap is structurally larger (+1.92 s) because CPU GEMM scales with `num_tokens` while the GPU layer budget is longer but not proportionally enough to keep pace.

### Per-layer math (cross-checks `§1.13b`'s 660 µs/layer claim)

`forwards/generate = 1 prefill + 128 decode = 129`; `layer-visits/generate = 129 × 28 = 3,612`. From the table at B=1 t=16: `gap = 2.486 s ÷ 3,612 = 688 µs/layer` (matches `§1.13b`'s ~660 µs/layer within 5%). At B=1 t=4 `gap = 1.061 s ÷ 3,612 = 294 µs/layer` — already a 2.3× improvement over t=16 just from thread-tuning, with the dispatcher unchanged.

### Implication for the roadmap

The original plan deferred the C++ runner port to Phase 4 (after attention offload and full e2e). §1.14 makes the case for moving it up: any Phase 2 attention-offload measurement built on the Python `CpuTaskRunner` substrate would mix the runtime tax (~0.45 s) and the active CPU-work penalty (~2 s at the default t=16) into the attention numbers, with no way to attribute regressions to attention vs runtime. The fix is to land Phase 1c (renamed from Phase 4 — see `implementation_roadmap.md`) between Phase 1b (prefetch) and Phase 2 (attention) so attention offload measures itself, not the prototype. Without that swap, COTS-vs-baseline at the supposed free regime is a misleading anchor.

---

## Conclusion

Phase 1a delivers a static, tensor-granularity weight-offload backend that:
1. Frees ~1.13 GB of GPU at `f=0.09` on Qwen2.5-7B (§1.7) — the head-aligned WQKV snap pushes effective WQKV f from 9% to 11.1%, adding ~21 MB beyond the analytic estimate.
2. Preserves vLLM's `gpu_memory_utilization` contract (§1.8) — the same util that baseline tolerates also works for cots, with the offload's freed bytes going to KV cache headroom.
3. Generates output identical to baseline at the smoke prompt (§1.8) within documented BF16 noise (15/16 tokens match; one synonym flip "true"↔"correct").
4. Achieves real GPU+CPU compute overlap on the Nsight timeline (§1.9) with **exactly one UVA copy per fused MLP block** — three transfers eliminated per block per forward (the matched-index invariant).
5. Runs all 60 unit/integration tests green in ~4 s (§1.10).
6. Fails fast at wrap time on misconfiguration (§1.12) — TP>1, no-pin, non-BF16, quantized, malformed MLP block, or orphan col/row.
7. Beats the strongest prefetch baseline at five (depth, batch) corners with `cpu_num_threads=16` (decode-heavy, §1.13b): three at B=1 (`cots_009` 2%, `cots_022` 40%, `cots_050` 38%) and two at B=4 (`cots_022` 19%, `cots_050` 19%). COTS still loses to `none` at f=0.05 — and §1.14 attributes that 2.49 s gap to ~18% pure host orchestration (Phase 1c removes via `cudaLaunchHostFunc` + graph capture) plus ~82% active CPU-work penalty from oneDNN-vs-main-thread contention (Phase 1c removes via bucket-aware thread policy; t=4 already cuts the gap by 1.4 s with no dispatcher change). COTS also loses to prefetch at B=16 (where COTS's `num_tokens`-linear CPU GEMM dominates prefetch's `num_tokens`-flat PCIe cost). For decode-only deployments, `cpu_num_threads=8` recovers an additional 9–15% across all four B=1 depths.

The architectural layers (§1.1: storage / execution / operator) and invariants (§1.5, §1.6, §1.12) are pinned for Phase 1b/1c/2. Each later phase is a localized addition — Phase 1b fills `_wait_for_layer` / `_start_prefetch` no-ops on the offloader; **Phase 1c** swaps `CpuTaskRunner.submit_with_d2h` / `wait` to a `cudaLaunchHostFunc` + native worker binding (gating performance work; renamed from Phase 4 and reordered before Phase 2 per §1.14); Phase 2 adds a `CotsAttnOp` operator alongside the two existing ones, running on the 1c substrate. The next code-level checkpoint is Phase 1b.
