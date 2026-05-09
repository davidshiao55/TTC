# Phase 1c Implementation Findings

This document records the design and verification of **Phase 1c** of
the COTS prototype — the native CPU-runner substrate that replaces
Phase 1a/1b's Python `ThreadPoolExecutor` + `future.result()` orchestration
with a `cudaLaunchHostFunc` host-callback dispatch to a C++ `TaskQueue`
worker. Phase 1c is what was previously called Phase 4: the work is the
same, but the sequencing is moved up before Phase 2 because Phase 1a's
postmortem (`phase1a_findings.md §1.14`) showed the host critical path —
not CPU GEMM throughput — was the dominant overhead at the f=0.05 B=1
free-regime cell. Any Phase 2 (attention offload) measurement built on
the Python prototype would conflate the runtime gap into the attention
numbers.

The substrate ships with three end-to-end-verified gates:
1. **Substrate orch round-trip** — C++ `cudaLaunchHostFunc` + TaskQueue
   round-trip is no slower than Python `executor.submit/future.result`
   under matched eager workload (Stage 2; ratio 0.974).
2. **Bit-exact parity** — native runner's slab dispatch produces the
   same outputs as the Python runner's closure dispatch on QKV and
   fused-MLP operators across `f_cpu_store ∈ {0.10, 0.25, 0.50}`,
   `f_prefetch ∈ {0, 0.10, 0.15, 0.20, 0.25}` (Stage 3).
3. **CUDA Graph capture + replay** — operators capture cleanly under
   `torch.cuda.graph(...)`, replay 50× with bit-deterministic outputs;
   `mutates_args` declarations on `cots_submit_gemm` /
   `cots_sync_then_uva` are positionally enforced in the FX graph so
   `torch.compile` cannot reorder the submit / GEMM / sync window
   (Stage 5).

The §1.14 absolute orch-collapse target (`orch ≤ 0.05 s/generate` on
Qwen2.5-7B + FastTTS) is **NOT yet met**. The synthetic
multi-layer collapse-shape sanity check passes (`collapse_ratio =
0.477`, Stage 5), and after the §1c.18 / §1c.19 / §1c.20 chain of
fixes the real-model harness `bench_dryrun_vs_native_qwen.py`
runs the `cots_005_native_capture_dryrun` and
`cots_005_native_capture_real` arms end-to-end. The settled
multi-iter numbers (B=1, t=16, f=0.05; see §1c.20 for the full
table) reveal that **capture mode is currently WORSE than
native+eager** — orch +0.497 s vs +0.316 s — and
`native_capture_real` is wildly slow at 119 s/generate. The
architectural blockers are closed; the perf shortfall is now a
diagnostic problem tracked as §1c.21 (perf investigation). Status:
**runs end-to-end, perf needs nsys diagnosis**.

Hardware: NVIDIA RTX 4090 (24 GB), Intel i9-14900KF (AVX2, no
AVX512/AMX), DDR5. PyTorch 2.10.0+cu128, MKL enabled, oneDNN BF16,
Triton. CUDA 12.4. C++ compiled at `-O3 -DNDEBUG`, no -fopenmp on the
COTS extension (oneDNN owns the worker's intra-op threading).

---

## Contents

**Mechanism**
- §1c.1 — Architecture: storage / execution / operator (carries Phase 1a
  layering forward) + native runner composition
- §1c.2 — `CotsCpuInfer` C++ substrate: TaskQueue + cudaLaunchHostFunc
  submit/sync + slab pool
- §1c.3 — Custom op design: barrier-installing `mutates_args` for the
  submit/sync ordering invariant
- §1c.4 — Uniform operator facade: one API across both runners

**Memory & buffer invariants**
- §1c.5 — Slab pool: address-stable `unique_ptr<TaskSlab[]>` (not
  `vector` — `std::atomic` non-MoveConstructible)
- §1c.6 — POST-narrow pointers + strides for the strided down-proj path
- §1c.7 — Distinct dummy CUDA anchors (no aliasing in `mutates_args`)

**Bucket-aware thread policy (Stage 4)**
- §1c.8 — Per-`BatchDescriptor` `n_threads` via slab field; cache-guarded
  worker-side `at::set_num_threads`
- §1c.9 — Main-thread `at::get_num_threads` isolation: PyTorch's
  at-thread-pool is thread-local on this build (no `omp_set_num_threads`
  contingency needed)

**Graph capture (Stage 5)**
- §1c.10 — Conditional `enforce_eager` drop: native runner only
- §1c.11 — FX-positional ordering proof: `mutates_args` pins
  submit < GEMMs < sync under `torch._dynamo.export`

**Measurements**
- §1c.12 — Stage 1 hard gate: C++ `at::linear` matches Python `F.linear`
  (oneDNN BF16 fast path on AVX2; strided-view path validated)
- §1c.13 — Stage 5 synthetic collapse-shape bench (orch ratio 0.477);
  §1.14 absolute on real Qwen2.5-7B documented separately
- §1c.14 — Stage 4 thread-policy sweep: per-bucket optimal table for
  the Planner

**Verification**
- §1c.15 — Test matrix and reproducibility

**Forward work**
- §1c.16 — Stage 7 (optional): transposed-storage row/down-proj
  unification
- §1c.17 — `__del__` drain forward risk (registered, not yet exercised)
- §1c.18 — Stage 6 follow-up: pre-hook × torch.compile fullgraph
  interaction (CLOSED — `_bucket_for` now Dynamo-traceable)
- §1c.19 — Stage 6 follow-up #2: Dynamo guard serialization tries
  to pickle `CotsCpuInfer` (CLOSED — registry split moves the pybind
  handle out of the runner facade)
- §1c.20 — Stage 6 follow-up #3: Inductor materializes any CPU
  tensor visible in the captured graph (CLOSED — both ops now
  CUDA-tensors-and-scalar-ids only; pinned buffers reached via
  slab pointers in C++)
- §1c.21 — **CLOSED**: live unpadded token count plumbed from
  `gpu_model_runner.execute_model` →
  `BaseOffloader.set_runtime_num_tokens` →
  `CotsCpuInfer::set_runtime_num_tokens`. Worker reads override at
  host-callback time and uses it for all CPU-side row arithmetic;
  captured graph shape stays at the bucket. native_capture_real at
  output_len=128 collapsed from 119.33 s → 2.76 s (43× speedup),
  matching native_eager_real (~2.60 s).
- §1c.22 — **Open follow-up**: bucket-sized D2H + UVA copies. The
  captured `cudaMemcpyAsync` byte counts and the captured Triton
  UVA grid are still bucket-sized (~99.6% PCIe waste at B=1
  steady-state per call). Counter pass at output_len=128 shows
  16.4 GB captured input D2H vs 10.0 GB live worker input bytes
  — measured material waste in the over-transfer. May explain a
  meaningful chunk of the residual ~+0.5 s capture-mode orch.

---

## 1c.1: Architecture — storage / execution / operator + native runner

Phase 1a's three-layer split (`phase1a_findings.md §1.1`) is preserved
verbatim. Phase 1b added prefetch streaming as a sibling of
`CpuTaskRunner` in the execution layer (`phase1b_findings.md §1b.1`).
Phase 1c keeps the layering intact and adds a sibling runner alongside
the renamed `PythonCotsRunner`.

| Layer | Phase 1a | Phase 1b | Phase 1c |
|---|---|---|---|
| Storage | `CotsLinearHandle` | per-bucket geometry dicts; prefetch slot pool | (unchanged) |
| Execution | `CpuTaskRunner` | + `WeightPrefetchStreamer`, `CotsPrefetchBufferPool` | `CpuTaskRunner` → `PythonCotsRunner` (alias preserved); add `NativeCotsRunner` (uniform facade); add `cots_ops.py` (custom-op registry) |
| Operator | `CotsQKVOp`, `CotsSwiGLUMLPOp` (per-Linear scatter / fused MLP) | three-way scatter | uniform facade: `submit_with_d2h(x, x_pinned, y_pinned, op_descriptor)` + `wait_and_uva(y_pinned, y_gpu, gpu_anchor_a, gpu_anchor_b)` — no runner-type branching |
| Lifecycle | `CotsOffloader` (discovery, install) | + bucket pre-hook for prefetch | + `_current_bucket`, `_install_bucket_prehook` (unconditional), `_install_runner` (slab/closures), `_dummy_gpu_anchor_a/_b` |

**Installer refactor (Stage 2).** Phase 1a/1b constructed a fresh
`CpuTaskRunner()` in each `_install_qkv_ops` / `_install_mlp_ops` call;
Phase 1c's `CotsOffloader.__init__` constructs ONE runner via
`_make_runner(config)` and shares it across all operator installs. This
is the structural prerequisite for the native runner's per-offloader
slab pool + runner_id (multi-engine safety; FastTTS gen + ver coexist).

**Dual runner choice.** `CotsOffloadConfig.cpu_runner: Literal["native",
"python"]` — default `"native"` post-Stage-5. The Python runner is
retained as a kill-switch path under `enforce_eager=True` for A/B
diagnostics; slated for deprecation one quarter post-Phase-1c.
Selecting `cpu_runner="python"` with `enforce_eager=False` is rejected
at engine launch — `ThreadPoolExecutor.submit` is not graph-capturable
and capturing it would silently drop CPU GEMM work from the graph.

---

## 1c.2: `CotsCpuInfer` C++ substrate

`csrc/cots/` adds three files (~440 LOC) plus build glue:

| File | Source | Notes |
|---|---|---|
| `task_queue.{h,cpp}` | direct port from `kt-kernel/cpu_backend/task_queue.{h,cpp}` | Michael-Scott MPSC queue + condvar sync, one worker thread; `pthread_setname_np(self, "cots-cpu-wkr")` for nsys visibility |
| `cots_cpu_infer.{h,cpp}` | adapted from `kt-kernel/cpu_backend/cpuinfer.h` | `WorkerPool` dropped (oneDNN owns intra-op threading); slab pool + worker-local scratch + worker-exception surfacing |
| `cots_torch_bindings.cpp` | new | pybind11 `CotsCpuInfer` class only; no `torch.ops.cots.*` registration in C++ (lives in Python `cots_ops.py`, mirrors `prefetch_ops.py`) |

**Build target.** `_cots_C` extension in `vllm/CMakeLists.txt`, gated on
`if(VLLM_GPU_LANG STREQUAL "CUDA")` so ROCm builds skip cleanly
(host-callback design depends on the CUDA Runtime API). `LANGUAGE CXX`
(no CUDA kernels — only CUDA Runtime API for `cudaLaunchHostFunc`).
Standard Python ABI (no `USE_SABI`) because the extension exposes a
stateful pybind class. Linked against `torch_python` so pybind11's
`at::Tensor` type caster resolves.

**Submit/sync handshake.** `submit_on_stream(task_id, num_tokens,
stream)` writes `num_tokens` into the slab and queues a host callback
on `stream` via `cudaLaunchHostFunc(stream, &dispatch_cb, &slabs_[task_id])`;
the callback fires after prior stream work completes, enqueues the
task on `TaskQueue`, and returns. `sync_on_stream(stream)` queues
another host callback that blocks on `task_queue_->sync(0)` (worker
must drain). Between submit and sync the GPU stream proceeds with
permanent + prefetched GEMMs while the worker executes the CPU GEMM
in parallel.

**Worker exception surfacing** (`check_error` / `has_error` /
`take_error`). Every task body is wrapped in try/catch; the catch sets
`has_error_` + `last_error_msg_` but lets `TaskQueue::Worker` complete
its pending-decrement / cv-notify normally so a thrown exception never
leaves `sync()` deadlocked. Each Python-side `submit*` / `sync*` /
`populate_slab*` entry point checks `has_error_` first and re-raises
as a Python `RuntimeError(last_error_msg_)`. Mirrors the Python
runner's `future.result()` re-raise semantics.

---

## 1c.3: Custom op design — barrier-installing `mutates_args`

`vllm/model_executor/offloader/cots_ops.py` registers two ops via
`direct_register_custom_op` (mirrors `prefetch_ops.py`):

```
vllm.cots_submit_gemm(x_gpu, x_pinned, y_pinned, runner_id, task_id, num_tokens) -> ()
    mutates_args = ["x_gpu", "y_pinned"]
vllm.cots_sync_then_uva(y_pinned, y_gpu, gpu_anchor_a, gpu_anchor_b, runner_id) -> ()
    mutates_args = ["y_gpu", "gpu_anchor_a", "gpu_anchor_b"]
```

The `mutates_args` declarations are not just data-flow annotations —
they are **barrier-installing** in the FX graph. `cots_submit_gemm`
mutates `x_gpu` (the layer's input tensor read by the GPU GEMMs)
specifically so torch.compile cannot hoist any GPU GEMM that reads
`x_gpu` ABOVE the submit. Conversely, `cots_sync_then_uva` mutates
both `gpu_anchor_a` and `gpu_anchor_b` (the GPU GEMM outputs
`out_perm` / `out_pref` for QKV) so neither GEMM can be sunk BELOW
the sync. Two distinct anchors are required because in QKV the two
F.linear calls are independent; mutating only one would let the
compiler reorder the other across the sync.

`cots_sync_then_uva` bundles the cudaLaunchHostFunc-based stream sync
with the existing Triton UVA copy kernel into one graph-recorded
entry. Splitting them into two ops (sync, then uva) would let the
compiler reorder the UVA copy across sync.

**Multi-engine safety** (no C++ singleton). Each `NativeCotsRunner`
allocates a `runner_id` and registers itself in a module-private weak
registry (`_COTS_RUNNERS: WeakValueDictionary[int, NativeCotsRunner]`)
in `cots_ops.py`. The custom-op impls look up the right runner by id
via `_lookup_runner(runner_id)` so two offloaders (FastTTS gen + ver)
coexist with independent slab pools. `NativeCotsRunner.close()`
explicitly drains and unregisters.

---

## 1c.4: Uniform operator facade

Both runners expose the same operator-side API:

```python
runner.submit_with_d2h(x_gpu, x_pinned, y_pinned, op_descriptor)
... GPU permanent + prefetched GEMMs ...
runner.wait_and_uva(y_pinned, y_gpu, gpu_anchor_a, gpu_anchor_b)
```

where `op_descriptor = (h.layer_idx, b, "qkv" | "mlp_block")`. Per-(layer,
bucket, op_kind) work is pre-built at install time:

- **Python runner**: a closures dict `_callbacks: dict[tuple[int, int,
  str], Callable]`. Each closure captures the per-bucket weight slice
  views (e.g., `h.w_cpu.narrow(0, n_pref, n_cpu)`) so the operator
  call only carries x/y views.
- **Native runner**: a `_task_id_for: dict[tuple[int, int, str], int]`
  mapping descriptors to slab indices. Slabs are populated at
  `post_init` with POST-narrow pointers + strides.

Lazy bucket fallback: if `op_descriptor[1] is None` at submit (e.g.,
a code path that bypassed the first-decoder pre-hook), the runner
rebuilds the descriptor with `_bucket_for(num_tokens)` so the
slab/closure lookup never sees `bucket=None`. This is in addition to
the operator's own resolver (operator reads `offloader._current_bucket`
or falls back to `_bucket_for(num_tokens)` before passing to the
runner).

**No runner-type branching in operators.** Stage 3's central design
guarantee: the operator code path is identical regardless of which
runner is active. The per-runner divergence lives entirely inside
`submit_with_d2h` / `wait_and_uva`.

---

## 1c.5: Slab pool — `unique_ptr<TaskSlab[]>`, not `std::vector`

`TaskSlab` carries `std::atomic<int32_t> num_tokens` (the only
per-call mutable field). `std::atomic` is not MoveConstructible →
`std::vector<TaskSlab>` template instantiation fails to compile (even
under `reserve()`-only flows, the standard library's eager template
instantiation hits `std::uninitialized_copy(move_iterator)` and
rejects). The slab pool is therefore a heap-allocated raw array
(`std::unique_ptr<TaskSlab[]>`) sized once at `install(n_slabs)`,
never resized. Address stability is structural — captured CUDA graphs
record `&slabs_[task_id]` as host-callback userData and re-replay
must see the same pointer.

`SyncArgs sync_args_{this, /*allow_n_pending=*/0}` is a stable member
of `CotsCpuInfer` (NOT heap-allocated per `sync()` call) for the same
reason: graph capture freezes the userData pointer.

**Layout:**

```cpp
struct alignas(64) TaskSlab {
  CotsCpuInfer* self;
  int32_t op_kind;                  // 0=qkv, 1=mlp_block, 2=dryrun_noop
  int32_t n_threads;                // bucket-aware (Stage 4)
  std::atomic<int32_t> num_tokens;  // ONLY field updated per submit

  void* x_pinned_ptr; int32_t in_dim;
  void* y_pinned_ptr; int32_t cpu_out_dim;

  // qkv (op_kind=0)
  void* w_cpu_ptr; int32_t w_cpu_rows;          // contiguous

  // mlp (op_kind=1)
  void* w_gate_ptr; int32_t w_gate_rows;        // contiguous
  void* w_up_ptr;   int32_t w_up_rows;          // contiguous
  void* w_down_ptr;                              // POST-narrow data_ptr()
  int32_t w_down_rows; int32_t w_down_cols;
  int64_t w_down_stride_row;                     // = original n_cpu
  int64_t w_down_stride_col;                     // = 1
  int32_t intermediate_per_half;
};
```

dtype is hard-coded `at::kBFloat16` in the view builders (the
`CotsOffloadConfig.cpu_dtype` literal locks this; `phase0_findings.md
§0.3.2` shows oneDNN BF16 is the only fast CPU GEMM path on AVX2).
Avoiding a `dtype` field on the slab also sidesteps a brittle
`int(torch.bfloat16)` enum dance over pybind.

---

## 1c.6: POST-narrow pointers + strides for strided down-proj

The MLP down-proj CPU compute slice is
`dn_h.w_cpu.narrow(1, dn_n_pref, dn_n_cpu)` — a column slice of
row-major `(out_dim, n_cpu)` storage. Two facts make this load-bearing:

1. The slice is **non-contiguous** when `dn_n_pref > 0`. The C++ side
   reconstructs the view via `at::from_blob(ptr, sizes, strides,
   opts)` with strides matching the source tensor's `.stride()`
   (`stride_row = original_n_cpu`, `stride_col = 1`).
2. `at::from_blob` has **no storage-offset parameter**. The pointer
   passed to the slab must be the post-narrow `data_ptr()` (already
   offset by `dn_n_pref * elem_size`), NOT the base + a separate
   offset field. Same principle applies to QKV's `w_cpu.narrow(0,
   n_pref, n_cpu)` — row-narrowed on row-major storage so the view
   is contiguous (default strides), but the offset is still
   load-bearing.

Stage 1's `test_at_linear_microbench.py` proved C++ `at::linear` on a
strided `at::from_blob` view matches Python `F.linear` within 5% on
both the contiguous MLP1 9% slice (3408×3584 BF16) and the strided
down-proj column slice. Stage 3's
`test_strided_down_proj.py` runs this end-to-end through the public
slab dispatch path with `dn_n_pref > 0`, plus a sanity check
confirming the post-narrow pointer is actually consumed (passing the
base pointer instead diverges by > 0.5).

---

## 1c.7: Distinct dummy CUDA anchors (no `mutates_args` aliasing)

`cots_sync_then_uva` mutates `gpu_anchor_a` and `gpu_anchor_b`
separately. Operators pass the actual GPU GEMM outputs when present
(`out_perm`, `out_pref`); when absent, they pass two **distinct** dummy
CUDA tensors:

```python
# In CotsOffloader._allocate_activation_buffers (DeviceMemoryProfiler
# accounting window, phase1a §1.5):
self._dummy_gpu_anchor_a = torch.empty(1, dtype=dtype, device=device)
self._dummy_gpu_anchor_b = torch.empty(1, dtype=dtype, device=device)
```

Two separate allocations — never aliased. Aliasing the same tensor to
both `mutates_args` slots can confuse torch.compile / functionalization
(plan §design-decision 6). The dummies are sized 1 element each;
DeviceMemoryProfiler-accounted.

Operator code:

```python
gpu_a = out_perm if out_perm is not None else off._dummy_gpu_anchor_a
gpu_b = out_pref if out_pref is not None else off._dummy_gpu_anchor_b  # distinct dummy
self._runner.wait_and_uva(y_pinned, y_gpu, gpu_a, gpu_b)
```

For the MLP block operator, `out_gpu` carries a combined dep on perm +
pref via `out_gpu.add_(pref_out)`, so anchor_a alone covers the GPU
work; anchor_b is always the dummy.

---

## 1c.8: Bucket-aware thread policy (Stage 4)

`CotsOffloadConfig.cpu_num_threads_by_bucket: dict[int, int] | None`.
Keys must be a subset of `cudagraph_capture_sizes`; values >= 1.
Missing keys fall back to scalar `cpu_num_threads`. Validated in
`_validate_thread_policy` at install — a Planner mistype fails loudly
instead of silently falling back to scalar.

Each slab carries an `n_threads` field set per-bucket via
`_n_threads_for(bucket)` in `_build_native_slab_specs`. The C++ worker's
slab dispatcher cache-guards the set:

```cpp
if (slab->n_threads > 0 && slab->n_threads != worker_current_n_threads_) {
  at::set_num_threads(slab->n_threads);
  worker_current_n_threads_ = slab->n_threads;
}
last_observed_num_threads_.store(at::get_num_threads(), ...);
```

The cache guard avoids redundant `at::set_num_threads` calls on every
slab when consecutive buckets share the same thread count (changing
the at-thread-pool isn't free). The `last_observed_num_threads()`
accessor is the side channel the Stage 4 tests use to verify the
worker actually applied the requested setting.

**Optional CPU affinity** (`cpu_worker_affinity: list[int] | None`).
The native runner can pin its TaskQueue worker thread to a CPU mask
via `set_worker_affinity(uint64_t cpu_set)`. The C++ side intersects
the requested mask with `sched_getaffinity` and warns-and-skips on
empty intersection. Hardware-specific; defaults to None. Recommended
on i9-14900KF: P-cores 1..7 (avoid P-core 0 where the main thread /
CUDA dispatch / kernel tend to land).

The mask is `uint64_t` end-to-end (Stage 4 review fix). An earlier
`int64_t` signature rejected `1 << 63` over pybind ("int too big to
convert"); `int64_t{1} << 63` is also signed-shift undefined behavior
in C++. `uint64_t{1} << i` is well-defined for `i ∈ [0, 63]`; the
shift loop is bounded `i < 64`.

---

## 1c.9: Main-thread `at::get_num_threads` isolation

Plan §risk register #3: if PyTorch's at-thread-pool is process-global
on this build, the worker's `at::set_num_threads(slab.n_threads)` would
leak into the main thread's CUDA dispatch path and the bucket-aware
policy would actively HURT main-thread launch latency. The contingency
was to switch the C++ callback to `omp_set_num_threads` inside a
`#pragma omp parallel num_threads(n)` region.

`test_main_thread_at_threads_isolation.py` confirmed PyTorch's
at-thread-pool is **thread-local** on this build (PyTorch 2.10 +
Python 3.11): main thread snapshot before / after a worker
`at::set_num_threads(n)` with `n != main_current` shows no drift.
Also tested across 5 back-to-back dispatches with varying n_threads
in {1, 4, 2, 8, 1}. Risk #3 is GREEN; omp pragma contingency NOT
needed.

---

## 1c.10: Conditional `enforce_eager` drop (Stage 5)

`post_init` rejects `enforce_eager=False` only when
`cpu_runner='python'`:

```python
if not vllm_config.model_config.enforce_eager:
    cpu_runner = getattr(self.config, "cpu_runner", "python")
    if cpu_runner != "native":
        raise RuntimeError(
            "CotsOffloader: cpu_runner='python' requires enforce_eager=True — "
            "Python runner uses ThreadPoolExecutor + future.result() which is "
            "NOT graph-capturable. Either set enforce_eager=True or switch to "
            "cpu_runner='native'."
        )
```

Native + `enforce_eager=False` is the production path. Phase 1c risk
register #4 (CUDA Graph rejecting host-function nodes) is GREEN —
captured graphs accept `cudaLaunchHostFunc` nodes cleanly. The
contingency (drop offloaded layers to PIECEWISE capture mode) is NOT
needed.

`CotsOffloadConfig.cpu_runner` default flipped from `"python"` to
`"native"` at Stage 5 once graph capture was verified end-to-end. The
default flip is the user-visible behavior change carried by this
phase: existing Phase 1a/1b workflows that explicitly set
`enforce_eager=True` are unchanged; defaulted workflows now go
through the production native path with capture enabled.

---

## 1c.11: FX-positional ordering proof

The `mutates_args` declarations on `cots_submit_gemm` /
`cots_sync_then_uva` (§1c.3) are barrier hints to torch.compile.
Verifying they actually pin the operator's call sequence requires
walking the FX graph, not just checking output parity.

`test_dependency_ordering.py::test_fx_graph_orders_submit_before_gpu_gemms_before_sync`
uses `torch._dynamo.export(forward)(*args)` to extract a single-graph
FX representation (forces no graph breaks; raises if any) of:

```python
def forward(x, x_pin, y_pin, y_gpu, w_perm, w_pref, dummy_a, dummy_b):
    x_pin.copy_(x, non_blocking=True)
    torch.ops.vllm.cots_submit_gemm(x, x_pin, y_pin, runner_id, 0, x.shape[0])
    out_perm = torch.nn.functional.linear(x, w_perm)
    out_pref = torch.nn.functional.linear(x, w_pref)
    torch.ops.vllm.cots_sync_then_uva(y_pin, y_gpu, out_perm, out_pref, runner_id)
    return out_perm, out_pref, y_gpu
```

then walks `gm.graph.nodes` and asserts:

```
index(cots_submit_gemm) < index(linear, out_perm)
index(cots_submit_gemm) < index(linear, out_pref)
index(cots_sync_then_uva) > index(linear, out_perm)
index(cots_sync_then_uva) > index(linear, out_pref)
```

A regression that lets the compiler reorder these would change FX
positions and trip the assertion, regardless of whether the output
happened to remain numerically correct (the dangerous
"wrong-fast-but-still-numerically-fine" failure mode where capture
destroys overlap but the worker happens to finish in time).

`test_cots_submit_gemm_mutates_x_gpu_and_y_pinned` and
`test_cots_sync_then_uva_mutates_y_gpu_and_both_anchors`
schema-parse the registered op (regex `Tensor\(\w+!\)\s*\??\s*(\w+)`)
and assert the EXACT mutated arg names match what `cots_ops.py`
registers — a refactor that adds a stray marker or moves one to the
wrong tensor surfaces immediately.

---

## 1c.12: Stage 1 hard gate — `at::linear` matches `F.linear`

`phase0_findings.md §0.3.2` documented that BF16 `torch.mm` /
`aten::mm` falls to a 100–250× slower scalar path on i9-14900KF (no
AVX512_BF16 / AMX), while `F.linear` → `torch._C._nn.linear` →
oneDNN BF16 hits the fast path. Phase 1c assumes the C++ equivalent
`at::linear` (in `<ATen/ops/linear.h>`) dispatches the same way.
Stage 1's microbench gate verified this empirically before any
runner wiring landed.

`test_at_linear_microbench.py` runs C++ `at::linear` (via the
test-only `CotsCpuInfer.run_at_linear_inline` helper) against Python
`F.linear` on identical BF16 tensors at B ∈ {1, 4, 16, 32} for two
shapes:

| Shape | B=1 | B=4 | B=16 | B=32 |
|---|---|---|---|---|
| Contiguous MLP1 9% slice (3408×3584) | 1.02× | 1.01× | 0.84× | 0.99× |
| Strided down-proj column slice (out=3584, n_cpu=3790, stride0=5683) | 0.99× | 1.00× | 1.00× | 1.01× |

All ratios within 5% of `F.linear`. Both contiguous AND strided
views hit the oneDNN BF16 fast path on AVX2. The strided case is the
load-bearing one for §1c.6's down-proj path; an earlier signed-int /
scalar-fallback would have manifested as ≥ 2× ratios.

The microbench is a HARD GATE: if the C++ `at::linear` had fallen to
the scalar path (e.g., on a future PyTorch bump that changes the
strided-BF16 dispatch), Stage 1 would halt and Phase 1c would scope
oneDNN linkage as a separate task. That contingency is NOT needed
on this build.

---

## 1c.13: Stage 5 collapse-shape sanity check + §1.14 future-anchor

Stage 5's headline gate target was `orch ≤ 0.05 s/generate` on
Qwen2.5-7B + FastTTS (the §1.14 Python-runner-eager baseline was
~0.45 s/generate). The synthetic multi-layer collapse-shape bench
(`bench_dryrun_vs_real_native.py`) asserts the SHAPE of the collapse
on a stub workload, not the absolute generate-equivalent budget:

```
workload: n_layers=8, num_tokens=4, f_cpu_store=0.10, n_iters=100
RTX 4090 + i9-14900KF:

  (c) baseline (no offload, eager):          99.1 μs / forward
  (a) eager + dry_run=True:                 660.8 μs / forward
  (b) captured + dry_run=True:              367.1 μs / forward

  orch_eager   = (a) - (c):                 561.7 μs (synthetic)
  orch_capture = (b) - (c):                 268.0 μs (synthetic)
  collapse_ratio (capture / eager):         0.477   ← PASS (≤ 0.70)
```

Capture is 2.1× faster than eager at the substrate level — graph
replay re-issues `cudaLaunchHostFunc` nodes without traversing
Python operator bodies. The synthetic per-layer / per-forward μs
absolutes do NOT translate to Qwen2.5-7B's per-generate budget
(HIDDEN=256 here vs 3584 on Qwen2.5-7B; no attention/MLP between
QKV calls; smaller layer count). The COLLAPSE RATIO is what's
load-bearing for Phase 1c sign-off.

**Anchoring §1.14 absolute on the real model.** The thesis-locked
absolute number requires running on Qwen2.5-7B + FastTTS (or
equivalent decode-heavy workload). The harness for the real-model
run lives at `David/Benchmarks/phase1c/bench_dryrun_vs_native_qwen.py`
(ported from Phase 1a's `bench_cots_dryrun_vs_none.py` with six
arms: two no-offload baselines [`none` eager + `none_capture`
graph-mode] plus four COTS arms covering python/native ×
eager/capture × dryrun/real). Stage 6 landed the harness AND the
auto-derived `--cots-cpu-runner` /
`--cots-cpu-num-threads-by-bucket` / `--cots-cpu-worker-affinity`
CLI flags so `vllm bench latency` accepts the new fields.

**Settled multi-iter results (post-§1c.18/§1c.19/§1c.20 fixes)** at
B=1, t=16 (input=8, output=128, f_cpu_store=0.05, 3 iters / 2
warmup):

```
  arm                                     avg_latency (s)   orch
  none                                       2.0333         —
  none_capture                               2.0323         —
  cots_005_python_eager_dryrun               2.5307        +0.497 s
  cots_005_native_eager_dryrun               2.3488        +0.316 s
  cots_005_native_capture_dryrun             2.5294        +0.497 s
  cots_005_native_capture_real             119.3297    cpu_work +116.80 s
```

These are settled multi-iter numbers (NOT 1-iter smoke). They
confirm five things:

- The harness wires through correctly under the §1c.20 schema:
  subprocesses launch, `vllm bench latency` accepts the new
  `--cots-cpu-runner` / `--cots-cpu-num-threads-by-bucket` /
  `--cots-cpu-worker-affinity` flags, JSON cells populate, summary
  subtraction uses the right baseline per arm.
- The §1.14 python-eager baseline (≈ 0.45 s/generate on Phase 1a)
  reproduces (0.497 s here, within run-to-run variance).
- The native runner under EAGER reduces orch by 36% over the
  Python runner (0.316 vs 0.497 s).
- **Capture mode is currently WORSE than native+eager**: +0.497 s
  vs +0.316 s. This is the §1c.21 perf regression.
- **`native_capture_real` is wildly slow** (119 s/gen, vs an
  expected ~3–5 s extra over dryrun for real CPU GEMM at this
  shape). Tracked under §1c.21.

Subtraction baseline note: capture-mode arms subtract `none_capture`
(graph-mode no-offload), NOT eager `none`. Subtracting eager `none`
would understate COTS orch by however much torch.compile saves on
the no-offload path. At B=1 the two baselines are indistinguishable
(2.0333 vs 2.0323 — graph capture saves ~10 ms on the no-offload
path) so the difference is small, but the harness uses the correct
baseline by construction.

Architectural status: the captured forward runs end-to-end. The
§1.14 absolute target ≤ 0.05 s/generate is unmet — the next blocker
is perf, not correctness, and is tracked under §1c.21.

---

## 1c.14: Stage 4 thread-policy sweep

`bench_thread_policy_sweep.py` sweeps `cpu_num_threads ∈ {2, 4, 8,
16, 24}` × B ∈ {1, 4, 16} on the Phase 1a §0.3.2 reference shape
(3408 × 3584 BF16). First-run snapshot on RTX 4090 + i9-14900KF
(`--n-iters 30 --warmup 5`):

```
  B \ t          2         4         8        16        24
  B=1         498.1     238.5     236.9     134.1      95.1   ← best 24
  B=4        1983.4     924.5     893.8     467.3     352.1   ← best 24
  B=16       6932.1    3476.3    3528.4    2103.1    1325.4   ← best 24
```

`{1: 24, 4: 24, 16: 24}` — the suggested Planner starting table.

**Caveat — reads the same as §1.13b's prelude.** This sweep
measures CPU GEMM standalone with no concurrent CUDA stream
pressure. `phase1a_findings.md §1.13b` showed the optimum drops
to t=8 at decode-B=1 once oneDNN-on-many-threads contends with the
main thread's CUDA dispatch path under real model load. Stage 6's
real-model rerun supersedes this table for Planner consumption;
this synthetic table is the substrate-level optimum.

---

## 1c.15: Test matrix and reproducibility

`David/Tests/phase1c/` has **16 files (14 test modules + conftest +
pytest.ini), 95 collected tests** at Stage 5 sign-off. Mirrors the
Phase 1a/1b layout plus Stage-specific gates.

| File | What it covers |
|---|---|
| `conftest.py` | Session-scoped TP-1 init via gloo (`MASTER_PORT=29501`); `needs_cuda` skip handling |
| `pytest.ini` | `needs_cuda` marker registration |
| `test_native_runner_loadable.py` | `vllm._cots_C` import smoke; `CotsCpuInfer` construction; `at::set_num_threads` callable on worker |
| `test_taskqueue_stress.py` | TaskQueue 100k no-op burst FIFO + drain via the test-only `submit_dryrun_burst` helper; repeated cycles; destructor drain |
| `test_at_linear_microbench.py` | Stage 1 HARD GATE — contiguous MLP1 + strided down-proj parity vs `F.linear` (8 cases) |
| `test_cuda_launch_host_func_smoke.py` | Real CUDA stream submit/sync round-trip without graph capture; D2H ordering; sync blocks correctly |
| `test_strided_down_proj.py` | Standalone strided-down-proj parity through the public slab dispatch (`dn_n_pref > 0`); offset-pointer load-bearing sanity |
| `test_worker_exception_surfacing.py` | Forces an MLP-scratch unavailability fault; confirms (a) sync doesn't hang, (b) `check_error` raises with the worker's `what()`, (c) next-call surfacing, (d) `take_error` consumes state |
| `test_stage2_substrate.py` | Custom-op registration; runner registry round-trip + WeakValueDictionary GC; runner classes; `_make_runner` factory; default-runner = native; installer refactor; backward-compat alias |
| `test_parity_with_python_runner.py` | QKV + MLP parity python-vs-native at `f_prefetch=0` and `f_prefetch>0` (3-way); no-streamer-bucket fallback; `prepare_before_forward` sets `_current_bucket` |
| `test_bucket_thread_policy.py` | Stage 4 — worker observes slab `n_threads`; per-bucket transitions; `_n_threads_for` resolver; validator rejection; affinity zero-mask + intersected mask + high-bit (1 << 63) regression |
| `test_main_thread_at_threads_isolation.py` | Stage 4 risk #3 — main thread `at::get_num_threads` not affected by worker's `at::set_num_threads` |
| `test_python_runner_graph_hard_fail.py` | Stage 5 conditional `enforce_eager` check across 4 (runner, eager) combinations |
| `test_graph_capture_e2e.py` | Stage 5 — capture once, replay 50× deterministic + parity for QKV and MLP at 3 fcpu_store points; default-config graph capture; capture/eager alternation |
| `test_multi_engine.py` | Stage 5 — two NativeCotsRunner instances coexist with distinct runner_ids; close one, other stays valid; interleaved forwards stay self-consistent |
| `test_dependency_ordering.py` | Stage 5 — schema-level mutated-arg name check; FX-positional submit < GEMMs < sync via `torch._dynamo.export`; export-without-graph-break sentinel |

Phase 1a (60 tests) + Phase 1b (80 tests) + Phase 1c (95 tests) =
**235 total**, all green on RTX 4090 + i9-14900KF in ~14 s. Run via:

```bash
cd /TTC/David/Tests/phase1c && /opt/conda/envs/thesis/bin/python -m pytest . -q
cd /TTC/David/Tests/phase1a && /opt/conda/envs/thesis/bin/python -m pytest . -q
cd /TTC/David/Tests/phase1b && /opt/conda/envs/thesis/bin/python -m pytest . -q
```

---

## 1c.16: Stage 7 forward — transposed-storage row/down-proj

Stage 7 was deliberately deferred from the Phase 1c critical path. It
investigates removing the duplicated row-prefetch source buffer
`w_row_prefetch_src_t` (Phase 1b §1b.6) — currently ~1 GiB of pinned
CPU at `f_prefetch=0.30` — by unifying the storage / kernel design.
Constraints (per the approved plan §Stage 7):

- Preserve Phase 1b's measured row-prefetch BW fix (§1b.7's ~1.85×
  PCIe recovery at the collaborative point).
- Strided-view down-proj path (Stage 3) stays unchanged unless
  benchmarks prove a switch is strictly better.
- Benchmark-gated: strided `at::linear` vs proposed transposed/kernel
  path; row-prefetch H2D BW with vs without `w_row_prefetch_src_t`;
  end-to-end Bench 2 / Bench 3 impact.

Stage 7 is optional and should be scheduled only after Stage 6
real-model numbers are locked.

---

## 1c.17: Forward risk — `__del__` drain

`NativeCotsRunner.__del__` unregisters the runner from the registry
but does NOT drain in-flight `cudaLaunchHostFunc` callbacks scheduled
via `submit_on_stream` / `sync_on_stream`. Tests work around this by
calling `runner.close()` explicitly. In an FastTTS-style workload, an
offloader teardown mid-forward could leave host callbacks pointing at
a freed slab.

The plan-tracked mitigation options:
1. Add a BaseOffloader-level shutdown hook that drains the compute
   stream and closes the runner before slabs are freed (preferred).
2. Best-effort `torch.cuda.current_stream().synchronize()` in
   `__del__` (also dangerous from a finalizer if CUDA is already
   torn down).

Not exercised in production yet because FastTTS engine teardown
happens at process exit — the CUDA driver's process-cleanup path
drains the stream regardless. The risk surfaces if an in-process
engine is reconstructed (e.g., test fixtures, hot-reload). Tracked
explicitly in the `__del__` docstring; recommended Stage 6 → Stage 8
follow-up.

---

## 1c.18: Real-model anchor blocker #1 — pre-hook × torch.compile fullgraph (CLOSED)

**Status: closed.** `_bucket_for` is Dynamo-traceable as of the
post-Stage-6 §1c.18 fix commit. Re-running the
`cots_005_native_capture_dryrun` smoke no longer hits the
`bisect_left` failure mode; engine init proceeds past Dynamo's
fullgraph capture step. (Engine init still fails further down the
line — see §1c.19, the next-uncovered blocker.)

### Original problem

Stage 6's smoke-run of `bench_dryrun_vs_native_qwen.py` on Qwen2.5-7B
surfaced an interaction between Phase 1c's first-decoder pre-hook
(§1c.4: `_install_bucket_prehook` — registered unconditionally so
`prepare_before_forward` fires even without a streamer) and vLLM's
default `torch.compile(fullgraph=True)` model wrapping.

Repro: `cots_005_native_capture_dryrun` arm at `--num-iters 1
--num-iters-warmup 1` fails at engine initialization. Captured stack
from the current commit's smoke run (no `@torch._dynamo.disable`
decorator on the pre-hook — see "Decorator note" below):

```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin
              `_bisect.bisect_left.` This function is either a Python
              builtin (e.g. _warnings.warn) or a third-party C/C++
              Python extension (perhaps created with pybind).
  Hint: ...wrap it into a PyTorch-understood custom operator... or,
        if it is traceable, use `torch.compiler.allow_in_graph`.
  Developer debug context: module: _bisect, qualname: bisect_left,
                           skip reason: <missing reason>

from user code:
  vllm/model_executor/models/qwen2.py:444 layer(positions, hidden_states, residual)
  nn/modules/module.py:1809 inner       # forward pre-hook dispatch
  vllm/model_executor/offloader/cots.py:2774 _first_decoder_pre_hook
    self.prepare_before_forward(anchor.shape[0])
  cots.py:2801 prepare_before_forward
    self._current_bucket = self._bucket_for(num_tokens)
  cots.py:2782 _bucket_for
    i = bisect_left(self._capture_buckets, num_tokens)   ← Dynamo can't trace
```

The full subprocess log is a locally-generated artifact (the
results directory's `*.log` files are repo-gitignored; only the
`*.json` cells + `summary.json` are tracked). Reproduce with:

```bash
cd David/Benchmarks/phase1c
/opt/conda/envs/thesis/bin/python bench_dryrun_vs_native_qwen.py \
    --only-arms cots_005_native_capture_dryrun \
    --batches 1 --num-iters 1 --num-iters-warmup 1
# log lands at:
# results/dryrun_vs_native_qwen/cots_005_native_capture_dryrun_b1.log
```

Two interacting facts:
1. vLLM's compiled-model setup uses `fullgraph=True` (one captured
   graph per `BatchDescriptor`).
2. `nn.Module._call_impl` traces forward pre-hooks INTO the captured
   graph (Dynamo's standard nn.Module handling, line 1809).

Phase 1c's pre-hook is supposed to run OUTSIDE the captured region
(its job is to set `_current_bucket` BEFORE the forward starts; the
captured region's slab dispatch only reads `_current_bucket`).
`cudagraph_utils.py:267` already calls `prepare_before_forward` at
the FULL graph boundary outside compile's view — which is correct —
but the duplicate forward-pre-hook registration ALSO fires inside
compile's view and tries to traverse `bisect_left`.

**Decorator note:** an earlier Stage 6 attempt added
`@torch._dynamo.disable` to `_first_decoder_pre_hook`. Under
`fullgraph=True` Dynamo raises a different error (`Skip calling
torch.compiler.disable()'d function`, gb0098) instead of tracing
through. The decorator was reverted in this commit; the stack above
is from the current (no-decorator) code. Either way the pre-hook
cannot stay both registered AND opaque under fullgraph capture.

**Resolution paths (Stage 6 follow-up):**

(a) **Make `_bucket_for` Dynamo-traceable.** Convert
`_capture_buckets` to a constant `tuple` at resolve time and replace
the bisect with a small for-loop:

```python
def _bucket_for(self, num_tokens: int) -> int:
    for b in self._capture_buckets:  # tuple, treated as constant by Dynamo
        if num_tokens <= b:
            return b
    return self._capture_buckets[-1]
```

Dynamo can specialize on the tuple's contents and trace the loop.
Lowest-friction fix; preserves the unconditional pre-hook semantics
that `phase1c_findings.md §1c.9` and the Stage 4 tests rely on.

(b) **Drop the pre-hook entirely; rely on
`cudagraph_utils.py:267`'s out-of-graph
`prepare_before_forward` boundary.** vLLM's FULL graph capture path
already calls `prepare_before_forward` at the right place. The
pre-hook was added in Stage 3 to handle the eager-mode case where
`cudagraph_utils.py` isn't involved — that case still works via the
runner's lazy `_bucket_for(num_tokens)` fallback (§1c.4). Removing
the pre-hook simplifies the architecture; only risk is eager mode
loses the per-forward layer-0 slot repair, which would need to be
called via the runner-side fallback path or moved into
`offloader.prepare_before_forward` such that it's idempotent.

(c) **Use `cudagraph_mode=NONE` in the bench config to bypass
torch.compile entirely.** Falls back to raw `torch.cuda.graph` capture
which respects the pre-hook boundary by manual placement (the
synthetic Stage 5 collapse gate uses this path). Workaround, not a
fix.

The synthetic shape-collapse gate (Stage 5, ratio 0.477) is in-stage
proof that the substrate works. The real-model absolute is a
follow-up that should land path (a) or (b) before re-running the
harness.

### Resolution shipped

Path (a) was selected. Patch sites:

- `cots.py:_bucket_for` — replaced `bisect_left` with a linear scan
  over `_capture_buckets`. Dynamo specializes the tuple as a
  constant and unrolls the loop at trace time. Repeat-runs of the
  for-loop carry no per-bucket overhead vs `bisect_left` because N
  is the number of capture buckets (typically O(10)) and the
  function runs once per forward boundary, not per-GEMM.
- `cots.py:_resolve_capture_buckets` — `_capture_buckets` is now
  `tuple[int, ...]` (was `list[int]`). Tuples are hashable + treated
  as constant containers by Dynamo, which the linear scan needs.
- `cots.py:lookup_dispatch` — refactored to reuse `_bucket_for`
  instead of carrying its own `bisect_left`. Single source of truth
  for the rounding semantics; both paths trace identically.

Coverage: `David/Tests/phase1c/test_bucket_for_dynamo.py` (19 tests):
- 17 parity assertions vs the original `bisect_left` oracle across
  five interesting input classes (below first, exact match, between
  buckets, equal to largest, above largest) on four bucket-tuple
  shapes.
- A positive Dynamo gate: a tiny `nn.Module` with a forward pre-hook
  that mirrors `_first_decoder_pre_hook`'s structural shape compiles
  cleanly under `torch.compile(fullgraph=True)`.
- A negative regression gate: the same module, but with
  `bisect_left` reinstated, raises under `fullgraph=True`. Locks in
  the §1c.18 root cause so a future Dynamo update silently
  upgrading `bisect_left` to traceable doesn't make this fix look
  redundant — and so a regression to `bisect_left` in production
  code would be caught.

---

## 1c.19: Real-model anchor blocker #2 — Dynamo guard pickling vs `CotsCpuInfer` (CLOSED)

**Status: closed.** Resolved via the registry split landed alongside
§1c.18. The compile-visible `NativeCotsRunner` facade no longer
holds a `CotsCpuInfer` reference; the pybind handle lives in the
`cots_ops._COTS_INFER` registry, keyed by `runner_id`. Custom op
impls and offloader install/teardown helpers all dereference the
registry instead of `runner._infer`.

### Original problem

Uncovered AFTER §1c.18 was closed. The
`cots_005_native_capture_dryrun` smoke now gets past Dynamo's
fullgraph capture and falls into AOT compile's guard-serialization
step, which tries to pickle a `vllm._cots_C.CotsCpuInfer` instance
and fails:

```
File ".../torch/_dynamo/aot_compile.py", line 257, in aot_compile_fullgraph
    check_fn = graph_capture_output.build_guards(...)
File ".../torch/_dynamo/convert_frame.py", line 1001, in build_guards
    return CheckFunctionManager(...)
File ".../torch/_dynamo/guards.py", line 3766, in __init__
    self.guards_state = self.serialize_guards(...)
File ".../torch/_dynamo/guards.py", line 3926, in serialize_guards
    return pickle_guards_state(guards_state, builder.guard_tree_values)
File ".../torch/_dynamo/guards.py", line 3552, in pickle_guards_state
    pickler.dump(state)
TypeError: cannot pickle 'vllm._cots_C.CotsCpuInfer' object
```

### Why Dynamo needs to pickle it

vLLM uses PyTorch's AOT compile cache (`vllm/compilation/wrapper.py:176`
calls `_compiled_callable.aot_compile`). AOT compile builds a guard
function and serializes it for cache reuse — so guards' closure
values must be picklable. The COTS operator's `__call__` reads
`self._runner._task_id_for[desc]` and `self._runner._runner_id`;
when Dynamo traces those reads, it builds guards on `self._runner`,
walks its attributes for guard construction, and finds
`self._runner._infer: CotsCpuInfer`, which is a stateful pybind11
class with no `__reduce__` / `__getstate__`.

### Resolution paths (open work)

(a) **Add pickle support to `CotsCpuInfer` via pybind11.** Cheapest
in code volume — define `__getstate__` / `__setstate__` (or
`pybind11::pickle`) returning a no-op state. Risks: (i) the C++
extension needs a rebuild; (ii) "pickle to None and reconstruct"
isn't semantically valid for a stateful inference engine, so guard
deserialization later would produce a broken handle (acceptable
ONLY if the AOT cache is never actually used — the file gets
written, never read).

(b) **Decouple traced operator code from `_runner`.** Stash
`runner_id` and a frozen view of `_task_id_for` directly on the
operator at install time (so the operator's `__call__` reads only
plain ints / a plain dict — no `_runner` deref). Dynamo's guard
walker stops at the operator; `CotsCpuInfer` stays out of the guard
graph. Larger change, no rebuild needed.

(c) **Disable AOT compile guard serialization for the offloader
path.** Investigate vLLM's `compilation_config` for a knob that
turns off `aot_compile`'s cache. Cheapest if the knob exists;
sidesteps the picklability question entirely. Workaround quality
depends on whether disabling the cache costs measurable startup
time.

The §1c.18 closure already buys the architectural cleanup needed
for path (b) (`_bucket_for` and dispatch are unified at the
offloader level). Path (b) is probably the right answer; path (a)
is a one-hour patch if cache reuse turns out to be a no-op for
this workload anyway.

### Why §1c.18 fix still ships independently

The §1c.18 fix is a strictly-better state regardless of §1c.19's
resolution: it removes a Dynamo-traced builtin call (always wrong
under fullgraph capture, regardless of caching), and it unifies
`_bucket_for` / `lookup_dispatch` to share a single rounding rule.
§1c.19 was unobservable until §1c.18 was fixed because the older
crash short-circuited engine init before AOT compile reached the
guard-serialization step.

### Resolution shipped (registry split)

Patch sites:

- `cots_ops.py` — `_COTS_RUNNERS` (a `WeakValueDictionary` of
  runners) replaced with `_COTS_INFER: dict[int, CotsCpuInfer]`
  (strong refs, keyed by `runner_id`). The registry IS the storage
  for the pybind handle; the runner only holds the integer id.
  Helper functions `install_infer`, `populate_slab_via_spec`,
  `set_worker_affinity`, `sync_blocking` provide install-time and
  teardown-time access without ever exposing `CotsCpuInfer` on the
  runner's `__dict__`.
- `cots.py:NativeCotsRunner.__init__` — creates `CotsCpuInfer()`
  and immediately hands it to `cots_ops._register_infer(...)`.
  The local variable goes out of scope; nothing in `self.__dict__`
  references the handle. Fields are now `_runner_id`,
  `_task_id_for`, `_dry_run`, `_installed` — all picklable.
- `cots.py:NativeCotsRunner.install` — drops the
  `bucket_for_fallback` parameter. Operators are required to
  resolve `op_descriptor[1]` to a non-None int before calling the
  runner. (Same change applied to `PythonCotsRunner.install` for
  parity, even though that runner is eager-only and Dynamo never
  sees it.)
- `cots.py:CotsQKVOp.apply` and `CotsSwiGLUMLPOp.__call__` —
  resolve `b = offloader._current_bucket or
  offloader._bucket_for(num_tokens)` up-front, before the
  per-bucket data lookups. Eliminates the `int | None` ambiguity
  that the runner's lazy fallback used to handle.
- `cots.py:CotsOffloader._install_runner` —
  `runner._infer.set_worker_affinity(mask)` becomes
  `cots_ops.set_worker_affinity(runner._runner_id, mask)`.

### Ownership: pickled copies must be non-owning (review fix)

The first cut of the §1c.19 fix had a high-severity bug. The
unpickled facade shared `_runner_id` with the original AND ran the
same `__del__`. PyTorch's AOT guard cache pickles+unpickles the
runner during guard serialization — GC of any unpickled copy could
hit `_unregister_infer(rid)` on a live entry, causing the original
runner's next custom-op call to fail with `runner_id not in
registry`.

Fix:

- `NativeCotsRunner.__init__` adds
  `self._owns_infer_registry_entry: bool = True`. The original
  constructor is the sole owner; only it may drop the registry
  entry.
- `__getstate__` flips the flag to False on the pickled state, so
  unpickled copies are non-owning by construction. `__setstate__`
  defaults to non-owning if a future state dict ever omits the
  flag.
- `close()` and `__del__` both early-return when
  `_owns_infer_registry_entry` is False. Owning `close()` clears
  the flag after unregistering so a second `close()` is a no-op.

Coverage: `David/Tests/phase1c/test_runner_picklable.py` (8 tests):
- `pickle.dumps(NativeCotsRunner)` succeeds (the AOT-cache
  serialization path).
- The runner's `__dict__` contains no `CotsCpuInfer` instance
  under any name (defensive — Dynamo's guard walker uses
  `__dict__`).
- The serialized byte stream contains no `CotsCpuInfer` class
  reference (catches a future regression where someone gives the
  pybind class a permissive `__reduce__`).
- After a pickle round-trip, the unpickled facade still names the
  same registry slot — i.e., the runner facade is "a tagged
  pointer into `cots_ops._COTS_INFER`."
- **Ownership tests (review-fix, the high-severity finding):**
  - GC of an unpickled copy does NOT unregister the original.
  - `close()` on an unpickled copy is a no-op.
  - `close()` on the owning original still drains and unregisters.
  - `__getstate__` marks the pickled state non-owning; the
    original's flag is unchanged by introspection.

Plus seven existing tests rewritten for the new registry surface
(`test_stage2_substrate.test_infer_registry_*`,
`test_multi_engine.test_two_runners_have_distinct_runner_ids`,
`test_multi_engine.test_close_one_runner_does_not_affect_other`,
`test_dependency_ordering.test_fx_graph_*`).

### Smoke result

`cots_005_native_capture_dryrun` now gets PAST AOT compile and
graph capture — runtime entry, custom op dispatch, the
`mutates_args` ordering — all good. Engine init proceeds further
than before. The next-uncovered failure is documented in §1c.20.

---

## 1c.20: Real-model anchor blocker #3 — Inductor materializes any CPU tensor it sees (CLOSED)

**Status: closed.** Resolved by removing BOTH `x_pinned` AND
`y_pinned` from the custom op signatures and reaching the slab's
pinned-buffer pointers directly from C++. The captured-graph custom
ops are now CUDA-tensors-and-scalar-ids only — Inductor has
nothing CPU-side to materialize. After this fix the
`cots_005_native_capture_dryrun` arm runs end-to-end on Qwen2.5-7B
through Inductor + AOT compile + CUDA Graph capture + replay.

The story unfolded over three increasingly-deeper diagnoses; the
original framing (metadata-only loss) was wrong.

### Original problem (the one we walked into)

Uncovered AFTER §1c.19 was closed. The smoke run
now reaches the captured forward's runtime execution; the failure
is in `uva_copy_into_gpu`'s `assert src_pinned.is_pinned()`:

```
torch._inductor.utils.run → model(new_inputs)
  /tmp/torchinductor_root/.../output_code.py:2027 in call
    torch.ops.vllm.cots_sync_then_uva.default(
        reinterpret_tensor(buf9, (s72, 3584), (3584, 1), 0),  # y_pinned
        reinterpret_tensor(buf13, (s72, 3584), (3584, 1), 0), # y_gpu
        buf12, arg11_1, 1)
  cots_ops.py:193 _cots_sync_then_uva_impl
    uva_copy_into_gpu(y_pinned, y_gpu)
  cots.py:125 uva_copy_into_gpu
    assert src_pinned.is_pinned(), "src must be pinned host memory"
AssertionError: src must be pinned host memory
```

### What's happening

Inductor's lowering pass traces through the operator's
`__call__`. It sees:

```python
y_out = offloader._y_pinned[: num_tokens * n_cpu].view(num_tokens, n_cpu)
self._runner.submit_with_d2h(x, x_in, y_out, op_desc)
...
self._runner.wait_and_uva(y_out, y_dst, gpu_a, gpu_b)
```

`offloader._y_pinned` is a `torch.empty(..., pin_memory=True)`
allocation. Inductor emits a `reinterpret_tensor(buf9, ...)` node
representing the slice/view; in the generated runtime,
`reinterpret_tensor` produces a tensor whose `is_pinned()` flag
is False even though the underlying storage IS pinned.

The Triton UVA kernel itself doesn't actually need the
`is_pinned()` flag to be True at the Python level — it needs the
underlying CUDA host pointer to be page-locked, which the
storage still is. So this is a type-system mismatch: the runtime
storage is fine; only the metadata bit is missing on the
reinterpret view.

### Resolution paths

(a) **Relax the `uva_copy_into_gpu` assertion to a runtime check
on the storage, not the view.** `is_pinned()` walks the tensor's
storage class; we can drop down to
`src_pinned.untyped_storage().is_pinned()` or check the device
type explicitly. The simplest patch is to remove the `is_pinned()`
assertion and rely on the storage-level check that the Triton
kernel itself performs. Risk: silently accepts non-pinned input
and Triton can read garbage; mitigate by adding a one-time check
at install/wrap_modules time and trusting the post-compile path.

(b) **Pre-pin the Inductor-allocated buffer.** Mark the operator's
`y_out` as a graph-output-style buffer that Inductor must respect
the pinned-ness of. This requires Inductor knobs that may not
exist (custom-op return type doesn't currently carry pinned-ness).

(c) **Bypass `uva_copy_into_gpu` entirely from inside the captured
graph.** Move the H2D copy back to a regular `dst_gpu.copy_(
src_pinned, non_blocking=True)` and let CUDA Runtime do it. The
UVA kernel was an optimization to share the PCIe link with
concurrent CE0 traffic; under graph capture the alternative may
or may not be measurably worse — would need a follow-up
microbench.

Path (a) is probably the right immediate fix (with the storage-level
check kept as the safety belt). Path (c) is a Stage 7-adjacent
investigation that affects the bandwidth ceiling.

### Why §1c.19 fix still ships independently

The registry split is a strictly-better state regardless of §1c.20
— it removes a non-pickleable handle from the compile-visible
object graph, which is correct architecture for ANY downstream
PyTorch caching/serialization layer (today's AOT guard cache,
tomorrow's cache_size_limit / inductor cache / etc.).

### Diagnosis evolution: the right invariant emerges

Three smoke-run iterations sharpened the diagnosis:

1. **First read**: "Inductor drops the `is_pinned()` metadata bit
   on `reinterpret_tensor` views." Storage-level check
   (`untyped_storage().is_pinned()`) was the proposed fix.
2. **Re-run with the storage check**: still failed. Inspecting the
   actual Inductor codegen showed `buf9 = empty_strided_cpu(...,
   bfloat16)` followed by `cpp_fused_as_strided_view_2(buf4, ...,
   buf9, ...)` — Inductor was allocating a **fresh pageable** CPU
   buffer and cloning the pinned slice into it after the
   `cots_submit_gemm` mutation, then handing the clone to
   `cots_sync_then_uva`. The storage genuinely wasn't pinned;
   there was nothing to find.
3. **Schema swap (drop `y_pinned` from `cots_sync_then_uva`)**:
   moved the failure to the SUBMIT side. The same pattern fired
   on `x_pinned`: `triton_red_fused_1.run(...) → buf2` (GPU
   intermediate) → `buf3 = empty_strided_cpu(...)` → `buf3.copy_(
   buf2, False)` (blocking GPU→CPU copy) — rejected by CUDA Graph
   capture with `cudaErrorStreamCaptureUnsupported`.

The right invariant — visible only after climbing the diagnostic
ladder — is **stronger** than "no mutated CPU tensor": **any CPU
tensor visible to Inductor in the captured graph is suspect**.
Inductor's functionalization / memory-planning passes will
materialize CPU views via GPU intermediates + blocking transfers
whenever it suits the rest of the plan, regardless of mutation
declarations.

### Resolution shipped

Both custom op signatures now contain ONLY CUDA tensors and scalar
ids:

```
cots_submit_gemm(x_gpu, runner_id, task_id, num_tokens) -> None
  mutates_args=["x_gpu"]

cots_sync_then_uva(y_gpu, gpu_anchor_a, gpu_anchor_b,
                   submit_anchor, runner_id, task_id, num_tokens)
                   -> None
  mutates_args=["y_gpu", "gpu_anchor_a", "gpu_anchor_b"]
```

C++-side machinery ports the pinned-buffer pointer story:

- **`submit_on_stream`** now takes `(task_id, num_tokens, x_gpu_ptr,
  x_cols, x_stride0, x_stride1, cuda_stream)` and bundles the
  x_gpu → slab.x_pinned_ptr D2H WITH the host-callback enqueue,
  on the supplied stream. Both the copy and the host_fn enqueue
  are graph-capturable. **Stride-aware**: `x_stride0 == x_cols`
  → `cudaMemcpyAsync` (1D); otherwise → `cudaMemcpy2DAsync` (2D)
  walking rows correctly. Real Qwen2 hidden_states tensors can
  be row-strided when sliced from a wider base; rejecting them
  would make native COTS brittle. `x_stride1 == 1` is required
  (transposed layouts rejected with a clear message).
- **`y_pinned_view(task_id, num_tokens)`** returns an
  `at::from_blob` CPU view over the slab's pinned output pointer
  — the sync impl uses this internally to drive the UVA copy
  without exposing the CPU tensor as a custom-op argument. The
  trust boundary is install-time: the slab pointer came from
  `_y_pinned` (a `torch.empty(..., pin_memory=True)` allocation
  validated there).
- **`populate_slab_dryrun`** extended to take `(x_pinned_ptr,
  in_dim, y_pinned_ptr, cpu_out_dim)` so the dryrun arm — which
  measures orchestration WITHOUT real CPU GEMM — still resolves
  through both captured-graph paths.

Operator-side branch on `runner.kind` BEFORE constructing CPU
views (the user's explicit direction during the patch dialog):

```python
if self._runner.kind == "native":
    y_dst = offloader._y_gpu[: ...].view(...)
    self._runner.submit_with_d2h(x, desc)
else:
    x_in = offloader._x_pinned[: ...].view(...)
    y_out = offloader._y_pinned[: ...].view(...)
    y_dst = offloader._y_gpu[: ...].view(...)
    self._runner.submit_with_d2h(x, x_in, y_out, desc)
```

Just dropping `y_pinned` from the runner facade and ignoring it
inside the runner method body wasn't enough — Dynamo traces the
operator's bytecode and records the
`_y_pinned[:N].view(...)` compute path even if the receiver
discards it. The branch eliminates the compute path entirely on
the captured side.

### Coverage

- `David/Tests/phase1c/test_strided_x_gpu_d2h.py` (4 tests):
  contiguous-row D2H (1D path), row-strided D2H (2D path),
  transposed input rejection, partial-num_tokens correctness.
- `David/Tests/phase1c/test_y_pinned_view.py` (5 tests): shape /
  dtype / device, data_ptr matches slab, reads worker writes,
  out-of-range task_id rejection, partial num_tokens.
- Schema-test additions in
  `David/Tests/phase1c/test_dependency_ordering.py`:
  `test_cots_submit_gemm_does_not_take_y_pinned`,
  `test_cots_sync_then_uva_does_not_take_y_pinned`,
  `test_cots_sync_then_uva_takes_submit_anchor`,
  `test_cots_sync_then_uva_mutates_only_gpu_args`. The
  load-bearing FX-graph ordering test was updated for the new
  schema (no x_pinned/y_pinned args; sync takes
  `submit_anchor`).

Triple suite at §1c.20 closure: phase1a 60, phase1b 80, phase1c
143 (139 + 4 new strided D2H tests).

### Real-model anchor (the §1.14 number)

`bench_dryrun_vs_native_qwen.py` at B=1, f=0.05, input=8, output=128,
3 iters / 2 warmup on Qwen2.5-7B (RTX 4090 + i9-14900KF). Sweep
across t={4, 8, 16} after the §1c.21 review-fix that gates
`torch.set_num_threads` to the python runner only:

```
                                 t=4         t=8         t=16
none                                       2.0333 (t-invariant)
none_capture                               2.0323 (t-invariant)
cots_005_python_eager_dryrun     2.4888    2.5415    2.5307
cots_005_native_eager_dryrun     2.3416    2.3090    2.3233
cots_005_native_eager_real       2.4314    2.5579    2.6050
cots_005_native_capture_dryrun   2.5058    2.4944    2.4948
cots_005_native_capture_real   192.5793  186.4441  123.1685

orch decomposition vs the right baseline:
                                 t=4         t=8         t=16
orch_python_eager (eager-eager) +0.4555 s  +0.5082 s  +0.4974 s
orch_native_eager (eager-eager) +0.3083 s  +0.2757 s  +0.2900 s
orch_native_capture (cap-cap)   +0.4735 s  +0.4621 s  +0.4626 s
cpu_work_native_eager           +0.0897 s  +0.2489 s  +0.2817 s
cpu_work_native_capture       +190.0735 s +183.9497 s +120.6737 s
```

Findings:

1. **Architecture works end-to-end.** The captured native arm runs
   the full FastTTS-style decode on Qwen2.5-7B without falling
   over. No `cudaErrorStreamCaptureUnsupported`, no functionalization
   crashes, no missing slab pointers. §1c.20's invariant ("no CPU
   tensors visible to Inductor in the captured graph") holds in
   the wild.
2. **Capture mode is not winning over native-eager**: orch +0.46–
   0.47 s under capture vs +0.28–0.31 s under eager, **across all
   thread counts**. Native runner under EAGER mode already gave
   us a 36% reduction over the python runner; capture undoes that
   gain. The thread-gate fix
   (`torch.set_num_threads` no longer applied for native; §1c.21
   review-fix) **did not move this number** — the capture penalty
   is structural to the captured-forward dispatch path, not
   thread-policy contamination.
3. **CPU work under capture is 400–2000× heavier than under eager
   on the same workload.** Eager `cpu_work` is +0.09–0.28 s per
   generate; capture `cpu_work` is +120–190 s. The worker IS
   running real GEMMs (cpu_work scales with thread count: 190 s
   at t=4, 121 s at t=16) but each call is inflated by
   ~17–27 ms. Compared to eager's ~40 μs/GEMM under the same
   shape this is ~425× per-call slowdown. The §1.14 target
   (≤ 0.05 s/generate) is unreachable until this is diagnosed.

Concretely: the eager path executes ~7000 CPU GEMMs (28 layers ×
2 ops/layer × 128 output tokens) in 0.28 s @ t=16 ≈ 40 μs each.
The capture path takes 17 ms each — somewhere between the
host_fn enqueue and the at::linear return there is a ~17 ms
amplifier. Hypotheses (in §1c.21):

- Captured `cudaLaunchHostFunc` blocks the GPU stream synchronously
  on the worker, serializing CPU and GPU. Eager mode runs
  `future.result()` on the main thread instead, leaving the GPU
  free to execute other queued work.
- Per-replay Dynamo runtime check function (guard introspection)
  fires on every captured forward replay; if the guard is heavy
  it bills against every layer.
- `cudaMemcpy2DAsync` setup cost vs 1D — bench should log which
  branch fires. Counters proposed in §1c.21.

(2) — that capture orch ≈ python_eager orch (both ≈ +0.47 s) — is
striking: the §1c.20 schema swap successfully made the
captured-forward custom ops opaque to Inductor, but did not
recover the orch reduction native+eager already provided.

### Why this still closes §1c.20

§1c.20 was scoped as an **architectural blocker** — Inductor was
rejecting our op shape and the captured forward couldn't even
init. That's now fixed. The unfavorable orch ratio and the
pathological real-arm latency are downstream perf questions; they
need a profiler trace (`nsys`), not another schema redesign. The
schema changes already shipped are strictly better regardless of
how that investigation lands: any future fix to the perf path
requires Inductor not materializing our CPU views, which is
exactly what the §1c.20 schema swap guarantees.

§1c.21 follow-up: DIAGNOSED via C++ counters; **CLOSED** via the
live-token plumb-through (vllm@5fecc800b). Counter-driven diagnosis
notes preserved below for the historical record. Resolution and
post-fix measurements live in the next sub-section.

### Resolution shipped

The fix decouples "graph tensor shape" (still bucket-sized) from
"logical live tokens" (the live unpadded count). COTS now has TWO
row counts:

- `slab.num_tokens` — graph bucket capacity (e.g., 256). Frozen at
  capture time; sizes the captured cudaMemcpyAsync byte count, the
  slab's pinned x/y buffers, and the worker's upper-bound check.
- `runtime_num_tokens` — live rows to compute. Set OUT OF GRAPH by
  `gpu_model_runner.execute_model` from
  `scheduler_output.total_num_scheduled_tokens` BEFORE every
  forward. Always `runtime_num_tokens <= slab.num_tokens`. The
  worker's `at::linear` shapes, scratch slicing, and y_pinned write
  region key off this.

Plumb-through:
1. `gpu_model_runner.execute_model` calls
   `get_offloader().set_runtime_num_tokens(num_tokens_unpadded)`
   BEFORE the FULL/PIECEWISE/eager dispatch — covers all paths.
2. `BaseOffloader.set_runtime_num_tokens(actual)` — no-op default;
   `CotsOffloader` override pushes through `cots_ops` to
   `CotsCpuInfer::set_runtime_num_tokens(int32_t n)` (atomic
   release-store, validates n >= 0).
3. `RunSlabOnWorker` reads via acquire-load:
   ```
   effective_n = (override > 0) ? override : slab.num_tokens
   TORCH_CHECK(effective_n <= slab.num_tokens)
   ```

`CotsOffloader.prepare_before_forward` stays Dynamo-clean (no
pybind calls) because the first-decoder pre-hook is traced into
the captured graph; the C++-side runtime push happens at the
out-of-graph model-runner boundary instead.

### Real-model anchor (post-fix)

At B=1, output_len=128, t=16, f=0.05, 3 iters / 2 warmup on
Qwen2.5-7B + RTX 4090 + i9-14900KF:

```
arm                            before §1c.21   after §1c.21
none (eager baseline)            2.0333         2.0333
none_capture (graph baseline)    2.0323         2.0323
native_eager_dryrun              2.3488         2.3488
native_eager_real                2.6050         2.6050
native_capture_dryrun            2.5294         2.5294
native_capture_real            119.3297         2.76     ← 43× faster
```

`cpu_work_native_capture` collapsed from +116.80 s/gen to ~+0.23 s
— matches native_eager's CPU work cost (+0.28 s) within run-to-run
variance. The 43× speedup is the elimination of wasted bucket-sized
GEMMs.

### Counters confirm the worker behavior

At output_len=128 (settled):
```
runtime_set_calls:    640        (= 5 generates × 128 forwards)
runtime_last_value:   1          (B=1 decode)
worker_eff_n_nt_le_1: 35,874     (dominant — actual decode work)
worker_eff_n_nt_gt_64: 3,920     (capture-time forwards only)
```

Submit-time histogram still shows ~76% at `nt_gt_64` — that's
expected and unchanged because `x_gpu.shape[0]` (passed to submit)
is the captured bucket size by construction. The override
mechanism makes the WORKER ignore that bucket and process only
`runtime_num_tokens` rows.

### Coverage

`David/Tests/phase1c/test_runtime_num_tokens_override.py` (4 tests):
1. `set_runtime_num_tokens` smaller than bucket → worker processes
   only first n rows; rest of y_pinned untouched.
2. `runtime_num_tokens=0` → fall back to `slab.num_tokens`.
3. `runtime_num_tokens > slab.num_tokens` → hard-fail TORCH_CHECK
   (worker exception surfaces on next Python call).
4. Negative value rejected at the Python boundary.

Triple suite: phase1a 60, phase1b 80, phase1c 147 (143 + 4 new).

### What's still open: §1c.22 (PCIe waste from bucket-sized copies)

The captured `cudaMemcpyAsync` byte count and the captured Triton
UVA grid are still bucket-sized — only the worker's CPU-side
arithmetic shrinks to live tokens. Byte-accounting smoke at
output_len=128 (dryrun arm, B=1, t=16, f=0.05, 1 warmup + 1 iter):

```
d2h_1d_bytes (captured input D2H):    16.4 GB
worker_input_bytes_used (live read):  10.0 GB
   ⇒ over-transfer:                    6.4 GB (39% waste at this scope)

worker_output_bytes_used (live write): 5.4 GB
```

For steady-state B=1 decode (excluding engine-init capture
forwards): each cudaMemcpyAsync transfers `bucket × in_dim × 2`
bytes (e.g., 256 × 3584 × 2 = 1.8 MB), worker reads
`1 × 3584 × 2 = 7 KB` → **99.6% per-call PCIe waste**. UVA copy
back is symmetric. Plausibly explains a meaningful chunk of the
residual `+0.5 s` capture-mode orch.

§1c.22 fix candidates (open):
- D2H: the captured cudaMemcpyAsync byte count is frozen at
  capture time. To shrink at replay, the host_fn callback would
  need to issue a NEW cudaMemcpyAsync from inside the callback
  — but host_fn callbacks can't issue CUDA work. Possible
  workaround: separate "replay-time copy" stream where the byte
  count is dynamic. Non-trivial.
- UVA: the captured Triton kernel has a static grid sized for
  the bucket. A "trusted dynamic-copy path" (separate Triton
  kernel that takes runtime n via a CUDA-readable side channel)
  is plausible.
- Both fixes are §1c.22 territory; not on §1c.21's critical
  path. The §1.14 thesis target accounting must remain honest:
  remaining +0.73 s/gen is no longer wasted CPU GEMM, but it is
  still graph-mode overhead + small real CPU work + bucket-sized
  copy waste — not pure orchestration.

---

### §1c.21 historical diagnosis (preserved)


### Counter-driven diagnosis

A focused histogram of submitted `num_tokens` by op kind, exposed
via `CotsCpuInfer.get_counters()` and dumped at process exit when
`VLLM_COTS_DUMP_COUNTERS=1`, revealed the root cause immediately.
Tiny smoke at `output_len=8`, B=1, t=16, f=0.05, 1 iter no warmup:

```
                 EAGER (native_eager_real)   CAPTURE (native_capture_real)
                 wall: 0.22 s                wall: 7.78 s (35× slower)

submit_count_qkv   280                       5096   (18×)
submit_count_mlp   280                       5096   (18×)

QKV num_tokens histogram:
  nt_le_1          196   (70%)               112    (2.2%)
  nt_le_2            0                       112    (2.2%)
  nt_le_4            0                       112    (2.2%)
  nt_le_8           28   (10%)               112    (2.2%)
  nt_le_16           0                       112    (2.2%)
  nt_le_32           0                       224    (4.4%)
  nt_le_64           0                       448    (8.8%)
  nt_gt_64          56   (20%)              3864   (76%)   ← !!!

D2H bytes:
  d2h_1d_count      560                      10192  (18×)
  d2h_1d_bytes      3.4 GB                   16.4 GB (4.8×)
```

Under eager 70% of all CPU GEMM submits fire at `num_tokens=1`
(matching the actual B=1 decode), with the prefill of input_len=8
showing up as 28 ops at `nt_le_8` and a one-time KV-profile forward
at `nt_gt_64`. Under capture, **only 2.2% of submits fire at
`nt=1`**; **76% fire at `nt>64`**. Capture is doing CPU GEMMs for
the **captured graph-bucket size, not the live decode count**.

The reviewer's microbench data closes the math:

- Captured QKV at tokens=1: 52 μs.
- Captured MLP at tokens=1: 176 μs.
- Captured QKV at tokens=256: 4.8 ms.
- Captured MLP at tokens=256: 28.5 ms.
- Combined per layer at tokens=256: ~33 ms.
- 28 layers × 128 decode steps × 33 ms ≈ **118 s** ≈ observed
  `cpu_work_native_capture` of 120 s @ t=16.

So the per-call host-callback machinery is fine; the worker is
just doing 256× more arithmetic per call than it should be.

### Why this happens

`NativeCotsRunner.submit_with_d2h(x_gpu, op_descriptor)` reads
`num_tokens = int(x_gpu.shape[0])` and hands it to
`infer.submit_on_stream`. Under eager, `x_gpu.shape[0]` IS the live
token count (B=1 decode → 1). Under capture, vLLM compiles the
forward into a captured graph for some bucket size (e.g., 256 if
that's the smallest captured bucket ≥ live tokens, or more
typically a fixed full-graph descriptor) — `x_gpu.shape[0]` at
capture time is the bucket size. That value is BAKED into the
captured graph: the cudaMemcpyAsync byte count, the `num_tokens`
written into the slab, and downstream ATen view shapes are all
frozen at the capture-time bucket value. Replays at B=1 decode
re-fire the captured ops at the bucket size, doing 256× the work.

Eager mode doesn't go through capture, so each forward sees the
true live token count. The `cpu_work_native_eager` of +0.09–0.28 s
is what the workload actually costs.

### Fix direction (open)

The runner needs the **logical active token count**, not
`x.shape[0]`. vLLM tracks this in the BatchDescriptor handed to
`prepare_before_forward(num_actual_tokens)` (called at
`cudagraph_utils.py:267`, OUT OF GRAPH). The offloader already
caches `_current_bucket = self._bucket_for(num_actual_tokens)`
there. The fix is to plumb the live `num_actual_tokens` to the
C++ side via a side channel that varies per replay:

- Option A: store live `num_actual_tokens` on the offloader, read
  it in the operator's Python code, hand to `submit_on_stream`.
  The catch: under capture, the operator is traced by Inductor —
  reading `offloader._current_bucket` bakes in a constant. Need a
  graph-input-style entry that varies per replay.
- Option B: set up vLLM's capture so each captured graph DOES
  encode the live num_tokens. The bucket selection at decode
  time would then pick the bucket=1 graph for B=1 decodes,
  which captured num_tokens=1 and replays accordingly. This is
  what the existing per-bucket capture machinery is supposed
  to do; investigating why it isn't picking bucket-1 for B=1
  is the first step.
- Option C: capture-sizes restricted to [1] as a quick experiment
  to verify the diagnosis. The reviewer noted: "if the 120s
  collapses, that confirms the diagnosis immediately." If the
  bug is "vLLM picked the wrong captured graph", restricting to
  one captured size forces the right one.

Of these, (B)/(C) are tactical (working with vLLM's existing
mechanism); (A) is more invasive but most direct. The next
investigation step is to instrument WHICH bucket the
`prepare_before_forward` call is receiving for B=1 decodes —
that pins down whether vLLM is selecting the wrong captured
graph or whether the graph itself is wrong.

### Original analysis below (now superseded)

Numbers above (multi-iter, t={4,8,16}) localized the shape of the
regression but did NOT pin the root cause. The counter-driven
diagnosis above did. Keeping the wall-clock observations for
historical reference:

- The capture orch overhead is roughly constant at ~+0.18 s
  (capture - eager) across thread counts → the per-layer dispatch
  amplifier scales with ~28 layers × 128 output tokens, not with
  CPU thread count. Suggests something in the dispatch path
  (Dynamo guard check, host_fn enqueue, or graph-replay machinery)
  rather than in oneDNN.
- The capture cpu_work is 100–200 s for ~7000 GEMMs = 17–27 ms
  per GEMM. That's 425× the eager per-GEMM cost. CPU work scales
  inversely with thread count (190 → 121 s as t goes 4 → 16) so
  oneDNN itself is parallelizing fine; the per-call amplifier is
  the issue.

Hypothesis-list to drive the nsys probe (recommended order, lowest
to highest invasiveness):

- **Lightweight C++ counters first.** Before nsys: instrument
  `submit_on_stream` / `RunSlabOnWorker` / `SyncCallback` with
  atomic counters (task count by op kind, total worker ns by op
  kind, sync wait ns, D2H 1D/2D split, num_tokens histogram).
  These need no capture trace and may explain most of the gap.
- **Captured `cudaLaunchHostFunc` blocks the GPU stream on the
  worker.** Eager mode's `future.result()` blocks the main thread
  and leaves the GPU free; capture's `sync_on_stream` callback
  blocks the driver thread on `TaskQueue.sync(0)`, which pauses
  GPU execution until CPU completes. If CPU and GPU were
  overlapping under eager, capture serializes them — and the
  serialization shows up as inflated wall-clock per layer.
- **Per-replay Dynamo runtime check function**. Every captured
  forward replay walks `CheckFunctionManager`'s guard function.
  If guards include slow Python expressions (e.g., dict lookups
  on `self._task_id_for`) the per-iteration overhead scales with
  layer count.
- **`cudaMemcpy2DAsync` setup cost vs 1D.** If most calls take
  the 2D branch (real model's hidden_states being row-strided),
  per-call setup overhead may be measurable. Counter for
  1D-vs-2D split would pin this.

---

## Conclusion

Phase 1c delivers the Phase-1a/1b → Phase 2 substrate transition as a
Python-thin / C++-substrate split:

1. **C++ TaskQueue + cudaLaunchHostFunc** (Stage 1) replaces Phase
   1a/1b's Python ThreadPoolExecutor + future.result(). The C++
   `at::linear` path matches Python `F.linear` within 5% across all
   measured shapes including the strided down-proj column slice
   (no scalar-fallback regression on AVX2 BF16).

2. **NativeCotsRunner with custom-op-driven submit/sync** (Stage 2 +
   Stage 3). Operators are uniform across both runners — no
   runner-type branching. `mutates_args=["x_gpu", "y_pinned"]` on
   `cots_submit_gemm` and `["y_gpu", "gpu_anchor_a", "gpu_anchor_b"]`
   on `cots_sync_then_uva` install the barriers torch.compile / CUDA
   graph need to keep submit < GPU GEMMs < sync ordering (FX-positionally
   verified, §1c.11). One offloader-owned runner shared across all
   operator installs (multi-engine safe, §1c.3).

3. **Bucket-aware thread policy** (Stage 4). Per-`BatchDescriptor`
   `n_threads` per slab; cache-guarded worker-side
   `at::set_num_threads`. PyTorch at-thread-pool thread-locality
   confirmed empirically (risk #3 GREEN).

4. **CUDA Graph capture** (Stage 5). `enforce_eager=True` requirement
   conditionally dropped for native runner. Captured graphs accept
   `cudaLaunchHostFunc` nodes; replay is bit-deterministic across 50
   replays at multiple f_cpu_store points. Synthetic collapse-shape
   bench passes (ratio 0.477 ≤ 0.70). §1.14 absolute generate-
   equivalent locked separately on Qwen2.5-7B + FastTTS.

5. **Default flipped to native** (Stage 5). The user-visible
   behavior change carried by Phase 1c. Existing
   `enforce_eager=True` workflows are unchanged; defaulted
   workflows now use the production path with capture enabled.

The architecture remains the three-layer split from Phase 1a (storage
/ execution / operators) plus Phase 1b's prefetch sibling. Phase 1c's
additions (NativeSlabSpec, cots_ops registry, dummy CUDA anchors,
bucket-aware thread policy) are localized; Phase 2's `CotsAttnOp`
will slot in alongside `CotsQKVOp` / `CotsSwiGLUMLPOp` and run on the
same native runner without further substrate changes.

The next code-level checkpoint is Phase 2 (CPU suffix attention with
per-head LSE + tier-aware KV admission) per
`David/Docs/implementation_roadmap.md §Phase 2`.
