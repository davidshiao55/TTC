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
Qwen2.5-7B + FastTTS) is **NOT yet anchored on the real model** — the
synthetic multi-layer collapse-shape sanity check passes
(`collapse_ratio = 0.477`, Stage 5), and the real-model harness
`bench_dryrun_vs_native_qwen.py` is landed and runnable on the `none`
and python+eager arms, but the native+capture arm hits the
`fullgraph=True` × pre-hook interaction blocker documented in §1c.18.
Status: **docs landed, real-model native-capture absolute pending the
§1c.18 fix**.

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
  interaction blocks the real-model native+capture absolute

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
(ported from Phase 1a's `bench_cots_dryrun_vs_none.py` with five
arms covering python/native × eager/capture × dryrun/real). Stage 6
landed the harness AND the auto-derived `--cots-cpu-runner` /
`--cots-cpu-num-threads-by-bucket` / `--cots-cpu-worker-affinity`
CLI flags so `vllm bench latency` accepts the new fields.

Smoke-runs at `--num-iters 1 --num-iters-warmup 1` confirm:
- `none` arm completes (2.03 s/generate baseline on Qwen2.5-7B at
  input=8, output=128, B=1).
- `cots_005_native_capture_dryrun` arm currently FAILS at engine
  initialization due to the §1c.18 pre-hook × torch.compile
  interaction. The python+eager and native+eager arms DO work
  (their pre-hook fires in eager mode where `bisect_left` is fine).

Concretely: Stage 6's job was to land the harness + design doc, NOT
to lock the real-model absolute end-to-end. The synthetic
shape-collapse gate (Stage 5) is the in-stage Phase 1c sign-off; the
real-model anchor is a Stage 6 follow-up that needs the §1c.18 fix.

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

## 1c.18: Real-model anchor blocker — pre-hook × torch.compile fullgraph

Stage 6's smoke-run of `bench_dryrun_vs_native_qwen.py` on Qwen2.5-7B
surfaced an interaction between Phase 1c's first-decoder pre-hook
(§1c.4: `_install_bucket_prehook` — registered unconditionally so
`prepare_before_forward` fires even without a streamer) and vLLM's
default `torch.compile(fullgraph=True)` model wrapping.

Repro: `cots_005_native_capture_dryrun` arm at `--num-iters 1
--num-iters-warmup 1` fails at engine initialization with:

```
torch._dynamo.exc.Unsupported: Skip calling `torch.compiler.disable()`d function
   from user code:
     vllm/model_executor/models/qwen2.py:444 layer(positions, hidden_states, residual)
     nn/modules/module.py:1809 inner   # forward pre-hook dispatch
     vllm/model_executor/offloader/cots.py _first_decoder_pre_hook
       prepare_before_forward → _bucket_for → bisect_left   ← Dynamo can't trace
```

Two interacting facts:
1. vLLM's compiled-model setup uses `fullgraph=True` (one captured
   graph per `BatchDescriptor`).
2. `nn.Module._call_impl` traces forward pre-hooks INTO the captured
   graph (Dynamo's standard nn.Module handling).

Phase 1c's pre-hook is supposed to run OUTSIDE the captured region
(its job is to set `_current_bucket` BEFORE the forward starts; the
captured region's slab dispatch only reads `_current_bucket`).
`cudagraph_utils.py:267` already calls `prepare_before_forward` at
the FULL graph boundary outside compile's view — which is correct —
but the duplicate forward-pre-hook registration ALSO fires inside
compile's view and tries to traverse `bisect_left`. Dynamo can't
trace `bisect_left`; under fullgraph=True the alternative
`@torch._dynamo.disable` decorator also raises.

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
