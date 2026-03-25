# llama.cpp Offloading Analysis

How weight and KV cache offloading work in llama.cpp, traced from source code.

---

## Foundation: ggml Multi-Backend Scheduler

All offloading in llama.cpp is built on ggml's backend abstraction (`ggml/src/ggml-backend.cpp`).

- Backends are registered in priority order (GPU first, CPU last — CPU is always the fallback).
- Each tensor lives in a **buffer** tied to a specific **device** (CPU, CUDA, Metal, etc.).
- The scheduler assigns operations to backends via a **5-pass algorithm**:
  1. Assign nodes with pre-allocated buffers to their buffer's backend.
  2. Assign nodes based on **weight affinity** (prefer running where weights live).
  3. Forward-propagate GPU assignments through the graph.
  4. Backward-propagate GPU assignments through the graph.
  5. **Split the graph at backend boundaries**, inserting cross-device tensor copies.

When a split boundary is hit (e.g., layer N on GPU, layer N+1 on CPU), the scheduler automatically creates **copy tensors** and issues `ggml_backend_tensor_copy_async()` to transfer activations between devices.

**Key files:**
- `ggml/src/ggml-backend.cpp:685-1850` — scheduler, graph splitting, split execution
- `ggml/src/ggml-backend-impl.h` — backend/buffer type interfaces
- `ggml/include/ggml-alloc.h` — graph allocator API

---

## Weight Offloading

### Mechanism: `n_gpu_layers`

Weight placement is controlled by a single parameter: **`n_gpu_layers`** (`include/llama.h:289`).

```
n_gpu_layers = K  (out of L total layers)
→ i_gpu_start = L + 1 - K
→ Layers 0 to i_gpu_start-1     : CPU
→ Layers i_gpu_start to L-1     : GPU
→ Output head                    : GPU (if K > L)
```

A negative value means "offload everything" (all L layers + output head).

### Loading Path (`src/llama-model.cpp:2693-2779`)

1. `load_tensors()` computes `i_gpu_start` from `n_gpu_layers`.
2. For each layer, `get_layer_buft_list(il)` returns CPU or GPU buffer type:
   ```cpp
   if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
       return {cpu_dev, &cpu_buft_list};      // CPU
   }
   // else → GPU device (distributed across GPUs by tensor_split)
   ```
3. Tensors are classified as INPUT (embeddings → always CPU), REPEATING (per-layer → depends on index), or OUTPUT (follows last GPU layer's device).
4. Each tensor is allocated into its assigned device's buffer.

### Multi-GPU Distribution

When multiple GPUs are present, GPU-assigned layers are distributed proportionally by free VRAM:
```
splits[] = normalized cumulative free memory across devices
layer → GPU index = upper_bound(splits, layer_idx / act_gpu_layers)
```

### Inference Behavior

- Weights are **statically placed** at load time — they never move during inference.
- The compute graph spans both CPU and GPU layers.
- The scheduler splits the graph at CPU↔GPU boundaries.
- Only **activation tensors** (small) are copied across PCIe at split points.
- This is **not** weight streaming/prefetching.

### Granularity

**Whole-layer, one device per layer.** There is no splitting of a single weight matrix across devices. A layer's weights are entirely on CPU or entirely on GPU.

**Key file:** `src/llama-model.cpp:2693-2779` (load_tensors), `8120-8121` (dev_layer accessor)

---

## KV Cache Offloading

### Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `offload_kqv` | **`true`** | When true, each layer's KV cache follows that layer's weight device. When false, all KV forced to CPU. |
| `type_k` | `F16` | K cache data type (F16, F32, BF16, Q8_0, etc.) |
| `type_v` | `F16` | V cache data type (independent from type_k) |
| `n_ctx` | model-dependent | Context length → determines KV cache size |
| `kv_unified` | `false` | Unified buffer across sequences vs per-sequence streams |

Default declared at `src/llama-context.cpp:2797`.

### Device Placement (`src/llama-kv-cache.cpp:102-154`)

KV cache is allocated **per-layer** in a loop:

```cpp
for (uint32_t il = 0; il < n_layer; il++) {
    buft = ggml_backend_cpu_buffer_type();       // default: CPU

    if (offload) {
        auto * dev = model.dev_layer(il);        // same device as layer weights
        buft = ggml_backend_dev_buffer_type(dev);
    }

    k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
    v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
}
```

### Granularity

**Layer-wise, partitioned across devices.** Each layer's KV cache independently lives on whatever device that layer's weights are assigned to (when `offload_kqv=true`). With `n_gpu_layers=20` out of 32:

```
offload_kqv=true:
  Layer  0-12: weights=CPU, KV=CPU
  Layer 13-31: weights=GPU, KV=GPU

offload_kqv=false:
  Layer  0-12: weights=CPU, KV=CPU
  Layer 13-31: weights=GPU, KV=CPU  ← KV forced to CPU
```

Within a single layer, the KV cache is entirely on one device (no splitting K on GPU and V on CPU, etc.).

### Attention Path When KV Is on CPU (`src/llama-graph.cpp:1856-1859`)

When `offload_kqv=false`, the attention compute nodes (QKV ops) are explicitly pinned to CPU:
```cpp
if (!cparams.offload_kqv) {
    ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
}
```
This keeps the entire attention path on CPU to avoid cross-device KV reads.

### KV Cache Structure

- **Ring buffer** design with slot allocation via `find_slot()`.
- K shape: `[n_embd_head_k, n_head_kv, n_kv, n_stream]`
- V shape: `[n_embd_v_gqa, n_kv, n_stream]` (transposed for flash attention)
- Multi-stream support for multiple independent sequences.
- Supports K-shift (RoPE position updates in-place) and defragmentation.

### Access Pattern During Attention

```cpp
// Store new KV projections into cache (scattered write via ggml_set_rows)
mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il);
mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il);

// Retrieve cached KV for attention (4D view with strides)
k = mctx_cur->get_k(ctx0, il);
v = mctx_cur->get_v(ctx0, il);
```

---

## Summary: Offloading Granularity

| Aspect | Weight | KV Cache |
|--------|--------|----------|
| **Granularity** | Whole layer | Whole layer |
| **Placement** | Static at load time | Static at init time |
| **Config** | `n_gpu_layers` (int) | `offload_kqv` (bool) |
| **Cross-device** | Activations copied at layer boundaries | Attention pinned to KV's device |
| **Within a layer** | All weights on one device | All K+V on one device |
| **Across layers** | Different layers on different devices | Each layer's KV follows its weights (or all CPU) |
