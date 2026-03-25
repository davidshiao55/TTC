# vLLM Benchmarking Findings

Experiments conducted on NVIDIA RTX 4090 (24 GB VRAM), vLLM v0.16.1, BF16 weights.
Workload: random dataset, input_len=512, output_len=128, 1024 prompts (unless noted).

## 1. Throughput Scales with Batch Size

### 1.5B Model (Qwen2.5-Math-1.5B-Instruct, gpu-mem=0.9)

Available KV cache: ~17.76 GiB. Weights: ~3 GiB.

| max-num-seqs | Throughput (tok/s) | Speedup vs 1 |
|---|---|---|
| 128 | 35,788 | — |
| 256 | 37,476 | 1.05x |
| 512 | 36,305 | 1.01x |

Saturates around 256 concurrent sequences. The 1.5B model leaves abundant KV cache so memory is never the bottleneck.

### 7B Model (Qwen2.5-Math-7B-Instruct, gpu-mem=0.9)

Available KV cache: ~8 GiB (from bench.log at gpu-mem=0.9). Weights: ~14 GiB.

| max-num-seqs | Throughput (tok/s) | Speedup vs 1 |
|---|---|---|
| 1 | 313 | 1.0x |
| 2 | 598 | 1.9x |
| 4 | 1,147 | 3.7x |
| 8 | 2,117 | 6.8x |
| 16 | 3,641 | 11.6x |
| 32 | 5,427 | 17.3x |
| 64 | 7,656 | 24.5x |
| 128 | 9,369 | 29.9x |
| 256 | 8,700 | 27.8x |

Peak at 128 sequences. **At 256, throughput drops** due to KV cache pressure — the scheduler cannot sustain 256 concurrent sequences and begins preempting, causing thrashing.

### Key Observation

Near-linear scaling up to saturation. The 7B model saturates earlier and actually regresses at high concurrency due to KV memory limits — something the 1.5B model never encounters.

## 2. KV Cache Is the Bottleneck for Larger Models

### 1.5B Model: gpu-memory-utilization sweep (max-num-seqs=256)

| gpu-mem-util | KV cache (GiB) | Throughput (tok/s) |
|---|---|---|
| 0.3 | — | FAILED |
| 0.4 | ~5.9 | 37,353 |
| 0.5 | ~8.2 | 37,467 |
| 0.6 | ~10.6 | 37,498 |
| 0.7 | ~12.9 | 37,505 |
| 0.8 | ~15.3 | 37,509 |
| 0.9 | ~17.6 | 37,294 |

Flat from 0.4 onward. The 1.5B model's KV cache needs are fully met at 40% GPU memory.

### 7B Model: gpu-memory-utilization sweep (max-num-seqs=256)

| gpu-mem-util | KV cache (GiB) | Throughput (tok/s) | vs peak |
|---|---|---|---|
| 0.5 | — | FAILED | — |
| 0.6 | — | FAILED | — |
| 0.7 | ~2.3 (est.) | 5,435 | -37% |
| 0.75 | ~3.1 (est.) | 7,435 | -14% |
| 0.8 | ~4.0 (est.) | 8,051 | -7.5% |
| 0.9 | ~8.0 | 8,703 | baseline |

The 7B model cannot even run below 0.7 gpu-mem-util. From 0.7 to 0.9, there is a **60% throughput gain** — entirely from having more KV cache. This directly contrasts with the 1.5B where this sweep was flat.

### FastTTS Context

FastTTS allocates gpu-memory-utilization=0.75 for the 7B generator (the remaining 0.15 goes to the 1.5B verifier). At 0.75, the 7B generator achieves 7,435 tok/s — a **20% throughput loss** compared to running solo at 0.9. This is the gap that weight offloading targets.

## 3. Scheduler Verification

Using `VLLM_LOGGING_LEVEL=DEBUG`, we confirmed the `BatchDescriptor` logs showing actual concurrent requests per step.

### 1.5B (max-num-seqs=128, num-prompts=1024)

| num_reqs | Steps |
|---|---|
| 128 | 944 (steady state) |
| 120-8 | 8 (drain) |

The scheduler fully saturates the max-num-seqs cap.

### 7B offload run (1/28 layers offloaded, gpu-mem=0.75)

From the engine logs, despite setting max-num-seqs=256:
- Running: 80-95 reqs (KV-cache-limited, not reaching 256)
- KV cache usage: 99-100% at steady state
- Generation throughput: ~290-310 tok/s

The scheduler is capped by KV memory, not by max-num-seqs.

## 4. vLLM Prefetch Offloader: Catastrophic Slowdown

Tested the built-in prefetch offloader on the 7B model (gpu-mem=0.75, max-num-seqs=256).

| Layers Offloaded | KV Cache (GiB) | Throughput (tok/s) | vs Baseline |
|---|---|---|---|
| 0 (baseline) | 2.32 | 7,442 | — |
| 1/28 | 2.70 | 1,345 | **-82%** |
| 2/28 | 3.14 | 829 | **-89%** |
| 4/28 | 4.00 | 577 | **-92%** |
| 7/28 | ~5.2 | 360 | **-95%** |

Each additional offloaded layer adds another ~18ms PCIe stall per decode step. With 7 offloaded layers, overhead is ~126ms per step vs ~2ms baseline compute. KV cache grows by ~0.5 GiB per offloaded layer but is completely negated by the throughput collapse.

Offloading even 1 layer out of 28 causes an **82% throughput collapse**, despite the KV cache growing from 2.32 to 2.70 GiB.

### Root Cause: PCIe Transfer Cannot Be Hidden

Per-layer weight size (7B / 28 layers, BF16): **~500 MB**

PCIe 4.0 x16 transfer time: **~20 ms**

Per-layer GPU compute time (decode, ~80 concurrent sequences): **~2 ms**

The prefetch offloader starts the async copy for the offloaded layer at the end of the previous step. The next step's non-offloaded layers (0-26) execute in ~1.8 ms, then the offloaded layer must `wait_prefetch` — but the 20 ms transfer is far from complete. This creates an **~18 ms stall per step**.

### CUDA Graph Evidence

| Run | Graph capture time | Per-graph speed |
|---|---|---|
| Baseline | 3 sec | 27-35 it/s |
| 1/28 offload | 26 sec | 3.3-3.5 it/s |
| 2/28 offload | 49 sec | 1.7-1.8 it/s |

The `wait_prefetch` and `start_prefetch` custom ops are captured inside the CUDA graphs. On every graph replay, the GPU stalls waiting for the PCIe transfer. The captured stream synchronization events (`wait_event`) cannot be overlapped across graph replays.

### Conclusion on Prefetch Offloading

Naive layer-level prefetch offloading is fundamentally incompatible with BF16 decode on consumer GPUs:
- The PCIe transfer time per layer (~20 ms) is 10x longer than the compute overlap window (~2 ms)
- CUDA graph replay serializes the stall into every step
- More layers offloaded = more stalls = worse throughput, while KV cache gains are marginal

## 5. Summary and Implications for Thesis

### The Problem Chain (Confirmed)

```
Large model weights (14 GiB for 7B BF16)
  → Less VRAM for KV cache (2.3 GiB at FastTTS's 0.75 allocation)
  → Fewer concurrent sequences (~80-95 vs 128+ possible)
  → Lower throughput (7,435 vs 9,369 tok/s = 20% loss)
```

### Why Naive Offloading Fails

The existing vLLM prefetch offloader moves entire layers to CPU and streams them back. For BF16:
- Transfer cost: ~500 MB/layer × 20 ms
- Compute cost: ~2 ms/layer (decode batch)
- **Transfer is 10x slower than compute — overlap is impossible**

### Why Hybrid Weight Computation Should Work

Instead of transferring full layer weights (500 MB), split the weight matrix:
- 90% stays on GPU, computed normally
- 10% on CPU, computed via CPU matmul
- Only **activation results** (~300 KB) transfer CPU→GPU

This eliminates the PCIe bottleneck:
- No large weight transfers
- Activation transfers are 1000x smaller than weight transfers
- CPU and GPU compute in parallel on their respective weight portions
- Transfer uses opposite PCIe direction from any weight prefetch (full duplex)

Expected outcome: freeing ~10% of weights across 28 layers = ~1.4 GiB more KV cache, with minimal latency impact (~10% from the CPU compute path).
