# §1c.28 Step 1 — value-signal smoke report

Standalone CUDA-Graph value-signal smoke. Self-contained CLI binary
(`value_signal_smoke.cu`); does NOT touch COTS / vLLM. Built with:

```
nvcc -O2 -std=c++17 value_signal_smoke.cu -o value_signal_smoke -lpthread
```

## What was tested

The §1c.28 design proposed `cuStreamWriteValue32` + monotonic
sequence numbers in host-mapped pinned slots, polled by a
persistent CPU worker, as the M2 replacement for the captured
`cudaLaunchHostFunc(dispatch_cb)`.

Because `cuStreamWriteValue32` takes a literal at capture time
(replays would re-fire the same value, defeating monotonicity),
this smoke uses the actual production-shaped primitive: a tiny
`atomicAdd`-and-write kernel. Each replay launches one such kernel
per task; the worker observes the resulting seq advances.

Two slot shapes:

* **`per_task`** — one slot per task_id, each task has its own
  device counter. Worker polls each slot for monotonic advance
  on its own seq counter.
* **`single_packed`** — one shared slot, payload =
  `(seq << 16) | task_id`. Worker polls the shared slot and
  decodes both fields per advance.

Two worker poll policies:

* `poll-ns=0` — busy-spin (no yield/sleep).
* `poll-ns=1` — `std::this_thread::yield()` between rounds.

Two graph-replay rates:

* default — 1,000 graph launches in a tight loop (no
  synchronization between replays). Latency under flood
  conditions; worker stack-up is severe.
* `--sync-each` — `cudaStreamSynchronize` between replays.
  Latency reflects single-fire end-to-end, the metric M2 actually
  cares about.

All on RTX 4090, CUDA 12.4.

## Results

### Correctness (1,000 replays × 56 tasks = 56,000 expected observations)

| shape | poll | sync_each | total_observed | stale | duplicate | invalid_task | exit |
|---|---|---|---|---|---|---|---|
| `per_task` | busy | no | **56,000** | 0 | 0 | 0 | 0 ✓ |
| `per_task` | yield | no | **56,000** | 0 | 0 | 0 | 0 ✓ |
| `per_task` | busy | yes | **56,000** | 0 | 0 | 0 | 0 ✓ |
| `single_packed` | busy | no | 55,273 (lost 727, ~1.3%) | 0 | 0 | 0 | 0 |
| `single_packed` | yield | no | 55,641 (lost 359, ~0.6%) | 0 | 0 | 0 | 0 |
| `single_packed` | busy | yes | 55,527 (lost 473, ~0.8%) | 0 | 0 | 0 | 0 |

**`per_task` is correctness-clean.** Every replay's contribution
to every task is observed. Coalescing on a per-task slot (the
worker observes seq jumping from N to N+k in one poll) is
benign — the worker's `task_observed.fetch_add(gap)` correctly
counts k replays-worth of work for the same task.

**`single_packed` loses ~1% of (seq, task_id) signals.** The
shared slot is overwritten before the worker observes
intermediate values; lost task_id signals cannot be recovered.
Tighter polling (`busy` vs `yield`) doesn't fix it — this is a
fundamental "elegant but fragile" property of the shared-slot
shape. The worker would need to poll faster than the GPU writes
across all 56 tasks per replay (~50 ns/write), which busy-spin
isn't fast enough to guarantee.

### Latency — `--sync-each` (single-fire approximation)

| shape | poll | p50 ns | p90 ns | max ns | replay_wall ms |
|---|---|---|---|---|---|
| `per_task` | busy | **25,888** | 43,835 | 61,726 | 49.2 |
| `single_packed` | busy | 25,411 | 43,938 | 74,206 | 49.4 |

p50 ≈ **25-26 μs** end-to-end from "graph launch" to "worker
observed advance", regardless of slot shape.

### Comparison to current `cots:dispatch_cb`

§1c.24's marker-filtered NVTX measured `cots:dispatch_cb` p50
= 1.45 μs, p90 = 1.99 μs (driver-thread cost of the existing
captured `cudaLaunchHostFunc` round-trip).

**The kernel-counter signaling adds ~24 μs/op of signal
latency vs the current host_fn pattern.** Per generate at
B=1: 24 μs × 56 ops × 128 forwards = ~172 ms/generate of
worker-start delay added.

This is the §1c.28 gate's "start latency" concern: a
mechanism that drops cudaGraphLaunch wall by N ms but adds
M ms of worker-start delay only wins if N − M > 0. The
§1c.27 `no_submit_hostfn` arm dropped cgl by 93 ms; if the
M2 mechanism replaces all 56 dispatch host_fns AND adds
24 μs/op of start delay, the net would be 93 − ~172 ms ≈
−79 ms (regression).

This finding is the headline of the smoke. **It is NOT a
correctness blocker (per_task is clean), but it strongly
suggests M2's expected wall-clock win is much smaller than
the §1c.27 no-submit ablation upper bound — and may be net
zero or negative — in real-mode operation.**

## Recommendation

### Slot shape: `per_task`

Per-task slots are the correctness contract. Single-packed loses
real signals at the tested poll rates and is rejected for
production. (The user's bias was right.)

### Worker poll policy: TBD

Both `busy` and `yield` give clean correctness on `per_task`.
Latency is similar (within run-to-run noise). For production
COTS, busy-spin is wasteful — burns one core continuously even
when no signals are arriving. `yield` is reasonable; `nanosleep`
with a short period (e.g., 100 ns) might be better still and
is worth a small follow-up sweep.

### M2 mechanism gate — recalibrate before implementation

The §1c.28 design said:

> `submit_signal_to_worker_start_ns` median ≤ today's baseline
> dispatch_cb-to-worker-start gap (estimate from §1c.24:
> dispatch_cb p50 1.45 μs + queue handoff ≈ 5 μs end-to-end).

The smoke measured ~25 μs. **The 5 μs target is not achievable
with the kernel-counter approach.** Two possibilities:

1. **The 5 μs target is too tight.** Re-derive the gate from
   real-mode CPU-GEMM tail risk: how much start-delay can the
   worker absorb before CPU GEMM leaks past the GPU window?
   §1c.24 measured per-fire `worker_qkv` p50 = 67 μs (eager),
   `worker_mlp` = 486 μs. A 25 μs start delay is small vs the
   MLP compute time; CPU GEMM finishes at 25+486 = 511 μs vs
   the GPU window of probably 500-700 μs of GEMMs. Marginal but
   possibly OK.

2. **A different mechanism is needed.** Options not yet
   explored in the smoke:
   - **Literal-value `cuStreamWriteValue32` + per-replay value
     update via `cudaGraphExecSetParams`.** Bypasses the
     kernel-counter overhead, but requires a host-side hook
     between every replay — defeats the host_fn-elimination
     purpose unless the param-update is cheaper than a host_fn.
   - **`cudaEventRecord` after D2H, worker
     `cudaEventSynchronize`.** The replay re-arm trap the user
     warned about needs a generation-counter scheme; smoke
     not yet written.
   - **`cudaStreamWaitValue32` only on sync side (M3-only) with
     submit kept as host_fn.** Inverts §1c.28's plan — keep
     submit cheap (host_fn), only replace sync. §1c.27 said
     sync-only ablation cut 273 ms cgl (vs 93 ms for submit-
     only); this asymmetric path might land bigger benefit
     from less invasive change.

### Stop before M2 implementation; design pass needed

This smoke wasn't a green light for M2 implementation as
designed. Before writing M2 code, the §1c.28 design should be
revised to:

1. Acknowledge the ~25 μs kernel-counter signal-latency
   floor.
2. Either accept it (with the worker-start gate widened to
   match) or change the primitive (literal-write + param-
   update, or events with generation counters, or M3-only).
3. Re-derive the wall-clock upper-bound estimate accounting
   for the start-delay tax.

## Update — M3 sync-side smoke (`m3_smoke.cu`)

After the M2 result above, the §1c.28 design was repivoted to
M3 (sync-side replacement only). A second standalone smoke
tests captured kernel-spin on a worker-written monotonic done
counter under repeated graph replay.

The kernel does atomicAdd to compute this fire's seq, writes
seq to `req_slot[t]` (host-mapped pinned), then spins on
`done_slot[t]` until `done >= seq`. Worker observes req
advance, runs fake CPU work, writes done back. Per-task slots
only.

### Correctness (1,000 replays × 56 tasks)

| poll | sync_each | total_observed | stale | per_task_min | per_task_max |
|---|---|---|---|---|---|
| busy | yes | **56,000** | 0 | 1,000 | 1,000 ✓ |
| busy | no  | **56,000** | 0 | 1,000 | 1,000 ✓ |
| yield | yes | **56,000** | 0 | 1,000 | 1,000 ✓ |

All clean; deterministic checksum 0x7bf0 across configs.

### Single-task latency (1,000 replays × 1 task — isolates
per-fire cost without 56-task stack-up)

| poll | per_replay_wall μs | request_to_obs p50 ns | p90 ns |
|---|---|---|---|
| busy-spin (poll=0) | **5.91** | **3,027** | 3,141 |
| yield (poll=1) | 5.83 | 3,226 | 3,274 |
| nanosleep 100 ns | 51.2 | 48,537 | 48,943 |

Single-fire M3 kernel-spin + busy-spin worker: ~5.9 μs
end-to-end. That's the per-fire overhead the captured graph
pays for sync.

### M3 viability — math closes positively

§1c.27 measured: removing sync host_fn cuts cgl by 273 ms over
156 launches × 56 sync fires per launch = **~31 μs per fire**
of cgl wall saved by removing the host_fn round-trip.

M3 kernel-spin replacement costs:
- Single-task busy-spin: 5.91 μs per fire
- 56-task amortized: 146 μs / 56 ≈ 2.6 μs per fire (in-stream
  pacing once kernels are queued)

**Per-fire delta: M3 saves ~25 μs vs host_fn(sync_cb).** Per
generate at B=1, 56 ops × 128 forwards × 25 μs ≈ **+179 ms
saved**. That's comparable to §1c.27's no_sync_hostfn cgl
delta (273 ms upper bound) after subtracting the M3 mechanism's
own overhead.

### Real-mode caveat

Both the existing host_fn(sync_cb) and M3 kernel-spin wait for
the worker's CPU GEMM (~500 μs/MLP) before the captured stream
proceeds. The savings come from the per-fire OVERHEAD on top
of the CPU GEMM wait, which is independent of CPU GEMM
duration. So the +179 ms estimate above should hold in
real-mode too, modulo run-to-run variance.

### M3 recommendation

**Prototype M3 in vLLM behind a feature flag.** The smoke
clears the reviewer's stated criteria:

* No stale waits, no drops, no duplicates ✓
* No deadlock ✓
* Host-mapped visibility works ✓
* cudaGraphLaunch impact: 5.9 μs per fire vs host_fn's 31 μs
  per fire = **5×** faster mechanism ✓
* Wait overhead: 3.0 μs request-to-obs p50 (busy-spin); ~25 μs
  net savings per fire ✓

Implementation will mirror the kernel-spin approach: each COTS
op's captured graph fires `m3_request_and_wait_kernel(req_counter[t],
req_slot[t], done_slot[t])` instead of the existing
`cudaLaunchHostFunc(sync_cb, ...)`. Submit-side stays as the
existing host_fn(dispatch_cb) — §1c.27 showed submit-only
ablation cuts only 93 ms cgl, and §1c.28 Step 1 already showed
M2 kernel-counter regresses on the submit side.

### Alternate paths (lower priority unless M3 prototype fails)

* **`cuStreamWaitValue32` with cyclic per-replay slots** — the
  literal-value variant works if we cycle slot IDs across
  replays modulo K. Requires K ≥ max-in-flight replays;
  complexity vs benefit unclear.
* **Switch to native_eager** — keep the existing host_fn path
  but disable graph capture for the COTS forward. §1c.25
  showed native_eager_dryrun is +0.382 s/gen vs none vs
  native_capture_dryrun's +0.584 s/gen. Eager is +0.694 s
  vs none on real-mode (§1c.25 wall-clock landscape) vs
  capture's +0.835 s — a 141 ms gap that M3 should be able
  to close.

## Artifacts

* `value_signal_smoke.cu` — M2-side (submit-replacement) test.
* `m3_smoke.cu` — M3-side (sync-replacement) test.
* `value_signal_smoke`, `m3_smoke` — built binaries (gitignored).
* `result_*.json` — machine-readable per-config outputs (M2 and
  M3 both).
* `result_*.txt` — full stdout per config (gitignored).
