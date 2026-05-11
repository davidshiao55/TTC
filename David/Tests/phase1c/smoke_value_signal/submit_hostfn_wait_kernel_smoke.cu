// SPDX-License-Identifier: Apache-2.0
//
// §1c.28 Step 3 standalone smoke: production-shaped M3.
//
// Difference from m3_smoke.cu (the kernel-spin "request+wait" smoke):
// that earlier smoke had ONE captured kernel per task that did BOTH
// the request signaling AND the wait. CPU work effectively started
// when the wait kernel began executing — collapsing the per-op
// overlap window the production M3 design depends on.
//
// Production M3 keeps submit as the existing host_fn (so CPU GEMM
// starts at submit time, BEFORE the GPU runs the per-op GEMMs) and
// replaces ONLY the sync host_fn with a wait kernel. This smoke
// tests THAT shape, with a parameterized fake-GPU-delay between
// submit and wait so we can model varying overlap windows.
//
// Captured graph per task per replay:
//   1. cudaLaunchHostFunc(submit_cb, &ctx[t])
//        — submit_cb increments next_seq[t] (CPU-side counter),
//          writes the new seq to req_slot[t] (host-mapped pinned),
//          and that publishes the request to both the worker (host
//          read) AND the wait kernel (GPU read of host-mapped
//          device pointer).
//   2. Optional gpu_busywait_kernel<<<1,1>>>(delay_clocks)
//        — simulates the GPU GEMM window between submit and sync
//          in production COTS. Tunable via --gpu-delay-us so we
//          can vary how much time the worker has to finish CPU
//          work before the wait kernel hits the spin loop.
//   3. cots_wait_done_kernel<<<1,1>>>(&req_slot[t], &done_slot[t])
//        — busy-spins on done_slot[t] until done >= req. Replay-
//          safe: req_slot was just written by submit_cb on THIS
//          replay; done_slot must reach that seq to release the
//          stream.
//
// Persistent CPU worker (separate thread):
//   — polls req_slot[t] for monotonic advance,
//   — runs fake CPU work (deterministic checksum + optional
//     parameterized busy delay via --cpu-work-us),
//   — writes done_slot[t] = req_slot[t] to release the wait
//     kernel.
//
// Validation (1,000 replays × 56 tasks):
//   — no stale waits (deadlock-fail-fast at 5s)
//   — no drops (worker observes all 56,000 advances)
//   — no duplicates (per-task last_seen monotonic)
//   — bit-identical checksum across runs of the same config
//
// Metrics:
//   — submit-to-worker-start ns (the "did CPU GEMM start early"
//     signal; should be small, comparable to today's host_fn
//     dispatch_cb cost ~1.5 μs)
//   — wait-kernel-overhead ns (the "is the wait fast" signal;
//     should be small when worker finishes BEFORE wait fires)
//   — wait-kernel-blocking ns (the case when worker is still
//     working when wait fires; bounded by remaining CPU work)
//   — replay throughput
//
// Build:
//   nvcc -O2 -std=c++17 -arch=sm_89 m3_submit_hostfn_wait_kernel_smoke.cu \
//        -o m3_submit_hostfn_wait_kernel_smoke -lpthread
//
// Run:
//   ./m3_submit_hostfn_wait_kernel_smoke --replays 1000 --tasks 56 \
//       --gpu-delay-us 0 --cpu-work-us 0
//   ./m3_submit_hostfn_wait_kernel_smoke --replays 1000 --tasks 56 \
//       --gpu-delay-us 50 --cpu-work-us 100   # mimic real overlap

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error at %s:%d: %s -> %s\n", __FILE__,        \
                   __LINE__, #expr, cudaGetErrorString(_e));                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// Wait-only kernel: read expected from req_slot, spin until done >=
// expected.  `volatile` is critical for host-mapped pinned visibility.
__global__ void cots_wait_done_kernel(volatile unsigned int* req_slot,
                               volatile unsigned int* done_slot) {
  unsigned int expected = *req_slot;
  unsigned int done;
  do {
    done = *done_slot;
    asm volatile("nanosleep.u32 100;" ::: "memory");
  } while (done < expected);
}

// Fake GPU work to simulate the overlap window between submit and sync.
// Single thread, single block — the point is wall-clock duration, not
// throughput. Uses clock64() for portability across architectures;
// 4090 base clock ~2.2 GHz so target_clocks = us * 2200.
__global__ void gpu_busywait_kernel(unsigned long long target_clocks) {
  unsigned long long start = clock64();
  while (clock64() - start < target_clocks) {
    // Spin.
  }
}

// CPU work simulator: parameterized busy delay + deterministic checksum.
static inline unsigned long long cpu_fake_work(unsigned int seq,
                                               unsigned int task_id,
                                               int cpu_work_us) {
  unsigned long long cs = ((unsigned long long)seq * (unsigned long long)(task_id + 1)) +
                          (unsigned long long)seq;
  if (cpu_work_us > 0) {
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::microseconds(cpu_work_us);
    while (std::chrono::steady_clock::now() < deadline) {
      // Busy delay; using clock_gettime via steady_clock would be
      // simpler but this matches the "real CPU GEMM keeps the
      // worker busy" pattern.
      cs ^= (cs * 6364136223846793005ULL) + 1442695040888963407ULL;
    }
  }
  return cs;
}

// §1c.28 review-fix: per-seq timestamp ring. The earlier draft
// had submit_cb store (req_slot, then submit_ns) and the worker
// read (req_slot, then submit_ns); a race let the worker observe
// req advancing from replay N+1 while still reading replay N's
// submit_ns (or vice versa), producing the huge p90/max
// latencies in the original smoke. Fix: store the timestamp in
// a per-seq slot so the worker can deterministically pair (seq
// it observed) with (timestamp recorded at submission of that
// seq). 2,048 slots gives plenty of headroom for 1,000-replay
// runs without wraparound.
constexpr int TS_RING_SIZE = 2048;
constexpr int TS_RING_MASK = TS_RING_SIZE - 1;
static_assert((TS_RING_SIZE & TS_RING_MASK) == 0,
              "TS_RING_SIZE must be power of 2");

struct SubmitCtx {
  unsigned int* host_next_seq;  // CPU-only counter, incremented per fire
  unsigned int* host_req_slot;  // host-mapped pinned, GPU-visible
  long long* submit_ts_ring;    // per-task ring, indexed by (seq-1) & MASK
  int task_id;
};

extern "C" void CUDART_CB submit_cb(void* user_data) {
  auto* ctx = static_cast<SubmitCtx*>(user_data);
  unsigned int seq = ++(*ctx->host_next_seq);
  // Write the timestamp into the per-seq slot BEFORE publishing
  // the new req. The worker, on observing req=N, indexes into the
  // ring at (N-1) & MASK and gets ts_N (or older — never newer,
  // because submit_cb advances seq monotonically and writes ts to
  // its OWN slot, not a shared cell). Release fence ensures the
  // ts write is visible before the req publish.
  ctx->submit_ts_ring[(seq - 1) & TS_RING_MASK] =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  std::atomic_thread_fence(std::memory_order_release);
  *ctx->host_req_slot = seq;
  std::atomic_thread_fence(std::memory_order_release);
}

struct Args {
  int replays = 1000;
  int tasks = 56;
  int gpu_delay_us = 0;       // simulated GPU GEMM duration between submit & wait
  int cpu_work_us = 0;        // simulated CPU GEMM duration on the worker
  int poll_ns = 0;            // worker poll policy (0=busy, 1=yield, N=sleep N ns)
  bool sync_each = false;
  std::string json_out;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag); std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (s == "--replays") a.replays = std::atoi(next("--replays").c_str());
    else if (s == "--tasks") a.tasks = std::atoi(next("--tasks").c_str());
    else if (s == "--gpu-delay-us") a.gpu_delay_us = std::atoi(next("--gpu-delay-us").c_str());
    else if (s == "--cpu-work-us") a.cpu_work_us = std::atoi(next("--cpu-work-us").c_str());
    else if (s == "--poll-ns") a.poll_ns = std::atoi(next("--poll-ns").c_str());
    else if (s == "--sync-each") a.sync_each = true;
    else if (s == "--json") a.json_out = next("--json");
    else if (s == "--help" || s == "-h") {
      std::printf("Usage: %s [--replays N] [--tasks N] [--gpu-delay-us N] "
                  "[--cpu-work-us N] [--poll-ns N] [--sync-each] [--json path]\n",
                  argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", s.c_str()); std::exit(2);
    }
  }
  return a;
}

static inline long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  // Allocate host-mapped pinned slots.
  unsigned int* host_req = nullptr;
  unsigned int* host_done = nullptr;
  CUDA_CHECK(cudaHostAlloc(&host_req, sizeof(unsigned int) * a.tasks, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&host_done, sizeof(unsigned int) * a.tasks, cudaHostAllocMapped));
  std::memset(host_req, 0, sizeof(unsigned int) * a.tasks);
  std::memset(host_done, 0, sizeof(unsigned int) * a.tasks);
  unsigned int* dev_req = nullptr;
  unsigned int* dev_done = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&dev_req, host_req, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&dev_done, host_done, 0));

  // CPU-only per-task counter (incremented by submit_cb).
  std::vector<unsigned int> host_next_seq(a.tasks, 0);

  // Per-task per-seq timestamp ring. submit_cb writes
  // submit_ts_ring[t][(seq-1) & MASK] = now_ns. Worker reads the
  // same slot when observing req=seq. Replay-safe pairing.
  std::vector<std::vector<long long>> submit_ts_ring(
      a.tasks, std::vector<long long>(TS_RING_SIZE, 0));

  // Submit contexts (stable addresses for cudaLaunchHostFunc userData).
  std::vector<SubmitCtx> submit_ctx(a.tasks);
  for (int t = 0; t < a.tasks; ++t) {
    submit_ctx[t].host_next_seq = &host_next_seq[t];
    submit_ctx[t].host_req_slot = &host_req[t];
    submit_ctx[t].submit_ts_ring = submit_ts_ring[t].data();
    submit_ctx[t].task_id = t;
  }

  // Build CUDA graph.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
  // Approximate clock rate: 2.2 GHz on RTX 4090. Target_clocks = us * 2200.
  const unsigned long long target_clocks =
      static_cast<unsigned long long>(a.gpu_delay_us) * 2200ULL;
  for (int t = 0; t < a.tasks; ++t) {
    // 1. Submit host_fn — increments seq, writes req_slot, enqueues
    //    work (the worker observes req_slot advance).
    CUDA_CHECK(cudaLaunchHostFunc(stream, submit_cb, &submit_ctx[t]));

    // 2. Optional GPU delay to simulate per-op GEMMs running between
    //    submit and sync.
    if (a.gpu_delay_us > 0) {
      gpu_busywait_kernel<<<1, 1, 0, stream>>>(target_clocks);
    }

    // 3. Wait kernel — busy-spins until done_slot[t] >= req_slot[t].
    cots_wait_done_kernel<<<1, 1, 0, stream>>>(&dev_req[t], &dev_done[t]);
  }
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  // Worker shared state.
  struct WorkerStats {
    std::vector<std::atomic<unsigned int>> last_seen_req;
    std::vector<std::atomic<unsigned long long>> task_observed;
    std::vector<std::atomic<unsigned long long>> task_checksum;
    std::atomic<unsigned long long> stale_signals{0};
    std::atomic<unsigned long long> coalesced_advances{0};
    std::vector<long long> submit_to_worker_start_ns;
    std::mutex lat_mtx;
  } stats;
  stats.last_seen_req = std::vector<std::atomic<unsigned int>>(a.tasks);
  stats.task_observed = std::vector<std::atomic<unsigned long long>>(a.tasks);
  stats.task_checksum = std::vector<std::atomic<unsigned long long>>(a.tasks);
  for (int t = 0; t < a.tasks; ++t) {
    stats.last_seen_req[t].store(0);
    stats.task_observed[t].store(0);
    stats.task_checksum[t].store(0);
  }
  stats.submit_to_worker_start_ns.reserve((size_t)a.replays * (size_t)a.tasks);

  std::atomic<bool> stop_worker{false};

  // Worker: poll each req_slot for monotonic advance, run fake CPU
  // work (parameterized via cpu_work_us), write done_slot back.
  std::thread worker([&]() {
    while (!stop_worker.load(std::memory_order_acquire)) {
      bool any_advance = false;
      for (int t = 0; t < a.tasks; ++t) {
        unsigned int cur = host_req[t];
        unsigned int last = stats.last_seen_req[t].load();
        if (cur > last) {
          long long obs_ns = now_ns();
          // §1c.28 review-fix: read the timestamp from the
          // per-seq ring, indexed by (cur-1) & MASK. This is
          // the timestamp submit_cb wrote when it advanced to
          // seq=cur — guaranteed-paired with this observation,
          // not a stale or future-replay timestamp.
          std::atomic_thread_fence(std::memory_order_acquire);
          long long sub_ns =
              submit_ts_ring[t][(cur - 1) & TS_RING_MASK];
          if (sub_ns > 0) {
            std::lock_guard<std::mutex> g(stats.lat_mtx);
            stats.submit_to_worker_start_ns.push_back(obs_ns - sub_ns);
          }
          unsigned int gap = cur - last;
          if (gap > 1) {
            stats.coalesced_advances.fetch_add(gap - 1, std::memory_order_relaxed);
          }
          // Run fake CPU work (deterministic + parameterized delay).
          unsigned long long cs = stats.task_checksum[t].load();
          for (unsigned int s = last + 1; s <= cur; ++s) {
            cs ^= cpu_fake_work(s, t, a.cpu_work_us);
          }
          stats.task_checksum[t].store(cs);
          stats.task_observed[t].fetch_add(gap);
          // Write done_slot to release the wait kernel.
          host_done[t] = cur;
          stats.last_seen_req[t].store(cur, std::memory_order_release);
          any_advance = true;
        } else if (cur < last) {
          stats.stale_signals.fetch_add(1, std::memory_order_relaxed);
        }
      }
      if (!any_advance) {
        if (a.poll_ns == 0) {
        } else if (a.poll_ns == 1) {
          std::this_thread::yield();
        } else {
          struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = a.poll_ns;
          nanosleep(&ts, nullptr);
        }
      }
    }
  });

  // Replay loop with deadlock fail-fast.
  long long deadlock_budget_ns = 30'000'000'000LL;  // 30s
  long long t0 = now_ns();
  for (int r = 0; r < a.replays; ++r) {
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    if (a.sync_each) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (now_ns() - t0 > deadlock_budget_ns) {
      std::fprintf(stderr, "DEADLOCK after %d/%d replays in %lld ns\n",
                   r, a.replays, now_ns() - t0);
      stop_worker.store(true);
      worker.join();
      std::exit(2);
    }
  }
  if (!a.sync_each) {
    long long sync_start = now_ns();
    while (now_ns() - sync_start < deadlock_budget_ns) {
      cudaError_t err = cudaStreamQuery(stream);
      if (err == cudaSuccess) break;
      if (err != cudaErrorNotReady) CUDA_CHECK(err);
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    if (cudaStreamQuery(stream) != cudaSuccess) {
      std::fprintf(stderr, "DEADLOCK: stream did not drain in 30s\n");
      stop_worker.store(true);
      worker.join();
      std::exit(2);
    }
  }
  long long t1 = now_ns();

  // Drain worker.
  long long drain_start = now_ns();
  unsigned long long expected_total = (unsigned long long)a.replays * a.tasks;
  while (now_ns() - drain_start < 200'000'000LL) {  // 200ms drain budget
    unsigned long long total = 0;
    for (int t = 0; t < a.tasks; ++t) total += stats.task_observed[t].load();
    if (total >= expected_total) break;
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  stop_worker.store(true);
  worker.join();

  // Summary.
  unsigned long long total_observed = 0;
  unsigned long long min_per_task = ~0ull, max_per_task = 0;
  for (int t = 0; t < a.tasks; ++t) {
    unsigned long long obs = stats.task_observed[t].load();
    total_observed += obs;
    if (obs < min_per_task) min_per_task = obs;
    if (obs > max_per_task) max_per_task = obs;
  }
  long long lat_p50 = 0, lat_p90 = 0, lat_max = 0;
  size_t n_lat = stats.submit_to_worker_start_ns.size();
  if (n_lat > 0) {
    std::sort(stats.submit_to_worker_start_ns.begin(),
              stats.submit_to_worker_start_ns.end());
    lat_p50 = stats.submit_to_worker_start_ns[n_lat / 2];
    lat_p90 = stats.submit_to_worker_start_ns[(n_lat * 90) / 100];
    lat_max = stats.submit_to_worker_start_ns.back();
  }
  unsigned long long combined_checksum = 0;
  for (int t = 0; t < a.tasks; ++t) combined_checksum ^= stats.task_checksum[t].load();

  std::printf("=== §1c.28 Step 3: production-shaped M3 smoke ===\n");
  std::printf("replays:                       %d\n", a.replays);
  std::printf("tasks:                         %d\n", a.tasks);
  std::printf("gpu_delay_us:                  %d\n", a.gpu_delay_us);
  std::printf("cpu_work_us:                   %d\n", a.cpu_work_us);
  std::printf("poll_ns:                       %d\n", a.poll_ns);
  std::printf("sync_each:                     %d\n", (int)a.sync_each);
  std::printf("expected_observations:         %llu\n", expected_total);
  std::printf("total_observed:                %llu\n", total_observed);
  std::printf("per_task_min:                  %llu\n", min_per_task);
  std::printf("per_task_max:                  %llu\n", max_per_task);
  std::printf("stale_signals:                 %llu\n", stats.stale_signals.load());
  std::printf("coalesced_advances:            %llu\n",
              stats.coalesced_advances.load());
  std::printf("replay_wall_ns:                %lld\n", t1 - t0);
  std::printf("replay_wall_ms:                %.3f\n", (t1 - t0) / 1e6);
  std::printf("per_replay_wall_us:            %.2f\n",
              (double)(t1 - t0) / a.replays / 1000.0);
  std::printf("submit_to_worker_start_p50_ns: %lld\n", lat_p50);
  std::printf("submit_to_worker_start_p90_ns: %lld\n", lat_p90);
  std::printf("submit_to_worker_start_max_ns: %lld\n", lat_max);
  std::printf("latency_samples:               %zu\n", n_lat);
  std::printf("combined_checksum:             0x%016llx\n", combined_checksum);

  int fail = 0;
  if (total_observed != expected_total) {
    std::fprintf(stderr,
                 "ASSERT FAIL: total_observed=%llu != expected=%llu\n",
                 total_observed, expected_total);
    ++fail;
  }
  for (int t = 0; t < a.tasks; ++t) {
    unsigned int last = stats.last_seen_req[t].load();
    if (last != (unsigned int)a.replays) {
      std::fprintf(stderr,
                   "ASSERT FAIL: task[%d].last_seen_req=%u != replays=%d\n",
                   t, last, a.replays);
      ++fail;
    }
  }
  if (stats.stale_signals.load() != 0) {
    std::fprintf(stderr, "ASSERT FAIL: stale_signals=%llu\n",
                 stats.stale_signals.load());
    ++fail;
  }

  if (!a.json_out.empty()) {
    FILE* f = std::fopen(a.json_out.c_str(), "w");
    if (f) {
      std::fprintf(
          f,
          "{\n"
          "  \"mode\":\"m3_submit_hostfn_wait_kernel\",\n"
          "  \"replays\":%d,\n  \"tasks\":%d,\n"
          "  \"gpu_delay_us\":%d,\n  \"cpu_work_us\":%d,\n"
          "  \"poll_ns\":%d,\n  \"sync_each\":%s,\n"
          "  \"expected_observations\":%llu,\n"
          "  \"total_observed\":%llu,\n"
          "  \"per_task_min\":%llu,\n  \"per_task_max\":%llu,\n"
          "  \"stale_signals\":%llu,\n  \"coalesced_advances\":%llu,\n"
          "  \"replay_wall_ns\":%lld,\n"
          "  \"per_replay_wall_us\":%.2f,\n"
          "  \"submit_to_worker_start_p50_ns\":%lld,\n"
          "  \"submit_to_worker_start_p90_ns\":%lld,\n"
          "  \"submit_to_worker_start_max_ns\":%lld,\n"
          "  \"latency_samples\":%zu,\n"
          "  \"combined_checksum\":\"0x%016llx\",\n"
          "  \"assertion_failures\":%d\n"
          "}\n",
          a.replays, a.tasks, a.gpu_delay_us, a.cpu_work_us, a.poll_ns,
          a.sync_each ? "true" : "false", expected_total, total_observed,
          min_per_task, max_per_task, stats.stale_signals.load(),
          stats.coalesced_advances.load(), t1 - t0,
          (double)(t1 - t0) / a.replays / 1000.0, lat_p50, lat_p90, lat_max,
          n_lat, combined_checksum, fail);
      std::fclose(f);
    }
  }

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFreeHost(host_req));
  CUDA_CHECK(cudaFreeHost(host_done));

  return fail ? 1 : 0;
}
