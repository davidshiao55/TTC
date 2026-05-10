// SPDX-License-Identifier: Apache-2.0
//
// §1c.28 M3 standalone smoke: prove that a captured CUDA graph can
// SAFELY wait on a worker-written monotonic done counter, replay-
// aware, with no stale/dup/drop/deadlock under 1,000 graph replays.
//
// Self-contained CLI binary; deliberately does NOT touch the COTS /
// vLLM code path. Pairs with `value_signal_smoke.cu` (M2-side smoke)
// — that one tested submit-side replacement; this one tests sync-
// side replacement only.
//
// What's tested
// -------------
// Each replay launches one captured kernel per task. The kernel does
// THREE things atomically per fire:
//   (1) atomicAdd(req_counter[t], 1)+1 to compute this fire's seq.
//   (2) Write seq to host-mapped pinned `req_slot[t]` (CPU sees it).
//   (3) Spin-wait on `done_slot[t]` until done >= seq.
// The kernel terminates only when the worker has acknowledged
// THIS fire by writing done >= seq. The captured graph holds the
// stream busy on this kernel until the worker drains.
//
// Worker thread polls `req_slot[t]` for monotonic advance, runs
// fake CPU work (deterministic checksum), then writes `done_slot[t]
// = req_slot[t]`. Per-task slots only — Step 1 already established
// per-task is the correctness contract.
//
// Why this shape and not literal `cuStreamWaitValue32(slot, value=N,
// GEQ)`:
//   `cuStreamWaitValue32` takes a literal at capture time. With GEQ
//   semantics, the wait succeeds whenever `done >= N` for some
//   captured-time N. After replay 1, done stays at N (or higher);
//   replay 2's wait then succeeds immediately, BEFORE the worker
//   has processed replay 2's request. That's the M3 stale-wait
//   trap. The kernel-spin pattern below is replay-safe by
//   construction: each replay's wait condition is `done >= THIS
//   replay's atomicAdd result`, which advances naturally.
//
// Build:
//   nvcc -O2 -std=c++17 m3_smoke.cu -o m3_smoke -lpthread
//
// Run:
//   ./m3_smoke --replays 1000 --tasks 56 --json result_m3_busy.json
//   ./m3_smoke --replays 1000 --tasks 56 --poll-ns 1 --json result_m3_yield.json

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

// Captured kernel: atomically advance request seq, publish, wait
// for matching done. Single thread, single block — minimal GPU
// resource use; this is a synchronization primitive, not compute.
//
// `volatile` on req_slot/done_slot is critical for host-mapped
// pinned visibility — without it, the compiler is allowed to
// cache the load.
__global__ void m3_request_and_wait_kernel(unsigned int* req_counter,
                                           volatile unsigned int* req_slot,
                                           volatile unsigned int* done_slot) {
  unsigned int seq = atomicAdd(req_counter, 1u) + 1u;
  *req_slot = seq;
  // Membar to publish the req_slot write to the host before the
  // wait loop begins.
  __threadfence_system();
  unsigned int d;
  do {
    d = *done_slot;
    // PTX nanosleep avoids burning SM cycles on a spin. Available
    // on sm_70+; on older arches, this is a no-op.
    asm volatile("nanosleep.u32 100;" ::: "memory");
  } while (d < seq);
}

enum class Mode { KernelSpin };  // Reserved for future modes (e.g.,
                                 // cuStreamWaitValue32 literal — but
                                 // its replay-stale-trap is already
                                 // documented above).

struct Args {
  int replays = 1000;
  int tasks = 56;
  // Worker poll policy for the request slots:
  //   0 = busy-spin
  //   1 = std::this_thread::yield() between rounds
  //   N>1 = nanosleep N ns between rounds
  int poll_ns = 0;
  // sync_each: like Step 1, force cudaStreamSynchronize between
  // graph launches so latency reflects single-fire end-to-end
  // rather than queue-depth latency.
  bool sync_each = false;
  std::string json_out;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", flag);
        std::exit(2);
      }
      return std::string(argv[++i]);
    };
    if (s == "--replays") a.replays = std::atoi(next("--replays").c_str());
    else if (s == "--tasks") a.tasks = std::atoi(next("--tasks").c_str());
    else if (s == "--poll-ns") a.poll_ns = std::atoi(next("--poll-ns").c_str());
    else if (s == "--sync-each") a.sync_each = true;
    else if (s == "--json") a.json_out = next("--json");
    else if (s == "--help" || s == "-h") {
      std::printf("Usage: %s [--replays N] [--tasks N] [--poll-ns N] "
                  "[--sync-each] [--json path]\n", argv[0]);
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

struct Stats {
  // Per-task last-seen request seq (monotonic).
  std::vector<std::atomic<unsigned int>> last_seen_req;
  // Per-task observation count (how many request advances the
  // worker processed; coalesced gaps add `gap` per observation).
  std::vector<std::atomic<unsigned long long>> task_observed;
  // Per-task deterministic checksum of seq values.
  std::vector<std::atomic<unsigned long long>> task_checksum;
  // Anomaly counters.
  std::atomic<unsigned long long> stale_signals{0};
  std::atomic<unsigned long long> coalesced_advances{0};
  // Per-task latency: (worker observation time) - (graph launch
  // time of the matching replay). Sampled over all observed
  // advances.
  std::vector<long long> request_latencies_ns;
  std::mutex request_latencies_mtx;
};

// CPU worker: poll request slots, on advance, run fake work
// (deterministic checksum), write done back to release the
// captured kernel's wait.
static void worker_per_task(volatile unsigned int* host_req,
                            volatile unsigned int* host_done, int n_slots,
                            std::atomic<bool>* stop_worker, Stats* stats,
                            std::atomic<long long>* last_launch_ns,
                            int poll_ns) {
  while (!stop_worker->load(std::memory_order_acquire)) {
    bool any_advance = false;
    for (int t = 0; t < n_slots; ++t) {
      unsigned int cur = host_req[t];
      unsigned int last = stats->last_seen_req[t].load(std::memory_order_acquire);
      if (cur > last) {
        long long obs_ns = now_ns();
        long long lln = last_launch_ns->load(std::memory_order_acquire);
        if (lln > 0) {
          std::lock_guard<std::mutex> g(stats->request_latencies_mtx);
          stats->request_latencies_ns.push_back(obs_ns - lln);
        }
        unsigned int gap = cur - last;
        if (gap > 1) {
          stats->coalesced_advances.fetch_add(gap - 1, std::memory_order_relaxed);
        }
        // Process: deterministic checksum.
        unsigned long long cs = stats->task_checksum[t].load();
        cs ^= ((unsigned long long)cur * (unsigned long long)(t + 1)) +
              (unsigned long long)cur;
        stats->task_checksum[t].store(cs);
        stats->task_observed[t].fetch_add(gap);
        // Acknowledge: release the GPU kernel's wait by writing
        // done = cur (the request seq we just observed).
        host_done[t] = cur;
        stats->last_seen_req[t].store(cur, std::memory_order_release);
        any_advance = true;
      } else if (cur < last) {
        stats->stale_signals.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (!any_advance) {
      if (poll_ns == 0) {
      } else if (poll_ns == 1) {
        std::this_thread::yield();
      } else {
        struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = poll_ns;
        nanosleep(&ts, nullptr);
      }
    }
  }
}

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  // Allocate host-mapped pinned slots: one req + one done per task.
  unsigned int* host_req = nullptr;
  unsigned int* host_done = nullptr;
  CUDA_CHECK(cudaHostAlloc(&host_req, sizeof(unsigned int) * a.tasks,
                           cudaHostAllocMapped));
  CUDA_CHECK(cudaHostAlloc(&host_done, sizeof(unsigned int) * a.tasks,
                           cudaHostAllocMapped));
  std::memset(host_req, 0, sizeof(unsigned int) * a.tasks);
  std::memset(host_done, 0, sizeof(unsigned int) * a.tasks);
  unsigned int* dev_req = nullptr;
  unsigned int* dev_done = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&dev_req, host_req, 0));
  CUDA_CHECK(cudaHostGetDevicePointer(&dev_done, host_done, 0));

  // Per-task GPU device counter (used by atomicAdd in the kernel).
  unsigned int* dev_req_counter = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_req_counter, sizeof(unsigned int) * a.tasks));
  CUDA_CHECK(cudaMemset(dev_req_counter, 0, sizeof(unsigned int) * a.tasks));

  // Stream + capture graph.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
  for (int t = 0; t < a.tasks; ++t) {
    m3_request_and_wait_kernel<<<1, 1, 0, stream>>>(
        &dev_req_counter[t], &dev_req[t], &dev_done[t]);
  }
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  Stats stats;
  stats.last_seen_req = std::vector<std::atomic<unsigned int>>(a.tasks);
  stats.task_observed = std::vector<std::atomic<unsigned long long>>(a.tasks);
  stats.task_checksum = std::vector<std::atomic<unsigned long long>>(a.tasks);
  for (int t = 0; t < a.tasks; ++t) {
    stats.last_seen_req[t].store(0);
    stats.task_observed[t].store(0);
    stats.task_checksum[t].store(0);
  }
  stats.request_latencies_ns.reserve((size_t)a.replays * (size_t)a.tasks);

  std::atomic<bool> stop_worker{false};
  std::atomic<long long> last_launch_ns{0};
  std::thread worker(worker_per_task, host_req, host_done, a.tasks,
                     &stop_worker, &stats, &last_launch_ns, a.poll_ns);

  // Replay loop. Watchdog timer fail-fasts on deadlock.
  long long deadlock_budget_ns = 5'000'000'000LL;  // 5 s for 1k replays
  long long t0 = now_ns();
  int replays_done = 0;
  for (int r = 0; r < a.replays; ++r) {
    last_launch_ns.store(now_ns(), std::memory_order_release);
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    if (a.sync_each) {
      // Wait for this replay's kernels to complete before launching
      // the next. The kernel only completes when the worker has
      // ack'd, so this is the natural M3 single-fire latency.
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    ++replays_done;
    if (now_ns() - t0 > deadlock_budget_ns) {
      std::fprintf(stderr,
                   "DEADLOCK: replays_done=%d/%d after %lld ns\n",
                   replays_done, a.replays, now_ns() - t0);
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
    cudaError_t final_err = cudaStreamQuery(stream);
    if (final_err != cudaSuccess) {
      std::fprintf(stderr, "DEADLOCK: stream did not drain in 5s\n");
      stop_worker.store(true);
      worker.join();
      std::exit(2);
    }
  }
  long long t1 = now_ns();

  // Allow worker to drain final advances.
  long long drain_budget_ns = 50'000'000;  // 50 ms
  long long drain_start = now_ns();
  unsigned long long expected_total = (unsigned long long)a.replays * a.tasks;
  while (now_ns() - drain_start < drain_budget_ns) {
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
  size_t n_lat = stats.request_latencies_ns.size();
  if (n_lat > 0) {
    std::sort(stats.request_latencies_ns.begin(),
              stats.request_latencies_ns.end());
    lat_p50 = stats.request_latencies_ns[n_lat / 2];
    lat_p90 = stats.request_latencies_ns[(n_lat * 90) / 100];
    lat_max = stats.request_latencies_ns.back();
  }
  unsigned long long combined_checksum = 0;
  for (int t = 0; t < a.tasks; ++t) {
    combined_checksum ^= stats.task_checksum[t].load();
  }

  std::printf("=== §1c.28 M3 sync-side smoke ===\n");
  std::printf("replays:                    %d\n", a.replays);
  std::printf("tasks:                      %d\n", a.tasks);
  std::printf("poll_ns:                    %d\n", a.poll_ns);
  std::printf("sync_each:                  %d\n", (int)a.sync_each);
  std::printf("expected_observations:      %llu\n", expected_total);
  std::printf("total_observed:             %llu\n", total_observed);
  std::printf("per_task_min:               %llu\n", min_per_task);
  std::printf("per_task_max:               %llu\n", max_per_task);
  std::printf("stale_signals:              %llu\n", stats.stale_signals.load());
  std::printf("coalesced_advances:         %llu\n",
              stats.coalesced_advances.load());
  std::printf("replay_wall_ns:             %lld\n", t1 - t0);
  std::printf("replay_wall_ms:             %.3f\n", (t1 - t0) / 1e6);
  std::printf("per_replay_wall_us:         %.2f\n",
              (double)(t1 - t0) / a.replays / 1000.0);
  std::printf("request_to_obs_p50_ns:      %lld\n", lat_p50);
  std::printf("request_to_obs_p90_ns:      %lld\n", lat_p90);
  std::printf("request_to_obs_max_ns:      %lld\n", lat_max);
  std::printf("latency_samples:            %zu\n", n_lat);
  std::printf("combined_checksum:          0x%016llx\n", combined_checksum);

  int fail = 0;
  if (total_observed != expected_total) {
    std::fprintf(stderr,
                 "ASSERT FAIL: total_observed=%llu != expected=%llu "
                 "(stale waits, drops, or stuck worker)\n",
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
          "  \"mode\":\"m3_kernel_spin_per_task\",\n"
          "  \"replays\":%d,\n  \"tasks\":%d,\n  \"poll_ns\":%d,\n"
          "  \"sync_each\":%s,\n"
          "  \"expected_observations\":%llu,\n"
          "  \"total_observed\":%llu,\n"
          "  \"per_task_min\":%llu,\n  \"per_task_max\":%llu,\n"
          "  \"stale_signals\":%llu,\n"
          "  \"coalesced_advances\":%llu,\n"
          "  \"replay_wall_ns\":%lld,\n"
          "  \"per_replay_wall_us\":%.2f,\n"
          "  \"request_to_obs_p50_ns\":%lld,\n"
          "  \"request_to_obs_p90_ns\":%lld,\n"
          "  \"request_to_obs_max_ns\":%lld,\n"
          "  \"latency_samples\":%zu,\n"
          "  \"combined_checksum\":\"0x%016llx\",\n"
          "  \"assertion_failures\":%d\n"
          "}\n",
          a.replays, a.tasks, a.poll_ns, a.sync_each ? "true" : "false",
          expected_total, total_observed, min_per_task, max_per_task,
          stats.stale_signals.load(), stats.coalesced_advances.load(),
          t1 - t0, (double)(t1 - t0) / a.replays / 1000.0, lat_p50, lat_p90,
          lat_max, n_lat, combined_checksum, fail);
      std::fclose(f);
    }
  }

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dev_req_counter));
  CUDA_CHECK(cudaFreeHost(host_req));
  CUDA_CHECK(cudaFreeHost(host_done));

  return fail ? 1 : 0;
}
