// SPDX-License-Identifier: Apache-2.0
//
// §1c.28 Step 1 standalone smoke: prove that a captured CUDA graph
// using value-write signaling over host-mapped pinned slots is
// replay-safe. Self-contained CLI binary; deliberately does NOT
// touch the COTS / vLLM code path.
//
// What's tested
// -------------
// Each replay fires N "task" writes from the captured stream into
// host-mapped pinned signal slots. A persistent CPU worker thread
// polls the slots, validates monotonic sequence numbers per slot,
// computes a deterministic per-task checksum, and times the gap
// between graph-launch and first observation.
//
// Two shapes are tested:
//
//   per_task     — one slot per task_id; each captured kernel
//                  atomically increments a per-task device counter
//                  and writes the new value into its own slot.
//                  Worker polls each slot for monotonic advance.
//
//   single_packed — one shared slot; each captured kernel atomically
//                  increments a single device counter and writes
//                  `(seq << 16) | task_id` into the shared slot.
//                  Worker polls the single slot, decodes both
//                  fields, advances when seq strictly increases.
//                  This shape is fragile by construction: if the
//                  GPU writes faster than the worker polls, some
//                  intermediate (seq, task_id) pairs are
//                  overwritten before observation.
//
// Why the kernel pattern (atomicAdd) instead of a literal-value
// cuStreamWriteValue32:
//   cuStreamWriteValue32 takes a literal at capture time. Replays
//   re-fire the same literal — the worker would see identical seq
//   across replays, defeating monotonicity. The production M2
//   design plan (§1c.28) accepts this and uses a tiny atomicAdd
//   kernel for the counter increment; this smoke validates that
//   primitive end-to-end under repeated graph replay.
//
// Build:
//   nvcc -O2 -std=c++17 value_signal_smoke.cu -o value_signal_smoke -lpthread
//
// Run:
//   ./value_signal_smoke --shape per_task     --replays 1000 --tasks 56
//   ./value_signal_smoke --shape single_packed --replays 1000 --tasks 56
//   ./value_signal_smoke --shape per_task --replays 1000 --tasks 56 --json out.json

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

// ------------------------- Captured kernels ------------------------- //

// Per-task: atomically increment counter[t]; write new value to slot[t].
__global__ void inc_per_task_kernel(unsigned int* counter,
                                    unsigned int* slot) {
  unsigned int v = atomicAdd(counter, 1u) + 1u;
  *slot = v;
}

// Single packed slot: atomically increment counter; write
// (seq << 16) | task_id to the shared slot.
__global__ void inc_packed_kernel(unsigned int* counter, unsigned int* slot,
                                  unsigned int task_id) {
  unsigned int v = atomicAdd(counter, 1u) + 1u;
  *slot = (v << 16) | (task_id & 0xFFFFu);
}

// ------------------------- CLI parsing ------------------------- //

enum class Shape { PerTask, SinglePacked };

struct Args {
  Shape shape = Shape::PerTask;
  int replays = 1000;
  int tasks = 56;
  // Worker poll policy:
  //   0 = busy-spin (no yield/sleep)
  //   1 = std::this_thread::yield() between polling rounds
  //   N>1 = nanosleep N ns between polling rounds
  int poll_ns = 0;
  // sync_each: if true, cudaStreamSynchronize between consecutive
  // graph launches. Without it, 1,000 launches stack up in <50 ms
  // and the worker can't keep pace, so the reported "latency" is
  // really queue-depth latency, not single-replay end-to-end.
  // With it, each replay is launched only after the previous one
  // drains — the gap between graph_launch and worker observation
  // approximates the real single-fire signal latency that M2 cares
  // about.
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
    if (s == "--shape") {
      std::string v = next("--shape");
      if (v == "per_task") a.shape = Shape::PerTask;
      else if (v == "single_packed") a.shape = Shape::SinglePacked;
      else { std::fprintf(stderr, "Unknown shape: %s\n", v.c_str()); std::exit(2); }
    } else if (s == "--replays") {
      a.replays = std::atoi(next("--replays").c_str());
    } else if (s == "--tasks") {
      a.tasks = std::atoi(next("--tasks").c_str());
    } else if (s == "--poll-ns") {
      a.poll_ns = std::atoi(next("--poll-ns").c_str());
    } else if (s == "--sync-each") {
      a.sync_each = true;
    } else if (s == "--json") {
      a.json_out = next("--json");
    } else if (s == "--help" || s == "-h") {
      std::printf(
          "Usage: %s [--shape per_task|single_packed] [--replays N] "
          "[--tasks N] [--poll-ns N] [--json path]\n", argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", s.c_str());
      std::exit(2);
    }
  }
  return a;
}

// ------------------------- Worker shared state ------------------------- //

struct Stats {
  // Per-slot last-seen monotonic seq.
  std::vector<std::atomic<unsigned int>> last_seen_seq;
  // Per-task observation count.
  std::vector<std::atomic<unsigned long long>> task_observed;
  // Per-task deterministic checksum of seq values seen.
  std::vector<std::atomic<unsigned long long>> task_checksum;
  // Anomaly counters.
  std::atomic<unsigned long long> stale_signals{0};
  std::atomic<unsigned long long> duplicate_signals{0};
  // For per_task: a gap > 1 just means the GPU advanced more
  // than once between two worker observations on the SAME slot
  // (same task). It is NOT a loss — task_observed correctly
  // adds `gap` per observation, so the total count of
  // replays-processed is preserved. Renamed from
  // `coalesced_advances` to make the semantic distinction
  // explicit. For single_packed shape, real drops would show
  // up as `total_observed < expected_observations` (visible in
  // the summary table), not as advances on the shared slot.
  std::atomic<unsigned long long> coalesced_advances{0};
  std::atomic<unsigned long long> invalid_task_ids{0};
  // Latency: observation_time - last_replay_launch_time, ns.
  std::vector<long long> latencies_ns;
  std::mutex latencies_mtx;
};

static inline long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ------------------------- Worker threads ------------------------- //

static void worker_per_task(volatile unsigned int* host_slots, int n_slots,
                            std::atomic<bool>* stop_worker, Stats* stats,
                            std::atomic<long long>* last_launch_ns,
                            int poll_ns) {
  while (!stop_worker->load(std::memory_order_acquire)) {
    bool any_advance = false;
    for (int t = 0; t < n_slots; ++t) {
      unsigned int cur = host_slots[t];
      unsigned int last = stats->last_seen_seq[t].load(std::memory_order_acquire);
      if (cur > last) {
        long long obs_ns = now_ns();
        long long lln = last_launch_ns->load(std::memory_order_acquire);
        if (lln > 0) {
          std::lock_guard<std::mutex> g(stats->latencies_mtx);
          stats->latencies_ns.push_back(obs_ns - lln);
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
        stats->last_seen_seq[t].store(cur, std::memory_order_release);
        any_advance = true;
      } else if (cur < last) {
        stats->stale_signals.fetch_add(1, std::memory_order_relaxed);
      }
    }
    if (!any_advance) {
      if (poll_ns == 0) {
        // Busy spin.
      } else if (poll_ns == 1) {
        std::this_thread::yield();
      } else {
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = poll_ns;
        nanosleep(&ts, nullptr);
      }
    }
  }
}

static void worker_single_packed(volatile unsigned int* host_slot, int n_tasks,
                                 std::atomic<bool>* stop_worker, Stats* stats,
                                 std::atomic<long long>* last_launch_ns,
                                 int poll_ns) {
  unsigned int last_seen = 0;  // packed (seq<<16)|task_id; 0 means unset
  while (!stop_worker->load(std::memory_order_acquire)) {
    unsigned int cur = *host_slot;
    if (cur != last_seen && cur != 0) {
      long long obs_ns = now_ns();
      long long lln = last_launch_ns->load(std::memory_order_acquire);
      if (lln > 0) {
        std::lock_guard<std::mutex> g(stats->latencies_mtx);
        stats->latencies_ns.push_back(obs_ns - lln);
      }
      unsigned int seq = cur >> 16;
      unsigned int task_id = cur & 0xFFFFu;
      if ((int)task_id >= n_tasks) {
        stats->invalid_task_ids.fetch_add(1, std::memory_order_relaxed);
      } else {
        // Deterministic checksum for this task; XOR in the seq.
        unsigned long long cs = stats->task_checksum[task_id].load();
        cs ^= ((unsigned long long)seq * (unsigned long long)(task_id + 1)) +
              (unsigned long long)seq;
        stats->task_checksum[task_id].store(cs);
        stats->task_observed[task_id].fetch_add(1);
      }
      // Detect stale/duplicate by comparing to last_seen's seq.
      unsigned int last_seq = last_seen >> 16;
      if (seq < last_seq) {
        stats->stale_signals.fetch_add(1, std::memory_order_relaxed);
      } else if (seq == last_seq) {
        stats->duplicate_signals.fetch_add(1, std::memory_order_relaxed);
      }
      last_seen = cur;
    }
    if (poll_ns == 0) {
      // Busy spin.
    } else if (poll_ns == 1) {
      std::this_thread::yield();
    } else {
      struct timespec ts;
      ts.tv_sec = 0;
      ts.tv_nsec = poll_ns;
      nanosleep(&ts, nullptr);
    }
  }
}

// ------------------------- Main ------------------------- //

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);

  CUDA_CHECK(cudaSetDevice(0));
  // Enable host-mapped allocation flag.
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  // ---- Allocate host-mapped pinned signal slots.
  int n_slots = (a.shape == Shape::PerTask) ? a.tasks : 1;
  unsigned int* host_slots = nullptr;
  CUDA_CHECK(cudaHostAlloc(&host_slots, sizeof(unsigned int) * n_slots,
                           cudaHostAllocMapped));
  std::memset(host_slots, 0, sizeof(unsigned int) * n_slots);
  unsigned int* dev_slots = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&dev_slots, host_slots, 0));

  // ---- Allocate per-task device counters (or single counter for packed).
  int n_counters = (a.shape == Shape::PerTask) ? a.tasks : 1;
  unsigned int* dev_counters = nullptr;
  CUDA_CHECK(cudaMalloc(&dev_counters, sizeof(unsigned int) * n_counters));
  CUDA_CHECK(cudaMemset(dev_counters, 0, sizeof(unsigned int) * n_counters));

  // ---- Stream + capture graph.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
  for (int t = 0; t < a.tasks; ++t) {
    if (a.shape == Shape::PerTask) {
      inc_per_task_kernel<<<1, 1, 0, stream>>>(&dev_counters[t], &dev_slots[t]);
    } else {
      inc_packed_kernel<<<1, 1, 0, stream>>>(&dev_counters[0], &dev_slots[0],
                                             (unsigned int)t);
    }
  }
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  // ---- Worker shared state.
  Stats stats;
  stats.last_seen_seq = std::vector<std::atomic<unsigned int>>(n_slots);
  stats.task_observed = std::vector<std::atomic<unsigned long long>>(a.tasks);
  stats.task_checksum = std::vector<std::atomic<unsigned long long>>(a.tasks);
  for (int i = 0; i < n_slots; ++i) stats.last_seen_seq[i].store(0);
  for (int i = 0; i < a.tasks; ++i) {
    stats.task_observed[i].store(0);
    stats.task_checksum[i].store(0);
  }
  stats.latencies_ns.reserve((size_t)a.replays * (size_t)a.tasks);

  std::atomic<bool> stop_worker{false};
  std::atomic<long long> last_launch_ns{0};

  std::thread worker;
  if (a.shape == Shape::PerTask) {
    worker = std::thread(worker_per_task, host_slots, n_slots, &stop_worker,
                         &stats, &last_launch_ns, a.poll_ns);
  } else {
    worker = std::thread(worker_single_packed, host_slots, a.tasks,
                         &stop_worker, &stats, &last_launch_ns, a.poll_ns);
  }

  // ---- Replay loop.
  long long t0 = now_ns();
  for (int r = 0; r < a.replays; ++r) {
    last_launch_ns.store(now_ns(), std::memory_order_release);
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    if (a.sync_each) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  if (!a.sync_each) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  long long t1 = now_ns();

  // Allow worker to drain any remaining advances; with busy-spin,
  // we wait a brief bounded time.
  long long drain_budget_ns = 50'000'000;  // 50 ms
  long long drain_start = now_ns();
  unsigned long long expected_total = (unsigned long long)a.replays * a.tasks;
  while (now_ns() - drain_start < drain_budget_ns) {
    unsigned long long total = 0;
    for (int t = 0; t < a.tasks; ++t) total += stats.task_observed[t].load();
    if (a.shape == Shape::PerTask) {
      if (total >= expected_total) break;
    } else {
      // Single packed: drops are expected, so drain when host_slot is stable.
      // Heuristic: if no advance in a few ms, stop draining.
      static long long last_total = -1;
      if ((long long)total == last_total) {
        // No progress for a while.
        break;
      }
      last_total = (long long)total;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  stop_worker.store(true, std::memory_order_release);
  worker.join();

  // ---- Per-task summary.
  unsigned long long total_observed = 0;
  unsigned long long min_per_task = ~0ull, max_per_task = 0;
  for (int t = 0; t < a.tasks; ++t) {
    unsigned long long obs = stats.task_observed[t].load();
    total_observed += obs;
    if (obs < min_per_task) min_per_task = obs;
    if (obs > max_per_task) max_per_task = obs;
  }

  // Latency stats.
  long long lat_p50 = 0, lat_p90 = 0, lat_max = 0;
  size_t n_lat = stats.latencies_ns.size();
  if (n_lat > 0) {
    std::sort(stats.latencies_ns.begin(), stats.latencies_ns.end());
    lat_p50 = stats.latencies_ns[n_lat / 2];
    lat_p90 = stats.latencies_ns[(n_lat * 90) / 100];
    lat_max = stats.latencies_ns.back();
  }

  // Checksum determinism is checked by re-running with same seed and
  // comparing — for now, dump the per-task checksums so runs can be
  // cross-checked outside.
  unsigned long long combined_checksum = 0;
  for (int t = 0; t < a.tasks; ++t) {
    combined_checksum ^= stats.task_checksum[t].load();
  }

  // ---- Output.
  std::printf("=== §1c.28 Step 1 value-signal smoke ===\n");
  std::printf("shape:                %s\n",
              a.shape == Shape::PerTask ? "per_task" : "single_packed");
  std::printf("replays:              %d\n", a.replays);
  std::printf("tasks:                %d\n", a.tasks);
  std::printf("poll_ns:              %d (0=busy, 1=yield, N=sleep N ns)\n",
              a.poll_ns);
  std::printf("expected_observations:%llu\n", expected_total);
  std::printf("total_observed:       %llu\n", total_observed);
  std::printf("per_task_min:         %llu\n", min_per_task);
  std::printf("per_task_max:         %llu\n", max_per_task);
  std::printf("stale_signals:        %llu\n", stats.stale_signals.load());
  std::printf("duplicate_signals:    %llu\n", stats.duplicate_signals.load());
  std::printf("coalesced_advances:      %llu\n", stats.coalesced_advances.load());
  std::printf("invalid_task_ids:     %llu\n", stats.invalid_task_ids.load());
  std::printf("replay_wall_ns:       %lld\n", t1 - t0);
  std::printf("replay_wall_ms:       %.3f\n", (t1 - t0) / 1e6);
  std::printf("signal_to_obs_p50_ns: %lld\n", lat_p50);
  std::printf("signal_to_obs_p90_ns: %lld\n", lat_p90);
  std::printf("signal_to_obs_max_ns: %lld\n", lat_max);
  std::printf("latency_samples:      %zu\n", n_lat);
  std::printf("combined_checksum:    0x%016llx\n", combined_checksum);

  // ---- Assertions (exit code reports fail count).
  // Per-task: correctness contract is "every replay's worth of work
  // is accounted for." Coalescing on a per-task slot is benign —
  // task_observed adds `gap` per observation, so total still equals
  // expected. No-loss invariant: total_observed == expected AND
  // last_seen[t] == replays AND no stale/invalid signals.
  // Single-packed: real drops manifest as total_observed < expected
  // (intermediate (seq, task_id) pairs overwritten on the shared
  // slot before the worker observed them).
  int fail = 0;
  if (a.shape == Shape::PerTask) {
    if (total_observed != expected_total) {
      std::fprintf(stderr,
                   "ASSERT FAIL [per_task]: total_observed=%llu != "
                   "expected=%llu (real drops or stuck worker)\n",
                   total_observed, expected_total);
      ++fail;
    }
    // last_seen must reach `replays` for every slot (worker
    // observed the final advance).
    for (int t = 0; t < n_slots; ++t) {
      unsigned int last = stats.last_seen_seq[t].load();
      if (last != (unsigned int)a.replays) {
        std::fprintf(stderr,
                     "ASSERT FAIL [per_task]: slot[%d].last_seen=%u "
                     "!= replays=%d (worker stuck or seq lost)\n",
                     t, last, a.replays);
        ++fail;
      }
    }
    if (stats.stale_signals.load() != 0) {
      std::fprintf(stderr, "ASSERT FAIL [per_task]: stale_signals=%llu\n",
                   stats.stale_signals.load());
      ++fail;
    }
    // coalesced_advances is REPORTED but not an assertion failure —
    // see the field's comment in Stats. Re-confirmed above by
    // total_observed == expected.
  } else {
    // single_packed: drops are expected unless polling matches GPU
    // rate. Strict assertions: no stale signals, no invalid task ids.
    if (stats.stale_signals.load() != 0) {
      std::fprintf(stderr,
                   "ASSERT FAIL [single_packed]: stale_signals=%llu\n",
                   stats.stale_signals.load());
      ++fail;
    }
    if (stats.invalid_task_ids.load() != 0) {
      std::fprintf(stderr,
                   "ASSERT FAIL [single_packed]: invalid_task_ids=%llu\n",
                   stats.invalid_task_ids.load());
      ++fail;
    }
    // Soft expectation: drops (total_observed < expected) occur. We
    // just report and let the report file recommend per_task.
  }

  if (!a.json_out.empty()) {
    FILE* f = std::fopen(a.json_out.c_str(), "w");
    if (f) {
      std::fprintf(
          f,
          "{\n"
          "  \"shape\":\"%s\",\n"
          "  \"replays\":%d,\n  \"tasks\":%d,\n  \"poll_ns\":%d,\n"
          "  \"expected_observations\":%llu,\n"
          "  \"total_observed\":%llu,\n"
          "  \"per_task_min\":%llu,\n  \"per_task_max\":%llu,\n"
          "  \"stale_signals\":%llu,\n"
          "  \"duplicate_signals\":%llu,\n"
          "  \"coalesced_advances\":%llu,\n"
          "  \"invalid_task_ids\":%llu,\n"
          "  \"replay_wall_ns\":%lld,\n"
          "  \"signal_to_obs_p50_ns\":%lld,\n"
          "  \"signal_to_obs_p90_ns\":%lld,\n"
          "  \"signal_to_obs_max_ns\":%lld,\n"
          "  \"latency_samples\":%zu,\n"
          "  \"combined_checksum\":\"0x%016llx\",\n"
          "  \"assertion_failures\":%d\n"
          "}\n",
          a.shape == Shape::PerTask ? "per_task" : "single_packed", a.replays,
          a.tasks, a.poll_ns, expected_total, total_observed, min_per_task,
          max_per_task, stats.stale_signals.load(),
          stats.duplicate_signals.load(), stats.coalesced_advances.load(),
          stats.invalid_task_ids.load(), t1 - t0, lat_p50, lat_p90, lat_max,
          n_lat, combined_checksum, fail);
      std::fclose(f);
    }
  }

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dev_counters));
  CUDA_CHECK(cudaFreeHost(host_slots));

  return fail ? 1 : 0;
}
