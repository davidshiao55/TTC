# SPDX-License-Identifier: Apache-2.0
"""§1c.29 commit 3 — wait-kernel sync A/B benchmark.

Compares the captured COTS path with `cots_capture_sync_mode="wait_kernel"` (wait-kernel sync
on, replaces the captured `cudaLaunchHostFunc(sync_cb)` node with a
`cots_wait_done_kernel<<<>>>` reading a host-mapped done_slot) vs
`cots_capture_sync_mode="host_callback"` (legacy sync_cb host_fn that blocks the
CUDA driver thread on `TaskQueue::sync(0)`). Submit-side host_fn
(dispatch_cb) is unchanged in both arms.

Upper bound recoverable per the §1c.27 split host_fn ablation:
~273 ms cgl / ~126 ms wall per generate at output_len=128, B=1,
t=16, f=0.05 (Qwen2.5-7B real model). This synthetic stub bench can
only measure the SHAPE of the win — micro-budget per replay — not
the generate-equivalent absolute, which lives in
bench_dryrun_vs_native_qwen.py (Stage 6).

Four arms × {dryrun, real}:
  * `dryrun_m3_off` — baseline. Captured graph replays sync_cb
    host_fn that blocks the driver thread.
  * `dryrun_m3_on`  — wait-kernel sync replaces sync_cb. Worker
    publishes done_slot=seq immediately (dryrun has nothing to do)
    so the wait kernel resumes immediately.
  * `real_m3_off`   — baseline with real CPU GEMM. The host_fn
    waits for the worker to finish at::linear.
  * `real_m3_on`    — wait-kernel sync with real CPU GEMM. Worker publishes
    done_slot=seq AFTER at::linear completes; if the GPU hits the
    wait kernel before the worker finishes, it spins on the
    host-mapped slot.

The dryrun A/B isolates the SUBSTRATE cost: without real CPU work,
the only difference between arms is sync_cb-host_fn vs
cots_wait_done_kernel. The real A/B reflects the production workload — if
wait-kernel sync helps in dryrun but loses in real, the spin cost dominates and
the whole approach should be revisited.

Diag mode: when run with VLLM_COTS_DIAG=1, the diag wait kernel
captures `wait_kernel_immediate_resume_count`, `wait_kernel_lagging_wait_count`, and
`wait_kernel_spin_iters_total`. The bench prints these as the canary
the §1c.29 design called out — high lagging count means CPU GEMM
exceeds the GPU window (wait-kernel sync has nothing to recover).

Run:
    /opt/conda/envs/thesis/bin/python bench_cots_wait_done_kernel_ab.py
    VLLM_COTS_DIAG=1 /opt/conda/envs/thesis/bin/python bench_cots_wait_done_kernel_ab.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from vllm.config import (
    CompilationConfig,
    CotsOffloadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.offloader import CotsOffloader, set_offloader
from vllm.model_executor.offloader import cots_ops as _cots_ops

HIDDEN = 256
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS


class _QkvLayer(nn.Module):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.qkv_proj = QKVParallelLinear(
            hidden_size=HIDDEN,
            head_size=HEAD_DIM,
            total_num_heads=NUM_HEADS,
            total_num_kv_heads=NUM_KV_HEADS,
            bias=False,
            disable_tp=True,
            params_dtype=torch.bfloat16,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_full, _ = self.qkv_proj(x)
        return x_full[:, :HIDDEN]


class _NLayerStub(nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_QkvLayer(prefix=f"layer{i}.qkv_proj") for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _make_vllm_config(*, max_num_tokens: int) -> VllmConfig:
    mc = ModelConfig.__new__(ModelConfig)
    object.__setattr__(mc, "enforce_eager", False)
    sc = SchedulerConfig.__new__(SchedulerConfig)
    object.__setattr__(sc, "max_num_batched_tokens", max_num_tokens)
    cc = CompilationConfig.__new__(CompilationConfig)
    object.__setattr__(cc, "cudagraph_capture_sizes", [max_num_tokens])
    object.__setattr__(cc, "custom_ops", ["none"])
    object.__setattr__(cc, "enabled_custom_ops", Counter())
    object.__setattr__(cc, "disabled_custom_ops", Counter())
    pc = ParallelConfig.__new__(ParallelConfig)
    object.__setattr__(pc, "tensor_parallel_size", 1)
    vc = VllmConfig.__new__(VllmConfig)
    object.__setattr__(vc, "model_config", mc)
    object.__setattr__(vc, "scheduler_config", sc)
    object.__setattr__(vc, "compilation_config", cc)
    object.__setattr__(vc, "parallel_config", pc)
    return vc


def _build_offloaded_stub(
    *,
    n_layers: int,
    max_num_tokens: int,
    f_cpu_store: float,
    dry_run: bool,
    m3: bool,
) -> tuple[_NLayerStub, CotsOffloader]:
    vc = _make_vllm_config(max_num_tokens=max_num_tokens)
    with set_current_vllm_config(vc):
        model = _NLayerStub(n_layers=n_layers).cuda()
        offloader = CotsOffloader(
            config=CotsOffloadConfig(
                f_cpu_store=f_cpu_store,
                cpu_runner="native",
                kv_biased=True,
                dry_run=dry_run,
                cots_capture_sync_mode=("wait_kernel" if m3 else "host_callback"),
            )
        )
        set_offloader(offloader)
        offloader.wrap_modules(iter(model.layers))
        torch.manual_seed(0)
        q_size = NUM_HEADS * HEAD_DIM
        kv_size = NUM_KV_HEADS * HEAD_DIM
        for layer in model.layers:
            q = torch.randn(q_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(kv_size, HIDDEN, dtype=torch.bfloat16, device="cuda")
            layer.qkv_proj.weight_loader(layer.qkv_proj.weight, q, "q")
            layer.qkv_proj.weight_loader(layer.qkv_proj.weight, k, "k")
            layer.qkv_proj.weight_loader(layer.qkv_proj.weight, v, "v")
        offloader.post_init()
    return model, offloader


def _bench_captured(
    *,
    model: _NLayerStub,
    offloader: CotsOffloader,
    x: torch.Tensor,
    n_iters: int,
    warmup: int,
) -> float:
    """Capture once, replay N times. Median wall-clock per replay (μs)."""
    for _ in range(warmup):
        offloader.prepare_before_forward(int(x.shape[0]))
        offloader.sync_prev_onload()
        _ = model(x)
        torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    offloader.prepare_before_forward(int(x.shape[0]))
    offloader.sync_prev_onload()
    with torch.cuda.graph(g):
        _ = model(x)
        offloader.join_after_forward()

    g.replay()
    torch.cuda.synchronize()  # discard first-replay one-shot

    samples: list[float] = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        offloader.prepare_before_forward(int(x.shape[0]))
        offloader.sync_prev_onload()
        g.replay()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    return statistics.median(samples)


def _read_diag_counters(offloader: CotsOffloader) -> dict[str, int]:
    """Snapshot the CotsCpuInfer counters via the cots_ops registry.
    Empty dict when wait-kernel sync is off or DIAG=0 (no counters increment)."""
    if offloader._runner is None:
        return {}
    rid = getattr(offloader._runner, "_runner_id", None)
    if rid is None:
        return {}
    try:
        infer = _cots_ops._lookup_infer(rid, "bench_m3_diag_read")
    except RuntimeError:
        return {}
    return dict(infer.get_counters())


def _init_distributed_once() -> None:
    import os

    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29503")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0,
            distributed_init_method="env://", backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)


def _run_arm(
    *, label: str, n_layers: int, num_tokens: int, f_cpu_store: float,
    dry_run: bool, m3: bool, n_iters: int, warmup: int,
    x: torch.Tensor,
) -> dict:
    """Build the offloader, run the captured replay bench, snapshot
    counters, tear down. Returns a dict for the JSON report."""
    model, off = _build_offloaded_stub(
        n_layers=n_layers,
        max_num_tokens=num_tokens,
        f_cpu_store=f_cpu_store,
        dry_run=dry_run,
        m3=m3,
    )
    # Reset counters AFTER capture-time activity so we measure replay
    # only. We do this by zeroing right before the timed loop inside
    # the bench helper — but simplest is reset post-capture-warmup
    # then sample after the timed run. For now we sample
    # cumulatively (capture + replays); the immediate-vs-lagging
    # ratio is insensitive to the constant capture-time fires.
    try:
        t_us = _bench_captured(
            model=model, offloader=off, x=x,
            n_iters=n_iters, warmup=warmup,
        )
        counters = _read_diag_counters(off)
    finally:
        if off._runner is not None:
            off._runner.close()
    return {
        "label": label,
        "dry_run": dry_run,
        "m3": m3,
        "t_us": t_us,
        "diag_counters": {
            k: v for k, v in counters.items()
            if k.startswith("m3_") or k in (
                "dispatch_cb_count",
                "sync_cb_count",
                "sync_cb_wait_total_ns",
                "worker_run_count",
                "worker_busy_total_ns",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--f-cpu-store", type=float, default=0.10)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — skipping wait-kernel sync A/B bench", file=sys.stderr)
        return 0

    _init_distributed_once()
    x = torch.randn(args.num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")

    arms = []
    for dry_run in (True, False):
        for m3 in (False, True):
            label = (
                f"{'dryrun' if dry_run else 'real'}_m3_"
                f"{'on' if m3 else 'off'}"
            )
            print(f"  running {label} ...", flush=True)
            arms.append(_run_arm(
                label=label,
                n_layers=args.n_layers,
                num_tokens=args.num_tokens,
                f_cpu_store=args.f_cpu_store,
                dry_run=dry_run,
                m3=m3,
                n_iters=args.n_iters,
                warmup=args.warmup,
                x=x,
            ))

    by_label = {a["label"]: a for a in arms}
    dryrun_off = by_label["dryrun_m3_off"]["t_us"]
    dryrun_on = by_label["dryrun_m3_on"]["t_us"]
    real_off = by_label["real_m3_off"]["t_us"]
    real_on = by_label["real_m3_on"]["t_us"]

    dryrun_delta = dryrun_off - dryrun_on  # positive = wait-kernel sync wins
    real_delta = real_off - real_on
    dryrun_per_layer = dryrun_delta / args.n_layers
    real_per_layer = real_delta / args.n_layers

    print()
    print("=" * 76)
    print("§1c.29 — wait-kernel sync A/B (synthetic stub)")
    print("=" * 76)
    print(f"  workload: n_layers={args.n_layers}, num_tokens={args.num_tokens}")
    print(f"  f_cpu_store={args.f_cpu_store}, n_iters={args.n_iters}")
    print()
    print(f"  dryrun, wait-kernel sync off: {dryrun_off:>9.1f} μs / forward")
    print(f"  dryrun, wait-kernel sync on:  {dryrun_on:>9.1f} μs / forward")
    print(f"  Δ (off - on):   {dryrun_delta:>+9.1f} μs   "
          f"({dryrun_per_layer:+.2f} μs/layer)")
    print()
    print(f"  real,   wait-kernel sync off: {real_off:>9.1f} μs / forward")
    print(f"  real,   wait-kernel sync on:  {real_on:>9.1f} μs / forward")
    print(f"  Δ (off - on):   {real_delta:>+9.1f} μs   "
          f"({real_per_layer:+.2f} μs/layer)")
    print()
    # Diag-mode counter dump (only meaningful with VLLM_COTS_DIAG=1).
    for arm in arms:
        if arm["m3"] and arm["diag_counters"]:
            imm = arm["diag_counters"].get("wait_kernel_immediate_resume_count", 0)
            lag = arm["diag_counters"].get("wait_kernel_lagging_wait_count", 0)
            spin = arm["diag_counters"].get("wait_kernel_spin_iters_total", 0)
            tot = imm + lag
            ratio = (lag / tot) if tot > 0 else 0.0
            print(f"  {arm['label']:>15} diag — immediate={imm}, lagging={lag} "
                  f"({ratio*100:.1f}%), spin_iters={spin}")
    print()
    if dryrun_delta > 0:
        print(f"  wait-kernel sync substrate gain (dryrun):  {dryrun_per_layer:+.2f} μs/layer "
              f"— sync_cb host_fn was costlier than the wait kernel.")
    else:
        print(f"  wait-kernel sync substrate cost (dryrun):  {-dryrun_per_layer:+.2f} μs/layer "
              f"— wait kernel overhead exceeds sync_cb's host_fn cost. "
              f"Consider reverting if real also loses.")
    print("=" * 76)
    print()
    print("  NB: this is a SHAPE check. The per-layer μs delta here does NOT")
    print("  translate directly to the §1c.27 upper bound (~273 ms cgl / 126")
    print("  ms wall per Qwen2.5-7B generate). Real-model anchor lives in")
    print("  bench_dryrun_vs_native_qwen.py and FastTTS bench harnesses.")

    import os as _os
    diag_on = _os.environ.get("VLLM_COTS_DIAG", "0") == "1"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_diag" if diag_on else ""
    out_path = args.results_dir / f"bench_cots_wait_done_kernel_ab{suffix}.json"
    out_path.write_text(json.dumps({
        "args": vars(args) | {"results_dir": str(args.results_dir)},
        "arms": arms,
        "deltas": {
            "dryrun_off_minus_on_us": dryrun_delta,
            "dryrun_per_layer_us": dryrun_per_layer,
            "real_off_minus_on_us": real_delta,
            "real_per_layer_us": real_per_layer,
        },
    }, indent=2, default=str))
    print(f"\n  results written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
