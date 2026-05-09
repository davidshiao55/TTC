# SPDX-License-Identifier: Apache-2.0
"""Stage 5 — Synthetic orch-collapse sanity check (§1.14 absolute
locked at Stage 6 on the real model).

The §1.14 finding measured `orch = T(dry_run=True) - T(no offload)`
≈ 0.45 s/generate on Phase 1a's Python runner under eager mode. That
gap is the per-operator Python orchestration: `executor.submit`,
`future.result()`, the operator's per-call view construction, the UVA
kernel launch — all of which run on the main Python thread between
GPU operations.

Stage 5's named target is **`orch ≤ 0.05 s/generate`** on Qwen2.5-7B
under `cpu_runner='native', enforce_eager=False`. THAT
absolute-budget gate is reserved for Stage 6 — it requires the real
model + the FastTTS workload, not a synthetic stub.

What THIS bench gates: the SHAPE of the collapse. Graph capture must
materially reduce the per-forward orch overhead vs eager on the same
workload. If capture didn't help, the §1.14 0.05 s/generate target
would be unreachable on Qwen2.5-7B too. The bench therefore asserts
ONE thing only:

  collapse_ratio = orch_capture / orch_eager < 1.0  (capture wins)

with a tighter informational threshold (collapse_ratio < 0.7) so a
near-no-op capture wouldn't sneak through. Absolute per-forward
microsecond numbers ARE printed but NOT gated — they're synthetic
and don't translate to the real model's generate-equivalent budget.

Workload: synthetic multi-layer QKV stub.
  * (a) eager dry_run=True — Python operator bodies run, host
        callbacks fire, no real GEMM.
  * (b) graph-capture dry_run=True — graph replay re-issues host
        callbacks; Python operator bodies are NOT traversed.
  * (c) eager no-offload — baseline.

Real-model anchor (Qwen2.5-7B + FastTTS) lives at
`bench_dryrun_vs_native_qwen.py`. Stage 6 landed that harness;
locking the §1.14 absolute there is pending the resolution of the
pre-hook × torch.compile fullgraph blocker documented in
`David/Docs/phase1c_findings.md §1c.18`.

Run:
    /opt/conda/envs/thesis/bin/python bench_dryrun_vs_real_native.py
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

HIDDEN = 256
NUM_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS


class _QkvLayer(nn.Module):
    """One offloadable layer = a wrapper Module containing a
    QKVParallelLinear. CotsOffloader's `_build_handles` walks each
    layer module and looks for `qkv_proj` / `gate_up_proj` / `down_proj`
    children — the bench mirrors phase1b/test_three_way_scatter.py's
    fabric so wrap_modules actually picks up the offload handle."""

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
        # Reduce back to HIDDEN for stacking; the actual transformation
        # is incidental — the orch we're measuring is per-call.
        return x_full[:, :HIDDEN]


class _NLayerStub(nn.Module):
    """N stacked offloadable `_QkvLayer` modules. Models the per-layer
    orch overhead that scales linearly with `n_layers` — Phase 1a §1.14
    measured ~28 layers (Qwen2.5-7B); we use a smaller stack here and
    compute per-layer orch."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_QkvLayer(prefix=f"layer{i}.qkv_proj") for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _make_vllm_config(*, max_num_tokens: int, enforce_eager: bool) -> VllmConfig:
    mc = ModelConfig.__new__(ModelConfig)
    object.__setattr__(mc, "enforce_eager", enforce_eager)
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
    enforce_eager: bool,
    f_cpu_store: float,
    dry_run: bool,
) -> tuple[_NLayerStub, CotsOffloader | None]:
    vc = _make_vllm_config(
        max_num_tokens=max_num_tokens, enforce_eager=enforce_eager
    )
    with set_current_vllm_config(vc):
        model = _NLayerStub(n_layers=n_layers).cuda()
        if f_cpu_store > 0:
            offloader = CotsOffloader(
                config=CotsOffloadConfig(
                    f_cpu_store=f_cpu_store,
                    cpu_runner="native",
                    kv_biased=True,
                    dry_run=dry_run,
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
        else:
            offloader = None
    return model, offloader


def _bench_eager(
    *,
    model: _NLayerStub,
    offloader: CotsOffloader | None,
    x: torch.Tensor,
    n_iters: int,
    warmup: int,
) -> float:
    """Median wall-clock per forward (μs)."""
    for _ in range(warmup):
        if offloader is not None:
            offloader.prepare_before_forward(int(x.shape[0]))
            offloader.sync_prev_onload()
        _ = model(x)
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        if offloader is not None:
            offloader.prepare_before_forward(int(x.shape[0]))
            offloader.sync_prev_onload()
        _ = model(x)
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    return statistics.median(samples)


def _bench_captured(
    *,
    model: _NLayerStub,
    offloader: CotsOffloader,
    x: torch.Tensor,
    n_iters: int,
    warmup: int,
) -> float:
    """Capture once, replay N times. Median wall-clock per replay (μs)."""
    # Pre-capture warmup.
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

    # Discard first replay (one-shot first-replay overhead).
    g.replay()
    torch.cuda.synchronize()

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


def _init_distributed_once() -> None:
    """Bring up a single-rank TP group (gloo) so vLLM's
    QKVParallelLinear constructor finds the tensor parallel group at
    init. Idempotent — no-op if already initialized."""
    import os

    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")  # distinct from test ports
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0,
            distributed_init_method="env://", backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)


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
    parser.add_argument(
        "--collapse-ratio-threshold",
        type=float,
        default=0.7,
        help=(
            "Stage 5 sanity threshold: orch_capture / orch_eager must be "
            "<= this value (default 0.7 — capture must reduce orch by at "
            "least 30%). The §1.14 absolute generate-equivalent budget is "
            "Stage 6 work on the real Qwen2.5-7B; this bench only proves "
            "the SHAPE of the collapse on a synthetic stub."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — skipping headline bench", file=sys.stderr)
        return 0

    _init_distributed_once()
    x = torch.randn(args.num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")

    # (c) Baseline: no offload, eager.
    model_none, _ = _build_offloaded_stub(
        n_layers=args.n_layers,
        max_num_tokens=args.num_tokens,
        enforce_eager=True,
        f_cpu_store=0.0,
        dry_run=False,
    )
    t_none_us = _bench_eager(
        model=model_none, offloader=None, x=x,
        n_iters=args.n_iters, warmup=args.warmup,
    )

    # (a) Eager + dry_run=True — substrate-included, no real GEMM.
    model_eager, off_eager = _build_offloaded_stub(
        n_layers=args.n_layers,
        max_num_tokens=args.num_tokens,
        enforce_eager=True,
        f_cpu_store=args.f_cpu_store,
        dry_run=True,
    )
    assert off_eager is not None
    try:
        t_eager_us = _bench_eager(
            model=model_eager, offloader=off_eager, x=x,
            n_iters=args.n_iters, warmup=args.warmup,
        )
    finally:
        off_eager._runner.close() if off_eager._runner else None

    # (b) Captured + dry_run=True — graph replay re-issues host callbacks
    #     without Python operator-body traversal.
    model_cap, off_cap = _build_offloaded_stub(
        n_layers=args.n_layers,
        max_num_tokens=args.num_tokens,
        enforce_eager=False,
        f_cpu_store=args.f_cpu_store,
        dry_run=True,
    )
    assert off_cap is not None
    try:
        t_capture_us = _bench_captured(
            model=model_cap, offloader=off_cap, x=x,
            n_iters=args.n_iters, warmup=args.warmup,
        )
    finally:
        off_cap._runner.close() if off_cap._runner else None

    orch_eager_us = t_eager_us - t_none_us
    orch_capture_us = t_capture_us - t_none_us
    collapse_ratio = (
        orch_capture_us / orch_eager_us if orch_eager_us > 0 else float("inf")
    )
    orch_capture_per_layer_us = orch_capture_us / args.n_layers

    print()
    print("=" * 76)
    print("Stage 5 — Synthetic orch-collapse sanity check")
    print("=" * 76)
    print(f"  workload: n_layers={args.n_layers}, num_tokens={args.num_tokens}")
    print(f"  f_cpu_store={args.f_cpu_store}, n_iters={args.n_iters}")
    print()
    print(f"  (c) baseline (no offload, eager):       {t_none_us:>9.1f} μs / forward")
    print(f"  (a) eager + dry_run=True:               {t_eager_us:>9.1f} μs / forward")
    print(f"  (b) captured + dry_run=True:            {t_capture_us:>9.1f} μs / forward")
    print()
    print(f"  orch_eager   = (a) - (c):               {orch_eager_us:>9.1f} μs (synthetic)")
    print(f"  orch_capture = (b) - (c):               {orch_capture_us:>9.1f} μs (synthetic)")
    print(f"  collapse ratio (capture / eager):       {collapse_ratio:>9.3f}")
    print(f"  orch_capture / n_layers:                {orch_capture_per_layer_us:>9.1f} μs/layer (informational)")
    print()
    verdict_collapse = (
        "PASS"
        if collapse_ratio <= args.collapse_ratio_threshold
        else f"FAIL (collapse_ratio {collapse_ratio:.3f} > "
        f"threshold {args.collapse_ratio_threshold:.3f})"
    )
    print(
        f"  collapse_ratio <= {args.collapse_ratio_threshold:.2f} "
        f"(capture must reduce orch):  {verdict_collapse}"
    )
    print("=" * 76)
    print()
    print("  NB: This is a SHAPE check on a synthetic multi-layer stub. The")
    print("  per-layer μs / per-forward μs absolutes here do NOT translate to")
    print("  Qwen2.5-7B's per-generate budget — HIDDEN=256 here vs 3584 on")
    print("  Qwen2.5-7B, smaller layer count, and no attention/MLP between")
    print("  QKV calls. Real-model anchor lives in bench_dryrun_vs_native_qwen.py")
    print("  (harness landed at Stage 6; absolute pending §1c.18 resolution).")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "bench_dryrun_vs_real_native.json"
    out_path.write_text(
        json.dumps(
            {
                "args": vars(args) | {"results_dir": str(args.results_dir)},
                "t_none_us": t_none_us,
                "t_eager_us": t_eager_us,
                "t_capture_us": t_capture_us,
                "orch_eager_us": orch_eager_us,
                "orch_capture_us": orch_capture_us,
                "collapse_ratio": collapse_ratio,
                "orch_capture_per_layer_us": orch_capture_per_layer_us,
                "collapse_ratio_threshold": args.collapse_ratio_threshold,
                "verdict_collapse": verdict_collapse,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\n  results written to {out_path}")

    return 0 if verdict_collapse == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
