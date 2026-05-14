from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parents[2] / "Benchmarks" / "phase1_analysis"
sys.path.insert(0, str(BENCH_DIR))

from bench_cots_free_regime import Arm as FreeArm  # noqa: E402
from bench_cots_free_regime import should_run_cell  # noqa: E402
from bench_cots_vs_native_prefetch import (  # noqa: E402
    DEPTH_PAIRS,
    build_arms,
    build_vllm_command,
)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        input_len=8,
        output_len=128,
        max_model_len=2048,
        gpu_memory_utilization=0.75,
        num_iters_warmup=2,
        num_iters=3,
        extra_vllm_args=[],
    )


def test_prefetch_comparison_command_modes_and_flags(tmp_path: Path) -> None:
    arms = build_arms(
        depth_pairs=[DEPTH_PAIRS[0]],
        native_backends=["prefetch_defer"],
        native_k_values=[1],
    )
    cots = next(arm for arm in arms if arm.name == "cots_01L")
    native = next(arm for arm in arms if arm.name == "native_prefetch_defer_k1_01L")
    args = _args(tmp_path)

    graph_cmd = build_vllm_command(
        mode="graph",
        arm=cots,
        batch=1,
        out_json=tmp_path / "graph.json",
        args=args,
    )
    eager_cmd = build_vllm_command(
        mode="eager",
        arm=native,
        batch=1,
        out_json=tmp_path / "eager.json",
        args=args,
    )

    assert "--enforce-eager" not in graph_cmd
    assert "--enforce-eager" in eager_cmd
    assert "--cots-f-cpu-store" in graph_cmd
    assert graph_cmd[graph_cmd.index("--cots-f-prefetch") + 1] == "0.035714"
    assert "--offload-backend" in eager_cmd
    assert eager_cmd[eager_cmd.index("--offload-backend") + 1] == "prefetch_defer"
    assert eager_cmd[eager_cmd.index("--offload-prefetch-step") + 1] == "1"


def test_free_regime_focused_grid_selects_expected_strategy_batches() -> None:
    args = Namespace(
        focused_grid=True,
        batch_sizes=[1, 64],
        focused_cpu_batches=[1],
        focused_prefetch_batches=[64],
        focused_collab_batches=[1],
    )

    assert should_run_cell(FreeArm("none", "none", 0.0, 0.0, ()), 64, args)
    assert should_run_cell(
        FreeArm("cpu", "cots_cpu_only", 0.01, 0.0, ()), 1, args
    )
    assert not should_run_cell(
        FreeArm("cpu", "cots_cpu_only", 0.01, 0.0, ()), 64, args
    )
    assert should_run_cell(
        FreeArm("pref", "cots_prefetch_only", 0.01, 0.01, ()), 64, args
    )
    assert not should_run_cell(
        FreeArm("pref", "cots_prefetch_only", 0.01, 0.01, ()), 1, args
    )
