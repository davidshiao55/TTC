from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parents[2] / "Benchmarks" / "phase1_analysis"
sys.path.insert(0, str(BENCH_DIR))

from bench_cots_kv_throughput import (  # noqa: E402
    Arm,
    Workload,
    build_arms,
    build_vllm_command,
    parse_log_text,
    should_run_cell,
)


def test_parse_log_text_extracts_kv_and_throughput_metrics() -> None:
    text = """
    INFO 05-13 10:00:00 [kv_cache_utils.py:1319] GPU KV cache size: 142,336 tokens
    INFO 05-13 10:00:00 [kv_cache_utils.py:1321] Maximum concurrency for 2,048 tokens per request: 69.50x
    Throughput: 3.25 requests/s, 4210.75 total tokens/s, 3980.50 output tokens/s
    Total num prompt tokens:  16,384
    Total num output tokens:  262,144
    """

    metrics = parse_log_text(text)

    assert metrics["kv_cache_tokens"] == 142336
    assert metrics["max_concurrency_request_tokens"] == 2048
    assert metrics["max_concurrency"] == 69.50
    assert metrics["requests_per_s_log"] == 3.25
    assert metrics["total_tokens_per_s_log"] == 4210.75
    assert metrics["output_tokens_per_s_log"] == 3980.50
    assert metrics["total_prompt_tokens_log"] == 16384
    assert metrics["total_output_tokens_log"] == 262144


def test_parse_log_text_tolerates_missing_metrics() -> None:
    metrics = parse_log_text("benchmark failed before engine init")

    assert metrics["kv_cache_tokens"] is None
    assert metrics["max_concurrency"] is None
    assert metrics["output_tokens_per_s_log"] is None


def test_build_vllm_command_uses_random_dataset_lengths(tmp_path: Path) -> None:
    args = Namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="bfloat16",
        num_prompts=512,
        gpu_memory_utilization=0.75,
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        max_model_len_slack=1,
        extra_vllm_args=[],
    )
    workload = Workload(name="short", input_len=8, output_len=128)
    arm = Arm(
        name="none",
        strategy="none",
        f_cpu_store=0.0,
        f_prefetch=0.0,
        f_prefetch_ratio=None,
        flags=(),
    )

    cmd = build_vllm_command(
        workload=workload,
        arm=arm,
        out_json=tmp_path / "out.json",
        args=args,
    )

    assert "--random-input-len" in cmd
    assert cmd[cmd.index("--random-input-len") + 1] == "8"
    assert "--random-output-len" in cmd
    assert cmd[cmd.index("--random-output-len") + 1] == "128"
    assert "--random-prefix-len" in cmd
    assert "--input-len" not in cmd
    assert "--output-len" not in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "137"


def test_focused_grid_keeps_only_plausible_kv_cells() -> None:
    args = Namespace(
        focused_grid=True,
        focused_short_prefetch_f_values=[0.02],
        focused_prefetch_f_values=[0.02, 0.05, 0.09],
        focused_collab_f_values=[0.09],
        focused_collab_ratios=[0.9],
    )
    short = Workload(name="short", input_len=8, output_len=128)
    medium = Workload(name="medium", input_len=32, output_len=512)
    long = Workload(name="long", input_len=32, output_len=1024)
    arms = {
        arm.name: arm
        for arm in build_arms(
            prefetch_f_values=[0.02, 0.05, 0.09],
            collab_f_values=[0.09],
            collab_ratios=[0.9],
        )
    }

    assert should_run_cell(short, arms["none"], args)
    assert should_run_cell(short, arms["cots_prefetch_only_f0p02"], args)
    assert not should_run_cell(short, arms["cots_prefetch_only_f0p05"], args)
    assert should_run_cell(medium, arms["cots_prefetch_only_f0p09"], args)
    assert not should_run_cell(medium, arms["cots_collab_f0p09_r0p9"], args)
    assert should_run_cell(long, arms["cots_collab_f0p09_r0p9"], args)
