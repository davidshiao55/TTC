import sys
from pathlib import Path

import pytest


TTC_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TTC_ROOT / "FastTTS-thesis"))
sys.path.insert(0, str(TTC_ROOT / "FastTTS-thesis" / "benchmarks"))

from config import FastTTSConfig, SearchConfig  # noqa: E402
from benchmark_config import build_benchmark_config_from_yaml  # noqa: E402
from planner import (  # noqa: E402
    DispatchProblem,
    ManualTTCPlanner,
    apply_ttc_plan_to_config,
    derive_weight_thread_policy,
    solve_per_bucket_dispatch,
)


class SyntheticDispatchProfile:
    """Tiny profile fixture for exercising the dispatch performance model."""

    def __init__(
        self,
        *,
        gpu_full_ms_by_bucket,
        cpu_full_ms_by_bucket,
        h2d_ms_per_byte,
        attention_gpu_ms_by_bucket=None,
        attention_cpu_ms_by_bucket=None,
        overhead_ms=0.0,
    ):
        self.gpu_full_ms_by_bucket = gpu_full_ms_by_bucket
        self.cpu_full_ms_by_bucket = cpu_full_ms_by_bucket
        self.h2d_ms_per_byte = h2d_ms_per_byte
        self.attention_gpu_ms_by_bucket = attention_gpu_ms_by_bucket or {}
        self.attention_cpu_ms_by_bucket = attention_cpu_ms_by_bucket or {}
        self._overhead_ms = overhead_ms

    def gpu_op_ms(self, op, bucket, gpu_fraction):
        if op == "attention":
            return self.attention_gpu_ms_by_bucket.get(bucket, 0.0) * gpu_fraction
        return self.gpu_full_ms_by_bucket[bucket] * gpu_fraction

    def cpu_op_ms(self, op, bucket, cpu_fraction):
        if op == "attention":
            return self.attention_cpu_ms_by_bucket.get(bucket, 0.0) * cpu_fraction
        return self.cpu_full_ms_by_bucket[bucket] * cpu_fraction

    def h2d_ms(self, transfer_bytes):
        return transfer_bytes * self.h2d_ms_per_byte

    def overhead_ms(self, bucket, f_cpu, f_prefetch):
        return self._overhead_ms


def _two_weight_op_problem(*, buckets=(1, 64), f_cpu_store=0.2):
    return DispatchProblem(
        buckets=buckets,
        f_cpu_store=f_cpu_store,
        num_layers=1,
        layer_ops=("qkv", "mlp1"),
        weight_bytes_per_layer={"qkv": 1000, "mlp1": 1000},
        candidate_f_cpu=(0.0, 0.1, 0.2),
    )


def test_dispatch_solver_chooses_cpu_for_small_and_prefetch_for_large_bucket():
    problem = _two_weight_op_problem()
    profile = SyntheticDispatchProfile(
        gpu_full_ms_by_bucket={1: 10.0, 64: 20.0},
        cpu_full_ms_by_bucket={1: 20.0, 64: 300.0},
        h2d_ms_per_byte=0.02,
    )

    result = solve_per_bucket_dispatch(problem, profile)

    assert result.dispatch_table == {
        1: (0.2, 0.0),
        64: (0.0, 0.2),
    }
    assert result.entries[1].predicted_ms == 16.0
    assert result.entries[64].predicted_ms == 40.0
    assert result.entries[1].bottleneck == "compute"
    assert result.entries[64].bottleneck == "compute"
    assert len(result.candidate_scores[1]) == 3
    assert derive_weight_thread_policy(result.dispatch_table) == {
        1: 16,
        64: 4,
    }


def test_dispatch_solver_slow_pcie_moves_solution_toward_cpu_compute():
    problem = _two_weight_op_problem(buckets=(64,))
    profile = SyntheticDispatchProfile(
        gpu_full_ms_by_bucket={64: 20.0},
        cpu_full_ms_by_bucket={64: 300.0},
        h2d_ms_per_byte=0.20,
    )

    result = solve_per_bucket_dispatch(problem, profile)
    entry = result.entries[64]

    assert entry.f_cpu == 0.1
    assert entry.f_prefetch == 0.1
    assert entry.predicted_ms == 60.0
    assert entry.f_cpu + entry.f_prefetch == problem.f_cpu_store


def test_dispatch_solver_attention_window_can_shift_weight_dispatch_to_prefetch():
    no_attention = DispatchProblem(
        buckets=(8,),
        f_cpu_store=0.2,
        num_layers=1,
        layer_ops=("qkv",),
        weight_bytes_per_layer={"qkv": 1000},
        candidate_f_cpu=(0.0, 0.1, 0.2),
    )
    with_attention = DispatchProblem(
        buckets=(8,),
        f_cpu_store=0.2,
        num_layers=1,
        layer_ops=("qkv", "attention"),
        weight_bytes_per_layer={"qkv": 1000},
        candidate_f_cpu=(0.0, 0.1, 0.2),
    )
    profile = SyntheticDispatchProfile(
        gpu_full_ms_by_bucket={8: 1.0},
        cpu_full_ms_by_bucket={8: 50.0},
        h2d_ms_per_byte=0.05,
        attention_gpu_ms_by_bucket={8: 10.0},
    )

    no_attention_result = solve_per_bucket_dispatch(no_attention, profile)
    with_attention_result = solve_per_bucket_dispatch(with_attention, profile)

    assert no_attention_result.dispatch_table[8] == (0.1, 0.1)
    assert with_attention_result.dispatch_table[8] == (0.0, 0.2)
    assert (
        with_attention_result.entries[8].f_prefetch
        > no_attention_result.entries[8].f_prefetch
    )


def test_dispatch_solver_rejects_invalid_problem():
    problem = DispatchProblem(
        buckets=(1,),
        f_cpu_store=1.2,
        num_layers=1,
        layer_ops=("qkv",),
        weight_bytes_per_layer={"qkv": 1000},
    )
    profile = SyntheticDispatchProfile(
        gpu_full_ms_by_bucket={1: 1.0},
        cpu_full_ms_by_bucket={1: 1.0},
        h2d_ms_per_byte=0.01,
    )

    with pytest.raises(ValueError, match="f_cpu_store"):
        solve_per_bucket_dispatch(problem, profile)


def test_manual_planner_applies_generator_and_verifier_overrides():
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={
            "model": "gen",
            "gpu_memory_utilization": 0.50,
        },
        verifier_vllm_config={
            "model": "ver",
            "gpu_memory_utilization": 0.20,
        },
        planner_config={
            "generator": {
                "gpu_memory_utilization": 0.68,
                "weight": {
                    "f_cpu_store": 0.02,
                    "f_prefetch": 0.01,
                    "modules": ["qkv", "mlp", "wo"],
                    "dispatch_table": {"64": [0.01, 0.01]},
                    "cpu_num_threads": 24,
                    "cpu_num_threads_by_bucket": {"64": 16},
                },
                "kv": {
                    "cpu_kv_bytes": 32 * (1 << 30),
                },
            },
            "verifier": {
                "gpu_memory_utilization": 0.22,
                "kv": {
                    "cpu_kv_gb": 8,
                },
            },
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=4))
    apply_ttc_plan_to_config(config, plan)

    gen = config.generator_vllm_config
    ver = config.verifier_vllm_config

    assert gen["gpu_memory_utilization"] == 0.68
    assert gen["offload_backend"] == "cots"
    assert gen["cots_f_cpu_store"] == 0.02
    assert gen["cots_f_prefetch"] == 0.01
    assert gen["cots_weight_modules"] == {"qkv", "mlp", "wo"}
    assert gen["cots_dispatch_table"] == {64: (0.01, 0.01)}
    assert gen["cots_cpu_num_threads"] == 24
    assert gen["cots_cpu_num_threads_by_bucket"] == {64: 16}
    assert gen["kv_offloading_size"] == 32.0
    assert gen["kv_offloading_backend"] == "native"

    assert ver["gpu_memory_utilization"] == 0.22
    assert "offload_backend" not in ver
    assert ver["kv_offloading_size"] == 8.0
    assert ver["kv_offloading_backend"] == "native"


def test_manual_planner_emits_cots_hybrid_kv_fields():
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={
            "model": "gen",
            "gpu_memory_utilization": 0.50,
        },
        planner_config={
            "generator": {
                "max_num_seqs": 512,
                "kv": {
                    "KV_gpu_bytes": 4 * (1 << 30),
                    "KV_cpu_bytes": 12 * (1 << 30),
                    "split_blocks": 128,
                },
            },
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=64))
    apply_ttc_plan_to_config(config, plan)

    gen = config.generator_vllm_config
    assert gen["offload_backend"] == "cots"
    assert gen["max_num_seqs"] == 512
    assert gen["kv_cache_memory_bytes"] == 4 * (1 << 30)
    assert gen["cots_kv_split_blocks"] == 128
    assert gen["cots_kv_cpu_pool_bytes"] == 12 * (1 << 30)
    assert gen["cots_kv_h2d_mode"] == "uva"
    assert "kv_offloading_size" not in gen
    assert "kv_offloading_backend" not in gen


def test_manual_planner_derives_weight_thread_policy_from_dispatch_table():
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        planner_config={
            "generator": {
                "weight": {
                    "f_cpu_store": 0.05,
                    "dispatch_table": {
                        "1": [0.02, 0.03],
                        "4": [0.05, 0.0],
                        "16": [0.05, 0.0],
                    },
                },
            },
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=4))
    apply_ttc_plan_to_config(config, plan)

    gen = config.generator_vllm_config
    assert gen["cots_cpu_num_threads_by_bucket"] == {
        1: 4,
        4: 16,
        16: 24,
    }


def test_manual_planner_accepts_comma_separated_weight_modules():
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        planner_config={
            "generator": {
                "weight": {
                    "f_cpu_store": 0.05,
                    "weight_modules": "qkv,wo",
                },
            },
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=4))
    apply_ttc_plan_to_config(config, plan)

    assert config.generator_vllm_config["cots_weight_modules"] == {"qkv", "wo"}


def test_manual_planner_rejects_invalid_max_num_seqs():
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        planner_config={
            "generator": {
                "max_num_seqs": 0,
            },
        },
    )

    try:
        ManualTTCPlanner(config).plan(SearchConfig(n=4))
    except ValueError as exc:
        assert "max_num_seqs must be positive" in str(exc)
    else:
        raise AssertionError("planner accepted max_num_seqs=0")


def test_benchmark_yaml_passes_planner_config_to_fasttts_config():
    benchmark = build_benchmark_config_from_yaml(
        {
            "name": "planner-smoke",
            "prefix_aware_scheduling": True,
            "dataset": {
                "name": "HuggingFaceH4/MATH-500",
                "split": "test",
                "limit": 1,
            },
            "generator_model": {
                "enforce_eager": True,
            },
            "planner_enabled": True,
            "planner_config": {
                "generator": {
                    "max_num_seqs": 128,
                    "kv": {
                        "KV_gpu_bytes": 1 << 30,
                        "KV_cpu_bytes": 2 << 30,
                        "split_blocks": 42,
                    },
                },
            },
            "search_config": {
                "approach": "beam_search",
                "beam_width": 4,
                "n": 4,
                "num_iterations": 1,
                "max_tokens": 32,
            },
        }
    )

    config = benchmark.fasttts_config
    assert benchmark.enable_prefix_aware_scheduling is True
    assert config.prefix_aware_scheduling is True
    assert config.planner_enabled is True
    assert config.planner_mode == "manual"
    assert config.planner_config["generator"]["max_num_seqs"] == 128
    assert config.planner_config["generator"]["kv"]["split_blocks"] == 42
    assert config.generator_vllm_config["enforce_eager"] is True


def test_manual_planner_defaults_to_existing_vllm_config():
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={
            "model": "gen",
            "gpu_memory_utilization": 0.74,
            "offload_backend": "cots",
            "cots_f_cpu_store": 0.03,
            "cots_f_prefetch": 0.02,
            "kv_offloading_size": 16.0,
            "kv_offloading_backend": "native",
        },
        verifier_vllm_config={
            "model": "ver",
            "gpu_memory_utilization": 0.16,
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=16))
    apply_ttc_plan_to_config(config, plan)

    gen = config.generator_vllm_config
    assert gen["gpu_memory_utilization"] == 0.74
    assert gen["offload_backend"] == "cots"
    assert gen["cots_f_cpu_store"] == 0.03
    assert gen["cots_f_prefetch"] == 0.02
    assert gen["kv_offloading_size"] == 16.0
