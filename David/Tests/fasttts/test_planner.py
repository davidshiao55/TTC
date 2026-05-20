import sys
from pathlib import Path


TTC_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TTC_ROOT / "FastTTS-thesis"))
sys.path.insert(0, str(TTC_ROOT / "FastTTS-thesis" / "benchmarks"))

from config import FastTTSConfig, SearchConfig  # noqa: E402
from benchmark_config import build_benchmark_config_from_yaml  # noqa: E402
from planner import ManualTTCPlanner, apply_ttc_plan_to_config  # noqa: E402


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
                    "dispatch_table": {"64": [0.01, 0.01]},
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
    assert gen["cots_dispatch_table"] == {64: (0.01, 0.01)}
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
