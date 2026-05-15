import sys
from pathlib import Path


TTC_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TTC_ROOT / "FastTTS-thesis"))

from config import FastTTSConfig, SearchConfig  # noqa: E402
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
