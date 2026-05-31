import sys
from pathlib import Path

FASTTTS_ROOT = Path(__file__).resolve().parents[3] / "FastTTS-thesis"
if str(FASTTTS_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTTTS_ROOT))

from config import FastTTSConfig, SearchConfig
from planner import ManualTTCPlanner, apply_ttc_plan_to_config


GIB = 1 << 30


def _plan_and_apply(config: FastTTSConfig) -> None:
    plan = ManualTTCPlanner(config).plan(SearchConfig())
    apply_ttc_plan_to_config(config, plan)


def test_manual_planner_applies_generator_and_verifier_overrides():
    config = FastTTSConfig(
        generator_vllm_config={
            "model": "gen",
            "gpu_memory_utilization": 0.40,
        },
        verifier_vllm_config={
            "model": "ver",
            "gpu_memory_utilization": 0.30,
        },
        planner_config={
            "generator": {
                "gpu_memory_utilization": 0.55,
                "max_num_seqs": 64,
                "weight": {
                    "f_cpu_store": 0.10,
                    "f_prefetch": 0.02,
                    "dispatch_table": {64: (0.08, 0.02)},
                    "cpu_num_threads": 24,
                    "cpu_num_threads_by_bucket": {64: 24},
                },
                "kv": {
                    "split_blocks": 32,
                    "cpu_kv_bytes": 8 * GIB,
                },
            },
            "verifier": {
                "gpu_memory_utilization": 0.35,
                "kv": {"cpu_kv_gb": 2},
            },
        },
    )

    _plan_and_apply(config)

    gen = config.generator_vllm_config
    ver = config.verifier_vllm_config
    assert gen["gpu_memory_utilization"] == 0.55
    assert gen["max_num_seqs"] == 64
    assert gen["offload_backend"] == "cots"
    assert gen["cots_f_cpu_store"] == 0.10
    assert gen["cots_f_prefetch"] == 0.02
    assert gen["cots_dispatch_table"] == {64: (0.08, 0.02)}
    assert gen["cots_cpu_num_threads"] == 24
    assert gen["cots_cpu_num_threads_by_bucket"] == {64: 24}
    assert gen["cots_kv_split_blocks"] == 32
    assert gen["cots_kv_cpu_pool_bytes"] == 8 * GIB
    assert gen["cots_kv_h2d_mode"] == "uva"

    assert ver["gpu_memory_utilization"] == 0.35
    assert "offload_backend" not in ver
    assert ver["kv_offloading_size"] == 2.0
    assert ver["kv_offloading_backend"] == "native"


def test_manual_planner_defaults_to_existing_vllm_config():
    config = FastTTSConfig(
        generator_vllm_config={
            "model": "gen",
            "gpu_memory_utilization": 0.42,
            "max_num_seqs": 16,
        },
        verifier_vllm_config={
            "model": "ver",
            "gpu_memory_utilization": 0.31,
        },
        planner_config={},
    )

    _plan_and_apply(config)

    gen = config.generator_vllm_config
    assert gen["model"] == "gen"
    assert gen["gpu_memory_utilization"] == 0.42
    assert gen["max_num_seqs"] == 16
    assert "offload_backend" not in gen


def test_manual_planner_derives_weight_thread_policy_from_dispatch_table():
    config = FastTTSConfig(
        generator_vllm_config={"model": "gen"},
        verifier_vllm_config={"model": "ver"},
        planner_config={
            "generator": {
                "weight": {
                    "f_cpu_store": 0.05,
                    "dispatch_table": {
                        1: (0.02, 0.03),
                        4: (0.05, 0.0),
                        16: (0.05, 0.0),
                    },
                },
            },
        },
    )

    _plan_and_apply(config)

    assert config.generator_vllm_config["cots_cpu_num_threads_by_bucket"] == {
        1: 4,
        4: 16,
        16: 24,
    }


def test_phase2_measurement_blocks_are_ignored_until_real_policy_lands():
    config = FastTTSConfig(
        generator_vllm_config={
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "dtype": "bfloat16",
            "max_model_len": 768,
            "gpu_memory_utilization": 0.67,
            "enforce_eager": False,
            "async_scheduling": False,
        },
        verifier_vllm_config={"model": "ver"},
        planner_config={
            "phase2_kv_profile": {
                "win_margin": 0.01,
                "cells": [
                    {
                        "roles": ["generator"],
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "dtype": "bfloat16",
                        "gpu_memory_utilization": 0.67,
                        "enforce_eager": False,
                        "max_model_len": 768,
                        "max_num_seqs": 512,
                        "split_blocks": 42,
                        "cpu_kv_gb": 12,
                        "gpu_only_out_tok_s": 2178.306,
                        "hybrid_out_tok_s": 2246.663,
                    }
                ],
            }
        },
    )

    _plan_and_apply(config)

    gen = config.generator_vllm_config
    assert "offload_backend" not in gen
    assert "cots_kv_split_blocks" not in gen
    assert "max_num_seqs" not in gen
