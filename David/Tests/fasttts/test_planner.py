import sys
from pathlib import Path

import pytest


TTC_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TTC_ROOT / "FastTTS-thesis"))
sys.path.insert(0, str(TTC_ROOT / "FastTTS-thesis" / "benchmarks"))
sys.path.insert(0, str(TTC_ROOT / "David" / "Benchmarks" / "planner"))

from config import FastTTSConfig, SearchConfig  # noqa: E402
from benchmark_config import build_benchmark_config_from_yaml  # noqa: E402
from planner import (  # noqa: E402
    CotsGPUBufferGeometry,
    DispatchProblem,
    DispatchCompiler,
    ManualTTCPlanner,
    ModelMemoryPartitioner,
    WeightDispatchBucketCost,
    WeightDispatchCostProfile,
    WeightKVCandidateScore,
    WeightKVPlacementFrontier,
    WeightKVPartitioner,
    WeightKVResourceEstimate,
    apply_ttc_plan_to_config,
    cots_snap_resource_maps_from_metadata,
    derive_gpu_buffer_bytes,
    derive_weight_store_candidates_from_profile,
    derive_weight_thread_policy,
    estimate_weight_kv_resources,
    score_weight_kv_candidate,
    solve_per_bucket_dispatch,
    solve_weight_kv_partition,
    solve_weight_dispatch_split,
    solve_weight_dispatch_table,
    weight_dispatch_table_from_splits,
)
from validate_runtime_memory_accounting import (  # noqa: E402
    parse_runtime_log_text,
    validate_runtime_memory_accounting,
)
from fit_dispatch_cost_model import build_weight_dispatch_profile  # noqa: E402


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


def test_weight_three_lane_solver_balances_cpu_and_h2d_for_decode_bucket():
    cost = WeightDispatchBucketCost(
        bucket=64,
        g_s_per_fraction=0.4698,
        c_s_per_fraction=49.4604,
        h_s_per_fraction=15.0140,
    )

    split = solve_weight_dispatch_split(cost, f_cpu_store=0.30)

    assert split.f_cpu == 0.075
    assert split.f_prefetch == 0.225
    assert split.bottleneck == "cpu"
    assert split.lane_scores_s["cpu"] == pytest.approx(3.70953)
    assert split.lane_scores_s["h2d"] == pytest.approx(3.37815)


def test_weight_three_lane_profile_loads_fit_summary_shape():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "real_fits": {
                "8": {
                    "g_s_per_fraction_fixed": 0.444,
                    "c_s_per_fraction": 8.332,
                    "h_s_per_fraction": 13.061,
                    "k_by_store_s": {"0.15": 0.592},
                    "rmse_s": 0.068,
                    "rank_exact": 4,
                    "rank_total": 5,
                    "rank_within_one_step": 5,
                }
            }
        }
    )

    cost = profile.cost_for_bucket(8)
    assert cost.g_s_per_fraction == 0.444
    assert cost.k_s(0.15) == 0.592
    assert cost.continuous_u_over_s_optimum == pytest.approx(
        13.061 / (13.061 + 8.332)
    )


def test_weight_three_lane_profile_preserves_weight_resource_model():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "schema_version": 1,
            "dispatch_model": "weight_three_lane_v1",
            "weight_resource_model": {
                "total_weight_bytes": 1000,
                "gpu_buffer_bytes_per_store_fraction": 100,
                "buffer_model": "cots_option_a_v1",
            },
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                }
            },
        }
    )

    resource_model = profile.metadata["weight_resource_model"]
    assert resource_model["total_weight_bytes"] == 1000
    assert resource_model["gpu_buffer_bytes_per_store_fraction"] == 100
    assert resource_model["buffer_model"] == "cots_option_a_v1"


def test_weight_three_lane_profile_preserves_cots_snap_realization():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "schema_version": 1,
            "dispatch_model": "weight_three_lane_v1",
            "cots_snap": {
                "schema_version": 1,
                "snap_model": "cots_snap_v1",
                "storage_by_store_fraction": {
                    "0.2": {
                        "cpu_weight_bytes": 190,
                        "gpu_buffer_bytes": 10,
                    }
                },
            },
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                }
            },
        }
    )

    maps = cots_snap_resource_maps_from_metadata(profile.metadata)

    assert maps["cpu_weight_bytes_by_store_fraction"] == {0.2: 190}
    assert maps["gpu_buffer_bytes_by_store_fraction"] == {0.2: 10}


def test_weight_dispatch_profile_export_includes_cots_snap_realization():
    profile = build_weight_dispatch_profile(
        {
            "sources": [],
            "metadata": {},
            "real_fits": {
                "8": {
                    "g_s_per_fraction_fixed": 0.0,
                    "c_s_per_fraction": 10.0,
                    "h_s_per_fraction": 10.0,
                    "k_by_store_s": {"0.2": 0.0},
                    "continuous_u_over_s_optimum": 0.5,
                    "rmse_s": 0.0,
                    "mae_s": 0.0,
                    "rank_exact": 1,
                    "rank_total": 1,
                    "rank_within_one_step": 1,
                    "winning_lane_counts": {"cpu": 1},
                }
            },
        },
        weight_resource_model={
            "total_weight_bytes": 1000,
            "gpu_buffer_bytes_per_store_fraction": 100,
            "cpu_weight_bytes_by_store_fraction": {"0.2": 190},
            "gpu_buffer_bytes_by_store_fraction": {"0.2": 10},
            "buffer_model": "cots_option_a_v1",
        },
    )

    assert profile["cots_snap"]["snap_model"] == "cots_snap_v1"
    assert profile["cots_snap"]["storage_by_store_fraction"] == {
        "0.2": {
            "cpu_weight_bytes": 190,
            "gpu_buffer_bytes": 10,
        }
    }


def test_weight_store_candidate_derivation_uses_common_k_support():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.1": 1.0, "0.2": 0.0},
                },
                "16": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 20.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.2": 0.0, "0.3": 1.0},
                },
            },
        }
    )

    assert derive_weight_store_candidates_from_profile(profile, (8, 16)) == (0.2,)


def test_weight_store_candidate_derivation_rejects_missing_common_k_support():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.1": 1.0},
                },
                "16": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 20.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.2": 0.0},
                },
            },
        }
    )

    with pytest.raises(ValueError, match="no common K"):
        derive_weight_store_candidates_from_profile(profile, (8, 16))


def test_weight_three_lane_profile_rejects_unknown_schema():
    with pytest.raises(ValueError, match="unsupported weight dispatch model"):
        WeightDispatchCostProfile.from_mapping(
            {
                "dispatch_model": "mystery_model",
                "buckets": {},
            }
        )


def test_weight_three_lane_table_export_shape():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.4440,
                    "C_s_per_fraction": 8.3320,
                    "H_s_per_fraction": 13.0612,
                },
                "32": {
                    "G_s_per_fraction": 0.4675,
                    "C_s_per_fraction": 25.5703,
                    "H_s_per_fraction": 14.6419,
                },
            },
        }
    )

    splits = solve_weight_dispatch_table(profile, buckets=(8, 32), f_cpu_store=0.4)

    assert weight_dispatch_table_from_splits(splits) == {
        8: (0.25, 0.15),
        32: (0.15, 0.25),
    }


def test_dispatch_compiler_class_exports_runtime_table():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "64": {
                    "G_s_per_fraction": 0.4698,
                    "C_s_per_fraction": 49.4604,
                    "H_s_per_fraction": 15.0140,
                },
            },
        }
    )
    compiler = DispatchCompiler(profile)

    assert compiler.compile_runtime_table(buckets=(64,), f_cpu_store=0.30) == {
        64: (0.075, 0.225)
    }


def test_weight_kv_partitioner_uses_k_to_compare_storage_candidates():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.1": 1.0, "0.2": 0.0},
                }
            },
        }
    )

    result = solve_weight_kv_partition(
        profile,
        buckets=(8,),
        f_cpu_store_candidates=(0.1, 0.2),
        candidate_ratio_step=0.5,
    )

    assert result.best.f_cpu_store == 0.2
    assert result.best.expected_s == pytest.approx(1.0)
    assert result.best.peak_prefetch_fraction == 0.1
    assert result.best.dispatch_table == {8: (0.1, 0.1)}
    assert [score.f_cpu_store for score in result.candidates] == [0.1, 0.2]


def test_weight_kv_partitioner_class_selects_static_storage_candidate():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.1": 1.0, "0.2": 0.0},
                }
            },
        }
    )
    partitioner = WeightKVPartitioner(
        profile=profile,
        buckets=(8,),
        candidate_ratio_step=0.5,
    )

    result = partitioner.solve(f_cpu_store_candidates=(0.1, 0.2))

    assert result.best.f_cpu_store == 0.2
    assert result.best.dispatch_table == {8: (0.1, 0.1)}
    assert result.frontier.best.f_cpu_store == 0.2


def test_weight_kv_score_attaches_weight_resource_estimate():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.25": 0.0},
                }
            },
        }
    )

    score = score_weight_kv_candidate(
        profile,
        buckets=(8,),
        f_cpu_store=0.25,
        candidate_ratio_step=0.5,
        total_weight_bytes=1000,
        engine_gpu_budget_bytes=1000,
        cpu_kv_bytes=300,
        gpu_buffer_bytes_per_store_fraction=200,
    )

    assert score.resources.gpu_weight_bytes == 750
    assert score.resources.cpu_weight_bytes == 250
    assert score.resources.gpu_buffer_bytes == 50
    assert score.resources.gpu_kv_bytes == 200
    assert score.resources.engine_gpu_budget_bytes == 1000
    assert score.gpu_bytes == 1000
    assert score.cpu_bytes == 550


def test_weight_kv_resource_estimate_prefers_profile_snapped_bytes():
    resources = estimate_weight_kv_resources(
        f_cpu_store=0.15,
        total_weight_bytes=1000,
        cpu_weight_bytes_by_store_fraction={0.15: 140},
        gpu_buffer_bytes_per_store_fraction=100,
        gpu_buffer_bytes_by_store_fraction={0.15: 13},
        engine_gpu_budget_bytes=900,
    )

    assert resources.cpu_weight_bytes == 140
    assert resources.gpu_weight_bytes == 860
    assert resources.gpu_buffer_bytes == 13
    assert resources.gpu_kv_bytes == 27


def test_cots_gpu_buffer_geometry_derives_full_store_coefficient():
    geometry = CotsGPUBufferGeometry(
        hidden_size=10,
        intermediate_size=20,
        qkv_output_size=30,
        dtype_bytes=2,
        prefetch_buffer_slots=2,
        max_num_batched_tokens=5,
        modules=frozenset({"qkv", "mlp"}),
    )

    assert geometry.prefetch_buffer_bytes_per_store_fraction == 3600
    assert geometry.output_scratch_bytes_per_store_fraction == 400
    assert geometry.gpu_buffer_bytes_per_store_fraction == 4000
    assert (
        derive_gpu_buffer_bytes(
            f_cpu_store=0.25,
            gpu_buffer_bytes_per_store_fraction=(
                geometry.gpu_buffer_bytes_per_store_fraction
            ),
        )
        == 1000
    )


def _frontier_candidate(*, f_cpu_store, expected_s, gpu_bytes, cpu_bytes):
    return WeightKVCandidateScore(
        f_cpu_store=f_cpu_store,
        expected_s=expected_s,
        bucket_weights={},
        per_bucket_s={},
        splits={},
        peak_prefetch_fraction=0.0,
        resources=WeightKVResourceEstimate(
            gpu_weight_bytes=gpu_bytes,
            cpu_weight_bytes=cpu_bytes,
        ),
    )


def test_weight_kv_frontier_prunes_dominated_candidates():
    keep_a = _frontier_candidate(
        f_cpu_store=0.1,
        expected_s=1.0,
        gpu_bytes=100,
        cpu_bytes=100,
    )
    drop_b = _frontier_candidate(
        f_cpu_store=0.2,
        expected_s=1.1,
        gpu_bytes=120,
        cpu_bytes=100,
    )
    keep_c = _frontier_candidate(
        f_cpu_store=0.3,
        expected_s=0.9,
        gpu_bytes=150,
        cpu_bytes=80,
    )

    frontier = WeightKVPlacementFrontier.from_candidates(
        [keep_a, drop_b, keep_c]
    )

    assert frontier.candidates == (keep_a, keep_c)
    assert frontier.best == keep_c


def test_weight_kv_partitioner_frontier_keeps_resource_tradeoffs():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.1": 1.0, "0.2": 0.0},
                }
            },
        }
    )
    partitioner = WeightKVPartitioner(
        profile=profile,
        buckets=(8,),
        candidate_ratio_step=0.5,
        total_weight_bytes=1000,
    )

    frontier = partitioner.frontier(f_cpu_store_candidates=(0.1, 0.2))

    assert [candidate.f_cpu_store for candidate in frontier.candidates] == [
        0.1,
        0.2,
    ]
    assert [
        (candidate.gpu_bytes, candidate.cpu_bytes)
        for candidate in frontier.candidates
    ] == [(900, 100), (800, 200)]


def test_weight_kv_partitioner_filters_infeasible_engine_gpu_budgets():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.1": 0.0, "0.2": 0.0},
                }
            },
        }
    )
    partitioner = WeightKVPartitioner(
        profile=profile,
        buckets=(8,),
        candidate_ratio_step=0.5,
        total_weight_bytes=1000,
        gpu_buffer_bytes_per_store_fraction=250,
        engine_gpu_budget_bytes=850,
    )

    frontier = partitioner.frontier(f_cpu_store_candidates=(0.1, 0.2))

    assert [candidate.f_cpu_store for candidate in frontier.candidates] == [0.2]
    assert frontier.candidates[0].resources.gpu_kv_bytes == 0
    assert frontier.candidates[0].gpu_bytes == 850


def test_weight_kv_scoring_requires_k_by_default():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                }
            },
        }
    )

    with pytest.raises(ValueError, match="requires K"):
        score_weight_kv_candidate(
            profile,
            buckets=(8,),
            f_cpu_store=0.1,
            candidate_ratio_step=0.5,
        )


def test_model_memory_partitioner_selects_best_feasible_pair():
    gen_fast = _frontier_candidate(
        f_cpu_store=0.1,
        expected_s=1.0,
        gpu_bytes=100,
        cpu_bytes=10,
    )
    gen_small = _frontier_candidate(
        f_cpu_store=0.2,
        expected_s=1.5,
        gpu_bytes=50,
        cpu_bytes=10,
    )
    ver_fast = _frontier_candidate(
        f_cpu_store=0.1,
        expected_s=0.5,
        gpu_bytes=100,
        cpu_bytes=20,
    )
    ver_small = _frontier_candidate(
        f_cpu_store=0.2,
        expected_s=0.8,
        gpu_bytes=40,
        cpu_bytes=20,
    )

    result = ModelMemoryPartitioner(
        gpu_budget_bytes=150,
        cpu_budget_bytes=100,
    ).solve(
        generator_frontier=WeightKVPlacementFrontier.from_candidates(
            [gen_fast, gen_small]
        ),
        verifier_frontier=WeightKVPlacementFrontier.from_candidates(
            [ver_fast, ver_small]
        ),
    )

    assert len(result.candidates) == 3
    assert result.best.generator == gen_fast
    assert result.best.verifier == ver_small
    assert result.best.gpu_bytes == 140
    assert result.best.cpu_bytes == 30
    assert result.best.objective_s == pytest.approx(1.8)


def test_model_memory_partitioner_solves_from_engine_budget_splits():
    profile = WeightDispatchCostProfile.from_mapping(
        {
            "dispatch_model": "weight_three_lane_v1",
            "buckets": {
                "8": {
                    "G_s_per_fraction": 0.0,
                    "C_s_per_fraction": 10.0,
                    "H_s_per_fraction": 10.0,
                    "K_by_store_s": {"0.2": 0.0},
                }
            },
        }
    )
    partitioner = WeightKVPartitioner(
        profile=profile,
        buckets=(8,),
        candidate_ratio_step=0.5,
        total_weight_bytes=1000,
        gpu_buffer_bytes_per_store_fraction=100,
    )

    assert partitioner.min_gpu_budget_breakpoints((0.2,)) == (820,)

    result = ModelMemoryPartitioner(
        gpu_budget_bytes=1840,
        cpu_budget_bytes=1000,
    ).solve_from_partitioners(
        generator_partitioner=partitioner,
        verifier_partitioner=partitioner,
        generator_f_cpu_store_candidates=(0.2,),
        verifier_f_cpu_store_candidates=(0.2,),
        engine_gpu_budget_step_bytes=100,
    )

    assert result.best.gpu_bytes == 1840
    assert result.best.cpu_bytes == 400
    assert result.best.generator.resources.gpu_weight_bytes == 800
    assert result.best.generator.resources.gpu_buffer_bytes == 20
    assert (
        result.best.generator.resources.gpu_kv_bytes
        + result.best.verifier.resources.gpu_kv_bytes
        == 200
    )


def _runtime_accounting_plan(*, gpu_buffer_bytes=75_000_000):
    return {
        "summary": {},
        "placements": [
            {
                "role": "generator",
                "f_cpu_store": 0.2,
                "gpu_weight_bytes": 800_000_000,
                "gpu_buffer_bytes": gpu_buffer_bytes,
                "gpu_kv_bytes": 1_234_567_890,
                "cpu_weight_bytes": 200_000_000,
                "cpu_kv_bytes": 987_654_321,
                "dispatch_table": {"8": [0.1, 0.1]},
            }
        ],
    }


def _runtime_accounting_log():
    return """
[CotsOffloader] dispatch policy: f_cpu_store=0.200000, dispatch_table={8: (0.1, 0.1)}
[CotsOffloader] ready: runner=native, sync=wait_kernel, modules=['mlp', 'qkv', 'wo'], wo_qkvo_granularity_multiplier=2, linears=10, mlp_blocks=1, wo_ops=0, weights_saved=0.2000 GB, buffers=0.0100 GB pinned_in + 0.0200 GB pinned_out + 0.0500 GB gpu_uva, prefetch_pool=0.0250 GB, graph_buckets=(8,), dispatch_buckets=(8,)
Initial free memory 9.0 GiB, reserved 1.15 GiB (kv_cache_memory_bytes=1234567890) memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling.
Initialized COTS hybrid CPU KV store: split_blocks=8, block_size=16, split_tokens=128, cpu_blocks=42, layers=28, cpu_pool_bytes=987654321
"""


def test_runtime_memory_accounting_validator_accepts_matching_logs():
    runtime_log = parse_runtime_log_text(_runtime_accounting_log())

    report = validate_runtime_memory_accounting(
        _runtime_accounting_plan(),
        runtime_log,
    )

    assert report["ok"]
    assert {check["field"] for check in report["checks"]} == {
        "cpu_weight_bytes",
        "gpu_buffer_bytes",
        "dispatch_buckets",
        "f_cpu_store",
        "dispatch_table",
        "gpu_kv_bytes",
        "cpu_kv_bytes",
    }


def test_runtime_memory_accounting_validator_flags_buffer_mismatch():
    runtime_log = parse_runtime_log_text(_runtime_accounting_log())

    report = validate_runtime_memory_accounting(
        _runtime_accounting_plan(gpu_buffer_bytes=90_000_000),
        runtime_log,
        cots_absolute_tolerance_bytes=1,
        relative_tolerance=0.0,
    )

    assert not report["ok"]
    failed_fields = {check["field"] for check in report["checks"] if not check["ok"]}
    assert failed_fields == {"gpu_buffer_bytes"}


def test_model_memory_partitioner_honors_engine_weights():
    gen_fast = _frontier_candidate(
        f_cpu_store=0.1,
        expected_s=1.0,
        gpu_bytes=100,
        cpu_bytes=10,
    )
    gen_small = _frontier_candidate(
        f_cpu_store=0.2,
        expected_s=1.1,
        gpu_bytes=50,
        cpu_bytes=10,
    )
    ver_fast = _frontier_candidate(
        f_cpu_store=0.1,
        expected_s=0.1,
        gpu_bytes=100,
        cpu_bytes=20,
    )
    ver_small = _frontier_candidate(
        f_cpu_store=0.2,
        expected_s=0.9,
        gpu_bytes=40,
        cpu_bytes=20,
    )

    result = ModelMemoryPartitioner(
        gpu_budget_bytes=150,
        cpu_budget_bytes=100,
        engine_weights={"generator": 10.0, "verifier": 1.0},
    ).solve(
        generator_frontier=WeightKVPlacementFrontier.from_candidates(
            [gen_fast, gen_small]
        ),
        verifier_frontier=WeightKVPlacementFrontier.from_candidates(
            [ver_fast, ver_small]
        ),
    )

    assert result.best.generator == gen_fast
    assert result.best.verifier == ver_small
    assert result.best.objective_s == pytest.approx(10.9)


def test_model_memory_partitioner_rejects_no_feasible_pair():
    gen = _frontier_candidate(
        f_cpu_store=0.1,
        expected_s=1.0,
        gpu_bytes=100,
        cpu_bytes=10,
    )
    ver = _frontier_candidate(
        f_cpu_store=0.1,
        expected_s=0.5,
        gpu_bytes=100,
        cpu_bytes=20,
    )

    with pytest.raises(ValueError, match="no feasible"):
        ModelMemoryPartitioner(
            gpu_budget_bytes=150,
            cpu_budget_bytes=100,
        ).solve(
            generator_frontier=WeightKVPlacementFrontier.from_candidates([gen]),
            verifier_frontier=WeightKVPlacementFrontier.from_candidates([ver]),
        )


def test_model_memory_partitioner_requires_resource_estimates():
    unknown = _frontier_candidate(
        f_cpu_store=0.1,
        expected_s=1.0,
        gpu_bytes=0,
        cpu_bytes=0,
    )
    known = _frontier_candidate(
        f_cpu_store=0.1,
        expected_s=0.5,
        gpu_bytes=100,
        cpu_bytes=20,
    )

    with pytest.raises(ValueError, match="resource estimates"):
        ModelMemoryPartitioner(
            gpu_budget_bytes=150,
            cpu_budget_bytes=100,
        ).solve(
            generator_frontier=WeightKVPlacementFrontier.from_candidates([unknown]),
            verifier_frontier=WeightKVPlacementFrontier.from_candidates([known]),
        )


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
                    "gpu_kv_bytes": 4 * (1 << 30),
                    "cpu_kv_bytes": 12 * (1 << 30),
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


def test_manual_planner_derives_dispatch_table_from_weight_cost_profile(tmp_path):
    profile_path = tmp_path / "weight_dispatch_profile.json"
    profile_path.write_text(
        """
{
  "dispatch_model": "weight_three_lane_v1",
  "buckets": {
    "8": {
      "G_s_per_fraction": 0.4440,
      "C_s_per_fraction": 8.3320,
      "H_s_per_fraction": 13.0612
    },
    "32": {
      "G_s_per_fraction": 0.4675,
      "C_s_per_fraction": 25.5703,
      "H_s_per_fraction": 14.6419
    }
  }
}
""".strip()
    )
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        planner_config={
            "generator": {
                "weight": {
                    "f_cpu_store": 0.4,
                    "dispatch_cost_profile_path": str(profile_path),
                    "dispatch_buckets": "8,32",
                },
            },
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=4))
    apply_ttc_plan_to_config(config, plan)

    assert config.generator_vllm_config["cots_dispatch_table"] == {
        8: (0.25, 0.15),
        32: (0.15, 0.25),
    }
    assert config.generator_vllm_config["cots_cpu_num_threads_by_bucket"] == {
        8: 24,
        32: 24,
    }


def test_manual_planner_solves_weight_store_candidates_from_profile(tmp_path):
    profile_path = tmp_path / "weight_dispatch_profile.json"
    profile_path.write_text(
        """
{
  "dispatch_model": "weight_three_lane_v1",
  "buckets": {
    "8": {
      "G_s_per_fraction": 0.0,
      "C_s_per_fraction": 10.0,
      "H_s_per_fraction": 10.0,
      "K_by_store_s": {
        "0.1": 1.0,
        "0.2": 0.0
      }
    }
  }
}
""".strip()
    )
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        planner_config={
            "generator": {
                "weight": {
                    "f_cpu_store_candidates": "0.1,0.2",
                    "dispatch_cost_profile_path": str(profile_path),
                    "dispatch_buckets": [8],
                    "dispatch_candidate_ratio_step": 0.5,
                },
            },
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=4))
    apply_ttc_plan_to_config(config, plan)

    assert config.generator_vllm_config["cots_f_cpu_store"] == 0.2
    assert config.generator_vllm_config["cots_dispatch_table"] == {8: (0.1, 0.1)}


def test_manual_planner_solves_global_model_memory_from_frontiers(tmp_path):
    profile_path = tmp_path / "weight_dispatch_profile.json"
    profile_path.write_text(
        """
{
  "dispatch_model": "weight_three_lane_v1",
  "buckets": {
    "8": {
      "G_s_per_fraction": 0.0,
      "C_s_per_fraction": 10.0,
      "H_s_per_fraction": 10.0,
      "K_by_store_s": {
        "0.1": 0.0,
        "0.2": 0.0
      }
    }
  }
}
""".strip()
    )
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        verifier_vllm_config={"model": "ver"},
        planner_config={
            "global": {
                "gpu_budget_bytes": 1700,
                "cpu_budget_bytes": 1000,
                "engine_gpu_budget_step_bytes": 100,
                "engine_weights": {
                    "generator": 2.0,
                    "verifier": 1.0,
                },
            },
            "generator": {
                "weight": {
                    "total_weight_bytes": 1000,
                    "dispatch_cost_profile_path": str(profile_path),
                    "dispatch_buckets": [8],
                    "dispatch_candidate_ratio_step": 0.5,
                },
            },
            "verifier": {
                "weight": {
                    "total_weight_bytes": 1000,
                    "dispatch_cost_profile_path": str(profile_path),
                    "dispatch_buckets": [8],
                    "dispatch_candidate_ratio_step": 0.5,
                },
            },
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=4))
    apply_ttc_plan_to_config(config, plan)

    assert plan.search["model_memory_gpu_bytes"] == 1700
    assert plan.search["model_memory_cpu_bytes"] == 300
    assert plan.search["model_memory_num_candidates"] == 5
    assert plan.search["model_memory_engine_gpu_budget_step_bytes"] == 100
    assert plan.search["model_memory_objective_s"] == pytest.approx(2.0)
    assert config.generator_vllm_config["cots_f_cpu_store"] == 0.1
    assert config.generator_vllm_config["cots_dispatch_table"] == {
        8: (0.05, 0.05)
    }
    assert config.verifier_vllm_config["cots_f_cpu_store"] == 0.2
    assert config.verifier_vllm_config["cots_dispatch_table"] == {
        8: (0.1, 0.1)
    }


def test_manual_planner_global_model_memory_assigns_spare_gpu_to_kv(tmp_path):
    profile_path = tmp_path / "weight_dispatch_profile.json"
    profile_path.write_text(
        """
{
  "dispatch_model": "weight_three_lane_v1",
  "buckets": {
    "8": {
      "G_s_per_fraction": 0.0,
      "C_s_per_fraction": 10.0,
      "H_s_per_fraction": 10.0,
      "K_by_store_s": {
        "0.2": 0.0
      }
    }
  }
}
""".strip()
    )
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        verifier_vllm_config={"model": "ver"},
        planner_config={
            "global": {
                "gpu_budget_bytes": 3400,
                "cpu_budget_bytes": 1000,
                "engine_gpu_budget_step_bytes": 100,
            },
            "generator": {
                "weight": {
                    "f_cpu_store_candidates": [0.2],
                    "total_weight_bytes": 1000,
                    "buffer_geometry": {
                        "hidden_size": 10,
                        "intermediate_size": 20,
                        "qkv_output_size": 30,
                        "dtype_bytes": 2,
                        "prefetch_buffer_slots": 2,
                        "max_num_batched_tokens": 5,
                    },
                    "dispatch_cost_profile_path": str(profile_path),
                    "dispatch_buckets": [8],
                    "dispatch_candidate_ratio_step": 0.5,
                },
            },
            "verifier": {
                "weight": {
                    "f_cpu_store_candidates": [0.2],
                    "total_weight_bytes": 1000,
                    "buffer_geometry": {
                        "hidden_size": 10,
                        "intermediate_size": 20,
                        "qkv_output_size": 30,
                        "dtype_bytes": 2,
                        "prefetch_buffer_slots": 2,
                        "max_num_batched_tokens": 5,
                    },
                    "dispatch_cost_profile_path": str(profile_path),
                    "dispatch_buckets": [8],
                    "dispatch_candidate_ratio_step": 0.5,
                },
            },
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=4))
    apply_ttc_plan_to_config(config, plan)

    assert plan.search["model_memory_gpu_bytes"] == 3400
    assert plan.search["model_memory_cpu_bytes"] == 400
    total_gpu_kv = config.generator_vllm_config.get(
        "kv_cache_memory_bytes", 0
    ) + config.verifier_vllm_config.get("kv_cache_memory_bytes", 0)
    assert total_gpu_kv == 200


def test_manual_planner_global_model_memory_uses_profile_resource_model(tmp_path):
    profile_path = tmp_path / "weight_dispatch_profile.json"
    profile_path.write_text(
        """
{
  "schema_version": 1,
  "dispatch_model": "weight_three_lane_v1",
  "weight_resource_model": {
    "total_weight_bytes": 1000,
    "gpu_buffer_bytes_per_store_fraction": 100,
    "cpu_weight_bytes_by_store_fraction": {
      "0.2": 190
    },
    "gpu_buffer_bytes_by_store_fraction": {
      "0.2": 10
    },
    "buffer_model": "cots_option_a_v1"
  },
  "buckets": {
    "8": {
      "G_s_per_fraction": 0.0,
      "C_s_per_fraction": 10.0,
      "H_s_per_fraction": 10.0,
      "K_by_store_s": {
        "0.2": 0.0
      }
    }
  }
}
""".strip()
    )
    weight_config = {
        "dispatch_cost_profile_path": str(profile_path),
        "dispatch_buckets": [8],
        "dispatch_candidate_ratio_step": 0.5,
    }
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        verifier_vllm_config={"model": "ver"},
        planner_config={
            "global": {
                "gpu_budget_bytes": 1840,
                "cpu_budget_bytes": 1000,
                "engine_gpu_budget_step_bytes": 100,
            },
            "generator": {"weight": weight_config},
            "verifier": {"weight": dict(weight_config)},
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=4))
    apply_ttc_plan_to_config(config, plan)

    assert plan.search["model_memory_gpu_bytes"] == 1840
    assert plan.search["model_memory_cpu_bytes"] == 380
    total_gpu_kv = config.generator_vllm_config.get(
        "kv_cache_memory_bytes", 0
    ) + config.verifier_vllm_config.get("kv_cache_memory_bytes", 0)
    assert total_gpu_kv == 200
    assert config.generator_vllm_config["cots_f_cpu_store"] == 0.2


def test_manual_planner_global_model_memory_uses_cots_snap_maps(tmp_path):
    profile_path = tmp_path / "weight_dispatch_profile.json"
    profile_path.write_text(
        """
{
  "schema_version": 1,
  "dispatch_model": "weight_three_lane_v1",
  "weight_resource_model": {
    "total_weight_bytes": 1000,
    "gpu_buffer_bytes_per_store_fraction": 100,
    "buffer_model": "cots_option_a_v1"
  },
  "cots_snap": {
    "schema_version": 1,
    "snap_model": "cots_snap_v1",
    "storage_by_store_fraction": {
      "0.2": {
        "cpu_weight_bytes": 190,
        "gpu_buffer_bytes": 10
      }
    }
  },
  "buckets": {
    "8": {
      "G_s_per_fraction": 0.0,
      "C_s_per_fraction": 10.0,
      "H_s_per_fraction": 10.0,
      "K_by_store_s": {
        "0.2": 0.0
      }
    }
  }
}
""".strip()
    )
    weight_config = {
        "dispatch_cost_profile_path": str(profile_path),
        "dispatch_buckets": [8],
        "dispatch_candidate_ratio_step": 0.5,
    }
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        verifier_vllm_config={"model": "ver"},
        planner_config={
            "global": {
                "gpu_budget_bytes": 1840,
                "cpu_budget_bytes": 1000,
                "engine_gpu_budget_step_bytes": 100,
            },
            "generator": {"weight": weight_config},
            "verifier": {"weight": dict(weight_config)},
        },
    )

    plan = ManualTTCPlanner(config).plan(SearchConfig(n=4))
    apply_ttc_plan_to_config(config, plan)

    assert plan.search["model_memory_cpu_bytes"] == 380
    total_gpu_kv = config.generator_vllm_config.get(
        "kv_cache_memory_bytes", 0
    ) + config.verifier_vllm_config.get("kv_cache_memory_bytes", 0)
    assert total_gpu_kv == 200


def test_manual_planner_global_model_memory_requires_resource_estimates(tmp_path):
    profile_path = tmp_path / "weight_dispatch_profile.json"
    profile_path.write_text(
        """
{
  "dispatch_model": "weight_three_lane_v1",
  "buckets": {
    "8": {
      "G_s_per_fraction": 0.0,
      "C_s_per_fraction": 10.0,
      "H_s_per_fraction": 10.0,
      "K_by_store_s": {
        "0.1": 0.0
      }
    }
  }
}
""".strip()
    )
    config = FastTTSConfig(
        planner_enabled=True,
        generator_vllm_config={"model": "gen"},
        verifier_vllm_config={"model": "ver"},
        planner_config={
            "global": {
                "gpu_budget_bytes": 1800,
                "cpu_budget_bytes": 1000,
            },
            "generator": {
                "weight": {
                    "f_cpu_store_candidates": [0.1],
                    "engine_gpu_budget_candidates": [900],
                    "dispatch_cost_profile_path": str(profile_path),
                    "dispatch_buckets": [8],
                    "dispatch_candidate_ratio_step": 0.5,
                },
            },
            "verifier": {
                "weight": {
                    "f_cpu_store_candidates": [0.1],
                    "total_weight_bytes": 1000,
                    "engine_gpu_budget_candidates": [900],
                    "dispatch_cost_profile_path": str(profile_path),
                    "dispatch_buckets": [8],
                    "dispatch_candidate_ratio_step": 0.5,
                },
            },
        },
    )

    with pytest.raises(ValueError, match="requires total_weight_bytes"):
        ManualTTCPlanner(config).plan(SearchConfig(n=4))


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
                        "gpu_kv_bytes": 1 << 30,
                        "cpu_kv_bytes": 2 << 30,
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
