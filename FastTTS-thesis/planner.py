#!/usr/bin/env python
"""Thin TTC planner interface for FastTTS.

This module implements the launch-time contract plus planner primitives. The
full Planner is broader than dispatch: it chooses static placement, evaluates
resource-lane costs, and emits runtime policy. The current prototype has
weight-only coefficients, but the stage-level API is intentionally general:
`ModelMemoryPartitioner` combines generator/verifier placement frontiers,
`WeightKVPartitioner` chooses per-engine static placement candidates, and
`DispatchCompiler` materializes runtime lookup tables.

FastTTS owns the two-engine memory decision and emits one engine-local plan per
model. vLLM consumes each engine-local plan through normal engine kwargs and
still owns tensor geometry, snapping, and runtime dispatch mechanics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Protocol, Sequence

from config import FastTTSConfig, SearchConfig


EngineRole = Literal["generator", "verifier"]
VALID_WEIGHT_MODULES = frozenset({"qkv", "mlp", "wo"})
DEFAULT_COTS_WEIGHT_MODULES = frozenset({"qkv", "mlp"})
DEFAULT_DISPATCH_LAYER_OPS = ("qkv", "attention", "wo", "mlp1", "mlp2")
DispatchBottleneck = Literal["compute", "prefetch"]
WeightDispatchLane = Literal["gpu", "cpu", "h2d"]
WEIGHT_THREE_LANE_MODEL = "weight_three_lane_v1"
SUPPORTED_WEIGHT_DISPATCH_MODELS = frozenset({WEIGHT_THREE_LANE_MODEL})
WEIGHT_RESOURCE_MODEL_KEYS = ("weight_resource_model", "resource_model")
COTS_SNAP_MODEL = "cots_snap_v1"
SUPPORTED_COTS_SNAP_MODELS = frozenset({COTS_SNAP_MODEL})
COTS_SNAP_PROFILE_KEYS = ("cots_snap", "cots_runtime_realization")


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1 << 30)


def _profile_metadata_from_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(raw.get("metadata", {}))
    for key in WEIGHT_RESOURCE_MODEL_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be a mapping")
        metadata["weight_resource_model"] = dict(value)
        break
    for key in COTS_SNAP_PROFILE_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"{key} must be a mapping")
        metadata["cots_snap"] = dict(value)
        break
    return metadata


def _normalize_dispatch_table(
    raw: Mapping[Any, Any] | None,
) -> dict[int, tuple[float, float]] | None:
    if raw is None:
        return None
    table: dict[int, tuple[float, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                "dispatch_table values must be two-item "
                "(f_cpu_compute, f_prefetch_compute) sequences"
            )
        bucket = int(key)
        f_cpu_compute = float(value[0])
        f_prefetch = float(value[1])
        if bucket <= 0:
            raise ValueError(f"dispatch_table bucket must be positive, got {bucket}")
        if f_cpu_compute < 0 or f_prefetch < 0:
            raise ValueError(
                "dispatch_table fractions must be non-negative, got "
                f"{value!r} for bucket {bucket}"
            )
        table[bucket] = (f_cpu_compute, f_prefetch)
    return table


def _normalize_thread_table(raw: Mapping[Any, Any] | None) -> dict[int, int] | None:
    if raw is None:
        return None
    table: dict[int, int] = {}
    for key, value in raw.items():
        bucket = int(key)
        n_threads = int(value)
        if bucket <= 0:
            raise ValueError(
                f"cpu_num_threads_by_bucket bucket must be positive, got {bucket}"
            )
        if n_threads < 1:
            raise ValueError(
                "cpu_num_threads_by_bucket values must be positive, got "
                f"{n_threads} for bucket {bucket}"
            )
        table[bucket] = n_threads
    return table


def _normalize_bucket_sequence(raw: Any, *, field_name: str) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    elif isinstance(raw, (int, float)):
        entries = [raw]
    else:
        entries = list(raw)
    buckets = tuple(sorted({int(entry) for entry in entries}))
    for bucket in buckets:
        if bucket <= 0:
            raise ValueError(f"{field_name} buckets must be positive, got {bucket}")
    return buckets


def _normalize_nonnegative_int_sequence(
    raw: Any,
    *,
    field_name: str,
) -> tuple[int, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    elif isinstance(raw, (int, float)):
        entries = [raw]
    else:
        entries = list(raw)
    values = tuple(sorted({int(entry) for entry in entries}))
    for value in values:
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative, got {value}")
    return values


def _normalize_fraction_sequence(
    raw: Any,
    *,
    field_name: str,
) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    elif isinstance(raw, (int, float)):
        entries = [raw]
    else:
        entries = list(raw)
    values = tuple(sorted({float(entry) for entry in entries}))
    for value in values:
        _validate_fraction(field_name, value)
    return values


def _normalize_fraction_int_mapping(
    raw: Any,
    *,
    field_name: str,
) -> dict[float, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    values: dict[float, int] = {}
    for key, value in raw.items():
        fraction = _round_fraction(_validate_fraction(field_name, float(key)))
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{field_name} values must be non-negative")
        values[fraction] = parsed
    return dict(sorted(values.items()))


def _fraction_mapping_value(
    mapping: Mapping[float, int] | None,
    fraction: float,
) -> int | None:
    if mapping is None:
        return None
    target = _round_fraction(_validate_fraction("f_cpu_store", fraction))
    for key, value in mapping.items():
        if abs(float(key) - target) <= 1e-9:
            return int(value)
    return None


def _normalize_bucket_weights(
    buckets: Sequence[int],
    raw: Mapping[Any, Any] | None = None,
) -> dict[int, float]:
    bucket_ids = tuple(int(bucket) for bucket in buckets)
    if not bucket_ids:
        raise ValueError("buckets must not be empty")
    if raw is None:
        weight = 1.0 / len(bucket_ids)
        return {bucket: weight for bucket in bucket_ids}

    parsed = {int(bucket): float(weight) for bucket, weight in raw.items()}
    weights = {bucket: parsed.get(bucket, 0.0) for bucket in bucket_ids}
    for bucket, weight in weights.items():
        if weight < 0:
            raise ValueError(
                f"bucket_weights values must be non-negative, got {weight} "
                f"for bucket {bucket}"
            )
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("bucket_weights must contain positive total weight")
    return {bucket: weight / total for bucket, weight in weights.items()}


def _normalize_weight_modules(raw: Any) -> set[str] | None:
    if raw is None:
        return None
    entries = (raw,) if isinstance(raw, str) else raw
    modules: set[str] = set()
    for entry in entries:
        for module in str(entry).split(","):
            module = module.strip().lower()
            if module:
                modules.add(module)
    unknown = modules - VALID_WEIGHT_MODULES
    if unknown:
        raise ValueError(
            f"weight.modules contains unsupported entries {sorted(unknown)}; "
            f"expected subset of {sorted(VALID_WEIGHT_MODULES)}"
        )
    return modules


def weight_thread_count_for_score(score: float) -> int:
    """Map bucket × CPU-compute fraction to the CPU GEMM thread count.

    This is a deterministic policy derived from the 2026-05-31 weight-thread
    experiment, not an additional Planner search axis.
    """
    if score <= 0.08:
        return 4
    if score <= 0.24:
        return 16
    return 24


def derive_weight_thread_policy(
    dispatch_table: Mapping[int, tuple[float, float]],
) -> dict[int, int]:
    return {
        int(bucket): weight_thread_count_for_score(int(bucket) * f_cpu_compute)
        for bucket, (f_cpu_compute, _) in dispatch_table.items()
    }


class DispatchProfileView(Protocol):
    """Profile lookup interface consumed by the reference dispatch solver.

    Real profiler JSON and synthetic unit-test fixtures can both implement this
    shape. The solver deliberately asks for operation-level costs instead of
    knowing how a table is stored on disk.
    """

    def gpu_op_ms(self, op: str, bucket: int, gpu_fraction: float) -> float:
        """GPU-side current-layer time for one operation."""
        ...

    def cpu_op_ms(self, op: str, bucket: int, cpu_fraction: float) -> float:
        """CPU-side current-layer time for one operation."""
        ...

    def h2d_ms(self, transfer_bytes: int) -> float:
        """Layer-ahead H2D time for a prefetched weight bundle."""
        ...


@dataclass(frozen=True)
class DispatchProblem:
    """Fixed-placement per-bucket dispatch problem.

    `f_cpu_store` is already chosen by the placement layer. This solver decides
    how those CPU-resident bytes are used in each bucket:

    * `f_cpu` is computed on CPU in the current layer.
    * `f_prefetch = f_cpu_store - f_cpu` is streamed layer-ahead and computed on
      GPU.

    Weight operations use the dispatch variable. Non-weight operations such as
    attention use `fixed_cpu_fractions_by_op`, because their split is controlled
    by KV placement rather than by the weight-dispatch table.
    """

    buckets: Sequence[int]
    f_cpu_store: float
    num_layers: int
    weight_bytes_per_layer: Mapping[str, int]
    layer_ops: Sequence[str] = DEFAULT_DISPATCH_LAYER_OPS
    candidate_f_cpu: Sequence[float] | None = None
    candidate_step: float = 0.01
    fixed_cpu_fractions_by_op: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DispatchEntry:
    """Selected or candidate dispatch score for one bucket."""

    bucket: int
    f_cpu: float
    f_prefetch: float
    predicted_ms: float
    layer_ms: float
    c_layer_ms: float
    p_layer_ms: float
    bottleneck: DispatchBottleneck


@dataclass(frozen=True)
class DispatchSolveResult:
    """Reference dispatch-solver output plus scoring diagnostics."""

    entries: dict[int, DispatchEntry]
    candidate_scores: dict[int, tuple[DispatchEntry, ...]]

    @property
    def dispatch_table(self) -> dict[int, tuple[float, float]]:
        return {
            bucket: (entry.f_cpu, entry.f_prefetch)
            for bucket, entry in self.entries.items()
        }


@dataclass(frozen=True)
class WeightDispatchBucketCost:
    """Calibrated three-lane weight-dispatch coefficients for one bucket.

    Units are seconds per fraction of the enabled weight set. The model is:

        K(B, s) + max(G_B * (1-u), C_B * u, H_B * (s-u))

    K(B, s) is optional for dispatch selection because it cancels when the
    storage fraction `s` is fixed and the solver only varies `u`.
    """

    bucket: int
    g_s_per_fraction: float
    c_s_per_fraction: float
    h_s_per_fraction: float
    k_by_store_s: Mapping[float, float] = field(default_factory=dict)
    rmse_s: float | None = None
    rank_exact: int | None = None
    rank_total: int | None = None
    rank_within_one_step: int | None = None

    def __post_init__(self) -> None:
        if int(self.bucket) <= 0:
            raise ValueError(f"bucket must be positive, got {self.bucket}")
        for name in ("g_s_per_fraction", "c_s_per_fraction", "h_s_per_fraction"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

    @property
    def continuous_u_over_s_optimum(self) -> float | None:
        denom = self.c_s_per_fraction + self.h_s_per_fraction
        if denom <= 0:
            return None
        return self.h_s_per_fraction / denom

    def lane_scores_s(
        self,
        *,
        f_cpu_store: float,
        f_cpu: float,
    ) -> dict[WeightDispatchLane, float]:
        s = _validate_fraction("f_cpu_store", f_cpu_store)
        u = _validate_fraction("f_cpu", f_cpu)
        if u > s + 1e-12:
            raise ValueError(f"f_cpu ({u}) must be <= f_cpu_store ({s})")
        p = _round_fraction(s - u)
        return {
            "gpu": float(self.g_s_per_fraction) * (1.0 - u),
            "cpu": float(self.c_s_per_fraction) * u,
            "h2d": float(self.h_s_per_fraction) * p,
        }

    def k_s(self, f_cpu_store: float) -> float | None:
        target = _round_fraction(f_cpu_store)
        for store, value in self.k_by_store_s.items():
            if abs(float(store) - target) <= 1e-9:
                return float(value)
        return None


@dataclass(frozen=True)
class WeightDispatchCostProfile:
    """Planner-facing profile artifact for calibrated weight dispatch.

    This is intentionally weight-only. Future KV terms belong in the outer
    resource-lane evaluator, which can call this solver after adding static
    KV placement costs to the GPU/CPU/PCIe lanes.
    """

    dispatch_model: str
    buckets: Mapping[int, WeightDispatchBucketCost]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WeightDispatchCostProfile":
        if "real_fits" in raw:
            return cls._from_fit_summary(raw)
        bucket_map = raw.get("buckets")
        if not isinstance(bucket_map, Mapping):
            raise ValueError("weight dispatch profile must contain a buckets map")
        dispatch_model = str(raw.get("dispatch_model", WEIGHT_THREE_LANE_MODEL))
        if dispatch_model not in SUPPORTED_WEIGHT_DISPATCH_MODELS:
            raise ValueError(
                f"unsupported weight dispatch model {dispatch_model!r}; "
                f"expected one of {sorted(SUPPORTED_WEIGHT_DISPATCH_MODELS)}"
            )
        return cls(
            dispatch_model=dispatch_model,
            buckets={
                int(bucket): cls._bucket_cost_from_mapping(int(bucket), value)
                for bucket, value in bucket_map.items()
            },
            metadata=_profile_metadata_from_mapping(raw),
        )

    @classmethod
    def _from_fit_summary(
        cls,
        raw: Mapping[str, Any],
    ) -> "WeightDispatchCostProfile":
        buckets: dict[int, WeightDispatchBucketCost] = {}
        for bucket, fit in raw.get("real_fits", {}).items():
            buckets[int(bucket)] = cls._bucket_cost_from_mapping(int(bucket), fit)
        metadata = _profile_metadata_from_mapping(raw)
        metadata.update(
            {
                "sources": list(raw.get("sources", [])),
                "baselines_by_batch": dict(raw.get("baselines_by_batch", {})),
            }
        )
        return cls(
            dispatch_model=WEIGHT_THREE_LANE_MODEL,
            buckets=buckets,
            metadata=metadata,
        )

    @classmethod
    def _bucket_cost_from_mapping(
        cls,
        bucket: int,
        raw: Mapping[str, Any],
    ) -> WeightDispatchBucketCost:
        g_value = raw.get("g_s_per_fraction", raw.get("g_s_per_fraction_fixed"))
        if g_value is None:
            g_value = raw.get("G_s_per_fraction", raw.get("G"))
        c_value = raw.get("c_s_per_fraction", raw.get("C_s_per_fraction", raw.get("C")))
        h_value = raw.get("h_s_per_fraction", raw.get("H_s_per_fraction", raw.get("H")))
        if g_value is None or c_value is None or h_value is None:
            raise ValueError(
                f"bucket {bucket} is missing one of G/C/H coefficients: {raw!r}"
            )
        k_raw = raw.get("k_by_store_s", raw.get("K_by_store_s", {}))
        if not isinstance(k_raw, Mapping):
            raise ValueError(f"bucket {bucket} K_by_store_s must be a mapping")
        return WeightDispatchBucketCost(
            bucket=int(bucket),
            g_s_per_fraction=float(g_value),
            c_s_per_fraction=float(c_value),
            h_s_per_fraction=float(h_value),
            k_by_store_s={float(store): float(value) for store, value in k_raw.items()},
            rmse_s=None if raw.get("rmse_s") is None else float(raw["rmse_s"]),
            rank_exact=(
                None if raw.get("rank_exact") is None else int(raw["rank_exact"])
            ),
            rank_total=(
                None if raw.get("rank_total") is None else int(raw["rank_total"])
            ),
            rank_within_one_step=(
                None
                if raw.get("rank_within_one_step") is None
                else int(raw["rank_within_one_step"])
            ),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "WeightDispatchCostProfile":
        return cls.from_mapping(json.loads(Path(path).read_text()))

    def cost_for_bucket(self, bucket: int) -> WeightDispatchBucketCost:
        bucket = int(bucket)
        if bucket not in self.buckets:
            raise KeyError(
                f"weight dispatch profile has no coefficients for bucket {bucket}"
            )
        return self.buckets[bucket]


@dataclass(frozen=True)
class WeightDispatchSplit:
    """Selected split for the calibrated weight-only dispatch model."""

    bucket: int
    f_cpu_store: float
    f_cpu: float
    f_prefetch: float
    resource_s: float
    predicted_s: float
    lane_scores_s: Mapping[WeightDispatchLane, float]
    bottleneck: WeightDispatchLane
    k_s: float | None = None


def _winning_weight_lane(
    lane_scores: Mapping[WeightDispatchLane, float],
) -> WeightDispatchLane:
    return max(sorted(lane_scores), key=lambda lane: lane_scores[lane])


def _candidate_weight_dispatch_f_cpu(
    *,
    f_cpu_store: float,
    candidate_f_cpu: Sequence[float] | None = None,
    candidate_ratios: Sequence[float] | None = None,
    candidate_ratio_step: float = 0.125,
) -> tuple[float, ...]:
    s = _validate_fraction("f_cpu_store", f_cpu_store)
    if candidate_f_cpu is not None and candidate_ratios is not None:
        raise ValueError("provide candidate_f_cpu or candidate_ratios, not both")
    if candidate_f_cpu is not None:
        raw_values = tuple(float(value) for value in candidate_f_cpu)
    else:
        if candidate_ratios is None:
            if candidate_ratio_step <= 0:
                raise ValueError(
                    "candidate_ratio_step must be positive, got "
                    f"{candidate_ratio_step}"
                )
            ratios = _default_fraction_grid(1.0, candidate_ratio_step)
        else:
            ratios = tuple(float(value) for value in candidate_ratios)
        raw_values = tuple(s * ratio for ratio in ratios)

    legal: dict[float, float] = {}
    for value in (0.0, s, *raw_values):
        if value < -1e-12 or value > s + 1e-12:
            continue
        clipped = min(s, max(0.0, float(value)))
        rounded = _round_fraction(clipped)
        legal[rounded] = rounded
    if not legal:
        raise ValueError("no legal weight-dispatch candidates remain")
    return tuple(sorted(legal))


def solve_weight_dispatch_split(
    bucket_cost: WeightDispatchBucketCost,
    *,
    f_cpu_store: float,
    candidate_f_cpu: Sequence[float] | None = None,
    candidate_ratios: Sequence[float] | None = None,
    candidate_ratio_step: float = 0.125,
    include_k: bool = False,
) -> WeightDispatchSplit:
    """Choose `(f_cpu_compute, f_prefetch_compute)` for fixed `(B, s)`.

    The selected split minimizes the calibrated three-lane bottleneck. K(B, s)
    is optionally added for absolute latency prediction, but it does not affect
    the minimizer because this function holds `B` and `s` fixed.
    """

    s = _validate_fraction("f_cpu_store", f_cpu_store)
    candidates = _candidate_weight_dispatch_f_cpu(
        f_cpu_store=s,
        candidate_f_cpu=candidate_f_cpu,
        candidate_ratios=candidate_ratios,
        candidate_ratio_step=candidate_ratio_step,
    )
    best: WeightDispatchSplit | None = None
    k_s = bucket_cost.k_s(s) if include_k else None
    k_addend = 0.0 if k_s is None else k_s
    for f_cpu in candidates:
        f_prefetch = _round_fraction(s - f_cpu)
        lane_scores = bucket_cost.lane_scores_s(f_cpu_store=s, f_cpu=f_cpu)
        resource_s = max(lane_scores.values())
        predicted_s = k_addend + resource_s
        split = WeightDispatchSplit(
            bucket=int(bucket_cost.bucket),
            f_cpu_store=s,
            f_cpu=f_cpu,
            f_prefetch=f_prefetch,
            resource_s=resource_s,
            predicted_s=predicted_s,
            lane_scores_s=lane_scores,
            bottleneck=_winning_weight_lane(lane_scores),
            k_s=k_s,
        )
        if best is None or split.predicted_s < best.predicted_s - 1e-12:
            best = split
        elif (
            best is not None
            and abs(split.predicted_s - best.predicted_s) <= 1e-12
            and split.f_cpu < best.f_cpu
        ):
            best = split
    assert best is not None
    return best


def solve_weight_dispatch_table(
    profile: WeightDispatchCostProfile,
    *,
    buckets: Sequence[int],
    f_cpu_store: float,
    candidate_ratios: Sequence[float] | None = None,
    candidate_ratio_step: float = 0.125,
    include_k: bool = False,
) -> dict[int, WeightDispatchSplit]:
    """Solve calibrated weight dispatch for a set of exact profile buckets."""

    return {
        int(bucket): solve_weight_dispatch_split(
            profile.cost_for_bucket(int(bucket)),
            f_cpu_store=f_cpu_store,
            candidate_ratios=candidate_ratios,
            candidate_ratio_step=candidate_ratio_step,
            include_k=include_k,
        )
        for bucket in buckets
    }


def weight_dispatch_table_from_splits(
    splits: Mapping[int, WeightDispatchSplit],
) -> dict[int, tuple[float, float]]:
    return {
        int(bucket): (split.f_cpu, split.f_prefetch)
        for bucket, split in splits.items()
    }


@dataclass(frozen=True)
class WeightKVResourceEstimate:
    """Memory resource estimate attached to one per-engine placement candidate.

    `gpu_buffer_bytes` is reserved COTS GPU workspace, including conservative
    prefetch slots and GPU output scratch. In the full placement model,
    `gpu_kv_bytes` should be derived from leftover engine GPU budget after
    weights and buffers are reserved.
    """

    gpu_weight_bytes: int = 0
    cpu_weight_bytes: int = 0
    gpu_kv_bytes: int = 0
    cpu_kv_bytes: int = 0
    gpu_buffer_bytes: int = 0
    engine_gpu_budget_bytes: int | None = None
    resource_known: bool = False

    def __post_init__(self) -> None:
        any_bytes = False
        for name in (
            "gpu_weight_bytes",
            "cpu_weight_bytes",
            "gpu_kv_bytes",
            "cpu_kv_bytes",
            "gpu_buffer_bytes",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            object.__setattr__(self, name, value)
            any_bytes = any_bytes or value > 0
        engine_gpu_budget = self.engine_gpu_budget_bytes
        if engine_gpu_budget is not None:
            engine_gpu_budget = int(engine_gpu_budget)
            if engine_gpu_budget < 0:
                raise ValueError(
                    "engine_gpu_budget_bytes must be non-negative, got "
                    f"{engine_gpu_budget}"
                )
            object.__setattr__(self, "engine_gpu_budget_bytes", engine_gpu_budget)
            any_bytes = True
        object.__setattr__(
            self,
            "resource_known",
            bool(self.resource_known or any_bytes),
        )

    @property
    def gpu_bytes(self) -> int:
        if self.engine_gpu_budget_bytes is not None:
            return self.engine_gpu_budget_bytes
        return self.gpu_weight_bytes + self.gpu_kv_bytes + self.gpu_buffer_bytes

    @property
    def cpu_bytes(self) -> int:
        return self.cpu_weight_bytes + self.cpu_kv_bytes


class _InfeasibleWeightKVPlacement(ValueError):
    """Placement candidate exceeds an engine-local resource budget."""


def estimate_weight_kv_resources(
    *,
    f_cpu_store: float,
    total_weight_bytes: int | None = None,
    cpu_weight_bytes_by_store_fraction: Mapping[float, int] | None = None,
    gpu_kv_bytes: int | None = None,
    cpu_kv_bytes: int = 0,
    gpu_buffer_bytes: int | None = None,
    gpu_buffer_bytes_per_store_fraction: int = 0,
    gpu_buffer_bytes_by_store_fraction: Mapping[float, int] | None = None,
    engine_gpu_budget_bytes: int | None = None,
) -> WeightKVResourceEstimate:
    """Estimate resource use for the current weight-only partition subset."""

    s = _validate_fraction("f_cpu_store", f_cpu_store)
    if engine_gpu_budget_bytes is not None and total_weight_bytes is None:
        raise ValueError("engine_gpu_budget_bytes requires total_weight_bytes")
    total_weight = 0 if total_weight_bytes is None else int(total_weight_bytes)
    if total_weight < 0:
        raise ValueError(
            f"total_weight_bytes must be non-negative, got {total_weight}"
        )
    engine_gpu_budget = (
        None
        if engine_gpu_budget_bytes is None
        else int(engine_gpu_budget_bytes)
    )
    if engine_gpu_budget is not None and engine_gpu_budget < 0:
        raise ValueError(
            "engine_gpu_budget_bytes must be non-negative, got "
            f"{engine_gpu_budget}"
        )
    cpu_kv = int(cpu_kv_bytes)
    gpu_buffer = derive_gpu_buffer_bytes(
        f_cpu_store=s,
        gpu_buffer_bytes=gpu_buffer_bytes,
        gpu_buffer_bytes_per_store_fraction=gpu_buffer_bytes_per_store_fraction,
        gpu_buffer_bytes_by_store_fraction=gpu_buffer_bytes_by_store_fraction,
    )
    if cpu_kv < 0:
        raise ValueError(f"cpu_kv_bytes must be non-negative, got {cpu_kv}")
    cpu_weight_bytes = derive_cpu_weight_bytes(
        f_cpu_store=s,
        total_weight_bytes=total_weight,
        cpu_weight_bytes_by_store_fraction=cpu_weight_bytes_by_store_fraction,
    )
    if cpu_weight_bytes > total_weight:
        raise ValueError(
            "cpu_weight_bytes_by_store_fraction entry exceeds total_weight_bytes: "
            f"cpu_weight_bytes={cpu_weight_bytes}, total_weight_bytes={total_weight}"
        )
    gpu_weight_bytes = total_weight - cpu_weight_bytes
    if engine_gpu_budget is not None:
        if gpu_kv_bytes is not None:
            raise ValueError(
                "gpu_kv_bytes is derived when engine_gpu_budget_bytes is set"
            )
        gpu_kv = engine_gpu_budget - gpu_weight_bytes - gpu_buffer
        if gpu_kv < 0:
            raise _InfeasibleWeightKVPlacement(
                "engine_gpu_budget_bytes is too small for resident weights "
                "and gpu_buffer_bytes: "
                f"budget={engine_gpu_budget}, "
                f"gpu_weight_bytes={gpu_weight_bytes}, "
                f"gpu_buffer_bytes={gpu_buffer}"
            )
    else:
        gpu_kv = 0 if gpu_kv_bytes is None else int(gpu_kv_bytes)
        if gpu_kv < 0:
            raise ValueError(f"gpu_kv_bytes must be non-negative, got {gpu_kv}")
    return WeightKVResourceEstimate(
        gpu_weight_bytes=gpu_weight_bytes,
        cpu_weight_bytes=cpu_weight_bytes,
        gpu_kv_bytes=gpu_kv,
        cpu_kv_bytes=cpu_kv,
        gpu_buffer_bytes=gpu_buffer,
        engine_gpu_budget_bytes=engine_gpu_budget,
        resource_known=(
            total_weight_bytes is not None
            or bool(cpu_weight_bytes_by_store_fraction)
            or engine_gpu_budget is not None
            or gpu_kv > 0
            or cpu_kv > 0
            or gpu_buffer > 0
            or int(gpu_buffer_bytes_per_store_fraction) > 0
            or bool(gpu_buffer_bytes_by_store_fraction)
        ),
    )


def derive_cpu_weight_bytes(
    *,
    f_cpu_store: float,
    total_weight_bytes: int,
    cpu_weight_bytes_by_store_fraction: Mapping[float, int] | None = None,
) -> int:
    """CPU-resident weight bytes, preferring profiler-measured snapped sizes."""

    s = _validate_fraction("f_cpu_store", f_cpu_store)
    exact = _fraction_mapping_value(cpu_weight_bytes_by_store_fraction, s)
    if exact is not None:
        return exact
    total_weight = int(total_weight_bytes)
    if total_weight < 0:
        raise ValueError(
            f"total_weight_bytes must be non-negative, got {total_weight}"
        )
    return int(round(total_weight * s))


def derive_gpu_buffer_bytes(
    *,
    f_cpu_store: float,
    gpu_buffer_bytes: int | None = None,
    gpu_buffer_bytes_per_store_fraction: int = 0,
    gpu_buffer_bytes_by_store_fraction: Mapping[float, int] | None = None,
) -> int:
    """Conservative option-A COTS GPU workspace reservation.

    vLLM's current prefetch pool is sized from the maximum effective prefetch
    rows across dispatch buckets. The planner intentionally reserves the
    full-store workspace instead, so buffer space depends on static storage
    fraction `s`, not the dispatch table.
    """

    s = _validate_fraction("f_cpu_store", f_cpu_store)
    if gpu_buffer_bytes is not None:
        explicit = int(gpu_buffer_bytes)
        if explicit < 0:
            raise ValueError(
                f"gpu_buffer_bytes must be non-negative, got {explicit}"
            )
        return explicit
    exact = _fraction_mapping_value(gpu_buffer_bytes_by_store_fraction, s)
    if exact is not None:
        return exact
    per_fraction = int(gpu_buffer_bytes_per_store_fraction)
    if per_fraction < 0:
        raise ValueError(
            "gpu_buffer_bytes_per_store_fraction must be non-negative, got "
            f"{per_fraction}"
        )
    return int(round(s * per_fraction))


@dataclass(frozen=True)
class CotsGPUBufferGeometry:
    """Shape-derived COTS GPU workspace estimate at full CPU storage.

    The prefetch pool is K slots per unique `(role, slot_shape)`, shared across
    decoder layers. It should therefore be derived from module geometry, not
    from total model weight bytes. `max_num_batched_tokens` only affects the
    reusable GPU output scratch used by CPU-compute returns.
    """

    hidden_size: int
    intermediate_size: int | None = None
    qkv_output_size: int | None = None
    dtype_bytes: int = 2
    prefetch_buffer_slots: int = 2
    max_num_batched_tokens: int = 0
    modules: frozenset[str] = field(default_factory=lambda: DEFAULT_COTS_WEIGHT_MODULES)

    def __post_init__(self) -> None:
        hidden_size = int(self.hidden_size)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        dtype_bytes = int(self.dtype_bytes)
        if dtype_bytes <= 0:
            raise ValueError(f"dtype_bytes must be positive, got {dtype_bytes}")
        prefetch_buffer_slots = int(self.prefetch_buffer_slots)
        if prefetch_buffer_slots <= 0:
            raise ValueError(
                "prefetch_buffer_slots must be positive, got "
                f"{prefetch_buffer_slots}"
            )
        max_num_batched_tokens = int(self.max_num_batched_tokens)
        if max_num_batched_tokens < 0:
            raise ValueError(
                "max_num_batched_tokens must be non-negative, got "
                f"{max_num_batched_tokens}"
            )
        modules = frozenset(self.modules)
        unknown = modules - VALID_WEIGHT_MODULES
        if unknown:
            raise ValueError(
                f"buffer geometry modules contain unsupported entries "
                f"{sorted(unknown)}; expected subset of "
                f"{sorted(VALID_WEIGHT_MODULES)}"
            )
        if "mlp" in modules:
            if self.intermediate_size is None:
                raise ValueError(
                    "buffer geometry for MLP requires intermediate_size"
                )
            intermediate_size = int(self.intermediate_size)
            if intermediate_size <= 0:
                raise ValueError(
                    "intermediate_size must be positive, got "
                    f"{intermediate_size}"
                )
            object.__setattr__(self, "intermediate_size", intermediate_size)
        if "qkv" in modules:
            if self.qkv_output_size is None:
                raise ValueError(
                    "buffer geometry for QKV requires qkv_output_size"
                )
            qkv_output_size = int(self.qkv_output_size)
            if qkv_output_size <= 0:
                raise ValueError(
                    f"qkv_output_size must be positive, got {qkv_output_size}"
                )
            object.__setattr__(self, "qkv_output_size", qkv_output_size)
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "dtype_bytes", dtype_bytes)
        object.__setattr__(self, "prefetch_buffer_slots", prefetch_buffer_slots)
        object.__setattr__(self, "max_num_batched_tokens", max_num_batched_tokens)
        object.__setattr__(self, "modules", modules)

    @property
    def prefetch_buffer_bytes_per_store_fraction(self) -> int:
        """Full-store prefetch pool bytes for the enabled COTS modules."""

        hidden = self.hidden_size
        unique_shape_numel = 0
        if "qkv" in self.modules:
            assert self.qkv_output_size is not None
            unique_shape_numel += self.qkv_output_size * hidden
        if "mlp" in self.modules:
            assert self.intermediate_size is not None
            intermediate = self.intermediate_size
            unique_shape_numel += (2 * intermediate) * hidden
            unique_shape_numel += intermediate * hidden
        if "wo" in self.modules:
            unique_shape_numel += hidden * hidden
        return unique_shape_numel * self.prefetch_buffer_slots * self.dtype_bytes

    @property
    def output_scratch_bytes_per_store_fraction(self) -> int:
        """Full-store GPU output scratch bytes for CPU-compute returns."""

        if self.max_num_batched_tokens == 0:
            return 0
        output_dims: list[int] = []
        if "qkv" in self.modules:
            assert self.qkv_output_size is not None
            output_dims.append(self.qkv_output_size)
        if "mlp" in self.modules:
            assert self.intermediate_size is not None
            output_dims.append(2 * self.intermediate_size)
            output_dims.append(self.hidden_size)
        if "wo" in self.modules:
            output_dims.append(self.hidden_size)
        max_output_dim = max(output_dims, default=0)
        return self.max_num_batched_tokens * max_output_dim * self.dtype_bytes

    @property
    def gpu_buffer_bytes_per_store_fraction(self) -> int:
        """Combined full-store workspace coefficient used by option A."""

        return (
            self.prefetch_buffer_bytes_per_store_fraction
            + self.output_scratch_bytes_per_store_fraction
        )


@dataclass(frozen=True)
class WeightKVCandidateScore:
    """Per-engine weight/KV placement cost after optimizing dispatch.

    This is the current Stage-1 placement-cost artifact. It scores a fixed
    per-engine placement candidate using the resource-lane model and bucket
    distribution. Today the only implemented static placement variable is
    weight `f_cpu_store`; KV terms will extend the same score object rather
    than changing the Stage-1/Stage-2 split. The embedded splits are the argmin
    realization used later by the Dispatch Compiler.
    """

    f_cpu_store: float
    expected_s: float
    bucket_weights: Mapping[int, float]
    per_bucket_s: Mapping[int, float]
    splits: Mapping[int, WeightDispatchSplit]
    peak_prefetch_fraction: float
    resources: WeightKVResourceEstimate = field(
        default_factory=WeightKVResourceEstimate
    )

    @property
    def dispatch_table(self) -> dict[int, tuple[float, float]]:
        return weight_dispatch_table_from_splits(self.splits)

    @property
    def gpu_bytes(self) -> int:
        return self.resources.gpu_bytes

    @property
    def cpu_bytes(self) -> int:
        return self.resources.cpu_bytes


def _weight_kv_candidate_dominates(
    left: WeightKVCandidateScore,
    right: WeightKVCandidateScore,
) -> bool:
    if not (left.resources.resource_known and right.resources.resource_known):
        return False
    resource_and_cost_le = (
        left.gpu_bytes <= right.gpu_bytes
        and left.cpu_bytes <= right.cpu_bytes
        and left.expected_s <= right.expected_s + 1e-12
        and left.resources.gpu_kv_bytes >= right.resources.gpu_kv_bytes
    )
    any_strict = (
        left.gpu_bytes < right.gpu_bytes
        or left.cpu_bytes < right.cpu_bytes
        or left.expected_s < right.expected_s - 1e-12
        or left.resources.gpu_kv_bytes > right.resources.gpu_kv_bytes
    )
    return resource_and_cost_le and any_strict


def pareto_prune_weight_kv_candidates(
    candidates: Sequence[WeightKVCandidateScore],
) -> tuple[WeightKVCandidateScore, ...]:
    """Keep only candidates not dominated in GPU bytes, CPU bytes, and cost."""

    candidate_tuple = tuple(candidates)
    frontier = []
    for idx, candidate in enumerate(candidate_tuple):
        dominated = any(
            other_idx != idx and _weight_kv_candidate_dominates(other, candidate)
            for other_idx, other in enumerate(candidate_tuple)
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)


@dataclass(frozen=True)
class WeightKVPlacementFrontier:
    """Pareto frontier of useful per-engine weight/KV placement candidates."""

    candidates: tuple[WeightKVCandidateScore, ...]

    @classmethod
    def from_candidates(
        cls,
        candidates: Sequence[WeightKVCandidateScore],
    ) -> "WeightKVPlacementFrontier":
        return cls(pareto_prune_weight_kv_candidates(candidates))

    @property
    def best(self) -> WeightKVCandidateScore:
        if not self.candidates:
            raise ValueError("WeightKVPlacementFrontier is empty")
        return min(
            self.candidates,
            key=lambda candidate: (
                candidate.expected_s,
                candidate.gpu_bytes,
                candidate.cpu_bytes,
                candidate.f_cpu_store,
            ),
        )


@dataclass(frozen=True)
class WeightKVPartitionResult:
    """WeightKVPartitioner result over feasible per-engine candidates."""

    best: WeightKVCandidateScore
    candidates: tuple[WeightKVCandidateScore, ...]
    frontier: WeightKVPlacementFrontier


@dataclass(frozen=True)
class DispatchCompiler:
    """Stage-2 compiler: materialize runtime dispatch tables.

    The compiler assumes static placement is already fixed. It does not choose
    memory placement; it compiles the selected placement into runtime policy
    rows. In the current weight-only prototype this maps `(bucket,
    f_cpu_store)` to `(f_cpu_compute, f_prefetch_compute)`.
    """

    profile: WeightDispatchCostProfile
    candidate_ratios: Sequence[float] | None = None
    candidate_ratio_step: float = 0.125

    def compile_split(
        self,
        *,
        bucket: int,
        f_cpu_store: float,
        include_k: bool = False,
    ) -> WeightDispatchSplit:
        return solve_weight_dispatch_split(
            self.profile.cost_for_bucket(bucket),
            f_cpu_store=f_cpu_store,
            candidate_ratios=self.candidate_ratios,
            candidate_ratio_step=self.candidate_ratio_step,
            include_k=include_k,
        )

    def compile_table(
        self,
        *,
        buckets: Sequence[int],
        f_cpu_store: float,
        include_k: bool = False,
    ) -> dict[int, WeightDispatchSplit]:
        return solve_weight_dispatch_table(
            self.profile,
            buckets=buckets,
            f_cpu_store=f_cpu_store,
            candidate_ratios=self.candidate_ratios,
            candidate_ratio_step=self.candidate_ratio_step,
            include_k=include_k,
        )

    def compile_runtime_table(
        self,
        *,
        buckets: Sequence[int],
        f_cpu_store: float,
    ) -> dict[int, tuple[float, float]]:
        return weight_dispatch_table_from_splits(
            self.compile_table(buckets=buckets, f_cpu_store=f_cpu_store)
        )


@dataclass(frozen=True)
class EngineMemoryBudget:
    """Per-engine memory budget assigned by the model-memory partitioner."""

    gpu_bytes: int
    cpu_bytes: int | None = None

    def __post_init__(self) -> None:
        gpu_bytes = int(self.gpu_bytes)
        if gpu_bytes < 0:
            raise ValueError(f"EngineMemoryBudget.gpu_bytes must be non-negative")
        object.__setattr__(self, "gpu_bytes", gpu_bytes)
        if self.cpu_bytes is not None:
            cpu_bytes = int(self.cpu_bytes)
            if cpu_bytes < 0:
                raise ValueError(
                    "EngineMemoryBudget.cpu_bytes must be non-negative"
                )
            object.__setattr__(self, "cpu_bytes", cpu_bytes)


@dataclass(frozen=True)
class ModelMemorySplit:
    """One generator/verifier budget split under shared global budgets."""

    generator: EngineMemoryBudget
    verifier: EngineMemoryBudget

    @property
    def gpu_bytes(self) -> int:
        return self.generator.gpu_bytes + self.verifier.gpu_bytes

    @property
    def cpu_bytes(self) -> int | None:
        if self.generator.cpu_bytes is None or self.verifier.cpu_bytes is None:
            return None
        return self.generator.cpu_bytes + self.verifier.cpu_bytes


@dataclass(frozen=True)
class ModelMemoryPartitioner:
    """Stage-0 combiner for generator/verifier placement frontiers.

    The current implementation solves the two-engine FastTTS case exactly by
    enumerating generator/verifier engine-budget splits. Each per-engine
    `WeightKVPartitioner` derives weight/KV placement under its assigned budget.
    """

    gpu_budget_bytes: int
    cpu_budget_bytes: int
    engine_weights: Mapping[EngineRole, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("gpu_budget_bytes", "cpu_budget_bytes"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "engine_weights",
            _normalize_engine_weights(self.engine_weights),
        )

    def solve(
        self,
        *,
        generator_frontier: WeightKVPlacementFrontier,
        verifier_frontier: WeightKVPlacementFrontier,
    ) -> "ModelMemoryPartitionResult":
        return solve_model_memory_partition(
            generator_frontier=generator_frontier,
            verifier_frontier=verifier_frontier,
            gpu_budget_bytes=self.gpu_budget_bytes,
            cpu_budget_bytes=self.cpu_budget_bytes,
            engine_weights=self.engine_weights,
        )

    def candidate_engine_budgets(
        self,
        *,
        min_gpu_breakpoints: Sequence[int],
        engine_gpu_budget_step_bytes: int,
    ) -> tuple[EngineMemoryBudget, ...]:
        """Generate meaningful per-engine GPU budgets from Stage-1 breakpoints."""

        step = int(engine_gpu_budget_step_bytes)
        if step <= 0:
            raise ValueError(
                "engine_gpu_budget_step_bytes must be positive, got "
                f"{engine_gpu_budget_step_bytes}"
            )
        breakpoints = tuple(
            sorted(
                {
                    int(value)
                    for value in min_gpu_breakpoints
                    if 0 <= int(value) <= self.gpu_budget_bytes
                }
            )
        )
        if not breakpoints:
            return ()
        candidates = set(breakpoints)
        budget = min(breakpoints)
        while budget <= self.gpu_budget_bytes:
            candidates.add(budget)
            budget += step
        candidates.add(self.gpu_budget_bytes)
        return tuple(
            EngineMemoryBudget(gpu_bytes=value)
            for value in sorted(
                candidate
                for candidate in candidates
                if candidate <= self.gpu_budget_bytes
            )
        )

    def candidate_splits(
        self,
        *,
        generator_budgets: Sequence[EngineMemoryBudget],
        verifier_budgets: Sequence[EngineMemoryBudget],
    ) -> tuple[ModelMemorySplit, ...]:
        """Enumerate generator/verifier budget splits under global budgets."""

        splits: list[ModelMemorySplit] = []
        for generator_budget in generator_budgets:
            for verifier_budget in verifier_budgets:
                split = ModelMemorySplit(
                    generator=generator_budget,
                    verifier=verifier_budget,
                )
                if split.gpu_bytes > self.gpu_budget_bytes:
                    continue
                cpu_bytes = split.cpu_bytes
                if cpu_bytes is not None and cpu_bytes > self.cpu_budget_bytes:
                    continue
                splits.append(split)
        return tuple(
            sorted(
                splits,
                key=lambda split: (
                    split.gpu_bytes,
                    split.generator.gpu_bytes,
                    split.verifier.gpu_bytes,
                ),
            )
        )

    def solve_from_partitioners(
        self,
        *,
        generator_partitioner: "WeightKVPartitioner",
        verifier_partitioner: "WeightKVPartitioner",
        generator_f_cpu_store_candidates: Sequence[float],
        verifier_f_cpu_store_candidates: Sequence[float],
        engine_gpu_budget_step_bytes: int,
        generator_engine_gpu_budget_candidates: Sequence[int] | None = None,
        verifier_engine_gpu_budget_candidates: Sequence[int] | None = None,
    ) -> "ModelMemoryPartitionResult":
        """Assign engine budgets, then solve per-engine weight/KV placement."""

        generator_budgets = _engine_memory_budgets_from_candidates(
            generator_engine_gpu_budget_candidates
        )
        if generator_budgets is None:
            generator_budgets = self.candidate_engine_budgets(
                min_gpu_breakpoints=generator_partitioner.min_gpu_budget_breakpoints(
                    generator_f_cpu_store_candidates
                ),
                engine_gpu_budget_step_bytes=engine_gpu_budget_step_bytes,
            )
        verifier_budgets = _engine_memory_budgets_from_candidates(
            verifier_engine_gpu_budget_candidates
        )
        if verifier_budgets is None:
            verifier_budgets = self.candidate_engine_budgets(
                min_gpu_breakpoints=verifier_partitioner.min_gpu_budget_breakpoints(
                    verifier_f_cpu_store_candidates
                ),
                engine_gpu_budget_step_bytes=engine_gpu_budget_step_bytes,
            )

        splits = self.candidate_splits(
            generator_budgets=generator_budgets,
            verifier_budgets=verifier_budgets,
        )
        if not splits:
            raise ValueError(
                "no feasible generator/verifier budget split fits "
                f"gpu_budget_bytes={self.gpu_budget_bytes}, "
                f"cpu_budget_bytes={self.cpu_budget_bytes}"
            )

        generator_frontier_cache: dict[int, WeightKVPlacementFrontier] = {}
        verifier_frontier_cache: dict[int, WeightKVPlacementFrontier] = {}
        feasible: list[ModelMemoryCandidate] = []
        for split in splits:
            try:
                if split.generator.gpu_bytes not in generator_frontier_cache:
                    generator_frontier_cache[split.generator.gpu_bytes] = (
                        generator_partitioner.frontier_for_engine_budget(
                            split.generator,
                            f_cpu_store_candidates=(
                                generator_f_cpu_store_candidates
                            ),
                        )
                    )
                generator_frontier = generator_frontier_cache[
                    split.generator.gpu_bytes
                ]
                if split.verifier.gpu_bytes not in verifier_frontier_cache:
                    verifier_frontier_cache[split.verifier.gpu_bytes] = (
                        verifier_partitioner.frontier_for_engine_budget(
                            split.verifier,
                            f_cpu_store_candidates=(
                                verifier_f_cpu_store_candidates
                            ),
                        )
                    )
                verifier_frontier = verifier_frontier_cache[
                    split.verifier.gpu_bytes
                ]
            except ValueError as exc:
                if "no feasible weight/KV placement candidates remain" in str(exc):
                    continue
                raise
            try:
                split_result = solve_model_memory_partition(
                    generator_frontier=generator_frontier,
                    verifier_frontier=verifier_frontier,
                    gpu_budget_bytes=self.gpu_budget_bytes,
                    cpu_budget_bytes=self.cpu_budget_bytes,
                    engine_weights=self.engine_weights,
                )
            except ValueError as exc:
                if "no feasible generator/verifier placement pair fits" in str(exc):
                    continue
                raise
            feasible.extend(split_result.candidates)

        if not feasible:
            raise ValueError(
                "no feasible generator/verifier placement pair fits "
                f"gpu_budget_bytes={self.gpu_budget_bytes}, "
                f"cpu_budget_bytes={self.cpu_budget_bytes}"
            )
        candidates = tuple(feasible)
        best = min(candidates, key=_model_memory_candidate_sort_key)
        return ModelMemoryPartitionResult(best=best, candidates=candidates)


@dataclass(frozen=True)
class ModelMemoryCandidate:
    """One feasible generator/verifier placement pair under shared budgets."""

    generator: WeightKVCandidateScore
    verifier: WeightKVCandidateScore
    gpu_bytes: int
    cpu_bytes: int
    objective_s: float
    engine_weights: Mapping[EngineRole, float]

    @property
    def placements(self) -> dict[EngineRole, WeightKVCandidateScore]:
        return {
            "generator": self.generator,
            "verifier": self.verifier,
        }


@dataclass(frozen=True)
class ModelMemoryPartitionResult:
    """Best feasible pair plus all feasible generator/verifier pairs."""

    best: ModelMemoryCandidate
    candidates: tuple[ModelMemoryCandidate, ...]


def _normalize_engine_weights(
    raw: Mapping[EngineRole, float] | None,
) -> dict[EngineRole, float]:
    weights: dict[EngineRole, float] = {
        "generator": 1.0,
        "verifier": 1.0,
    }
    if raw is None:
        return weights
    for role, value in raw.items():
        if role not in weights:
            raise ValueError(f"unsupported engine role for weight: {role!r}")
        parsed = float(value)
        if parsed < 0:
            raise ValueError(f"engine weight for {role} must be non-negative")
        weights[role] = parsed
    return weights


def _engine_memory_budgets_from_candidates(
    raw: Sequence[int] | None,
) -> tuple[EngineMemoryBudget, ...] | None:
    values = _normalize_nonnegative_int_sequence(
        raw,
        field_name="engine_gpu_budget_candidates",
    )
    if values is None:
        return None
    return tuple(EngineMemoryBudget(gpu_bytes=value) for value in values)


def _validate_frontier_for_model_memory(
    role: EngineRole,
    frontier: WeightKVPlacementFrontier,
) -> None:
    if not frontier.candidates:
        raise ValueError(f"{role} frontier must not be empty")
    unknown = [
        candidate.f_cpu_store
        for candidate in frontier.candidates
        if not candidate.resources.resource_known
    ]
    if unknown:
        raise ValueError(
            f"{role} frontier has candidates without resource estimates: {unknown}"
        )


def _score_model_memory_pair(
    *,
    generator: WeightKVCandidateScore,
    verifier: WeightKVCandidateScore,
    engine_weights: Mapping[EngineRole, float],
) -> float:
    return (
        float(engine_weights["generator"]) * generator.expected_s
        + float(engine_weights["verifier"]) * verifier.expected_s
    )


def _model_memory_candidate_sort_key(
    candidate: ModelMemoryCandidate,
) -> tuple[float, int, int, int, int, float, float]:
    generator_kv = candidate.generator.resources.gpu_kv_bytes
    verifier_kv = candidate.verifier.resources.gpu_kv_bytes
    return (
        candidate.objective_s,
        -min(generator_kv, verifier_kv),
        -candidate.gpu_bytes,
        abs(generator_kv - verifier_kv),
        candidate.cpu_bytes,
        candidate.generator.f_cpu_store,
        candidate.verifier.f_cpu_store,
    )


def solve_model_memory_partition(
    *,
    generator_frontier: WeightKVPlacementFrontier,
    verifier_frontier: WeightKVPlacementFrontier,
    gpu_budget_bytes: int,
    cpu_budget_bytes: int,
    engine_weights: Mapping[EngineRole, float] | None = None,
) -> ModelMemoryPartitionResult:
    """Combine generator/verifier frontiers under shared GPU and CPU budgets."""

    gpu_budget = int(gpu_budget_bytes)
    cpu_budget = int(cpu_budget_bytes)
    if gpu_budget < 0:
        raise ValueError(f"gpu_budget_bytes must be non-negative, got {gpu_budget}")
    if cpu_budget < 0:
        raise ValueError(f"cpu_budget_bytes must be non-negative, got {cpu_budget}")

    weights = _normalize_engine_weights(engine_weights)
    _validate_frontier_for_model_memory("generator", generator_frontier)
    _validate_frontier_for_model_memory("verifier", verifier_frontier)

    feasible: list[ModelMemoryCandidate] = []
    for generator in generator_frontier.candidates:
        for verifier in verifier_frontier.candidates:
            gpu_bytes = generator.gpu_bytes + verifier.gpu_bytes
            cpu_bytes = generator.cpu_bytes + verifier.cpu_bytes
            if gpu_bytes > gpu_budget or cpu_bytes > cpu_budget:
                continue
            feasible.append(
                ModelMemoryCandidate(
                    generator=generator,
                    verifier=verifier,
                    gpu_bytes=gpu_bytes,
                    cpu_bytes=cpu_bytes,
                    objective_s=_score_model_memory_pair(
                        generator=generator,
                        verifier=verifier,
                        engine_weights=weights,
                    ),
                    engine_weights=weights,
                )
            )

    if not feasible:
        raise ValueError(
            "no feasible generator/verifier placement pair fits "
            f"gpu_budget_bytes={gpu_budget}, cpu_budget_bytes={cpu_budget}"
        )

    candidates = tuple(feasible)
    best = min(candidates, key=_model_memory_candidate_sort_key)
    return ModelMemoryPartitionResult(best=best, candidates=candidates)


@dataclass(frozen=True)
class WeightKVPartitioner:
    """Stage-1 per-engine partitioner: choose static placement candidates.

    This is the current prototype of the thesis weight/KV partitioner. Its
    profile inputs are weight-only today, but the abstraction is the general
    per-engine contract: score feasible static candidates with a placement cost
    model that internally optimizes dispatch before comparing placements.
    """

    profile: WeightDispatchCostProfile
    buckets: Sequence[int]
    bucket_weights: Mapping[int, float] | None = None
    candidate_ratios: Sequence[float] | None = None
    candidate_ratio_step: float = 0.125
    require_k: bool = True
    total_weight_bytes: int | None = None
    cpu_weight_bytes_by_store_fraction: Mapping[float, int] | None = None
    gpu_kv_bytes: int | None = None
    cpu_kv_bytes: int = 0
    gpu_buffer_bytes: int | None = None
    gpu_buffer_bytes_per_store_fraction: int = 0
    gpu_buffer_bytes_by_store_fraction: Mapping[float, int] | None = None
    engine_gpu_budget_bytes: int | None = None
    engine_gpu_budget_candidates: Sequence[int] | None = None

    def min_gpu_budget_breakpoints(
        self,
        f_cpu_store_candidates: Sequence[float],
    ) -> tuple[int, ...]:
        """Minimum assigned GPU budget needed for each storage candidate.

        These are Stage-1 facts exposed upward to Stage 0. They include
        resident GPU weights and COTS GPU buffers, but no KV assignment.
        """

        candidates = _normalize_fraction_sequence(
            f_cpu_store_candidates,
            field_name="f_cpu_store_candidates",
        )
        if not candidates:
            raise ValueError("f_cpu_store_candidates must not be empty")
        if self.total_weight_bytes is None:
            raise ValueError("min_gpu_budget_breakpoints requires total_weight_bytes")
        total_weight = int(self.total_weight_bytes)
        if total_weight < 0:
            raise ValueError(
                f"total_weight_bytes must be non-negative, got {total_weight}"
            )

        breakpoints: set[int] = set()
        for s in candidates:
            cpu_weight_bytes = derive_cpu_weight_bytes(
                f_cpu_store=s,
                total_weight_bytes=total_weight,
                cpu_weight_bytes_by_store_fraction=(
                    self.cpu_weight_bytes_by_store_fraction
                ),
            )
            gpu_weight_bytes = total_weight - cpu_weight_bytes
            gpu_buffer_bytes = derive_gpu_buffer_bytes(
                f_cpu_store=s,
                gpu_buffer_bytes=self.gpu_buffer_bytes,
                gpu_buffer_bytes_per_store_fraction=(
                    self.gpu_buffer_bytes_per_store_fraction
                ),
                gpu_buffer_bytes_by_store_fraction=(
                    self.gpu_buffer_bytes_by_store_fraction
                ),
            )
            breakpoints.add(gpu_weight_bytes + gpu_buffer_bytes)
        return tuple(sorted(breakpoints))

    def frontier_for_engine_budget(
        self,
        budget: EngineMemoryBudget,
        *,
        f_cpu_store_candidates: Sequence[float],
    ) -> WeightKVPlacementFrontier:
        """Solve weight/KV placement under one assigned engine budget."""

        if self.engine_gpu_budget_candidates is not None:
            raise ValueError(
                "frontier_for_engine_budget expects the assigned budget "
                "argument, not engine_gpu_budget_candidates on the partitioner"
            )
        return build_weight_kv_placement_frontier(
            self.profile,
            buckets=self.buckets,
            f_cpu_store_candidates=f_cpu_store_candidates,
            bucket_weights=self.bucket_weights,
            candidate_ratios=self.candidate_ratios,
            candidate_ratio_step=self.candidate_ratio_step,
            require_k=self.require_k,
            total_weight_bytes=self.total_weight_bytes,
            cpu_weight_bytes_by_store_fraction=(
                self.cpu_weight_bytes_by_store_fraction
            ),
            gpu_kv_bytes=self.gpu_kv_bytes,
            cpu_kv_bytes=self.cpu_kv_bytes,
            gpu_buffer_bytes=self.gpu_buffer_bytes,
            gpu_buffer_bytes_per_store_fraction=(
                self.gpu_buffer_bytes_per_store_fraction
            ),
            gpu_buffer_bytes_by_store_fraction=(
                self.gpu_buffer_bytes_by_store_fraction
            ),
            engine_gpu_budget_bytes=budget.gpu_bytes,
        )

    def score(self, *, f_cpu_store: float) -> WeightKVCandidateScore:
        if self.engine_gpu_budget_candidates is not None:
            raise ValueError(
                "WeightKVPartitioner.score requires engine_gpu_budget_bytes, "
                "not engine_gpu_budget_candidates"
            )
        return score_weight_kv_candidate(
            self.profile,
            buckets=self.buckets,
            f_cpu_store=f_cpu_store,
            bucket_weights=self.bucket_weights,
            candidate_ratios=self.candidate_ratios,
            candidate_ratio_step=self.candidate_ratio_step,
            require_k=self.require_k,
            total_weight_bytes=self.total_weight_bytes,
            cpu_weight_bytes_by_store_fraction=(
                self.cpu_weight_bytes_by_store_fraction
            ),
            gpu_kv_bytes=self.gpu_kv_bytes,
            cpu_kv_bytes=self.cpu_kv_bytes,
            gpu_buffer_bytes=self.gpu_buffer_bytes,
            gpu_buffer_bytes_per_store_fraction=(
                self.gpu_buffer_bytes_per_store_fraction
            ),
            gpu_buffer_bytes_by_store_fraction=(
                self.gpu_buffer_bytes_by_store_fraction
            ),
            engine_gpu_budget_bytes=self.engine_gpu_budget_bytes,
        )

    def frontier(
        self,
        *,
        f_cpu_store_candidates: Sequence[float],
    ) -> WeightKVPlacementFrontier:
        return build_weight_kv_placement_frontier(
            self.profile,
            buckets=self.buckets,
            f_cpu_store_candidates=f_cpu_store_candidates,
            bucket_weights=self.bucket_weights,
            candidate_ratios=self.candidate_ratios,
            candidate_ratio_step=self.candidate_ratio_step,
            require_k=self.require_k,
            total_weight_bytes=self.total_weight_bytes,
            cpu_weight_bytes_by_store_fraction=(
                self.cpu_weight_bytes_by_store_fraction
            ),
            gpu_kv_bytes=self.gpu_kv_bytes,
            cpu_kv_bytes=self.cpu_kv_bytes,
            gpu_buffer_bytes=self.gpu_buffer_bytes,
            gpu_buffer_bytes_per_store_fraction=(
                self.gpu_buffer_bytes_per_store_fraction
            ),
            gpu_buffer_bytes_by_store_fraction=(
                self.gpu_buffer_bytes_by_store_fraction
            ),
            engine_gpu_budget_bytes=self.engine_gpu_budget_bytes,
            engine_gpu_budget_candidates=self.engine_gpu_budget_candidates,
        )

    def solve(
        self,
        *,
        f_cpu_store_candidates: Sequence[float],
    ) -> WeightKVPartitionResult:
        return solve_weight_kv_partition(
            self.profile,
            buckets=self.buckets,
            f_cpu_store_candidates=f_cpu_store_candidates,
            bucket_weights=self.bucket_weights,
            candidate_ratios=self.candidate_ratios,
            candidate_ratio_step=self.candidate_ratio_step,
            require_k=self.require_k,
            total_weight_bytes=self.total_weight_bytes,
            cpu_weight_bytes_by_store_fraction=(
                self.cpu_weight_bytes_by_store_fraction
            ),
            gpu_kv_bytes=self.gpu_kv_bytes,
            cpu_kv_bytes=self.cpu_kv_bytes,
            gpu_buffer_bytes=self.gpu_buffer_bytes,
            gpu_buffer_bytes_per_store_fraction=(
                self.gpu_buffer_bytes_per_store_fraction
            ),
            gpu_buffer_bytes_by_store_fraction=(
                self.gpu_buffer_bytes_by_store_fraction
            ),
            engine_gpu_budget_bytes=self.engine_gpu_budget_bytes,
            engine_gpu_budget_candidates=self.engine_gpu_budget_candidates,
        )


def score_weight_kv_candidate(
    profile: WeightDispatchCostProfile,
    *,
    buckets: Sequence[int],
    f_cpu_store: float,
    bucket_weights: Mapping[int, float] | None = None,
    candidate_ratios: Sequence[float] | None = None,
    candidate_ratio_step: float = 0.125,
    require_k: bool = True,
    total_weight_bytes: int | None = None,
    cpu_weight_bytes_by_store_fraction: Mapping[float, int] | None = None,
    gpu_kv_bytes: int | None = None,
    cpu_kv_bytes: int = 0,
    gpu_buffer_bytes: int | None = None,
    gpu_buffer_bytes_per_store_fraction: int = 0,
    gpu_buffer_bytes_by_store_fraction: Mapping[float, int] | None = None,
    engine_gpu_budget_bytes: int | None = None,
) -> WeightKVCandidateScore:
    """Score a fixed per-engine weight/KV placement candidate.

    Stage 1 compares different static placements, so `K(B,s)` matters here.
    `require_k=True` keeps that comparison honest: every profiled bucket must
    have a split-invariant term for the candidate `s`. Callers can disable it
    only for resource-only diagnostics.
    """

    s = _validate_fraction("f_cpu_store", f_cpu_store)
    bucket_ids = tuple(int(bucket) for bucket in buckets)
    weights = _normalize_bucket_weights(bucket_ids, bucket_weights)
    splits = solve_weight_dispatch_table(
        profile,
        buckets=bucket_ids,
        f_cpu_store=s,
        candidate_ratios=candidate_ratios,
        candidate_ratio_step=candidate_ratio_step,
        include_k=True,
    )
    missing_k = sorted(
        bucket for bucket, split in splits.items() if split.k_s is None
    )
    if require_k and missing_k:
        raise ValueError(
            "placement scoring requires K(B,s) for every bucket; "
            f"missing buckets={missing_k}, f_cpu_store={s}"
        )
    per_bucket_s = {
        bucket: split.predicted_s for bucket, split in sorted(splits.items())
    }
    expected_s = sum(weights[bucket] * per_bucket_s[bucket] for bucket in bucket_ids)
    peak_prefetch = max((split.f_prefetch for split in splits.values()), default=0.0)
    return WeightKVCandidateScore(
        f_cpu_store=s,
        expected_s=expected_s,
        bucket_weights=weights,
        per_bucket_s=per_bucket_s,
        splits=splits,
        peak_prefetch_fraction=peak_prefetch,
        resources=estimate_weight_kv_resources(
            f_cpu_store=s,
            total_weight_bytes=total_weight_bytes,
            cpu_weight_bytes_by_store_fraction=(
                cpu_weight_bytes_by_store_fraction
            ),
            gpu_kv_bytes=gpu_kv_bytes,
            cpu_kv_bytes=cpu_kv_bytes,
            gpu_buffer_bytes=gpu_buffer_bytes,
            gpu_buffer_bytes_per_store_fraction=(
                gpu_buffer_bytes_per_store_fraction
            ),
            gpu_buffer_bytes_by_store_fraction=(
                gpu_buffer_bytes_by_store_fraction
            ),
            engine_gpu_budget_bytes=engine_gpu_budget_bytes,
        ),
    )


def _score_weight_kv_candidates(
    profile: WeightDispatchCostProfile,
    *,
    buckets: Sequence[int],
    f_cpu_store_candidates: Sequence[float],
    bucket_weights: Mapping[int, float] | None = None,
    candidate_ratios: Sequence[float] | None = None,
    candidate_ratio_step: float = 0.125,
    require_k: bool = True,
    total_weight_bytes: int | None = None,
    cpu_weight_bytes_by_store_fraction: Mapping[float, int] | None = None,
    gpu_kv_bytes: int | None = None,
    cpu_kv_bytes: int = 0,
    gpu_buffer_bytes: int | None = None,
    gpu_buffer_bytes_per_store_fraction: int = 0,
    gpu_buffer_bytes_by_store_fraction: Mapping[float, int] | None = None,
    engine_gpu_budget_bytes: int | None = None,
    engine_gpu_budget_candidates: Sequence[int] | None = None,
) -> tuple[WeightKVCandidateScore, ...]:
    candidates = _normalize_fraction_sequence(
        f_cpu_store_candidates,
        field_name="f_cpu_store_candidates",
    )
    if not candidates:
        raise ValueError("f_cpu_store_candidates must not be empty")
    if (
        engine_gpu_budget_bytes is not None
        and engine_gpu_budget_candidates is not None
    ):
        raise ValueError(
            "provide engine_gpu_budget_bytes or "
            "engine_gpu_budget_candidates, not both"
        )
    budgets = _normalize_nonnegative_int_sequence(
        engine_gpu_budget_candidates,
        field_name="engine_gpu_budget_candidates",
    )
    if budgets is None:
        budgets = (
            None
            if engine_gpu_budget_bytes is None
            else int(engine_gpu_budget_bytes),
        )

    scored: list[WeightKVCandidateScore] = []
    for budget in budgets:
        for s in candidates:
            try:
                scored.append(
                    score_weight_kv_candidate(
                        profile,
                        buckets=buckets,
                        f_cpu_store=s,
                        bucket_weights=bucket_weights,
                        candidate_ratios=candidate_ratios,
                        candidate_ratio_step=candidate_ratio_step,
                        require_k=require_k,
                        total_weight_bytes=total_weight_bytes,
                        cpu_weight_bytes_by_store_fraction=(
                            cpu_weight_bytes_by_store_fraction
                        ),
                        gpu_kv_bytes=gpu_kv_bytes,
                        cpu_kv_bytes=cpu_kv_bytes,
                        gpu_buffer_bytes=gpu_buffer_bytes,
                        gpu_buffer_bytes_per_store_fraction=(
                            gpu_buffer_bytes_per_store_fraction
                        ),
                        gpu_buffer_bytes_by_store_fraction=(
                            gpu_buffer_bytes_by_store_fraction
                        ),
                        engine_gpu_budget_bytes=budget,
                    )
                )
            except _InfeasibleWeightKVPlacement:
                continue
    if not scored:
        raise ValueError("no feasible weight/KV placement candidates remain")
    return tuple(scored)


def build_weight_kv_placement_frontier(
    profile: WeightDispatchCostProfile,
    *,
    buckets: Sequence[int],
    f_cpu_store_candidates: Sequence[float],
    bucket_weights: Mapping[int, float] | None = None,
    candidate_ratios: Sequence[float] | None = None,
    candidate_ratio_step: float = 0.125,
    require_k: bool = True,
    total_weight_bytes: int | None = None,
    cpu_weight_bytes_by_store_fraction: Mapping[float, int] | None = None,
    gpu_kv_bytes: int | None = None,
    cpu_kv_bytes: int = 0,
    gpu_buffer_bytes: int | None = None,
    gpu_buffer_bytes_per_store_fraction: int = 0,
    gpu_buffer_bytes_by_store_fraction: Mapping[float, int] | None = None,
    engine_gpu_budget_bytes: int | None = None,
    engine_gpu_budget_candidates: Sequence[int] | None = None,
) -> WeightKVPlacementFrontier:
    """Build the Pareto frontier for feasible per-engine placement candidates."""

    scores = _score_weight_kv_candidates(
        profile,
        buckets=buckets,
        f_cpu_store_candidates=f_cpu_store_candidates,
        bucket_weights=bucket_weights,
        candidate_ratios=candidate_ratios,
        candidate_ratio_step=candidate_ratio_step,
        require_k=require_k,
        total_weight_bytes=total_weight_bytes,
        cpu_weight_bytes_by_store_fraction=cpu_weight_bytes_by_store_fraction,
        gpu_kv_bytes=gpu_kv_bytes,
        cpu_kv_bytes=cpu_kv_bytes,
        gpu_buffer_bytes=gpu_buffer_bytes,
        gpu_buffer_bytes_per_store_fraction=gpu_buffer_bytes_per_store_fraction,
        gpu_buffer_bytes_by_store_fraction=gpu_buffer_bytes_by_store_fraction,
        engine_gpu_budget_bytes=engine_gpu_budget_bytes,
        engine_gpu_budget_candidates=engine_gpu_budget_candidates,
    )
    return WeightKVPlacementFrontier.from_candidates(scores)


def solve_weight_kv_partition(
    profile: WeightDispatchCostProfile,
    *,
    buckets: Sequence[int],
    f_cpu_store_candidates: Sequence[float],
    bucket_weights: Mapping[int, float] | None = None,
    candidate_ratios: Sequence[float] | None = None,
    candidate_ratio_step: float = 0.125,
    require_k: bool = True,
    total_weight_bytes: int | None = None,
    cpu_weight_bytes_by_store_fraction: Mapping[float, int] | None = None,
    gpu_kv_bytes: int | None = None,
    cpu_kv_bytes: int = 0,
    gpu_buffer_bytes: int | None = None,
    gpu_buffer_bytes_per_store_fraction: int = 0,
    gpu_buffer_bytes_by_store_fraction: Mapping[float, int] | None = None,
    engine_gpu_budget_bytes: int | None = None,
    engine_gpu_budget_candidates: Sequence[int] | None = None,
) -> WeightKVPartitionResult:
    """Choose the best per-engine static placement from feasible candidates.

    The current candidate set is `f_cpu_store` only; future KV/static memory
    variables should extend this Stage-1 entry point instead of introducing a
    separate runtime decision.
    """

    scores = _score_weight_kv_candidates(
        profile,
        buckets=buckets,
        f_cpu_store_candidates=f_cpu_store_candidates,
        bucket_weights=bucket_weights,
        candidate_ratios=candidate_ratios,
        candidate_ratio_step=candidate_ratio_step,
        require_k=require_k,
        total_weight_bytes=total_weight_bytes,
        cpu_weight_bytes_by_store_fraction=cpu_weight_bytes_by_store_fraction,
        gpu_kv_bytes=gpu_kv_bytes,
        cpu_kv_bytes=cpu_kv_bytes,
        gpu_buffer_bytes=gpu_buffer_bytes,
        gpu_buffer_bytes_per_store_fraction=gpu_buffer_bytes_per_store_fraction,
        gpu_buffer_bytes_by_store_fraction=gpu_buffer_bytes_by_store_fraction,
        engine_gpu_budget_bytes=engine_gpu_budget_bytes,
        engine_gpu_budget_candidates=engine_gpu_budget_candidates,
    )
    frontier = WeightKVPlacementFrontier.from_candidates(scores)
    return WeightKVPartitionResult(
        best=frontier.best,
        candidates=scores,
        frontier=frontier,
    )


def _weight_dispatch_profile_path(
    raw: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    required_for: str,
) -> Any:
    profile_path = raw.get(
        "dispatch_cost_profile_path",
        raw.get(
            "dispatch_profile_path",
            raw.get(
                "weight_dispatch_profile_path",
                defaults.get("cots_dispatch_cost_profile_path"),
            ),
        ),
    )
    if profile_path is None and required_for:
        raise ValueError(f"{required_for} requires weight.dispatch_cost_profile_path")
    return profile_path


def _weight_dispatch_buckets(
    raw: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    required_for: str,
) -> tuple[int, ...] | None:
    buckets = _normalize_bucket_sequence(
        raw.get("dispatch_buckets", defaults.get("cots_dispatch_buckets")),
        field_name="dispatch_buckets",
    )
    if not buckets and required_for:
        raise ValueError(f"{required_for} requires weight.dispatch_buckets")
    return buckets


def _weight_dispatch_candidate_ratios(raw: Mapping[str, Any]) -> tuple[float, ...] | None:
    return _normalize_fraction_sequence(
        raw.get("dispatch_candidate_ratios"),
        field_name="dispatch_candidate_ratios",
    )


def _optional_nonnegative_int_from_keys(
    raw: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    keys: Sequence[str],
    field_name: str,
) -> int | None:
    value = None
    for key in keys:
        if key in raw:
            value = raw[key]
            break
    if value is None:
        for key in keys:
            if key in defaults:
                value = defaults[key]
                break
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative, got {parsed}")
    return parsed


ENGINE_GPU_BUDGET_BYTE_KEYS = (
    "engine_gpu_budget_bytes",
    "gpu_budget_bytes",
    "engine_vram_budget_bytes",
    "cots_engine_gpu_budget_bytes",
    "cots_gpu_budget_bytes",
)

ENGINE_GPU_BUDGET_CANDIDATE_KEYS = (
    "engine_gpu_budget_candidates",
    "engine_gpu_budget_byte_candidates",
    "gpu_budget_candidates",
    "gpu_budget_byte_candidates",
    "engine_gpu_budgets",
)

DEFAULT_MAX_ENGINE_GPU_BUDGET_STEP_BYTES = 1 << 30


def _engine_gpu_budget_bytes_from_config(
    raw: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    field_name: str,
) -> int | None:
    return _optional_nonnegative_int_from_keys(
        raw,
        defaults,
        keys=ENGINE_GPU_BUDGET_BYTE_KEYS,
        field_name=field_name,
    )


def _value_from_keys(
    raw: Mapping[str, Any],
    defaults: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    for key in keys:
        if key in defaults:
            return defaults[key]
    return None


def _mapping_from_keys(
    raw: Mapping[str, Any],
    defaults: Mapping[str, Any],
    keys: Sequence[str],
    *,
    field_name: str,
) -> Mapping[str, Any] | None:
    value = _value_from_keys(raw, defaults, keys)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _engine_gpu_budget_candidates_from_config(
    raw: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    field_name: str,
) -> tuple[int, ...] | None:
    candidate_raw = _value_from_keys(
        raw,
        defaults,
        ENGINE_GPU_BUDGET_CANDIDATE_KEYS,
    )
    candidates = _normalize_nonnegative_int_sequence(
        candidate_raw,
        field_name=field_name,
    )
    singular = _engine_gpu_budget_bytes_from_config(
        raw,
        defaults,
        field_name=field_name,
    )
    if candidates is not None and singular is not None:
        raise ValueError(
            f"{field_name} cannot set both engine_gpu_budget_bytes and "
            "engine_gpu_budget_candidates"
        )
    if candidates is not None:
        return candidates
    if singular is not None:
        return (singular,)
    return None


def _model_memory_engine_budget_step_bytes(
    raw: Mapping[str, Any],
    *,
    gpu_budget_bytes: int,
) -> int:
    step = _optional_budget_bytes_from_mapping(
        raw,
        byte_keys=(
            "engine_gpu_budget_step_bytes",
            "gpu_budget_step_bytes",
            "budget_step_bytes",
        ),
        gib_keys=(
            "engine_gpu_budget_step_gb",
            "gpu_budget_step_gb",
            "budget_step_gb",
        ),
        field_name="global.engine_gpu_budget_step_bytes",
    )
    if step is None:
        step = max(
            1,
            min(
                DEFAULT_MAX_ENGINE_GPU_BUDGET_STEP_BYTES,
                max(1, int(gpu_budget_bytes) // 16),
            ),
        )
    if step <= 0:
        raise ValueError(
            "global.engine_gpu_budget_step_bytes must be positive, got "
            f"{step}"
        )
    return int(step)


def _optional_budget_bytes_from_mapping(
    raw: Mapping[str, Any],
    *,
    byte_keys: Sequence[str],
    gib_keys: Sequence[str],
    field_name: str,
) -> int | None:
    for key in byte_keys:
        if key in raw:
            value = int(raw[key])
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
            return value
    for key in gib_keys:
        if key in raw:
            value = int(float(raw[key]) * (1 << 30))
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
            return value
    return None


BUFFER_GEOMETRY_KEYS = (
    "buffer_geometry",
    "gpu_buffer_geometry",
    "cots_buffer_geometry",
)

BUFFER_GEOMETRY_SIGNAL_KEYS = (
    "hidden_size",
    "model_hidden_size",
    "intermediate_size",
    "mlp_intermediate_size",
    "qkv_output_size",
    "qkv_out_features",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)


def _value_from_sources(
    sources: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> Any:
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
    return None


def _optional_nonnegative_int_from_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    field_name: str,
) -> int | None:
    value = _value_from_sources(sources, keys)
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative, got {parsed}")
    return parsed


def _optional_fraction_int_mapping_from_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    field_name: str,
) -> dict[float, int] | None:
    value = _value_from_sources(sources, keys)
    return _normalize_fraction_int_mapping(value, field_name=field_name)


def _cots_snap_profile_from_metadata(
    metadata: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    raw = _mapping_from_keys(
        metadata,
        {},
        COTS_SNAP_PROFILE_KEYS,
        field_name="profile.metadata.cots_snap",
    )
    if raw is None:
        return None
    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(
            f"unsupported cots_snap schema_version {schema_version}; expected 1"
        )
    snap_model = str(
        raw.get("snap_model", raw.get("realization_model", COTS_SNAP_MODEL))
    )
    if snap_model not in SUPPORTED_COTS_SNAP_MODELS:
        raise ValueError(
            f"unsupported COTS snap model {snap_model!r}; "
            f"expected one of {sorted(SUPPORTED_COTS_SNAP_MODELS)}"
        )
    return raw


def _cots_snap_storage_rows(
    metadata: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    snap = _cots_snap_profile_from_metadata(metadata)
    if snap is None:
        return None
    rows = _value_from_sources(
        (snap,),
        (
            "storage_by_store_fraction",
            "placement_by_store_fraction",
            "store_by_fraction",
        ),
    )
    if rows is None:
        return None
    if not isinstance(rows, Mapping):
        raise ValueError(
            "profile.metadata.cots_snap.storage_by_store_fraction "
            "must be a mapping"
        )
    return rows


def _fraction_int_mapping_from_cots_snap(
    metadata: Mapping[str, Any],
    *,
    value_keys: Sequence[str],
    field_name: str,
) -> dict[float, int] | None:
    rows = _cots_snap_storage_rows(metadata)
    if rows is None:
        return None
    parsed: dict[float, int] = {}
    for fraction, row in rows.items():
        store = _round_fraction(_validate_fraction("f_cpu_store", float(fraction)))
        if not isinstance(row, Mapping):
            raise ValueError(
                f"profile.metadata.cots_snap.storage_by_store_fraction[{fraction!r}] "
                "must be a mapping"
            )
        value = _value_from_sources((row,), value_keys)
        if value is None:
            continue
        byte_value = int(value)
        if byte_value < 0:
            raise ValueError(f"{field_name} values must be non-negative")
        parsed[store] = byte_value
    return dict(sorted(parsed.items())) if parsed else None


def cots_snap_resource_maps_from_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, dict[float, int]]:
    """Extract planner resource maps from a COTS runtime realization profile.

    vLLM COTS owns tensor snapping. The profiler can record those realized
    snapped bytes in `cots_snap`, and the planner consumes them as calibrated
    facts instead of reimplementing the tensor geometry.
    """

    maps: dict[str, dict[float, int]] = {}
    cpu_weight = _fraction_int_mapping_from_cots_snap(
        metadata,
        value_keys=(
            "cpu_weight_bytes",
            "weights_saved_bytes",
            "cots_cpu_weight_bytes",
        ),
        field_name="cpu_weight_bytes_by_store_fraction",
    )
    if cpu_weight is not None:
        maps["cpu_weight_bytes_by_store_fraction"] = cpu_weight
    gpu_buffer = _fraction_int_mapping_from_cots_snap(
        metadata,
        value_keys=(
            "gpu_buffer_bytes",
            "gpu_workspace_bytes",
            "gpu_uva_bytes",
            "cots_gpu_buffer_bytes",
        ),
        field_name="gpu_buffer_bytes_by_store_fraction",
    )
    if gpu_buffer is not None:
        maps["gpu_buffer_bytes_by_store_fraction"] = gpu_buffer
    return maps


def _positive_int_from_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    field_name: str,
) -> int | None:
    value = _value_from_sources(sources, keys)
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive, got {parsed}")
    return parsed


def _dtype_bytes_from_sources(sources: Sequence[Mapping[str, Any]]) -> int:
    dtype_bytes = _positive_int_from_sources(
        sources,
        keys=("dtype_bytes", "weight_dtype_bytes", "cots_dtype_bytes"),
        field_name="buffer_geometry.dtype_bytes",
    )
    if dtype_bytes is not None:
        return dtype_bytes
    dtype = _value_from_sources(
        sources,
        keys=("dtype", "weight_dtype", "cots_weight_dtype", "cpu_dtype"),
    )
    if dtype is None:
        return 2
    normalized = str(dtype).lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16", "float16", "fp16", "half"}:
        return 2
    if normalized in {"float32", "fp32", "single"}:
        return 4
    raise ValueError(f"unsupported buffer geometry dtype: {dtype!r}")


def _normalized_weight_modules_from_config(
    raw: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> frozenset[str]:
    modules = _normalize_weight_modules(
        raw.get(
            "modules",
            raw.get("weight_modules", defaults.get("cots_weight_modules")),
        )
    )
    if modules is None:
        return DEFAULT_COTS_WEIGHT_MODULES
    return frozenset(modules)


def _qkv_output_size_from_sources(
    sources: Sequence[Mapping[str, Any]],
) -> int | None:
    direct = _positive_int_from_sources(
        sources,
        keys=("qkv_output_size", "qkv_out_features", "qkv_out_dim"),
        field_name="buffer_geometry.qkv_output_size",
    )
    if direct is not None:
        return direct
    num_attention_heads = _positive_int_from_sources(
        sources,
        keys=("num_attention_heads", "n_heads", "attention_heads"),
        field_name="buffer_geometry.num_attention_heads",
    )
    num_key_value_heads = _positive_int_from_sources(
        sources,
        keys=("num_key_value_heads", "num_kv_heads", "n_kv_heads"),
        field_name="buffer_geometry.num_key_value_heads",
    )
    head_dim = _positive_int_from_sources(
        sources,
        keys=("head_dim", "attention_head_dim"),
        field_name="buffer_geometry.head_dim",
    )
    if (
        num_attention_heads is None
        or num_key_value_heads is None
        or head_dim is None
    ):
        return None
    return (num_attention_heads + 2 * num_key_value_heads) * head_dim


def _cots_gpu_buffer_geometry_from_config(
    raw: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    profile_metadata: Mapping[str, Any] | None = None,
) -> CotsGPUBufferGeometry | None:
    metadata = profile_metadata or {}
    raw_geometry = _mapping_from_keys(
        raw,
        {},
        BUFFER_GEOMETRY_KEYS,
        field_name="weight.buffer_geometry",
    )
    default_geometry = _mapping_from_keys(
        defaults,
        {},
        BUFFER_GEOMETRY_KEYS,
        field_name="defaults.cots_buffer_geometry",
    )
    metadata_geometry = _mapping_from_keys(
        metadata,
        {},
        BUFFER_GEOMETRY_KEYS,
        field_name="profile.metadata.buffer_geometry",
    )
    sources = tuple(
        source
        for source in (
            raw,
            raw_geometry,
            defaults,
            default_geometry,
            metadata,
            metadata_geometry,
        )
        if isinstance(source, Mapping)
    )
    if not any(
        any(key in source for key in BUFFER_GEOMETRY_SIGNAL_KEYS)
        for source in sources
    ):
        return None

    hidden_size = _positive_int_from_sources(
        sources,
        keys=("hidden_size", "model_hidden_size"),
        field_name="buffer_geometry.hidden_size",
    )
    if hidden_size is None:
        raise ValueError("buffer geometry requires hidden_size")
    modules = _normalized_weight_modules_from_config(raw, defaults)
    intermediate_size = _positive_int_from_sources(
        sources,
        keys=("intermediate_size", "mlp_intermediate_size", "ffn_hidden_size"),
        field_name="buffer_geometry.intermediate_size",
    )
    qkv_output_size = _qkv_output_size_from_sources(sources)
    prefetch_buffer_slots = _positive_int_from_sources(
        sources,
        keys=(
            "prefetch_buffer_slots",
            "gpu_prefetch_buffer_slots",
            "cots_prefetch_buffer_slots",
        ),
        field_name="buffer_geometry.prefetch_buffer_slots",
    )
    max_num_batched_tokens = _optional_nonnegative_int_from_sources(
        sources,
        keys=(
            "max_num_batched_tokens",
            "max_num_tokens",
            "cots_max_num_batched_tokens",
        ),
        field_name="buffer_geometry.max_num_batched_tokens",
    )
    return CotsGPUBufferGeometry(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        qkv_output_size=qkv_output_size,
        dtype_bytes=_dtype_bytes_from_sources(sources),
        prefetch_buffer_slots=(
            2 if prefetch_buffer_slots is None else prefetch_buffer_slots
        ),
        max_num_batched_tokens=(
            0 if max_num_batched_tokens is None else max_num_batched_tokens
        ),
        modules=modules,
    )


def _weight_kv_resource_kwargs(
    raw: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    profile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = profile_metadata or {}
    profile_resource_model = _mapping_from_keys(
        metadata,
        {},
        WEIGHT_RESOURCE_MODEL_KEYS,
        field_name="profile.metadata.weight_resource_model",
    )
    snap_resource_maps = cots_snap_resource_maps_from_metadata(metadata)
    config_resource_sources = (raw, defaults)
    profile_resource_sources = tuple(
        source
        for source in (snap_resource_maps, profile_resource_model)
        if isinstance(source, Mapping) and source
    )
    weight_resource_sources = tuple(
        source
        for source in (*config_resource_sources, *profile_resource_sources)
        if isinstance(source, Mapping)
    )
    total_weight_bytes = _optional_nonnegative_int_from_sources(
        weight_resource_sources,
        keys=(
            "total_weight_bytes",
            "model_weight_bytes",
            "weight_bytes",
            "cots_total_weight_bytes",
            "cots_model_weight_bytes",
            "cots_weight_bytes",
        ),
        field_name="total_weight_bytes",
    )
    cpu_weight_bytes_by_store = _optional_fraction_int_mapping_from_sources(
        weight_resource_sources,
        keys=(
            "cpu_weight_bytes_by_store_fraction",
            "cpu_weight_bytes_by_store",
            "weights_saved_bytes_by_store_fraction",
            "cots_cpu_weight_bytes_by_store_fraction",
            "cots_weights_saved_bytes_by_store_fraction",
        ),
        field_name="cpu_weight_bytes_by_store_fraction",
    )
    gpu_kv_bytes = _optional_nonnegative_int_from_keys(
        raw,
        defaults,
        keys=(
            "gpu_kv_bytes",
            "kv_gpu_bytes",
            "KV_gpu_bytes",
            "cots_gpu_kv_bytes",
            "cots_kv_gpu_bytes",
        ),
        field_name="gpu_kv_bytes",
    )
    cpu_kv_bytes = _optional_nonnegative_int_from_keys(
        raw,
        defaults,
        keys=(
            "cpu_kv_bytes",
            "kv_cpu_bytes",
            "KV_cpu_bytes",
            "cots_cpu_kv_bytes",
            "cots_kv_cpu_pool_bytes",
        ),
        field_name="cpu_kv_bytes",
    )
    gpu_buffer_bytes = _optional_nonnegative_int_from_keys(
        raw,
        defaults,
        keys=(
            "gpu_buffer_bytes",
            "prefetch_gpu_bytes",
            "cots_gpu_buffer_bytes",
            "cots_prefetch_gpu_bytes",
        ),
        field_name="gpu_buffer_bytes",
    )
    gpu_buffer_full_store_bytes = _optional_nonnegative_int_from_sources(
        config_resource_sources,
        keys=(
            "gpu_buffer_bytes_per_store_fraction",
            "gpu_buffer_bytes_at_full_store",
            "gpu_buffer_full_store_bytes",
            "cots_gpu_buffer_bytes_per_store_fraction",
            "cots_gpu_buffer_bytes_at_full_store",
        ),
        field_name="gpu_buffer_bytes_per_store_fraction",
    )
    gpu_prefetch_full_store_bytes = _optional_nonnegative_int_from_sources(
        config_resource_sources,
        keys=(
            "gpu_prefetch_buffer_bytes_per_store_fraction",
            "gpu_prefetch_buffer_bytes_at_full_store",
            "gpu_prefetch_buffer_full_store_bytes",
            "cots_gpu_prefetch_buffer_bytes_per_store_fraction",
            "cots_gpu_prefetch_buffer_bytes_at_full_store",
        ),
        field_name="gpu_prefetch_buffer_bytes_per_store_fraction",
    )
    gpu_output_scratch_full_store_bytes = _optional_nonnegative_int_from_sources(
        config_resource_sources,
        keys=(
            "gpu_output_scratch_bytes_per_store_fraction",
            "gpu_output_scratch_bytes_at_full_store",
            "gpu_uva_output_scratch_bytes_per_store_fraction",
            "gpu_uva_output_scratch_bytes_at_full_store",
            "cots_gpu_output_scratch_bytes_per_store_fraction",
            "cots_gpu_uva_output_scratch_bytes_at_full_store",
        ),
        field_name="gpu_output_scratch_bytes_per_store_fraction",
    )
    derived_buffer_coefficients = (
        gpu_buffer_full_store_bytes,
        gpu_prefetch_full_store_bytes,
        gpu_output_scratch_full_store_bytes,
    )
    has_derived_buffer_override = any(
        value is not None for value in derived_buffer_coefficients
    )
    if gpu_buffer_bytes is not None and has_derived_buffer_override:
        raise ValueError(
            "weight config cannot set both fixed gpu_buffer_bytes and "
            "derived gpu buffer full-store coefficients"
        )
    derived_gpu_buffer = sum(
        value for value in derived_buffer_coefficients if value is not None
    )
    if gpu_buffer_bytes is None and not has_derived_buffer_override:
        profile_gpu_buffer_full_store_bytes = _optional_nonnegative_int_from_sources(
            profile_resource_sources,
            keys=(
                "gpu_buffer_bytes_per_store_fraction",
                "gpu_buffer_bytes_at_full_store",
                "gpu_buffer_full_store_bytes",
                "cots_gpu_buffer_bytes_per_store_fraction",
                "cots_gpu_buffer_bytes_at_full_store",
            ),
            field_name="gpu_buffer_bytes_per_store_fraction",
        )
        profile_gpu_prefetch_full_store_bytes = (
            _optional_nonnegative_int_from_sources(
                profile_resource_sources,
                keys=(
                    "gpu_prefetch_buffer_bytes_per_store_fraction",
                    "gpu_prefetch_buffer_bytes_at_full_store",
                    "gpu_prefetch_buffer_full_store_bytes",
                    "cots_gpu_prefetch_buffer_bytes_per_store_fraction",
                    "cots_gpu_prefetch_buffer_bytes_at_full_store",
                ),
                field_name="gpu_prefetch_buffer_bytes_per_store_fraction",
            )
        )
        profile_gpu_output_scratch_full_store_bytes = (
            _optional_nonnegative_int_from_sources(
                profile_resource_sources,
                keys=(
                    "gpu_output_scratch_bytes_per_store_fraction",
                    "gpu_output_scratch_bytes_at_full_store",
                    "gpu_uva_output_scratch_bytes_per_store_fraction",
                    "gpu_uva_output_scratch_bytes_at_full_store",
                    "cots_gpu_output_scratch_bytes_per_store_fraction",
                    "cots_gpu_uva_output_scratch_bytes_at_full_store",
                ),
                field_name="gpu_output_scratch_bytes_per_store_fraction",
            )
        )
        profile_buffer_coefficients = (
            profile_gpu_buffer_full_store_bytes,
            profile_gpu_prefetch_full_store_bytes,
            profile_gpu_output_scratch_full_store_bytes,
        )
        if any(value is not None for value in profile_buffer_coefficients):
            derived_gpu_buffer = sum(
                value
                for value in profile_buffer_coefficients
                if value is not None
            )
        else:
            geometry = _cots_gpu_buffer_geometry_from_config(
                raw,
                defaults=defaults,
                profile_metadata=profile_metadata,
            )
            if geometry is not None:
                derived_gpu_buffer = geometry.gpu_buffer_bytes_per_store_fraction
    gpu_buffer_bytes_by_store = _optional_fraction_int_mapping_from_sources(
        weight_resource_sources,
        keys=(
            "gpu_buffer_bytes_by_store_fraction",
            "gpu_buffer_bytes_by_store",
            "cots_gpu_buffer_bytes_by_store_fraction",
            "cots_gpu_buffer_bytes_by_store",
        ),
        field_name="gpu_buffer_bytes_by_store_fraction",
    )
    engine_gpu_budget_bytes = _engine_gpu_budget_bytes_from_config(
        raw,
        defaults,
        field_name="engine_gpu_budget_bytes",
    )
    return {
        "total_weight_bytes": total_weight_bytes,
        "cpu_weight_bytes_by_store_fraction": cpu_weight_bytes_by_store,
        "gpu_kv_bytes": gpu_kv_bytes,
        "cpu_kv_bytes": 0 if cpu_kv_bytes is None else cpu_kv_bytes,
        "gpu_buffer_bytes": gpu_buffer_bytes,
        "gpu_buffer_bytes_per_store_fraction": derived_gpu_buffer,
        "gpu_buffer_bytes_by_store_fraction": gpu_buffer_bytes_by_store,
        "engine_gpu_budget_bytes": engine_gpu_budget_bytes,
    }


def _derive_weight_dispatch_table_from_cost_profile(
    raw: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    f_cpu_store: float,
) -> dict[int, tuple[float, float]] | None:
    profile_path = _weight_dispatch_profile_path(
        raw,
        defaults=defaults,
        required_for="",
    )
    if profile_path is None:
        return None
    buckets = _weight_dispatch_buckets(
        raw,
        defaults=defaults,
        required_for="weight.dispatch_cost_profile_path",
    )
    assert buckets is not None
    candidate_ratios = _weight_dispatch_candidate_ratios(raw)
    candidate_ratio_step = float(raw.get("dispatch_candidate_ratio_step", 0.125))
    profile = WeightDispatchCostProfile.load_json(profile_path)
    compiler = DispatchCompiler(
        profile=profile,
        candidate_ratios=candidate_ratios,
        candidate_ratio_step=candidate_ratio_step,
    )
    return compiler.compile_runtime_table(
        f_cpu_store=f_cpu_store,
        buckets=buckets,
    )


def _derive_weight_kv_partition_from_cost_profile(
    raw: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    f_cpu_store_candidates: Sequence[float],
) -> WeightKVPartitionResult:
    profile_path = _weight_dispatch_profile_path(
        raw,
        defaults=defaults,
        required_for="weight.f_cpu_store_candidates",
    )
    buckets = _weight_dispatch_buckets(
        raw,
        defaults=defaults,
        required_for="weight.f_cpu_store_candidates",
    )
    assert buckets is not None
    bucket_weights_raw = raw.get("dispatch_bucket_weights", raw.get("bucket_weights"))
    bucket_weights = None
    if bucket_weights_raw is not None:
        if not isinstance(bucket_weights_raw, Mapping):
            raise ValueError("weight.dispatch_bucket_weights must be a mapping")
        bucket_weights = {int(k): float(v) for k, v in bucket_weights_raw.items()}
    profile = WeightDispatchCostProfile.load_json(profile_path)
    resource_kwargs = _weight_kv_resource_kwargs(
        raw,
        defaults=defaults,
        profile_metadata=profile.metadata,
    )
    engine_gpu_budget_candidates = _engine_gpu_budget_candidates_from_config(
        raw,
        defaults,
        field_name="engine_gpu_budget_candidates",
    )
    if engine_gpu_budget_candidates is not None:
        resource_kwargs["engine_gpu_budget_bytes"] = None
    partitioner = WeightKVPartitioner(
        profile,
        buckets=buckets,
        bucket_weights=bucket_weights,
        candidate_ratios=_weight_dispatch_candidate_ratios(raw),
        candidate_ratio_step=float(raw.get("dispatch_candidate_ratio_step", 0.125)),
        require_k=bool(raw.get("require_k", True)),
        engine_gpu_budget_candidates=engine_gpu_budget_candidates,
        **resource_kwargs,
    )
    return partitioner.solve(f_cpu_store_candidates=f_cpu_store_candidates)


def _weight_dispatch_bucket_weights(
    raw: Mapping[str, Any],
) -> dict[int, float] | None:
    bucket_weights_raw = raw.get("dispatch_bucket_weights", raw.get("bucket_weights"))
    if bucket_weights_raw is None:
        return None
    if not isinstance(bucket_weights_raw, Mapping):
        raise ValueError("weight.dispatch_bucket_weights must be a mapping")
    return {int(k): float(v) for k, v in bucket_weights_raw.items()}


def _weight_store_candidates_from_config(
    raw: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    role: EngineRole,
) -> tuple[float, ...] | None:
    return _normalize_fraction_sequence(
        raw.get(
            "f_cpu_store_candidates",
            defaults.get("cots_f_cpu_store_candidates"),
        ),
        field_name=f"{role}.weight.f_cpu_store_candidates",
    )


def derive_weight_store_candidates_from_profile(
    profile: WeightDispatchCostProfile,
    buckets: Sequence[int],
    *,
    field_name: str = "f_cpu_store_candidates",
) -> tuple[float, ...]:
    """Derive the static-store search grid from calibrated K(B, s) support.

    Stage-1 placement compares different `s = f_cpu_store` values, so every
    candidate must have the split-invariant K(B, s) term for every bucket in
    the workload contract. The profiler can expose more store fractions per
    bucket, but the planner's unconstrained grid is their intersection.
    """

    bucket_ids = _normalize_bucket_sequence(buckets, field_name="dispatch_buckets")
    if not bucket_ids:
        raise ValueError(f"{field_name} derivation requires dispatch buckets")

    common: set[float] | None = None
    empty_buckets: list[int] = []
    for bucket in bucket_ids:
        cost = profile.cost_for_bucket(bucket)
        values: set[float] = set()
        for store in cost.k_by_store_s:
            values.add(_round_fraction(_validate_fraction("f_cpu_store", store)))
        if not values:
            empty_buckets.append(bucket)
        common = values if common is None else common & values

    if empty_buckets:
        raise ValueError(
            f"{field_name} omitted but profile has no K(B,s) store "
            f"fractions for buckets={empty_buckets}"
        )
    if not common:
        raise ValueError(
            f"{field_name} omitted but profile has no common K(B,s) "
            f"store fraction across buckets={list(bucket_ids)}"
        )
    return tuple(sorted(common))


@dataclass(frozen=True)
class _ConfiguredWeightKVPartitioner:
    partitioner: WeightKVPartitioner
    f_cpu_store_candidates: tuple[float, ...]
    engine_gpu_budget_candidates: tuple[int, ...] | None = None


def _build_weight_kv_partitioner_from_config(
    role: EngineRole,
    raw: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
) -> _ConfiguredWeightKVPartitioner:
    weight_raw = raw.get("weight", {})
    if not isinstance(weight_raw, Mapping):
        raise ValueError(f"{role}.weight must be a mapping")
    candidates = _weight_store_candidates_from_config(
        weight_raw,
        defaults=defaults,
        role=role,
    )
    if "f_cpu_store" in weight_raw:
        raise ValueError(
            "global model-memory planning derives "
            f"{role}.weight.f_cpu_store; use f_cpu_store_candidates only "
            "to constrain the profiled store-fraction grid"
        )
    if _normalize_dispatch_table(weight_raw.get("dispatch_table")) is not None:
        raise ValueError(
            "global model-memory planning cannot use an explicit "
            f"{role}.weight.dispatch_table"
        )
    profile_path = _weight_dispatch_profile_path(
        weight_raw,
        defaults=defaults,
        required_for=f"{role}.weight placement planning",
    )
    buckets = _weight_dispatch_buckets(
        weight_raw,
        defaults=defaults,
        required_for=f"{role}.weight placement planning",
    )
    assert buckets is not None
    profile = WeightDispatchCostProfile.load_json(profile_path)
    if candidates is None:
        candidates = derive_weight_store_candidates_from_profile(
            profile,
            buckets,
            field_name=f"{role}.weight.f_cpu_store_candidates",
        )
    resource_kwargs = _weight_kv_resource_kwargs(
        weight_raw,
        defaults=defaults,
        profile_metadata=profile.metadata,
    )
    engine_gpu_budget_candidates = _engine_gpu_budget_candidates_from_config(
        weight_raw,
        defaults,
        field_name=f"{role}.weight.engine_gpu_budget_candidates",
    )
    resource_kwargs["engine_gpu_budget_bytes"] = None
    return _ConfiguredWeightKVPartitioner(
        partitioner=WeightKVPartitioner(
            profile,
            buckets=buckets,
            bucket_weights=_weight_dispatch_bucket_weights(weight_raw),
            candidate_ratios=_weight_dispatch_candidate_ratios(weight_raw),
            candidate_ratio_step=float(
                weight_raw.get("dispatch_candidate_ratio_step", 0.125)
            ),
            require_k=bool(weight_raw.get("require_k", True)),
            **resource_kwargs,
        ),
        f_cpu_store_candidates=candidates,
        engine_gpu_budget_candidates=engine_gpu_budget_candidates,
    )


def _global_model_memory_config(
    raw: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for key in ("global", "model_memory", "model_memory_partition"):
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"planner_config.{key} must be a mapping")
        return value
    return None


def _model_memory_budget_bytes(
    raw: Mapping[str, Any],
    *,
    role: Literal["gpu", "cpu"],
) -> int:
    if role == "gpu":
        value = _optional_budget_bytes_from_mapping(
            raw,
            byte_keys=("gpu_budget_bytes", "vram_budget_bytes", "GPU_budget_bytes"),
            gib_keys=("gpu_budget_gb", "vram_budget_gb", "GPU_budget_gb"),
            field_name="global.gpu_budget_bytes",
        )
    else:
        value = _optional_budget_bytes_from_mapping(
            raw,
            byte_keys=("cpu_budget_bytes", "ram_budget_bytes", "CPU_budget_bytes"),
            gib_keys=("cpu_budget_gb", "ram_budget_gb", "CPU_budget_gb"),
            field_name="global.cpu_budget_bytes",
        )
    if value is None:
        raise ValueError(f"planner_config.global requires {role}_budget_bytes")
    return value


def _engine_config_mapping(
    raw: Mapping[str, Any],
    role: EngineRole,
) -> Mapping[str, Any]:
    value = raw.get(role, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"planner_config.{role} must be a mapping")
    return value


def _engine_config_with_selected_weight(
    raw: Mapping[str, Any],
    selected: WeightKVCandidateScore,
) -> dict[str, Any]:
    engine_raw = dict(raw)
    weight_raw = dict(engine_raw.get("weight", {}) or {})
    weight_raw.pop("f_cpu_store_candidates", None)
    for key in (*ENGINE_GPU_BUDGET_BYTE_KEYS, *ENGINE_GPU_BUDGET_CANDIDATE_KEYS):
        weight_raw.pop(key, None)
    weight_raw["f_cpu_store"] = selected.f_cpu_store
    weight_raw["dispatch_table"] = selected.dispatch_table
    engine_raw["weight"] = weight_raw
    if selected.resources.resource_known:
        kv_raw = dict(engine_raw.get("kv", {}) or {})
        kv_raw["gpu_kv_bytes"] = selected.resources.gpu_kv_bytes
        kv_raw["cpu_kv_bytes"] = selected.resources.cpu_kv_bytes
        engine_raw["kv"] = kv_raw
    return engine_raw


def _validate_fraction(name: str, value: float) -> float:
    value = float(value)
    if value < -1e-12 or value > 1.0 + 1e-12:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return min(1.0, max(0.0, value))


def _round_fraction(value: float) -> float:
    value = round(float(value), 12)
    return 0.0 if abs(value) <= 1e-12 else value


def _default_fraction_grid(max_fraction: float, step: float) -> tuple[float, ...]:
    values = [0.0]
    i = 1
    while i * step < max_fraction - 1e-12:
        values.append(i * step)
        i += 1
    values.append(max_fraction)
    return tuple(values)


def _candidate_f_cpu_values(problem: DispatchProblem) -> tuple[float, ...]:
    f_cpu_store = _validate_fraction("f_cpu_store", problem.f_cpu_store)
    if problem.candidate_step <= 0:
        raise ValueError(
            f"candidate_step must be positive, got {problem.candidate_step}"
        )

    raw_candidates = (
        _default_fraction_grid(f_cpu_store, problem.candidate_step)
        if problem.candidate_f_cpu is None
        else tuple(float(value) for value in problem.candidate_f_cpu)
    )
    candidates = [0.0, f_cpu_store, *raw_candidates]
    legal: dict[float, float] = {}
    for candidate in candidates:
        if candidate < -1e-12 or candidate > f_cpu_store + 1e-12:
            continue
        clipped = min(f_cpu_store, max(0.0, float(candidate)))
        rounded = _round_fraction(clipped)
        legal[rounded] = rounded
    if not legal:
        raise ValueError("no legal f_cpu candidates remain after snapping")
    return tuple(sorted(legal))


def _validate_dispatch_problem(problem: DispatchProblem) -> None:
    _validate_fraction("f_cpu_store", problem.f_cpu_store)
    if problem.num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {problem.num_layers}")
    if not problem.buckets:
        raise ValueError("buckets must not be empty")
    for bucket in problem.buckets:
        if int(bucket) <= 0:
            raise ValueError(f"bucket values must be positive, got {bucket}")
    for op, weight_bytes in problem.weight_bytes_per_layer.items():
        if int(weight_bytes) < 0:
            raise ValueError(
                f"weight_bytes_per_layer[{op!r}] must be non-negative, "
                f"got {weight_bytes}"
            )
    for op, fraction in problem.fixed_cpu_fractions_by_op.items():
        _validate_fraction(f"fixed_cpu_fractions_by_op[{op!r}]", float(fraction))
    _candidate_f_cpu_values(problem)


def _op_split_fractions(
    problem: DispatchProblem,
    op: str,
    f_cpu: float,
) -> tuple[float, float]:
    if op in problem.weight_bytes_per_layer:
        cpu_fraction = f_cpu
    else:
        cpu_fraction = float(problem.fixed_cpu_fractions_by_op.get(op, 0.0))
    cpu_fraction = _validate_fraction(f"cpu_fraction[{op}]", cpu_fraction)
    return 1.0 - cpu_fraction, cpu_fraction


def _profile_overhead_ms(
    profile: DispatchProfileView,
    bucket: int,
    f_cpu: float,
    f_prefetch: float,
) -> float:
    overhead_fn = getattr(profile, "overhead_ms", None)
    if overhead_fn is None:
        return 0.0
    return float(overhead_fn(bucket, f_cpu, f_prefetch))


def _score_dispatch_candidate(
    problem: DispatchProblem,
    profile: DispatchProfileView,
    bucket: int,
    f_cpu: float,
) -> DispatchEntry:
    f_cpu_store = _validate_fraction("f_cpu_store", problem.f_cpu_store)
    f_cpu = _round_fraction(f_cpu)
    f_prefetch = _round_fraction(f_cpu_store - f_cpu)
    c_layer_ms = 0.0

    for op in problem.layer_ops:
        gpu_fraction, cpu_fraction = _op_split_fractions(problem, op, f_cpu)
        gpu_ms = float(profile.gpu_op_ms(op, bucket, gpu_fraction))
        cpu_ms = float(profile.cpu_op_ms(op, bucket, cpu_fraction))
        c_layer_ms += max(gpu_ms, cpu_ms)

    c_layer_ms += _profile_overhead_ms(profile, bucket, f_cpu, f_prefetch)
    prefetch_bytes_per_layer = sum(
        int(problem.weight_bytes_per_layer.get(op, 0)) for op in problem.layer_ops
    )
    p_layer_ms = float(
        profile.h2d_ms(int(round(f_prefetch * prefetch_bytes_per_layer)))
    )
    layer_ms = max(c_layer_ms, p_layer_ms)
    predicted_ms = float(problem.num_layers) * layer_ms
    bottleneck: DispatchBottleneck = (
        "compute" if c_layer_ms >= p_layer_ms else "prefetch"
    )
    return DispatchEntry(
        bucket=int(bucket),
        f_cpu=f_cpu,
        f_prefetch=f_prefetch,
        predicted_ms=predicted_ms,
        layer_ms=layer_ms,
        c_layer_ms=c_layer_ms,
        p_layer_ms=p_layer_ms,
        bottleneck=bottleneck,
    )


def _better_dispatch_entry(
    candidate: DispatchEntry,
    incumbent: DispatchEntry | None,
) -> bool:
    if incumbent is None:
        return True
    if candidate.predicted_ms < incumbent.predicted_ms - 1e-9:
        return True
    if abs(candidate.predicted_ms - incumbent.predicted_ms) <= 1e-9:
        return candidate.f_cpu < incumbent.f_cpu - 1e-12
    return False


def solve_per_bucket_dispatch(
    problem: DispatchProblem,
    profile: DispatchProfileView,
) -> DispatchSolveResult:
    """Reference minimizer for the thesis per-bucket layer model.

    The solver does a bounded snapped search over `f_cpu` for each bucket. It
    is intentionally simple and deterministic: this is the validation target
    that future closed-form dispatch heuristics must match before becoming a
    fast path.
    """

    _validate_dispatch_problem(problem)
    candidates = _candidate_f_cpu_values(problem)
    entries: dict[int, DispatchEntry] = {}
    candidate_scores: dict[int, tuple[DispatchEntry, ...]] = {}

    for raw_bucket in problem.buckets:
        bucket = int(raw_bucket)
        scores = tuple(
            _score_dispatch_candidate(problem, profile, bucket, f_cpu)
            for f_cpu in candidates
        )
        best: DispatchEntry | None = None
        for score in scores:
            if _better_dispatch_entry(score, best):
                best = score
        assert best is not None
        entries[bucket] = best
        candidate_scores[bucket] = scores

    return DispatchSolveResult(entries=entries, candidate_scores=candidate_scores)


@dataclass(frozen=True)
class WeightPlacementPlan:
    """Per-engine static weight placement plus optional dispatch table."""

    f_cpu_store: float = 0.0
    f_prefetch: float = 0.0
    dispatch_table: dict[int, tuple[float, float]] | None = None
    modules: set[str] | None = None
    cpu_num_threads: int | None = None
    cpu_num_threads_by_bucket: dict[int, int] | None = None
    backend: Literal["auto", "cots"] = "auto"

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        defaults: Mapping[str, Any],
    ) -> "WeightPlacementPlan":
        raw = raw or {}
        f_cpu_store_candidates = _normalize_fraction_sequence(
            raw.get(
                "f_cpu_store_candidates",
                defaults.get("cots_f_cpu_store_candidates"),
            ),
            field_name="f_cpu_store_candidates",
        )
        if f_cpu_store_candidates is not None and "f_cpu_store" in raw:
            raise ValueError(
                "weight config cannot set both f_cpu_store and "
                "f_cpu_store_candidates"
            )
        f_cpu_store = float(
            raw.get("f_cpu_store", defaults.get("cots_f_cpu_store", 0.0))
        )
        f_prefetch = float(raw.get("f_prefetch", defaults.get("cots_f_prefetch", 0.0)))

        explicit_dispatch_table = _normalize_dispatch_table(raw.get("dispatch_table"))
        default_dispatch_table = _normalize_dispatch_table(
            defaults.get("cots_dispatch_table")
        )
        dispatch_table = explicit_dispatch_table or default_dispatch_table
        if f_cpu_store_candidates is not None:
            if explicit_dispatch_table is not None:
                raise ValueError(
                    "weight.f_cpu_store_candidates cannot be combined with "
                    "an explicit dispatch_table"
                )
            partition_result = _derive_weight_kv_partition_from_cost_profile(
                raw,
                defaults=defaults,
                f_cpu_store_candidates=f_cpu_store_candidates,
            )
            f_cpu_store = partition_result.best.f_cpu_store
            dispatch_table = partition_result.best.dispatch_table
        elif dispatch_table is None:
            dispatch_table = _derive_weight_dispatch_table_from_cost_profile(
                raw,
                defaults=defaults,
                f_cpu_store=f_cpu_store,
            )

        backend = raw.get("backend")
        if backend is None:
            backend = (
                "cots"
                if f_cpu_store > 0 or defaults.get("offload_backend") == "cots"
                else "auto"
            )
        if backend not in {"auto", "cots"}:
            raise ValueError(f"Unsupported weight backend: {backend!r}")
        modules = _normalize_weight_modules(
            raw.get(
                "modules",
                raw.get("weight_modules", defaults.get("cots_weight_modules")),
            )
        )
        cpu_num_threads_raw = raw.get(
            "cpu_num_threads", defaults.get("cots_cpu_num_threads")
        )
        cpu_num_threads = (
            None if cpu_num_threads_raw is None else int(cpu_num_threads_raw)
        )
        cpu_num_threads_by_bucket = _normalize_thread_table(
            raw.get(
                "cpu_num_threads_by_bucket",
                defaults.get("cots_cpu_num_threads_by_bucket"),
            )
        )
        derive_threads = bool(raw.get("derive_cpu_num_threads_by_bucket", True))
        if (
            cpu_num_threads_by_bucket is None
            and dispatch_table is not None
            and derive_threads
        ):
            cpu_num_threads_by_bucket = derive_weight_thread_policy(dispatch_table)
        plan = cls(
            f_cpu_store=f_cpu_store,
            f_prefetch=f_prefetch,
            dispatch_table=dispatch_table,
            modules=modules,
            cpu_num_threads=cpu_num_threads,
            cpu_num_threads_by_bucket=cpu_num_threads_by_bucket,
            backend=backend,
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if not (0.0 <= self.f_cpu_store <= 1.0):
            raise ValueError(f"f_cpu_store must be in [0, 1], got {self.f_cpu_store}")
        if not (0.0 <= self.f_prefetch <= 1.0):
            raise ValueError(f"f_prefetch must be in [0, 1], got {self.f_prefetch}")
        if self.f_prefetch > self.f_cpu_store:
            raise ValueError(
                f"f_prefetch ({self.f_prefetch}) must be <= "
                f"f_cpu_store ({self.f_cpu_store})"
            )
        if self.cpu_num_threads is not None and self.cpu_num_threads < 1:
            raise ValueError(
                f"cpu_num_threads must be positive, got {self.cpu_num_threads}"
            )
        if self.cpu_num_threads_by_bucket is not None:
            for bucket, n_threads in self.cpu_num_threads_by_bucket.items():
                if bucket <= 0:
                    raise ValueError(
                        "cpu_num_threads_by_bucket bucket must be positive, "
                        f"got {bucket}"
                    )
                if n_threads < 1:
                    raise ValueError(
                        "cpu_num_threads_by_bucket values must be positive, "
                        f"got {n_threads} for bucket {bucket}"
                    )
        if self.modules is not None:
            unknown = self.modules - VALID_WEIGHT_MODULES
            if unknown:
                raise ValueError(
                    f"weight.modules contains unsupported entries "
                    f"{sorted(unknown)}"
                )
        if self.dispatch_table is None:
            return
        for bucket, (f_cpu_compute, f_prefetch) in self.dispatch_table.items():
            if f_cpu_compute + f_prefetch > self.f_cpu_store + 1e-9:
                raise ValueError(
                    "dispatch_table entry exceeds f_cpu_store: "
                    f"bucket={bucket}, entry={(f_cpu_compute, f_prefetch)}, "
                    f"f_cpu_store={self.f_cpu_store}"
                )


@dataclass(frozen=True)
class KVPlacementPlan:
    """Per-engine KV placement budget.

    Phase 2 COTS hybrid KV uses `cpu_kv_bytes` + `split_blocks` directly.
    The legacy vLLM KV-offload fields are preserved for old planner configs
    but are not emitted for COTS hybrid KV.
    """

    gpu_kv_bytes: int | None = None
    cpu_kv_bytes: int | None = None
    split_blocks: int = 0
    backend: str = "native"

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        defaults: Mapping[str, Any],
    ) -> "KVPlacementPlan":
        raw = raw or {}
        gpu_kv_bytes = raw.get(
            "gpu_kv_bytes",
            raw.get(
                "kv_gpu_bytes",
                raw.get(
                    "KV_gpu_bytes",
                    raw.get(
                        "cots_gpu_kv_bytes",
                        raw.get("cots_kv_gpu_bytes"),
                    ),
                ),
            ),
        )
        if gpu_kv_bytes is None:
            gpu_kv_bytes = defaults.get("kv_cache_memory_bytes")
        if gpu_kv_bytes is None and "gpu_kv_gb" in raw:
            gpu_kv_bytes = int(float(raw["gpu_kv_gb"]) * (1 << 30))
        cpu_kv_bytes = raw.get(
            "cpu_kv_bytes",
            raw.get(
                "kv_cpu_bytes",
                raw.get(
                    "KV_cpu_bytes",
                    raw.get(
                        "cots_cpu_kv_bytes",
                        raw.get("cots_kv_cpu_pool_bytes"),
                    ),
                ),
            ),
        )
        if cpu_kv_bytes is None and "cpu_kv_gb" in raw:
            cpu_kv_bytes = int(float(raw["cpu_kv_gb"]) * (1 << 30))
        if cpu_kv_bytes is None and "cots_kv_cpu_pool_bytes" in defaults:
            cpu_kv_bytes = defaults.get("cots_kv_cpu_pool_bytes")
        elif cpu_kv_bytes is None and "kv_offloading_size" in defaults:
            size_gib = defaults.get("kv_offloading_size")
            if size_gib is not None:
                cpu_kv_bytes = int(float(size_gib) * (1 << 30))
        split_blocks = int(
            raw.get(
                "split_blocks",
                raw.get(
                    "kv_split_blocks",
                    raw.get(
                        "cots_kv_split_blocks",
                        defaults.get("cots_kv_split_blocks", 0),
                    ),
                ),
            )
        )
        if gpu_kv_bytes is not None and int(gpu_kv_bytes) < 0:
            raise ValueError(f"gpu_kv_bytes must be non-negative, got {gpu_kv_bytes}")
        if cpu_kv_bytes is not None and int(cpu_kv_bytes) < 0:
            raise ValueError(f"cpu_kv_bytes must be non-negative, got {cpu_kv_bytes}")
        if split_blocks < 0:
            raise ValueError(f"split_blocks must be non-negative, got {split_blocks}")
        return cls(
            gpu_kv_bytes=None if gpu_kv_bytes is None else int(gpu_kv_bytes),
            cpu_kv_bytes=None if cpu_kv_bytes is None else int(cpu_kv_bytes),
            split_blocks=split_blocks,
            backend=str(
                raw.get("backend", defaults.get("kv_offloading_backend", "native"))
            ),
        )


@dataclass(frozen=True)
class EnginePlan:
    """Launch-time plan for one vLLM engine."""

    role: EngineRole
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    weight: WeightPlacementPlan = field(default_factory=WeightPlacementPlan)
    kv: KVPlacementPlan = field(default_factory=KVPlacementPlan)

    @classmethod
    def from_mapping(
        cls,
        role: EngineRole,
        raw: Mapping[str, Any] | None,
        *,
        defaults: Mapping[str, Any],
    ) -> "EnginePlan":
        raw = raw or {}
        gpu_memory_utilization = raw.get(
            "gpu_memory_utilization", defaults.get("gpu_memory_utilization")
        )
        max_num_seqs = raw.get("max_num_seqs")
        if max_num_seqs is not None and int(max_num_seqs) <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        weight_raw = raw.get("weight", {})
        kv_raw = raw.get("kv", {})
        return cls(
            role=role,
            gpu_memory_utilization=(
                None
                if gpu_memory_utilization is None
                else float(gpu_memory_utilization)
            ),
            max_num_seqs=None if max_num_seqs is None else int(max_num_seqs),
            weight=WeightPlacementPlan.from_mapping(weight_raw, defaults=defaults),
            kv=KVPlacementPlan.from_mapping(kv_raw, defaults=defaults),
        )

    def to_vllm_overrides(self) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        if self.gpu_memory_utilization is not None:
            overrides["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.max_num_seqs is not None:
            overrides["max_num_seqs"] = self.max_num_seqs
        if self.kv.gpu_kv_bytes is not None and self.kv.gpu_kv_bytes > 0:
            overrides["kv_cache_memory_bytes"] = self.kv.gpu_kv_bytes
        hybrid_kv = (
            self.kv.split_blocks > 0
            and self.kv.cpu_kv_bytes is not None
            and self.kv.cpu_kv_bytes > 0
        )
        if (
            not hybrid_kv
            and self.kv.cpu_kv_bytes is not None
            and self.kv.cpu_kv_bytes > 0
        ):
            overrides["kv_offloading_size"] = _bytes_to_gib(self.kv.cpu_kv_bytes)
            overrides["kv_offloading_backend"] = self.kv.backend
        if self.weight.f_cpu_store > 0 or self.weight.backend == "cots" or hybrid_kv:
            overrides["offload_backend"] = "cots"
            overrides["cots_f_cpu_store"] = self.weight.f_cpu_store
            overrides["cots_f_prefetch"] = self.weight.f_prefetch
            if self.weight.dispatch_table is not None:
                overrides["cots_dispatch_table"] = self.weight.dispatch_table
            if self.weight.modules is not None:
                overrides["cots_weight_modules"] = self.weight.modules
            if self.weight.cpu_num_threads is not None:
                overrides["cots_cpu_num_threads"] = self.weight.cpu_num_threads
            if self.weight.cpu_num_threads_by_bucket is not None:
                overrides["cots_cpu_num_threads_by_bucket"] = (
                    self.weight.cpu_num_threads_by_bucket
                )
            if hybrid_kv:
                overrides["cots_kv_split_blocks"] = self.kv.split_blocks
                overrides["cots_kv_cpu_pool_bytes"] = self.kv.cpu_kv_bytes
                overrides["cots_kv_h2d_mode"] = "uva"
        return overrides


@dataclass(frozen=True)
class TTCSystemPlan:
    """Global two-engine FastTTS plan."""

    generator: EnginePlan
    verifier: EnginePlan
    search: dict[str, Any] = field(default_factory=dict)


class ManualTTCPlanner:
    """Config-driven planner.

    The default path normalizes user-supplied per-engine config and falls back
    to existing vLLM values. When `planner_config.global` supplies shared
    budgets, it uses the frontier-based ModelMemoryPartitioner to select
    generator/verifier weight placement candidates before emitting engine-local
    plans.
    """

    def __init__(self, config: FastTTSConfig):
        self.config = config

    def plan(self, search_config: SearchConfig) -> TTCSystemPlan:
        raw = self.config.planner_config or {}
        global_raw = _global_model_memory_config(raw)
        if global_raw is not None:
            return self._plan_with_model_memory(raw, global_raw, search_config)
        return TTCSystemPlan(
            generator=EnginePlan.from_mapping(
                "generator",
                raw.get("generator"),
                defaults=self.config.generator_vllm_config or {},
            ),
            verifier=EnginePlan.from_mapping(
                "verifier",
                raw.get("verifier"),
                defaults=self.config.verifier_vllm_config or {},
            ),
            search={
                "approach": search_config.approach,
                "n": search_config.n,
                "beam_width": search_config.beam_width,
                "max_tokens": search_config.max_tokens,
                "num_iterations": search_config.num_iterations,
            },
        )

    def _plan_with_model_memory(
        self,
        raw: Mapping[str, Any],
        global_raw: Mapping[str, Any],
        search_config: SearchConfig,
    ) -> TTCSystemPlan:
        generator_raw = _engine_config_mapping(raw, "generator")
        verifier_raw = _engine_config_mapping(raw, "verifier")
        gpu_budget_bytes = _model_memory_budget_bytes(global_raw, role="gpu")
        cpu_budget_bytes = _model_memory_budget_bytes(global_raw, role="cpu")
        engine_gpu_budget_step_bytes = _model_memory_engine_budget_step_bytes(
            global_raw,
            gpu_budget_bytes=gpu_budget_bytes,
        )
        generator_configured = _build_weight_kv_partitioner_from_config(
            "generator",
            generator_raw,
            defaults=self.config.generator_vllm_config or {},
        )
        verifier_configured = _build_weight_kv_partitioner_from_config(
            "verifier",
            verifier_raw,
            defaults=self.config.verifier_vllm_config or {},
        )
        partition_result = ModelMemoryPartitioner(
            gpu_budget_bytes=gpu_budget_bytes,
            cpu_budget_bytes=cpu_budget_bytes,
            engine_weights=global_raw.get("engine_weights", {}),
        ).solve_from_partitioners(
            generator_partitioner=generator_configured.partitioner,
            verifier_partitioner=verifier_configured.partitioner,
            generator_f_cpu_store_candidates=(
                generator_configured.f_cpu_store_candidates
            ),
            verifier_f_cpu_store_candidates=(
                verifier_configured.f_cpu_store_candidates
            ),
            engine_gpu_budget_step_bytes=engine_gpu_budget_step_bytes,
            generator_engine_gpu_budget_candidates=(
                generator_configured.engine_gpu_budget_candidates
            ),
            verifier_engine_gpu_budget_candidates=(
                verifier_configured.engine_gpu_budget_candidates
            ),
        )
        generator_selected_raw = _engine_config_with_selected_weight(
            generator_raw,
            partition_result.best.generator,
        )
        verifier_selected_raw = _engine_config_with_selected_weight(
            verifier_raw,
            partition_result.best.verifier,
        )
        return TTCSystemPlan(
            generator=EnginePlan.from_mapping(
                "generator",
                generator_selected_raw,
                defaults=self.config.generator_vllm_config or {},
            ),
            verifier=EnginePlan.from_mapping(
                "verifier",
                verifier_selected_raw,
                defaults=self.config.verifier_vllm_config or {},
            ),
            search={
                "approach": search_config.approach,
                "n": search_config.n,
                "beam_width": search_config.beam_width,
                "max_tokens": search_config.max_tokens,
                "num_iterations": search_config.num_iterations,
                "model_memory_objective_s": partition_result.best.objective_s,
                "model_memory_gpu_bytes": partition_result.best.gpu_bytes,
                "model_memory_cpu_bytes": partition_result.best.cpu_bytes,
                "model_memory_num_candidates": len(partition_result.candidates),
                "model_memory_engine_gpu_budget_step_bytes": (
                    engine_gpu_budget_step_bytes
                ),
            },
        )


def apply_ttc_plan_to_config(config: FastTTSConfig, plan: TTCSystemPlan) -> None:
    """Apply a system plan to FastTTS's per-engine vLLM config dictionaries."""
    generator_cfg = dict(config.generator_vllm_config or {})
    verifier_cfg = dict(config.verifier_vllm_config or {})
    generator_cfg.update(plan.generator.to_vllm_overrides())
    verifier_cfg.update(plan.verifier.to_vllm_overrides())
    config.generator_vllm_config = generator_cfg
    config.verifier_vllm_config = verifier_cfg
