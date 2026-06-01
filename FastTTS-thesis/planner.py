#!/usr/bin/env python
"""Thin TTC planner interface for FastTTS.

This module implements the launch-time contract plus the first reference
performance-model primitive: a per-bucket dispatch solver for fixed placement.
FastTTS owns the two-engine memory decision and emits one engine-local plan per
model. vLLM consumes each engine-local plan through normal engine kwargs and
still owns tensor geometry, snapping, and runtime dispatch mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Protocol, Sequence

from config import FastTTSConfig, SearchConfig


EngineRole = Literal["generator", "verifier"]
VALID_WEIGHT_MODULES = frozenset({"qkv", "mlp", "wo"})
DEFAULT_DISPATCH_LAYER_OPS = ("qkv", "attention", "wo", "mlp1", "mlp2")
DispatchBottleneck = Literal["compute", "prefetch"]


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1 << 30)


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
        f_cpu_store = float(
            raw.get("f_cpu_store", defaults.get("cots_f_cpu_store", 0.0))
        )
        f_prefetch = float(raw.get("f_prefetch", defaults.get("cots_f_prefetch", 0.0)))
        backend = raw.get("backend")
        if backend is None:
            backend = (
                "cots"
                if f_cpu_store > 0 or defaults.get("offload_backend") == "cots"
                else "auto"
            )
        if backend not in {"auto", "cots"}:
            raise ValueError(f"Unsupported weight backend: {backend!r}")
        dispatch_table = _normalize_dispatch_table(
            raw.get("dispatch_table", defaults.get("cots_dispatch_table"))
        )
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
        gpu_kv_bytes = raw.get("gpu_kv_bytes", raw.get("KV_gpu_bytes"))
        if gpu_kv_bytes is None and "gpu_kv_gb" in raw:
            gpu_kv_bytes = int(float(raw["gpu_kv_gb"]) * (1 << 30))
        cpu_kv_bytes = raw.get("cpu_kv_bytes")
        if cpu_kv_bytes is None:
            cpu_kv_bytes = raw.get("KV_cpu_bytes")
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
    """Manual/static planner.

    This is the pre-Phase-2 interface implementation. It does not optimize; it
    normalizes user-supplied planner config into a system plan and falls back to
    the existing vLLM config values when fields are omitted.
    """

    def __init__(self, config: FastTTSConfig):
        self.config = config

    def plan(self, search_config: SearchConfig) -> TTCSystemPlan:
        raw = self.config.planner_config or {}
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


def apply_ttc_plan_to_config(config: FastTTSConfig, plan: TTCSystemPlan) -> None:
    """Apply a system plan to FastTTS's per-engine vLLM config dictionaries."""
    generator_cfg = dict(config.generator_vllm_config or {})
    verifier_cfg = dict(config.verifier_vllm_config or {})
    generator_cfg.update(plan.generator.to_vllm_overrides())
    verifier_cfg.update(plan.verifier.to_vllm_overrides())
    config.generator_vllm_config = generator_cfg
    config.verifier_vllm_config = verifier_cfg
