#!/usr/bin/env python
"""Thin TTC planner interface for FastTTS.

This module intentionally implements only the launch-time contract, not the
final performance-model optimizer. FastTTS owns the two-engine memory decision
and emits one engine-local plan per model. vLLM consumes each engine-local plan
through normal engine kwargs and still owns tensor geometry, snapping, and
runtime dispatch mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional

from config import FastTTSConfig, SearchConfig


EngineRole = Literal["generator", "verifier"]


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


@dataclass(frozen=True)
class WeightPlacementPlan:
    """Per-engine static weight placement plus optional dispatch table."""

    f_cpu_store: float = 0.0
    f_prefetch: float = 0.0
    dispatch_table: dict[int, tuple[float, float]] | None = None
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
