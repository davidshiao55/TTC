#!/usr/bin/env python3
"""Dry-run the TTC planner from profiler weight-profile artifacts.

This is a planner-facing smoke tool: it consumes compact profiler output
(`weight_dispatch_profile.json` with `weight_resource_model`), global budgets,
and a workload bucket contract, then prints the selected placement and dispatch
table. Store-fraction candidate flags are optional experiment constraints; by
default the planner derives them from common K(B, s) support in the profile. It
does not launch vLLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


TTC_ROOT = Path(__file__).resolve().parents[3]
FASTTTS_ROOT = TTC_ROOT / "FastTTS-thesis"
sys.path.insert(0, str(FASTTTS_ROOT))

from planner import (  # noqa: E402
    ModelMemoryPartitioner,
    WeightDispatchCostProfile,
    WeightKVCandidateScore,
    WeightKVPartitioner,
    cots_snap_resource_maps_from_metadata,
    derive_weight_store_candidates_from_profile,
)


def _parse_fraction_list(raw: str, *, field_name: str) -> tuple[float, ...]:
    values = tuple(
        sorted({float(entry.strip()) for entry in raw.split(",") if entry.strip()})
    )
    if not values:
        raise argparse.ArgumentTypeError(f"{field_name} must not be empty")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError(
                f"{field_name} values must be in [0, 1], got {value}"
            )
    return values


def _parse_int_list(raw: str, *, field_name: str) -> tuple[int, ...]:
    values = tuple(
        sorted({int(entry.strip()) for entry in raw.split(",") if entry.strip()})
    )
    if not values:
        raise argparse.ArgumentTypeError(f"{field_name} must not be empty")
    for value in values:
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"{field_name} values must be positive, got {value}"
            )
    return values


def _parse_bucket_weights(raw: str | None) -> dict[int, float] | None:
    if raw is None:
        return None
    text = Path(raw[1:]).read_text() if raw.startswith("@") else raw
    parsed = json.loads(text) if text.strip().startswith("{") else None
    if parsed is None:
        parsed = {}
        for entry in text.split(","):
            entry = entry.strip()
            if not entry:
                continue
            bucket, weight = entry.split(":", 1)
            parsed[int(bucket.strip())] = float(weight.strip())
    weights = {int(bucket): float(weight) for bucket, weight in parsed.items()}
    if any(weight < 0 for weight in weights.values()):
        raise argparse.ArgumentTypeError("bucket weights must be non-negative")
    if sum(weights.values()) <= 0:
        raise argparse.ArgumentTypeError("bucket weights must have positive total")
    return weights


def _bytes_from_args(
    *,
    byte_value: int | None,
    gb_value: float | None,
    field_name: str,
) -> int:
    if byte_value is not None and gb_value is not None:
        raise ValueError(f"provide {field_name}-bytes or {field_name}-gb, not both")
    if byte_value is not None:
        if byte_value < 0:
            raise ValueError(f"{field_name}-bytes must be non-negative")
        return int(byte_value)
    if gb_value is None:
        raise ValueError(f"{field_name} budget is required")
    if gb_value < 0:
        raise ValueError(f"{field_name}-gb must be non-negative")
    return int(float(gb_value) * (1 << 30))


def _format_bytes(value: int) -> str:
    if value >= 1 << 30:
        return f"{value / (1 << 30):.3f} GiB"
    if value >= 1 << 20:
        return f"{value / (1 << 20):.3f} MiB"
    if value >= 1 << 10:
        return f"{value / (1 << 10):.3f} KiB"
    return f"{value} B"


def _resource_model(profile: WeightDispatchCostProfile) -> Mapping[str, Any]:
    raw = profile.metadata.get("weight_resource_model", {})
    if not isinstance(raw, Mapping):
        raise ValueError("profile.metadata.weight_resource_model must be a mapping")
    return raw


def _resource_int(
    resource_model: Mapping[str, Any],
    *,
    keys: Sequence[str],
    field_name: str,
    override: int | None = None,
) -> int:
    if override is not None:
        if override < 0:
            raise ValueError(f"{field_name} override must be non-negative")
        return int(override)
    for key in keys:
        if key in resource_model:
            value = int(resource_model[key])
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
            return value
    raise ValueError(
        f"profile weight_resource_model is missing {field_name}; "
        "provide it in the profile or use the matching CLI override"
    )


def _buffer_coefficient(
    resource_model: Mapping[str, Any],
    *,
    override: int | None = None,
) -> int:
    if override is not None:
        if override < 0:
            raise ValueError(
                "gpu_buffer_bytes_per_store_fraction override must be non-negative"
            )
        return int(override)
    for key in (
        "gpu_buffer_bytes_per_store_fraction",
        "gpu_buffer_bytes_at_full_store",
        "gpu_buffer_full_store_bytes",
    ):
        if key in resource_model:
            value = int(resource_model[key])
            if value < 0:
                raise ValueError(f"{key} must be non-negative")
            return value
    prefetch = int(
        resource_model.get("gpu_prefetch_buffer_bytes_per_store_fraction", 0)
    )
    scratch = int(
        resource_model.get("gpu_output_scratch_bytes_per_store_fraction", 0)
    )
    if prefetch < 0 or scratch < 0:
        raise ValueError("component GPU buffer coefficients must be non-negative")
    if prefetch or scratch:
        return prefetch + scratch
    raise ValueError(
        "profile weight_resource_model is missing "
        "gpu_buffer_bytes_per_store_fraction"
    )


def _fraction_int_map(
    resource_model: Mapping[str, Any],
    *,
    keys: Sequence[str],
    field_name: str,
) -> dict[float, int] | None:
    for key in keys:
        if key not in resource_model:
            continue
        raw = resource_model[key]
        if not isinstance(raw, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        parsed: dict[float, int] = {}
        for fraction, value in raw.items():
            parsed_fraction = float(fraction)
            if parsed_fraction < 0.0 or parsed_fraction > 1.0:
                raise ValueError(
                    f"{field_name} keys must be in [0, 1], got {parsed_fraction}"
                )
            parsed_value = int(value)
            if parsed_value < 0:
                raise ValueError(f"{field_name} values must be non-negative")
            parsed[round(parsed_fraction, 12)] = parsed_value
        return dict(sorted(parsed.items()))
    return None


def _default_engine_budget_step(gpu_budget_bytes: int) -> int:
    return max(1, min(1 << 30, max(1, int(gpu_budget_bytes) // 16)))


def _partitioner_for_role(
    *,
    profile_path: Path,
    buckets: Sequence[int],
    bucket_weights: Mapping[int, float] | None,
    candidate_ratio_step: float,
    total_weight_override: int | None,
    buffer_override: int | None,
):
    profile = WeightDispatchCostProfile.load_json(profile_path)
    resource_model = _resource_model(profile)
    total_weight_bytes = _resource_int(
        resource_model,
        keys=("total_weight_bytes", "model_weight_bytes", "weight_bytes"),
        field_name="total_weight_bytes",
        override=total_weight_override,
    )
    gpu_buffer_coeff = _buffer_coefficient(
        resource_model,
        override=buffer_override,
    )
    snap_maps = cots_snap_resource_maps_from_metadata(profile.metadata)
    cpu_weight_bytes_by_store = _fraction_int_map(
        resource_model,
        keys=(
            "cpu_weight_bytes_by_store_fraction",
            "cpu_weight_bytes_by_store",
            "weights_saved_bytes_by_store_fraction",
        ),
        field_name="cpu_weight_bytes_by_store_fraction",
    ) or snap_maps.get("cpu_weight_bytes_by_store_fraction")
    gpu_buffer_bytes_by_store = _fraction_int_map(
        resource_model,
        keys=(
            "gpu_buffer_bytes_by_store_fraction",
            "gpu_buffer_bytes_by_store",
        ),
        field_name="gpu_buffer_bytes_by_store_fraction",
    ) or snap_maps.get("gpu_buffer_bytes_by_store_fraction")
    return WeightKVPartitioner(
        profile=profile,
        buckets=buckets,
        bucket_weights=bucket_weights,
        candidate_ratio_step=candidate_ratio_step,
        total_weight_bytes=total_weight_bytes,
        cpu_weight_bytes_by_store_fraction=cpu_weight_bytes_by_store,
        gpu_buffer_bytes_per_store_fraction=gpu_buffer_coeff,
        gpu_buffer_bytes_by_store_fraction=gpu_buffer_bytes_by_store,
    )


def _row_for_candidate(role: str, candidate: WeightKVCandidateScore) -> dict[str, Any]:
    resources = candidate.resources
    return {
        "role": role,
        "f_cpu_store": candidate.f_cpu_store,
        "expected_s": candidate.expected_s,
        "gpu_budget_bytes": resources.gpu_bytes,
        "gpu_weight_bytes": resources.gpu_weight_bytes,
        "gpu_buffer_bytes": resources.gpu_buffer_bytes,
        "gpu_kv_bytes": resources.gpu_kv_bytes,
        "cpu_weight_bytes": resources.cpu_weight_bytes,
        "cpu_kv_bytes": resources.cpu_kv_bytes,
        "peak_prefetch_fraction": candidate.peak_prefetch_fraction,
        "dispatch_table": {
            str(bucket): [f_cpu, f_prefetch]
            for bucket, (f_cpu, f_prefetch) in sorted(
                candidate.dispatch_table.items()
            )
        },
    }


def _print_plan(payload: Mapping[str, Any]) -> None:
    summary = payload["summary"]
    print("Planner dry run")
    print(
        "  objective_s={objective_s:.6f}  gpu={gpu}  cpu={cpu}  "
        "candidates={num_candidates}".format(
            objective_s=summary["objective_s"],
            gpu=_format_bytes(summary["gpu_bytes"]),
            cpu=_format_bytes(summary["cpu_bytes"]),
            num_candidates=summary["num_candidates"],
        )
    )
    print()
    header = (
        "role       s      expected_s  gpu_budget  gpu_weight  gpu_buffer  "
        "gpu_kv     cpu_weight  cpu_kv     peak_pref"
    )
    print(header)
    print("-" * len(header))
    for row in payload["placements"]:
        print(
            f"{row['role']:<9} "
            f"{row['f_cpu_store']:<6.3f} "
            f"{row['expected_s']:<11.6f} "
            f"{_format_bytes(row['gpu_budget_bytes']):<11} "
            f"{_format_bytes(row['gpu_weight_bytes']):<11} "
            f"{_format_bytes(row['gpu_buffer_bytes']):<11} "
            f"{_format_bytes(row['gpu_kv_bytes']):<10} "
            f"{_format_bytes(row['cpu_weight_bytes']):<11} "
            f"{_format_bytes(row['cpu_kv_bytes']):<10} "
            f"{row['peak_prefetch_fraction']:.3f}"
        )
    print()
    for row in payload["placements"]:
        print(f"{row['role']} dispatch")
        for bucket, split in row["dispatch_table"].items():
            f_cpu, f_prefetch = split
            print(
                f"  B={int(bucket):<5} "
                f"f_cpu={f_cpu:.6f}  f_prefetch={f_prefetch:.6f}"
            )
        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-profile", type=Path, required=True)
    parser.add_argument("--verifier-profile", type=Path, required=True)
    parser.add_argument("--gpu-budget-bytes", type=int)
    parser.add_argument("--gpu-budget-gb", type=float)
    parser.add_argument("--cpu-budget-bytes", type=int)
    parser.add_argument("--cpu-budget-gb", type=float)
    parser.add_argument("--engine-gpu-budget-step-bytes", type=int)
    parser.add_argument("--engine-gpu-budget-step-gb", type=float)
    parser.add_argument(
        "--generator-f-cpu-store-candidates",
        type=lambda value: _parse_fraction_list(
            value,
            field_name="generator f_cpu_store candidates",
        ),
        help=(
            "Optional comma-separated store-fraction grid. Defaults to the "
            "profile's common K(B,s) support over the selected buckets."
        ),
    )
    parser.add_argument(
        "--verifier-f-cpu-store-candidates",
        type=lambda value: _parse_fraction_list(
            value,
            field_name="verifier f_cpu_store candidates",
        ),
        help=(
            "Optional comma-separated store-fraction grid. Defaults to the "
            "profile's common K(B,s) support over the selected buckets."
        ),
    )
    parser.add_argument(
        "--dispatch-buckets",
        type=lambda value: _parse_int_list(value, field_name="dispatch buckets"),
        help="Comma-separated bucket list. Defaults to each profile's buckets.",
    )
    parser.add_argument(
        "--bucket-weights",
        help='JSON object, @json-file, or "8:0.2,16:0.8". Applied to both roles.',
    )
    parser.add_argument("--candidate-ratio-step", type=float, default=0.125)
    parser.add_argument("--generator-objective-weight", type=float, default=1.0)
    parser.add_argument("--verifier-objective-weight", type=float, default=1.0)
    parser.add_argument("--generator-total-weight-bytes", type=int)
    parser.add_argument("--verifier-total-weight-bytes", type=int)
    parser.add_argument("--generator-gpu-buffer-bytes-per-store-fraction", type=int)
    parser.add_argument("--verifier-gpu-buffer-bytes-per-store-fraction", type=int)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    gpu_budget_bytes = _bytes_from_args(
        byte_value=args.gpu_budget_bytes,
        gb_value=args.gpu_budget_gb,
        field_name="gpu",
    )
    cpu_budget_bytes = _bytes_from_args(
        byte_value=args.cpu_budget_bytes,
        gb_value=args.cpu_budget_gb,
        field_name="cpu",
    )
    if args.engine_gpu_budget_step_bytes is not None:
        engine_budget_step_bytes = int(args.engine_gpu_budget_step_bytes)
    elif args.engine_gpu_budget_step_gb is not None:
        engine_budget_step_bytes = int(float(args.engine_gpu_budget_step_gb) * (1 << 30))
    else:
        engine_budget_step_bytes = _default_engine_budget_step(gpu_budget_bytes)
    if engine_budget_step_bytes <= 0:
        raise ValueError("engine GPU budget step must be positive")

    generator_profile = WeightDispatchCostProfile.load_json(args.generator_profile)
    verifier_profile = WeightDispatchCostProfile.load_json(args.verifier_profile)
    generator_buckets = (
        args.dispatch_buckets
        if args.dispatch_buckets is not None
        else tuple(sorted(generator_profile.buckets))
    )
    verifier_buckets = (
        args.dispatch_buckets
        if args.dispatch_buckets is not None
        else tuple(sorted(verifier_profile.buckets))
    )
    generator_f_cpu_store_candidates = (
        args.generator_f_cpu_store_candidates
        if args.generator_f_cpu_store_candidates is not None
        else derive_weight_store_candidates_from_profile(
            generator_profile,
            generator_buckets,
            field_name="generator f_cpu_store candidates",
        )
    )
    verifier_f_cpu_store_candidates = (
        args.verifier_f_cpu_store_candidates
        if args.verifier_f_cpu_store_candidates is not None
        else derive_weight_store_candidates_from_profile(
            verifier_profile,
            verifier_buckets,
            field_name="verifier f_cpu_store candidates",
        )
    )
    bucket_weights = _parse_bucket_weights(args.bucket_weights)

    generator_partitioner = _partitioner_for_role(
        profile_path=args.generator_profile,
        buckets=generator_buckets,
        bucket_weights=bucket_weights,
        candidate_ratio_step=args.candidate_ratio_step,
        total_weight_override=args.generator_total_weight_bytes,
        buffer_override=args.generator_gpu_buffer_bytes_per_store_fraction,
    )
    verifier_partitioner = _partitioner_for_role(
        profile_path=args.verifier_profile,
        buckets=verifier_buckets,
        bucket_weights=bucket_weights,
        candidate_ratio_step=args.candidate_ratio_step,
        total_weight_override=args.verifier_total_weight_bytes,
        buffer_override=args.verifier_gpu_buffer_bytes_per_store_fraction,
    )
    result = ModelMemoryPartitioner(
        gpu_budget_bytes=gpu_budget_bytes,
        cpu_budget_bytes=cpu_budget_bytes,
        engine_weights={
            "generator": args.generator_objective_weight,
            "verifier": args.verifier_objective_weight,
        },
    ).solve_from_partitioners(
        generator_partitioner=generator_partitioner,
        verifier_partitioner=verifier_partitioner,
        generator_f_cpu_store_candidates=generator_f_cpu_store_candidates,
        verifier_f_cpu_store_candidates=verifier_f_cpu_store_candidates,
        engine_gpu_budget_step_bytes=engine_budget_step_bytes,
    )

    payload = {
        "summary": {
            "objective_s": result.best.objective_s,
            "gpu_bytes": result.best.gpu_bytes,
            "cpu_bytes": result.best.cpu_bytes,
            "num_candidates": len(result.candidates),
            "engine_gpu_budget_step_bytes": engine_budget_step_bytes,
        },
        "placements": [
            _row_for_candidate("generator", result.best.generator),
            _row_for_candidate("verifier", result.best.verifier),
        ],
    }
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_plan(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
