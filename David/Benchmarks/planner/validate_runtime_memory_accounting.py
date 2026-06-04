#!/usr/bin/env python3
"""Validate planner memory accounting against vLLM startup logs.

This tool consumes the JSON emitted by ``plan_from_profiles.py --output-json``
and a vLLM/FastTTS runtime log. It checks the load-time accounting terms that
the planner now owns:

* CPU weight bytes against COTS ``weights_saved``.
* GPU COTS buffer bytes against ``gpu_uva + prefetch_pool``.
* GPU KV bytes against the manual ``kv_cache_memory_bytes`` reservation.
* CPU KV bytes against the COTS hybrid KV CPU pool, when present.
* Dispatch table and dispatch bucket coverage, when the runtime policy log is
  available.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

COTS_READY_RE = re.compile(
    r"\[CotsOffloader\] ready: .*?"
    rf"weights_saved=(?P<weights>{FLOAT_RE}) GB, "
    rf"buffers=(?P<x_pinned>{FLOAT_RE}) GB pinned_in "
    rf"\+ (?P<y_pinned>{FLOAT_RE}) GB pinned_out "
    rf"\+ (?P<gpu_uva>{FLOAT_RE}) GB gpu_uva, "
    rf"prefetch_pool=(?P<prefetch_pool>{FLOAT_RE}) GB, "
    r"graph_buckets=(?P<graph_buckets>\([^)]*\)), "
    r"dispatch_buckets=(?P<dispatch_buckets>\([^)]*\))"
)

COTS_POLICY_RE = re.compile(
    r"\[CotsOffloader\] dispatch policy: "
    rf"f_cpu_store=(?P<f_cpu_store>{FLOAT_RE}), "
    r"dispatch_table=(?P<dispatch_table>\{.*?\})"
    r"(?:, dispatch_table_by_module=(?P<dispatch_table_by_module>\{.*\}))?$"
)

KV_RESERVED_RAW_RE = re.compile(
    r"reserved "
    rf"(?P<gib>{FLOAT_RE}) GiB "
    r"\(kv_cache_memory_bytes=(?P<bytes>\d+)\) memory for KV Cache"
)
KV_RESERVED_GIB_RE = re.compile(
    r"reserved " rf"(?P<gib>{FLOAT_RE}) GiB memory for KV Cache"
)

HYBRID_KV_RE = re.compile(
    r"Initialized COTS hybrid CPU KV store: .*?"
    r"cpu_pool_bytes=(?P<cpu_pool_bytes>\d+)"
)


@dataclass(frozen=True)
class CotsReadyRecord:
    line_number: int
    cpu_weight_bytes: int
    gpu_uva_bytes: int
    prefetch_pool_bytes: int
    graph_buckets: tuple[int, ...]
    dispatch_buckets: tuple[int, ...]

    @property
    def gpu_buffer_bytes(self) -> int:
        return self.gpu_uva_bytes + self.prefetch_pool_bytes


@dataclass(frozen=True)
class CotsPolicyRecord:
    line_number: int
    f_cpu_store: float
    dispatch_table: dict[int, tuple[float, float]]
    dispatch_table_by_module: dict[str, dict[int, tuple[float, float]]]


@dataclass(frozen=True)
class KVReservationRecord:
    line_number: int
    gpu_kv_bytes: int
    exact_bytes: bool


@dataclass(frozen=True)
class HybridKVRecord:
    line_number: int
    cpu_kv_bytes: int


@dataclass(frozen=True)
class RuntimeMemoryLog:
    cots_ready: tuple[CotsReadyRecord, ...]
    cots_policy: tuple[CotsPolicyRecord, ...]
    kv_reservations: tuple[KVReservationRecord, ...]
    hybrid_kv: tuple[HybridKVRecord, ...]


def _decimal_gb_to_bytes(value: str) -> int:
    return int(round(float(value) * 1_000_000_000))


def _gib_to_bytes(value: str) -> int:
    return int(round(float(value) * (1 << 30)))


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    parsed = ast.literal_eval(raw)
    if isinstance(parsed, int):
        return (int(parsed),)
    if not isinstance(parsed, tuple):
        raise ValueError(f"expected tuple, got {type(parsed).__name__}: {raw}")
    return tuple(int(value) for value in parsed)


def _normalize_dispatch_table(raw: Mapping[Any, Any]) -> dict[int, tuple[float, float]]:
    table: dict[int, tuple[float, float]] = {}
    for bucket, value in raw.items():
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(
                "dispatch_table values must be two-item "
                "(f_cpu_compute, f_prefetch_compute) sequences"
            )
        table[int(bucket)] = (float(value[0]), float(value[1]))
    return dict(sorted(table.items()))


def _parse_policy_record(match: re.Match[str], line_number: int) -> CotsPolicyRecord:
    raw_table = ast.literal_eval(match.group("dispatch_table"))
    if not isinstance(raw_table, Mapping):
        raise ValueError("runtime dispatch_table must be a mapping")
    raw_by_module_text = match.group("dispatch_table_by_module")
    if raw_by_module_text is None:
        raw_by_module: Mapping[str, Any] = {}
    else:
        parsed = ast.literal_eval(raw_by_module_text)
        if not isinstance(parsed, Mapping):
            raise ValueError("runtime dispatch_table_by_module must be a mapping")
        raw_by_module = parsed
    by_module = {
        str(module): _normalize_dispatch_table(table)
        for module, table in raw_by_module.items()
    }
    return CotsPolicyRecord(
        line_number=line_number,
        f_cpu_store=float(match.group("f_cpu_store")),
        dispatch_table=_normalize_dispatch_table(raw_table),
        dispatch_table_by_module=by_module,
    )


def parse_runtime_log_text(text: str) -> RuntimeMemoryLog:
    cots_ready: list[CotsReadyRecord] = []
    cots_policy: list[CotsPolicyRecord] = []
    kv_reservations: list[KVReservationRecord] = []
    hybrid_kv: list[HybridKVRecord] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if match := COTS_READY_RE.search(line):
            cots_ready.append(
                CotsReadyRecord(
                    line_number=line_number,
                    cpu_weight_bytes=_decimal_gb_to_bytes(match.group("weights")),
                    gpu_uva_bytes=_decimal_gb_to_bytes(match.group("gpu_uva")),
                    prefetch_pool_bytes=_decimal_gb_to_bytes(
                        match.group("prefetch_pool")
                    ),
                    graph_buckets=_parse_int_tuple(match.group("graph_buckets")),
                    dispatch_buckets=_parse_int_tuple(match.group("dispatch_buckets")),
                )
            )
            continue
        if match := COTS_POLICY_RE.search(line):
            cots_policy.append(_parse_policy_record(match, line_number))
            continue
        if match := KV_RESERVED_RAW_RE.search(line):
            kv_reservations.append(
                KVReservationRecord(
                    line_number=line_number,
                    gpu_kv_bytes=int(match.group("bytes")),
                    exact_bytes=True,
                )
            )
            continue
        if match := KV_RESERVED_GIB_RE.search(line):
            kv_reservations.append(
                KVReservationRecord(
                    line_number=line_number,
                    gpu_kv_bytes=_gib_to_bytes(match.group("gib")),
                    exact_bytes=False,
                )
            )
            continue
        if match := HYBRID_KV_RE.search(line):
            hybrid_kv.append(
                HybridKVRecord(
                    line_number=line_number,
                    cpu_kv_bytes=int(match.group("cpu_pool_bytes")),
                )
            )

    return RuntimeMemoryLog(
        cots_ready=tuple(cots_ready),
        cots_policy=tuple(cots_policy),
        kv_reservations=tuple(kv_reservations),
        hybrid_kv=tuple(hybrid_kv),
    )


def parse_runtime_log(path: Path) -> RuntimeMemoryLog:
    return parse_runtime_log_text(path.read_text())


def _placement_role(placement: Mapping[str, Any]) -> str:
    role = str(placement.get("role", "")).strip()
    if not role:
        raise ValueError("each plan placement must include a non-empty role")
    return role


def _placement_int(placement: Mapping[str, Any], key: str) -> int:
    value = int(placement.get(key, 0))
    if value < 0:
        raise ValueError(f"{_placement_role(placement)}.{key} must be non-negative")
    return value


def _placement_f_cpu_store(placement: Mapping[str, Any]) -> float:
    return float(placement.get("f_cpu_store", 0.0))


def _placement_dispatch_table(
    placement: Mapping[str, Any],
) -> dict[int, tuple[float, float]]:
    raw = placement.get("dispatch_table", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{_placement_role(placement)}.dispatch_table must be a mapping")
    return _normalize_dispatch_table(raw)


def _is_weight_active(placement: Mapping[str, Any]) -> bool:
    return (
        _placement_f_cpu_store(placement) > 0.0
        or _placement_int(placement, "cpu_weight_bytes") > 0
        or _placement_int(placement, "gpu_buffer_bytes") > 0
    )


def _byte_tolerance(expected: int, *, absolute: int, relative: float) -> int:
    return max(int(absolute), int(round(abs(expected) * relative)))


def _append_byte_check(
    checks: list[dict[str, Any]],
    *,
    role: str,
    field: str,
    expected: int,
    observed: int,
    absolute_tolerance: int,
    relative_tolerance: float,
    source: str,
) -> None:
    tolerance = _byte_tolerance(
        expected,
        absolute=absolute_tolerance,
        relative=relative_tolerance,
    )
    delta = observed - expected
    checks.append(
        {
            "role": role,
            "field": field,
            "expected": expected,
            "observed": observed,
            "delta": delta,
            "tolerance": tolerance,
            "ok": abs(delta) <= tolerance,
            "source": source,
        }
    )


def _append_value_check(
    checks: list[dict[str, Any]],
    *,
    role: str,
    field: str,
    expected: Any,
    observed: Any,
    ok: bool,
    source: str,
    detail: str | None = None,
) -> None:
    check = {
        "role": role,
        "field": field,
        "expected": expected,
        "observed": observed,
        "ok": ok,
        "source": source,
    }
    if detail:
        check["detail"] = detail
    checks.append(check)


def _dispatch_tables_close(
    expected: Mapping[int, tuple[float, float]],
    observed: Mapping[int, tuple[float, float]],
    *,
    fraction_tolerance: float,
) -> tuple[bool, str | None]:
    if set(expected) != set(observed):
        return False, (
            "bucket keys differ: "
            f"expected={sorted(expected)}, observed={sorted(observed)}"
        )
    for bucket, expected_pair in sorted(expected.items()):
        observed_pair = observed[bucket]
        for index, name in enumerate(("f_cpu", "f_prefetch")):
            if abs(observed_pair[index] - expected_pair[index]) > fraction_tolerance:
                return False, (
                    f"bucket={bucket} {name} differs: "
                    f"expected={expected_pair[index]}, observed={observed_pair[index]}"
                )
    return True, None


def validate_runtime_memory_accounting(
    plan_payload: Mapping[str, Any],
    runtime_log: RuntimeMemoryLog,
    *,
    role_order: Sequence[str] | None = None,
    cots_absolute_tolerance_bytes: int = 1 << 20,
    kv_absolute_tolerance_bytes: int = 8 << 20,
    relative_tolerance: float = 1e-3,
    fraction_tolerance: float = 1e-6,
    require_policy: bool = True,
) -> dict[str, Any]:
    placements_raw = plan_payload.get("placements")
    if not isinstance(placements_raw, Sequence):
        raise ValueError("plan JSON must contain a placements array")
    placements = [placement for placement in placements_raw if isinstance(placement, Mapping)]
    if len(placements) != len(placements_raw):
        raise ValueError("all plan placements must be objects")

    role_to_placement = {_placement_role(placement): placement for placement in placements}
    if len(role_to_placement) != len(placements):
        raise ValueError("plan placements must have unique role names")

    roles = list(role_order) if role_order is not None else list(role_to_placement)
    missing_roles = sorted(set(roles) - set(role_to_placement))
    if missing_roles:
        raise ValueError(f"role_order contains roles missing from plan: {missing_roles}")
    extra_roles = sorted(set(role_to_placement) - set(roles))
    roles.extend(extra_roles)

    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    ready_index = 0
    policy_index = 0
    kv_index = 0
    hybrid_index = 0

    for role in roles:
        placement = role_to_placement[role]
        dispatch_table = _placement_dispatch_table(placement)

        if _is_weight_active(placement):
            if ready_index >= len(runtime_log.cots_ready):
                errors.append(f"{role}: missing COTS ready log record")
            else:
                ready = runtime_log.cots_ready[ready_index]
                ready_index += 1
                _append_byte_check(
                    checks,
                    role=role,
                    field="cpu_weight_bytes",
                    expected=_placement_int(placement, "cpu_weight_bytes"),
                    observed=ready.cpu_weight_bytes,
                    absolute_tolerance=cots_absolute_tolerance_bytes,
                    relative_tolerance=relative_tolerance,
                    source=f"CotsOffloader ready line {ready.line_number}",
                )
                _append_byte_check(
                    checks,
                    role=role,
                    field="gpu_buffer_bytes",
                    expected=_placement_int(placement, "gpu_buffer_bytes"),
                    observed=ready.gpu_buffer_bytes,
                    absolute_tolerance=cots_absolute_tolerance_bytes,
                    relative_tolerance=relative_tolerance,
                    source=f"CotsOffloader ready line {ready.line_number}",
                )
                expected_buckets = tuple(sorted(dispatch_table))
                _append_value_check(
                    checks,
                    role=role,
                    field="dispatch_buckets",
                    expected=expected_buckets,
                    observed=ready.dispatch_buckets,
                    ok=tuple(ready.dispatch_buckets) == expected_buckets,
                    source=f"CotsOffloader ready line {ready.line_number}",
                )

            if require_policy:
                if policy_index >= len(runtime_log.cots_policy):
                    errors.append(f"{role}: missing COTS dispatch policy log record")
                else:
                    policy = runtime_log.cots_policy[policy_index]
                    policy_index += 1
                    expected_s = _placement_f_cpu_store(placement)
                    observed_s = policy.f_cpu_store
                    _append_value_check(
                        checks,
                        role=role,
                        field="f_cpu_store",
                        expected=expected_s,
                        observed=observed_s,
                        ok=abs(observed_s - expected_s) <= fraction_tolerance,
                        source=f"CotsOffloader policy line {policy.line_number}",
                    )
                    ok, detail = _dispatch_tables_close(
                        dispatch_table,
                        policy.dispatch_table,
                        fraction_tolerance=fraction_tolerance,
                    )
                    _append_value_check(
                        checks,
                        role=role,
                        field="dispatch_table",
                        expected={
                            bucket: list(pair)
                            for bucket, pair in sorted(dispatch_table.items())
                        },
                        observed={
                            bucket: list(pair)
                            for bucket, pair in sorted(policy.dispatch_table.items())
                        },
                        ok=ok,
                        source=f"CotsOffloader policy line {policy.line_number}",
                        detail=detail,
                    )

        expected_gpu_kv = _placement_int(placement, "gpu_kv_bytes")
        if expected_gpu_kv > 0:
            if kv_index >= len(runtime_log.kv_reservations):
                errors.append(f"{role}: missing GPU KV reservation log record")
            else:
                kv = runtime_log.kv_reservations[kv_index]
                kv_index += 1
                _append_byte_check(
                    checks,
                    role=role,
                    field="gpu_kv_bytes",
                    expected=expected_gpu_kv,
                    observed=kv.gpu_kv_bytes,
                    absolute_tolerance=0
                    if kv.exact_bytes
                    else kv_absolute_tolerance_bytes,
                    relative_tolerance=0.0 if kv.exact_bytes else relative_tolerance,
                    source=f"GPU worker KV line {kv.line_number}",
                )

        expected_cpu_kv = _placement_int(placement, "cpu_kv_bytes")
        if expected_cpu_kv > 0:
            if hybrid_index >= len(runtime_log.hybrid_kv):
                errors.append(f"{role}: missing COTS hybrid KV log record")
            else:
                hybrid = runtime_log.hybrid_kv[hybrid_index]
                hybrid_index += 1
                _append_byte_check(
                    checks,
                    role=role,
                    field="cpu_kv_bytes",
                    expected=expected_cpu_kv,
                    observed=hybrid.cpu_kv_bytes,
                    absolute_tolerance=0,
                    relative_tolerance=0.0,
                    source=f"COTS hybrid KV line {hybrid.line_number}",
                )

    failed = [check for check in checks if not check["ok"]]
    ok = not errors and not failed
    return {
        "ok": ok,
        "errors": errors,
        "checks": checks,
        "runtime_records": {
            "cots_ready": len(runtime_log.cots_ready),
            "cots_policy": len(runtime_log.cots_policy),
            "kv_reservations": len(runtime_log.kv_reservations),
            "hybrid_kv": len(runtime_log.hybrid_kv),
        },
    }


def _format_bytes(value: int) -> str:
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1 << 30:
        return f"{sign}{value / (1 << 30):.3f} GiB"
    if value >= 1 << 20:
        return f"{sign}{value / (1 << 20):.3f} MiB"
    if value >= 1 << 10:
        return f"{sign}{value / (1 << 10):.3f} KiB"
    return f"{sign}{value} B"


def _print_report(report: Mapping[str, Any]) -> None:
    print("Runtime memory accounting validation")
    print("  status:", "PASS" if report["ok"] else "FAIL")
    print(
        "  records: "
        + ", ".join(
            f"{name}={count}"
            for name, count in report.get("runtime_records", {}).items()
        )
    )
    print()

    if report.get("errors"):
        print("Errors")
        for error in report["errors"]:
            print(f"  - {error}")
        print()

    checks = list(report.get("checks", ()))
    if not checks:
        return
    header = "role       field               status  delta       tolerance   source"
    print(header)
    print("-" * len(header))
    for check in checks:
        status = "ok" if check["ok"] else "FAIL"
        if "delta" in check:
            delta = _format_bytes(int(check["delta"]))
            tolerance = _format_bytes(int(check["tolerance"]))
        else:
            delta = "-"
            tolerance = "-"
        print(
            f"{check['role']:<10} "
            f"{check['field']:<19} "
            f"{status:<6} "
            f"{delta:<11} "
            f"{tolerance:<11} "
            f"{check['source']}"
        )
        if check.get("detail"):
            print(f"  detail: {check['detail']}")


def _bytes_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("byte tolerances must be non-negative")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument(
        "--role-order",
        help="Comma-separated role order for matching repeated runtime records.",
    )
    parser.add_argument(
        "--cots-byte-tolerance",
        type=_bytes_arg,
        default=1 << 20,
        help="Absolute tolerance for decimal-GB COTS weight/buffer logs.",
    )
    parser.add_argument(
        "--kv-byte-tolerance",
        type=_bytes_arg,
        default=8 << 20,
        help="Absolute tolerance for legacy rounded-GiB KV reservation logs.",
    )
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--fraction-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--no-require-policy",
        action="store_true",
        help="Skip dispatch policy checks for logs produced before that line existed.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    plan_payload = json.loads(args.plan_json.read_text())
    runtime_log = parse_runtime_log(args.runtime_log)
    role_order = None
    if args.role_order:
        role_order = tuple(
            role.strip() for role in args.role_order.split(",") if role.strip()
        )
        if not role_order:
            raise ValueError("--role-order must not be empty")
    report = validate_runtime_memory_accounting(
        plan_payload,
        runtime_log,
        role_order=role_order,
        cots_absolute_tolerance_bytes=args.cots_byte_tolerance,
        kv_absolute_tolerance_bytes=args.kv_byte_tolerance,
        relative_tolerance=args.relative_tolerance,
        fraction_tolerance=args.fraction_tolerance,
        require_policy=not args.no_require_policy,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
