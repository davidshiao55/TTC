#!/usr/bin/env python3
"""Validate the Planner weight-dispatch solver against measured grid summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TTC_ROOT = Path(__file__).resolve().parents[3]
FASTTTS_ROOT = TTC_ROOT / "FastTTS-thesis"
if str(FASTTTS_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTTTS_ROOT))

from planner import (  # noqa: E402
    WeightDispatchCostProfile,
    solve_weight_dispatch_split,
)


@dataclass(frozen=True)
class MeasuredRow:
    source: str
    mode: str
    batch: int
    f_cpu_store: float
    f_cpu: float
    f_prefetch: float
    latency_s: float


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results/planner/weight_dispatch_solver_validation") / stamp


def ftag(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.5f}".rstrip("0").rstrip(".")


def load_measured_rows(paths: Iterable[Path]) -> list[MeasuredRow]:
    rows: list[MeasuredRow] = []
    for path in paths:
        data = json.loads(path.read_text())
        extra_args = data.get("config", {}).get("extra_vllm_args", [])
        if "--cots-dry-run" in extra_args:
            continue
        for raw in data.get("rows", []):
            latency = raw.get("mean_latency_s")
            if (
                latency is None
                or raw.get("f_cpu_store") is None
                or raw.get("f_cpu") is None
                or raw.get("f_prefetch") is None
            ):
                continue
            rows.append(
                MeasuredRow(
                    source=str(path),
                    mode=str(raw["mode"]),
                    batch=int(raw["batch"]),
                    f_cpu_store=float(raw["f_cpu_store"]),
                    f_cpu=float(raw["f_cpu"]),
                    f_prefetch=float(raw["f_prefetch"]),
                    latency_s=float(latency),
                )
            )
    return rows


def average_duplicate_rows(rows: Iterable[MeasuredRow]) -> list[MeasuredRow]:
    grouped: dict[tuple[Any, ...], list[MeasuredRow]] = defaultdict(list)
    for row in rows:
        key = (row.mode, row.batch, row.f_cpu_store, row.f_cpu, row.f_prefetch)
        grouped[key].append(row)

    averaged = []
    for key, group in grouped.items():
        mode, batch, f_cpu_store, f_cpu, f_prefetch = key
        averaged.append(
            MeasuredRow(
                source="+".join(sorted({row.source for row in group})),
                mode=mode,
                batch=batch,
                f_cpu_store=f_cpu_store,
                f_cpu=f_cpu,
                f_prefetch=f_prefetch,
                latency_s=sum(row.latency_s for row in group) / len(group),
            )
        )
    return sorted(
        averaged,
        key=lambda row: (row.mode, row.batch, row.f_cpu_store, row.f_cpu),
    )


def min_grid_step(values: list[float]) -> float:
    values = sorted(set(values))
    if len(values) < 2:
        return 0.0
    return min(
        abs(b - a)
        for a, b in zip(values, values[1:])
        if abs(b - a) > 1e-12
    )


def validate_solver(
    profile: WeightDispatchCostProfile,
    rows: list[MeasuredRow],
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, float], list[MeasuredRow]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    for row in rows:
        if row.batch not in profile.buckets:
            key = {
                "mode": row.mode,
                "batch": row.batch,
                "f_cpu_store": row.f_cpu_store,
                "reason": "missing profile bucket",
            }
            if key not in skipped:
                skipped.append(key)
            continue
        grouped[(row.mode, row.batch, row.f_cpu_store)].append(row)

    checks = []
    exact = 0
    within_one_step = 0
    for (mode, batch, f_cpu_store), group in sorted(grouped.items()):
        candidate_us = sorted({row.f_cpu for row in group})
        measured_best = min(group, key=lambda row: row.latency_s)
        split = solve_weight_dispatch_split(
            profile.cost_for_bucket(batch),
            f_cpu_store=f_cpu_store,
            candidate_f_cpu=candidate_us,
        )
        predicted_cell = min(group, key=lambda row: abs(row.f_cpu - split.f_cpu))
        step = min_grid_step(candidate_us)
        delta_u = abs(measured_best.f_cpu - split.f_cpu)
        is_exact = delta_u <= 1e-9
        is_within = delta_u <= step + 1e-9
        exact += int(is_exact)
        within_one_step += int(is_within)
        checks.append(
            {
                "mode": mode,
                "batch": batch,
                "f_cpu_store": f_cpu_store,
                "measured_best_f_cpu": measured_best.f_cpu,
                "measured_best_f_prefetch": measured_best.f_prefetch,
                "measured_best_s": measured_best.latency_s,
                "predicted_f_cpu": split.f_cpu,
                "predicted_f_prefetch": split.f_prefetch,
                "predicted_cell_measured_s": predicted_cell.latency_s,
                "predicted_resource_s": split.resource_s,
                "predicted_bottleneck": split.bottleneck,
                "lane_scores_s": dict(split.lane_scores_s),
                "grid_step": step,
                "delta_f_cpu": delta_u,
                "exact": is_exact,
                "within_one_step": is_within,
            }
        )

    return {
        "dispatch_model": profile.dispatch_model,
        "num_checks": len(checks),
        "exact": exact,
        "within_one_step": within_one_step,
        "skipped": skipped,
        "checks": checks,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Weight Dispatch Solver Validation", ""]
    lines.append(f"Model: `{report['dispatch_model']}`")
    lines.append(
        f"Checks: {report['exact']}/{report['num_checks']} exact, "
        f"{report['within_one_step']}/{report['num_checks']} within one grid step."
    )
    lines.append("")
    lines.append(
        "| mode | B | f_store | measured u | planner u | planner p | "
        "bottleneck | measured best s | planner-cell measured s | exact | +/-1 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|")
    for item in report["checks"]:
        lines.append(
            f"| `{item['mode']}` | {item['batch']} | "
            f"{ftag(item['f_cpu_store'])} | "
            f"{ftag(item['measured_best_f_cpu'])} | "
            f"{ftag(item['predicted_f_cpu'])} | "
            f"{ftag(item['predicted_f_prefetch'])} | "
            f"{item['predicted_bottleneck']} | "
            f"{item['measured_best_s']:.4f} | "
            f"{item['predicted_cell_measured_s']:.4f} | "
            f"{item['exact']} | {item['within_one_step']} |"
        )
    if report["skipped"]:
        lines.append("")
        lines.append("## Skipped")
        lines.append("")
        lines.append("| mode | B | f_store | reason |")
        lines.append("|---|---:|---:|---|")
        for item in report["skipped"]:
            lines.append(
                f"| `{item['mode']}` | {item['batch']} | "
                f"{ftag(item['f_cpu_store'])} | {item['reason']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("summary_json", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--no-fail-on-miss",
        action="store_true",
        help="Write the report but return success even if a check misses by >1 grid step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = WeightDispatchCostProfile.load_json(args.profile)
    rows = average_duplicate_rows(load_measured_rows(args.summary_json))
    if not rows:
        raise SystemExit("no measured real rows found")
    report = validate_solver(profile, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "solver_validation.json"
    out_md = args.output_dir / "solver_validation.md"
    out_json.write_text(json.dumps(report, indent=2))
    write_markdown(report, out_md)
    print(f"[validate] wrote {out_json}")
    print(f"[validate] wrote {out_md}")
    print(
        "[validate] exact="
        f"{report['exact']}/{report['num_checks']} "
        "within_one_step="
        f"{report['within_one_step']}/{report['num_checks']}"
    )
    if (
        not args.no_fail_on_miss
        and report["within_one_step"] != report["num_checks"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
