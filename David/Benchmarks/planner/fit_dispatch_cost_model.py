#!/usr/bin/env python3
"""Fit the Planner dispatch cost model from validation summary files.

This script is intentionally lightweight: it reads one or more
``bench_dispatch_model_validation.py`` summary JSON files, averages duplicate
cells, and fits bucket-local effective slopes for the decode dispatch model.

Real rows:

    T(B, s, u) = K(B, s) + max(
        G_B * (1 - u),
        C_B * u,
        H_B * (s - u),
    )

Dry-run rows:

    T_dry(B, s, route) = D(B, route) - G_B * s

The real fit is meant to validate dispatch ranking first. ``G_B`` comes from
dry-run when available; otherwise the fitter falls back to the CPU/H2D active
subset. The per-store K(B, s) term absorbs split-invariant setup/prefill cost
so the slopes focus on the split choice inside a fixed (B, s) case.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Row:
    mode: str
    batch: int
    f_cpu_store: float | None
    f_cpu: float | None
    f_prefetch: float | None
    latency_s: float
    source: str
    dry_run: bool

    @property
    def is_baseline(self) -> bool:
        return self.f_cpu_store is None


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results/planner/dispatch_cost_model_fit") / stamp


def ftag(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.5f}".rstrip("0").rstrip(".")


def load_rows(paths: Iterable[Path]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        data = json.loads(path.read_text())
        extra_args = data.get("config", {}).get("extra_vllm_args", [])
        dry_run = "--cots-dry-run" in extra_args
        for raw in data.get("rows", []):
            latency = raw.get("mean_latency_s")
            if latency is None:
                continue
            rows.append(
                Row(
                    mode=str(raw["mode"]),
                    batch=int(raw["batch"]),
                    f_cpu_store=raw.get("f_cpu_store"),
                    f_cpu=raw.get("f_cpu"),
                    f_prefetch=raw.get("f_prefetch"),
                    latency_s=float(latency),
                    source=str(path),
                    dry_run=dry_run,
                )
            )
    return rows


def average_duplicate_rows(rows: Iterable[Row]) -> list[Row]:
    grouped: dict[tuple[Any, ...], list[Row]] = defaultdict(list)
    for row in rows:
        key = (
            row.mode,
            row.batch,
            row.f_cpu_store,
            row.f_cpu,
            row.f_prefetch,
            row.dry_run,
        )
        grouped[key].append(row)

    merged: list[Row] = []
    for key, group in grouped.items():
        mode, batch, s, u, p, dry_run = key
        merged.append(
            Row(
                mode=mode,
                batch=batch,
                f_cpu_store=s,
                f_cpu=u,
                f_prefetch=p,
                latency_s=float(np.mean([row.latency_s for row in group])),
                source="+".join(sorted({row.source for row in group})),
                dry_run=dry_run,
            )
        )
    return sorted(
        merged,
        key=lambda row: (
            row.mode,
            row.batch,
            -1.0 if row.f_cpu_store is None else row.f_cpu_store,
            -1.0 if row.f_cpu is None else row.f_cpu,
        ),
    )


def group_by_batch(rows: Iterable[Row]) -> dict[int, list[Row]]:
    grouped: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[row.batch].append(row)
    return dict(sorted(grouped.items()))


def real_rows(rows: Iterable[Row]) -> list[Row]:
    return [
        row
        for row in rows
        if not row.is_baseline
        and not row.dry_run
        and row.f_cpu_store is not None
        and row.f_cpu is not None
        and row.f_prefetch is not None
    ]


def dry_rows(rows: Iterable[Row]) -> list[Row]:
    return [
        row
        for row in rows
        if not row.is_baseline
        and row.dry_run
        and row.f_cpu_store is not None
        and row.f_cpu is not None
        and row.f_prefetch is not None
    ]


def baseline_by_batch(rows: Iterable[Row]) -> dict[int, float]:
    baselines: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.is_baseline:
            baselines[row.batch].append(row.latency_s)
    return {batch: float(np.mean(values)) for batch, values in baselines.items()}


def common_value(values: Iterable[Any]) -> Any:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    first = filtered[0]
    if all(value == first for value in filtered):
        return first
    return sorted({str(value) for value in filtered})


def load_source_metadata(paths: Iterable[Path]) -> dict[str, Any]:
    summaries = []
    for path in paths:
        data = json.loads(path.read_text())
        env = data.get("env", {})
        config = data.get("config", {})
        summaries.append(
            {
                "path": str(path),
                "gpu": env.get("gpu"),
                "model": config.get("model"),
                "dtype": config.get("dtype"),
                "modes": config.get("modes"),
                "input_len": config.get("input_len"),
                "output_len": config.get("output_len"),
                "dispatch_layout": config.get("dispatch_layout"),
            }
        )

    return {
        "source_summaries": summaries,
        "gpu": common_value(item.get("gpu") for item in summaries),
        "model": common_value(item.get("model") for item in summaries),
        "dtype": common_value(item.get("dtype") for item in summaries),
        "modes": common_value(
            tuple(item.get("modes") or []) for item in summaries
        ),
        "input_len": common_value(item.get("input_len") for item in summaries),
        "output_len": common_value(item.get("output_len") for item in summaries),
        "dispatch_layout": common_value(
            item.get("dispatch_layout") for item in summaries
        ),
    }


def _per_store_k_terms(
    rows: list[Row],
    z_values: list[float],
) -> dict[float, float]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row, z in zip(rows, z_values):
        assert row.f_cpu_store is not None
        grouped[float(row.f_cpu_store)].append(row.latency_s - z)
    return {s: float(np.mean(values)) for s, values in grouped.items()}


def _resource_terms(
    row: Row,
    *,
    c_slope: float,
    h_slope: float,
    g_slope: float | None,
) -> dict[str, float]:
    assert row.f_cpu is not None
    assert row.f_prefetch is not None
    terms = {
        "cpu": c_slope * float(row.f_cpu),
        "h2d": h_slope * float(row.f_prefetch),
    }
    if g_slope is not None:
        terms["gpu"] = g_slope * (1.0 - float(row.f_cpu))
    return terms


def _winning_lane(terms: dict[str, float]) -> str:
    return max(sorted(terms), key=lambda key: terms[key])


def _score_resource_model(
    rows: list[Row],
    c_slope: float,
    h_slope: float,
    g_slope: float | None,
) -> tuple[float, dict[float, float], list[float], list[dict[str, float]]]:
    terms_by_row = [
        _resource_terms(
            row,
            c_slope=c_slope,
            h_slope=h_slope,
            g_slope=g_slope,
        )
        for row in rows
    ]
    z_values = [max(terms.values()) for terms in terms_by_row]
    k_terms = _per_store_k_terms(rows, z_values)
    preds = [
        k_terms[float(row.f_cpu_store)] + z
        for row, z in zip(rows, z_values)
    ]
    errors = [pred - row.latency_s for pred, row in zip(preds, rows)]
    return (
        float(np.mean([err * err for err in errors])),
        k_terms,
        preds,
        terms_by_row,
    )


def fit_resource_model(rows: list[Row], g_slope: float | None) -> dict[str, Any]:
    if not rows:
        raise ValueError("no rows to fit")
    max_latency = max(row.latency_s for row in rows)
    max_store = max(float(row.f_cpu_store) for row in rows if row.f_cpu_store)
    upper = max(80.0, 2.5 * max_latency / max_store)

    best: tuple[float, float, float] | None = None
    c_lo, c_hi = 0.0, upper
    h_lo, h_hi = 0.0, upper
    for round_idx in range(6):
        n_grid = 81 if round_idx == 0 else 61
        c_grid = np.linspace(c_lo, c_hi, n_grid)
        h_grid = np.linspace(h_lo, h_hi, n_grid)
        for c_slope in c_grid:
            for h_slope in h_grid:
                mse, _, _, _ = _score_resource_model(
                    rows,
                    float(c_slope),
                    float(h_slope),
                    g_slope,
                )
                if best is None or mse < best[0]:
                    best = (mse, float(c_slope), float(h_slope))
        assert best is not None
        _, best_c, best_h = best
        c_span = max((c_hi - c_lo) / 8.0, 1e-6)
        h_span = max((h_hi - h_lo) / 8.0, 1e-6)
        c_lo = max(0.0, best_c - c_span)
        c_hi = best_c + c_span
        h_lo = max(0.0, best_h - h_span)
        h_hi = best_h + h_span

    assert best is not None
    mse, c_slope, h_slope = best
    mse, k_terms, preds, terms_by_row = _score_resource_model(
        rows,
        c_slope,
        h_slope,
        g_slope,
    )
    rmse = math.sqrt(mse)
    mae = float(np.mean([abs(pred - row.latency_s) for pred, row in zip(preds, rows)]))

    row_predictions = []
    lane_counts: dict[str, int] = defaultdict(int)
    for row, pred, terms in zip(rows, preds, terms_by_row):
        lane = _winning_lane(terms)
        lane_counts[lane] += 1
        row_predictions.append(
            {
                "f_cpu_store": row.f_cpu_store,
                "f_cpu": row.f_cpu,
                "f_prefetch": row.f_prefetch,
                "measured_s": row.latency_s,
                "predicted_s": pred,
                "error_s": pred - row.latency_s,
                "resource_terms_s": terms,
                "winning_lane": lane,
            }
        )

    by_store: dict[float, list[tuple[Row, float]]] = defaultdict(list)
    for row, pred in zip(rows, preds):
        by_store[float(row.f_cpu_store)].append((row, pred))

    rank_checks = []
    exact = 0
    within_one_step = 0
    for store, entries in sorted(by_store.items()):
        measured_best = min(entries, key=lambda item: item[0].latency_s)[0]
        predicted_best = min(entries, key=lambda item: item[1])[0]
        predicted_best_terms = _resource_terms(
            predicted_best,
            c_slope=c_slope,
            h_slope=h_slope,
            g_slope=g_slope,
        )
        candidate_us = sorted({float(row.f_cpu) for row, _ in entries})
        if len(candidate_us) > 1:
            min_step = min(
                abs(b - a)
                for a, b in zip(candidate_us, candidate_us[1:])
                if abs(b - a) > 1e-12
            )
        else:
            min_step = 0.0
        delta = abs(float(measured_best.f_cpu) - float(predicted_best.f_cpu))
        is_exact = delta <= 1e-9
        is_within = delta <= min_step + 1e-9
        exact += int(is_exact)
        within_one_step += int(is_within)
        rank_checks.append(
            {
                "f_cpu_store": store,
                "measured_best_f_cpu": measured_best.f_cpu,
                "measured_best_s": measured_best.latency_s,
                "predicted_best_f_cpu": predicted_best.f_cpu,
                "predicted_best_measured_s": predicted_best.latency_s,
                "predicted_best_winning_lane": _winning_lane(predicted_best_terms),
                "grid_step": min_step,
                "exact": is_exact,
                "within_one_step": is_within,
            }
        )

    optimum_ratio = h_slope / (h_slope + c_slope) if h_slope + c_slope > 0 else None
    model = (
        "K(B,s) + max(G_B*(1-u), C_B*u, H_B*(s-u))"
        if g_slope is not None
        else "K(B,s) + max(C_B*u, H_B*(s-u))"
    )
    return {
        "model": model,
        "g_s_per_fraction_fixed": g_slope,
        "c_s_per_fraction": c_slope,
        "h_s_per_fraction": h_slope,
        "cpu_to_h2d_slope_ratio": c_slope / h_slope if h_slope > 0 else None,
        "continuous_u_over_s_optimum": optimum_ratio,
        "rmse_s": rmse,
        "mae_s": mae,
        "rank_exact": exact,
        "rank_total": len(rank_checks),
        "rank_within_one_step": within_one_step,
        "winning_lane_counts": dict(sorted(lane_counts.items())),
        "k_by_store_s": {ftag(k): v for k, v in k_terms.items()},
        "rank_checks": rank_checks,
        "row_predictions": row_predictions,
    }


def fit_dry_gpu(rows: list[Row]) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if row.is_baseline or row.f_cpu_store is None or row.f_cpu is None:
            continue
        s = float(row.f_cpu_store)
        u = float(row.f_cpu)
        p = float(row.f_prefetch)
        if abs(u - s) <= 1e-9 and abs(p) <= 1e-9:
            route = "cpu"
        elif abs(u) <= 1e-9 and abs(p - s) <= 1e-9:
            route = "prefetch"
        else:
            continue
        candidates.append((row, route))

    if len(candidates) < 3:
        return None

    routes = sorted({route for _, route in candidates})
    if not routes:
        return None

    # y = K_route - G*s. Columns are route constants plus (-s).
    x = np.zeros((len(candidates), len(routes) + 1), dtype=np.float64)
    y = np.zeros((len(candidates),), dtype=np.float64)
    route_index = {route: i for i, route in enumerate(routes)}
    for i, (row, route) in enumerate(candidates):
        x[i, route_index[route]] = 1.0
        x[i, -1] = -float(row.f_cpu_store)
        y[i] = row.latency_s
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    preds = x @ beta
    errors = preds - y

    row_predictions = []
    for (row, route), pred in zip(candidates, preds):
        row_predictions.append(
            {
                "route": route,
                "f_cpu_store": row.f_cpu_store,
                "measured_s": row.latency_s,
                "predicted_s": float(pred),
                "error_s": float(pred - row.latency_s),
            }
        )

    route_constants = {
        route: float(beta[idx]) for route, idx in route_index.items()
    }
    return {
        "model": "D_route - G_B*s",
        "g_s_per_fraction": float(beta[-1]),
        "route_constants_s": route_constants,
        "route_constant_gap_s": (
            route_constants.get("cpu", 0.0)
            - route_constants.get("prefetch", 0.0)
            if "cpu" in route_constants and "prefetch" in route_constants
            else None
        ),
        "rmse_s": float(math.sqrt(np.mean(errors * errors))),
        "mae_s": float(np.mean(np.abs(errors))),
        "row_predictions": row_predictions,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Dispatch Cost Model Fit", ""]
    lines.append("## Inputs")
    for source in report["sources"]:
        lines.append(f"- `{source}`")
    lines.append("")

    lines.append("## Real Fit")
    lines.append("")
    lines.append(
        "| B | G s/frac | C s/frac | H s/frac | C/H | u*/s | RMSE s | lane counts | rank exact | rank +/-1 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for batch, fit in report["real_fits"].items():
        ratio = fit["cpu_to_h2d_slope_ratio"]
        opt = fit["continuous_u_over_s_optimum"]
        g = fit.get("g_s_per_fraction_fixed")
        g_text = "-" if g is None else f"{g:.4f}"
        lanes = ", ".join(
            f"{lane}={count}"
            for lane, count in sorted(fit["winning_lane_counts"].items())
        )
        lines.append(
            f"| {batch} | {g_text} | {fit['c_s_per_fraction']:.4f} | "
            f"{fit['h_s_per_fraction']:.4f} | "
            f"{ratio:.3f} | {opt:.3f} | {fit['rmse_s']:.4f} | "
            f"{lanes} | "
            f"{fit['rank_exact']}/{fit['rank_total']} | "
            f"{fit['rank_within_one_step']}/{fit['rank_total']} |"
        )
    lines.append("")

    for batch, fit in report["real_fits"].items():
        lines.append(f"### B={batch} Rank Checks")
        lines.append("")
        lines.append(
            "| f_store | measured best u | predicted best u | predicted lane | measured best s | predicted-cell measured s | exact | +/-1 |"
        )
        lines.append("|---:|---:|---:|---|---:|---:|---|---|")
        for item in fit["rank_checks"]:
            lines.append(
                f"| {item['f_cpu_store']:.5f} | "
                f"{item['measured_best_f_cpu']:.5f} | "
                f"{item['predicted_best_f_cpu']:.5f} | "
                f"{item['predicted_best_winning_lane']} | "
                f"{item['measured_best_s']:.4f} | "
                f"{item['predicted_best_measured_s']:.4f} | "
                f"{item['exact']} | {item['within_one_step']} |"
            )
        lines.append("")

    if report["dry_fits"]:
        lines.append("## Dry-Run GPU-Slope Fit")
        lines.append("")
        lines.append("| B | G s/frac | RMSE s | route K terms | CPU-pref gap s |")
        lines.append("|---:|---:|---:|---|---:|")
        for batch, fit in report["dry_fits"].items():
            route_terms = ", ".join(
                f"{route}={value:.4f}"
                for route, value in sorted(fit["route_constants_s"].items())
            )
            gap = fit["route_constant_gap_s"]
            gap_text = "-" if gap is None else f"{gap:.4f}"
            lines.append(
                f"| {batch} | {fit['g_s_per_fraction']:.4f} | "
                f"{fit['rmse_s']:.4f} | {route_terms} | {gap_text} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def parse_fraction_int_mapping(raw: str | None, *, field_name: str) -> dict[str, int] | None:
    if raw is None:
        return None
    text = Path(raw[1:]).read_text() if raw.startswith("@") else raw
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    values: dict[str, int] = {}
    for key, value in parsed.items():
        fraction = float(key)
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"{field_name} keys must be in [0, 1], got {fraction}")
        byte_value = int(value)
        if byte_value < 0:
            raise ValueError(f"{field_name} values must be non-negative")
        values[f"{fraction:.12g}"] = byte_value
    return dict(sorted(values.items(), key=lambda item: float(item[0])))


def build_weight_resource_model(args: argparse.Namespace) -> dict[str, Any] | None:
    buffer_fields = (
        args.gpu_buffer_bytes_per_store_fraction,
        args.gpu_prefetch_buffer_bytes_per_store_fraction,
        args.gpu_output_scratch_bytes_per_store_fraction,
    )
    if args.total_weight_bytes is None and all(
        value is None for value in buffer_fields
    ):
        return None
    if args.total_weight_bytes is None:
        raise ValueError("--total-weight-bytes is required for weight_resource_model")
    if args.total_weight_bytes < 0:
        raise ValueError("--total-weight-bytes must be non-negative")
    if all(value is None for value in buffer_fields):
        raise ValueError(
            "weight_resource_model requires a GPU buffer coefficient: "
            "--gpu-buffer-bytes-per-store-fraction or component split fields"
        )
    if (
        args.gpu_buffer_bytes_per_store_fraction is not None
        and (
            args.gpu_prefetch_buffer_bytes_per_store_fraction is not None
            or args.gpu_output_scratch_bytes_per_store_fraction is not None
        )
    ):
        raise ValueError(
            "provide combined GPU buffer coefficient or component split, not both"
        )
    for value in buffer_fields:
        if value is not None and value < 0:
            raise ValueError("GPU buffer coefficients must be non-negative")

    model: dict[str, Any] = {
        "total_weight_bytes": int(args.total_weight_bytes),
        "buffer_model": args.buffer_model,
    }
    if args.gpu_buffer_bytes_per_store_fraction is not None:
        model["gpu_buffer_bytes_per_store_fraction"] = int(
            args.gpu_buffer_bytes_per_store_fraction
        )
    else:
        model["gpu_prefetch_buffer_bytes_per_store_fraction"] = int(
            args.gpu_prefetch_buffer_bytes_per_store_fraction or 0
        )
        model["gpu_output_scratch_bytes_per_store_fraction"] = int(
            args.gpu_output_scratch_bytes_per_store_fraction or 0
        )
    cpu_weight_map = parse_fraction_int_mapping(
        args.cpu_weight_bytes_by_store_fraction_json,
        field_name="cpu_weight_bytes_by_store_fraction",
    )
    if cpu_weight_map is not None:
        model["cpu_weight_bytes_by_store_fraction"] = cpu_weight_map
    gpu_buffer_map = parse_fraction_int_mapping(
        args.gpu_buffer_bytes_by_store_fraction_json,
        field_name="gpu_buffer_bytes_by_store_fraction",
    )
    if gpu_buffer_map is not None:
        model["gpu_buffer_bytes_by_store_fraction"] = gpu_buffer_map
    return model


def build_cots_snap_profile(
    weight_resource_model: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if weight_resource_model is None:
        return None
    cpu_weight_map = weight_resource_model.get("cpu_weight_bytes_by_store_fraction")
    gpu_buffer_map = weight_resource_model.get("gpu_buffer_bytes_by_store_fraction")
    if not isinstance(cpu_weight_map, dict) and not isinstance(gpu_buffer_map, dict):
        return None
    cpu_weight_by_fraction = (
        {float(fraction): int(value) for fraction, value in cpu_weight_map.items()}
        if isinstance(cpu_weight_map, dict)
        else {}
    )
    gpu_buffer_by_fraction = (
        {float(fraction): int(value) for fraction, value in gpu_buffer_map.items()}
        if isinstance(gpu_buffer_map, dict)
        else {}
    )

    fractions = sorted({*cpu_weight_by_fraction, *gpu_buffer_by_fraction})
    storage_by_store_fraction: dict[str, dict[str, int]] = {}
    for fraction in fractions:
        key = f"{fraction:.12g}"
        row: dict[str, int] = {}
        if fraction in cpu_weight_by_fraction:
            row["cpu_weight_bytes"] = cpu_weight_by_fraction[fraction]
        if fraction in gpu_buffer_by_fraction:
            row["gpu_buffer_bytes"] = gpu_buffer_by_fraction[fraction]
        if row:
            storage_by_store_fraction[key] = row

    if not storage_by_store_fraction:
        return None

    profile = {
        "schema_version": 1,
        "snap_model": "cots_snap_v1",
        "storage_by_store_fraction": storage_by_store_fraction,
    }
    if "buffer_model" in weight_resource_model:
        profile["buffer_model"] = weight_resource_model["buffer_model"]
    return profile


def build_weight_dispatch_profile(
    report: dict[str, Any],
    *,
    weight_resource_model: dict[str, Any] | None = None,
    cots_snap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    buckets: dict[str, Any] = {}
    for batch, fit in report["real_fits"].items():
        g_value = fit.get("g_s_per_fraction_fixed")
        if g_value is None:
            continue
        buckets[str(batch)] = {
            "G_s_per_fraction": g_value,
            "C_s_per_fraction": fit["c_s_per_fraction"],
            "H_s_per_fraction": fit["h_s_per_fraction"],
            "K_by_store_s": fit.get("k_by_store_s", {}),
            "continuous_u_over_s_optimum": fit[
                "continuous_u_over_s_optimum"
            ],
            "rmse_s": fit["rmse_s"],
            "mae_s": fit["mae_s"],
            "rank_exact": fit["rank_exact"],
            "rank_total": fit["rank_total"],
            "rank_within_one_step": fit["rank_within_one_step"],
            "winning_lane_counts": fit["winning_lane_counts"],
        }

    profile: dict[str, Any] = {
        "schema_version": 1,
        "dispatch_model": "weight_three_lane_v1",
        "coefficient_units": "seconds_per_fraction",
        "metadata": {
            **dict(report.get("metadata", {})),
            "fit_sources": list(report.get("sources", [])),
        },
        "buckets": buckets,
    }
    if weight_resource_model is not None:
        profile["weight_resource_model"] = weight_resource_model
    if cots_snap is None:
        cots_snap = build_cots_snap_profile(weight_resource_model)
    if cots_snap is not None:
        profile["cots_snap"] = cots_snap
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--total-weight-bytes", type=int)
    parser.add_argument("--gpu-buffer-bytes-per-store-fraction", type=int)
    parser.add_argument("--gpu-prefetch-buffer-bytes-per-store-fraction", type=int)
    parser.add_argument("--gpu-output-scratch-bytes-per-store-fraction", type=int)
    parser.add_argument(
        "--cpu-weight-bytes-by-store-fraction-json",
        help='JSON object or @file mapping store fraction to snapped CPU weight bytes.',
    )
    parser.add_argument(
        "--gpu-buffer-bytes-by-store-fraction-json",
        help='JSON object or @file mapping store fraction to snapped GPU buffer bytes.',
    )
    parser.add_argument("--buffer-model", default="cots_option_a_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = average_duplicate_rows(load_rows(args.summary_json))
    if not rows:
        raise SystemExit("no usable rows found")

    report: dict[str, Any] = {
        "sources": [str(path) for path in args.summary_json],
        "metadata": load_source_metadata(args.summary_json),
        "num_rows": len(rows),
        "baselines_by_batch": baseline_by_batch(rows),
        "real_fits": {},
        "dry_fits": {},
    }

    for batch, batch_rows in group_by_batch(dry_rows(rows)).items():
        dry_fit = fit_dry_gpu(batch_rows)
        if dry_fit is not None:
            report["dry_fits"][str(batch)] = dry_fit
    for batch, batch_rows in group_by_batch(real_rows(rows)).items():
        dry_fit = report["dry_fits"].get(str(batch))
        g_slope = None if dry_fit is None else float(dry_fit["g_s_per_fraction"])
        report["real_fits"][str(batch)] = fit_resource_model(batch_rows, g_slope)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "fit_summary.json"
    out_md = args.output_dir / "fit_summary.md"
    out_profile = args.output_dir / "weight_dispatch_profile.json"
    out_json.write_text(json.dumps(report, indent=2))
    write_markdown(report, out_md)
    weight_resource_model = build_weight_resource_model(args)
    cots_snap = build_cots_snap_profile(weight_resource_model)
    out_profile.write_text(
        json.dumps(
            build_weight_dispatch_profile(
                report,
                weight_resource_model=weight_resource_model,
                cots_snap=cots_snap,
            ),
            indent=2,
        )
    )
    print(f"[fit] wrote {out_json}")
    print(f"[fit] wrote {out_md}")
    print(f"[fit] wrote {out_profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
