#!/usr/bin/env python3
"""Run the Phase 1c COTS sync-mode matrix.

This is a thin orchestrator around ``bench_dispatch_model_validation.py``.
It keeps the experiment knobs identical while crossing:

* runtime family: eager, current piecewise graph, legacy/full graph;
* sync mode: eager host_callback, graph host_callback, graph wait_kernel;
* workload cell: B=64 short decode and B=1 long decode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TTC_ROOT = Path(__file__).resolve().parents[3]
HARNESS = TTC_ROOT / "David/Benchmarks/planner/bench_dispatch_model_validation.py"

ATTENTION_SPLITTING_OPS = [
    "vllm::unified_attention",
    "vllm::unified_attention_with_output",
    "vllm::unified_mla_attention",
    "vllm::unified_mla_attention_with_output",
    "vllm::mamba_mixer2",
    "vllm::mamba_mixer",
    "vllm::short_conv",
    "vllm::linear_attention",
    "vllm::plamo2_mamba_mixer",
    "vllm::gdn_attention_core",
    "vllm::olmo_hybrid_gdn_full_forward",
    "vllm::kda_attention",
    "vllm::sparse_attn_indexer",
    "vllm::rocm_aiter_sparse_attn_indexer",
    "vllm::unified_kv_cache_update",
    "vllm::unified_mla_kv_cache_update",
]

COTS_SPLITTING_OPS = [
    "vllm::wait_prefetch",
    "vllm::start_prefetch",
    "vllm::cots_submit_gemm",
    "vllm::cots_sync_then_uva",
]


@dataclass(frozen=True)
class Scenario:
    name: str
    batch: int
    output_len: int
    f_cpu_store: float
    dispatch_layout: str
    cpu_threads: int


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    graph_family: str
    sync_mode: str
    extra_vllm_args: tuple[str, ...]


SCENARIOS = (
    Scenario(
        name="b64_short_decode_f030",
        batch=64,
        output_len=4,
        f_cpu_store=0.30,
        dispatch_layout="decode-only",
        cpu_threads=24,
    ),
    Scenario(
        name="b1_long_decode_f005",
        batch=1,
        output_len=128,
        f_cpu_store=0.05,
        dispatch_layout="uniform",
        cpu_threads=16,
    ),
)


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/planner/sync_mode_matrix") / stamp


def piecewise_host_compilation_config() -> str:
    return json.dumps(
        {
            "cudagraph_mode": "PIECEWISE",
            "splitting_ops": ATTENTION_SPLITTING_OPS + COTS_SPLITTING_OPS,
        },
        separators=(",", ":"),
    )


def arms() -> tuple[Arm, ...]:
    piecewise_host = (
        "--no-cots-auto-graph-split",
        "--cots-weight-capture-sync-mode",
        "host_callback",
        "--compilation-config",
        piecewise_host_compilation_config(),
    )
    full_host = (
        "--no-cots-auto-graph-split",
        "--cots-weight-capture-sync-mode",
        "host_callback",
    )
    full_wait = (
        "--no-cots-auto-graph-split",
        "--cots-weight-capture-sync-mode",
        "wait_kernel",
    )
    return (
        Arm("eager_host", "eager", "eager", "host_callback", ()),
        Arm("piecewise_wait", "graph", "piecewise", "wait_kernel", ()),
        Arm("piecewise_host", "graph", "piecewise", "host_callback", piecewise_host),
        Arm("full_host", "graph", "full", "host_callback", full_host),
        Arm("full_wait", "graph", "full", "wait_kernel", full_wait),
    )


def run_arm(args: argparse.Namespace, scenario: Scenario, arm: Arm) -> int:
    out_dir = args.results_dir / scenario.name / arm.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(HARNESS),
        "--exp",
        "--force",
        "--keep-going",
        "--results-dir",
        str(out_dir),
        "--modes",
        arm.mode,
        "--dispatch-layout",
        scenario.dispatch_layout,
        "--batches",
        str(scenario.batch),
        "--f-cpu-store-values",
        f"{scenario.f_cpu_store:.12g}",
        "--f-cpu-ratios",
        "0",
        "1",
        "--output-len",
        str(scenario.output_len),
        "--num-iters-warmup",
        str(args.num_iters_warmup),
        "--num-iters",
        str(args.num_iters),
        "--thread-policy",
        "scalar",
        "--cots-cpu-num-threads",
        str(scenario.cpu_threads),
        "--cell-timeout-s",
        str(args.cell_timeout_s),
    ]
    if arm.extra_vllm_args:
        cmd += ["--extra-vllm-args", *arm.extra_vllm_args]

    env = os.environ.copy()
    env.update(
        {
            "VLLM_CACHE_ROOT": f"/tmp/ttc-sync-mode-matrix/{scenario.name}/{arm.name}",
            "VLLM_COTS_COUNTERS": "1",
            "VLLM_COTS_DUMP_COUNTERS_ON_SHUTDOWN": "1",
            "VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE": "1",
        }
    )
    print(f"[run] {scenario.name} {arm.name}", flush=True)
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        print(
            f"[fail] {scenario.name} {arm.name} rc={proc.returncode}",
            flush=True,
        )
    return proc.returncode


def route_name(row: dict[str, Any]) -> str:
    if row["f_cpu_store"] is None:
        return "none"
    if abs(float(row["f_cpu"] or 0.0)) < 1e-12:
        return "prefetch"
    if abs(float(row["f_prefetch"] or 0.0)) < 1e-12:
        return "cpu"
    return f"cpu{row['f_cpu']}_pf{row['f_prefetch']}"


def collect(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for arm in arms():
            summary_path = args.results_dir / scenario.name / arm.name / "summary.json"
            if not summary_path.exists():
                failures.append(
                    {
                        "scenario": scenario.name,
                        "arm": arm.name,
                        "reason": "missing_summary",
                    }
                )
                continue
            summary = json.loads(summary_path.read_text())
            for row in summary.get("rows", []):
                rows.append(
                    {
                        "scenario": scenario.name,
                        "batch": scenario.batch,
                        "output_len": scenario.output_len,
                        "dispatch_layout": scenario.dispatch_layout,
                        "graph_family": arm.graph_family,
                        "sync_mode": arm.sync_mode,
                        "arm": arm.name,
                        "runtime_mode": row["mode"],
                        "route": route_name(row),
                        "mean_latency_s": row["mean_latency_s"],
                        "cv": row["cv"],
                    }
                )
            for failed in summary.get("failed_cells", []):
                failures.append(
                    {
                        "scenario": scenario.name,
                        "arm": arm.name,
                        **failed,
                    }
                )

    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['scenario']}:{row['route']}"
        if row["mean_latency_s"] is None:
            continue
        old = best.get(key)
        if old is None or row["mean_latency_s"] < old["mean_latency_s"]:
            best[key] = row

    return {
        "config": {
            "num_iters_warmup": args.num_iters_warmup,
            "num_iters": args.num_iters,
            "scenarios": [scenario.__dict__ for scenario in SCENARIOS],
            "arms": [arm.__dict__ for arm in arms()],
        },
        "rows": rows,
        "best_by_scenario_route": best,
        "failures": failures,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# COTS Sync Mode Matrix",
        "",
        "## Best By Scenario And Route",
        "",
        "| scenario | route | best arm | graph family | sync | mean s |",
        "|---|---|---|---|---|---:|",
    ]
    for key, row in sorted(summary["best_by_scenario_route"].items()):
        scenario, route = key.split(":", 1)
        lines.append(
            f"| `{scenario}` | `{route}` | `{row['arm']}` | "
            f"`{row['graph_family']}` | `{row['sync_mode']}` | "
            f"{fmt(row['mean_latency_s'])} |"
        )

    lines.extend(
        [
            "",
            "## All Rows",
            "",
            "| scenario | route | arm | graph family | sync | mean s | CV |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    for row in sorted(
        summary["rows"],
        key=lambda r: (r["scenario"], r["route"], r["graph_family"], r["sync_mode"]),
    ):
        lines.append(
            f"| `{row['scenario']}` | `{row['route']}` | `{row['arm']}` | "
            f"`{row['graph_family']}` | `{row['sync_mode']}` | "
            f"{fmt(row['mean_latency_s'])} | {fmt(row['cv'], 3)} |"
        )

    if summary["failures"]:
        lines.extend(
            [
                "",
                "## Failures",
                "",
                "| scenario | arm | reason |",
                "|---|---|---|",
            ]
        )
        for failure in summary["failures"]:
            lines.append(
                f"| `{failure.get('scenario')}` | `{failure.get('arm')}` | "
                f"`{failure.get('reason')}` |"
            )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--num-iters-warmup", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--cell-timeout-s", type=float, default=900)
    parser.add_argument("--exp", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    if args.exp:
        for scenario in SCENARIOS:
            for arm in arms():
                rc = run_arm(args, scenario, arm)
                if rc != 0:
                    exit_code = rc
                    if not args.keep_going:
                        break
            if exit_code != 0 and not args.keep_going:
                break

    summary = collect(args)
    summary_json = args.results_dir / "summary.json"
    summary_md = args.results_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, summary_md)
    print(f"[summary] wrote {summary_json}", flush=True)
    print(f"[summary] wrote {summary_md}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
