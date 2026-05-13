#!/usr/bin/env python3
"""Focused Phase 1c capture-gap benchmark for Qwen2.5-7B.

Fresh-run harness for the capture-vs-eager decision gate:

  * native_eager_real is the bar to beat.
  * capture_wait_kernel_real must beat native_eager_real to keep chasing
    capture mode.
  * capture_wait_uva_real is the experimental fused wait+UVA prototype.

Default output directory:
    /TTC/results/phase1c_capture_gap/<timestamp>/

Run from /TTC/FastTTS-thesis with the thesis env, for example:
    /opt/conda/envs/thesis/bin/python \
        /TTC/David/Benchmarks/phase1c/bench_capture_gap_qwen.py --force
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
BATCH_SIZE = 1
F_CPU_STORE = 0.05
CPU_THREADS = 16

DEFAULT_PIECEWISE_SPLITTING_OPS = [
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
COTS_SPLITTING_OPS = DEFAULT_PIECEWISE_SPLITTING_OPS + [
    "vllm::cots_submit_gemm",
    "vllm::cots_sync_then_uva",
]


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1c_capture_gap") / stamp


def arms(*, f_cpu_store: float, cpu_threads: int) -> dict[str, list[str]]:
    cots_base = [
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        str(f_cpu_store),
        "--cots-cpu-runner",
        "native",
        "--cots-cpu-num-threads",
        str(cpu_threads),
    ]
    legacy_cots_base = cots_base + ["--no-cots-auto-graph-split"]
    piecewise_graph = ["--compilation-config", '{"cudagraph_mode":"PIECEWISE"}']
    piecewise_cots_split_graph = [
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "PIECEWISE",
                "splitting_ops": COTS_SPLITTING_OPS,
            },
            separators=(",", ":"),
        ),
    ]
    piecewise_cots_split_inductor_graph = [
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "PIECEWISE",
                "use_inductor_graph_partition": True,
                "splitting_ops": COTS_SPLITTING_OPS,
            },
            separators=(",", ":"),
        ),
    ]
    return {
        "none_capture": [],
        "native_eager_dryrun": cots_base + ["--cots-dry-run"],
        "native_eager_real": cots_base,
        "cots_default_dryrun": cots_base + ["--cots-dry-run"],
        "cots_default_real": cots_base,
        "capture_host_callback_dryrun": legacy_cots_base + ["--cots-dry-run"],
        "capture_host_callback_real": legacy_cots_base,
        "capture_wait_kernel_dryrun": legacy_cots_base
        + ["--cots-dry-run", "--cots-capture-sync-mode", "wait_kernel"],
        "capture_wait_kernel_real": legacy_cots_base
        + ["--cots-capture-sync-mode", "wait_kernel"],
        "capture_wait_uva_dryrun": legacy_cots_base
        + ["--cots-dry-run", "--cots-capture-sync-mode", "wait_uva_kernel"],
        "capture_wait_uva_real": legacy_cots_base
        + ["--cots-capture-sync-mode", "wait_uva_kernel"],
        "piecewise_host_callback_dryrun": legacy_cots_base
        + ["--cots-dry-run"]
        + piecewise_graph,
        "piecewise_host_callback_real": legacy_cots_base + piecewise_graph,
        "piecewise_wait_kernel_dryrun": legacy_cots_base
        + ["--cots-dry-run", "--cots-capture-sync-mode", "wait_kernel"]
        + piecewise_graph,
        "piecewise_wait_kernel_real": legacy_cots_base
        + ["--cots-capture-sync-mode", "wait_kernel"]
        + piecewise_graph,
        "piecewise_cots_split_host_callback_dryrun": legacy_cots_base
        + ["--cots-dry-run"]
        + piecewise_cots_split_graph,
        "piecewise_cots_split_host_callback_real": legacy_cots_base
        + piecewise_cots_split_graph,
        "piecewise_cots_split_wait_kernel_dryrun": legacy_cots_base
        + ["--cots-dry-run", "--cots-capture-sync-mode", "wait_kernel"]
        + piecewise_cots_split_graph,
        "piecewise_cots_split_wait_kernel_real": legacy_cots_base
        + ["--cots-capture-sync-mode", "wait_kernel"]
        + piecewise_cots_split_graph,
        "piecewise_cots_split_wait_uva_dryrun": legacy_cots_base
        + ["--cots-dry-run", "--cots-capture-sync-mode", "wait_uva_kernel"]
        + piecewise_cots_split_graph,
        "piecewise_cots_split_wait_uva_real": legacy_cots_base
        + ["--cots-capture-sync-mode", "wait_uva_kernel"]
        + piecewise_cots_split_graph,
        "piecewise_cots_split_inductor_host_callback_dryrun": legacy_cots_base
        + ["--cots-dry-run"]
        + piecewise_cots_split_inductor_graph,
        "piecewise_cots_split_inductor_host_callback_real": legacy_cots_base
        + piecewise_cots_split_inductor_graph,
        "piecewise_cots_split_inductor_wait_kernel_dryrun": legacy_cots_base
        + ["--cots-dry-run", "--cots-capture-sync-mode", "wait_kernel"]
        + piecewise_cots_split_inductor_graph,
        "piecewise_cots_split_inductor_wait_kernel_real": legacy_cots_base
        + ["--cots-capture-sync-mode", "wait_kernel"]
        + piecewise_cots_split_inductor_graph,
    }


EAGER_ARMS = frozenset({"native_eager_dryrun", "native_eager_real"})


def run_cell(
    *,
    arm: str,
    flags: list[str],
    repeat: int,
    results_dir: Path,
    num_iters: int,
    num_iters_warmup: int,
    force: bool,
    extra_flags: list[str],
    model: str,
    dtype: str,
    input_len: int,
    output_len: int,
    batch_size: int,
) -> Path:
    out_json = results_dir / f"r{repeat:02d}_{arm}.json"
    out_log = out_json.with_suffix(".log")
    if out_json.exists() and not force:
        print(f"  [skip] r={repeat} {arm} (cached)")
        return out_json
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model",
        model,
        "--dtype",
        dtype,
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--batch-size",
        str(batch_size),
        "--num-iters-warmup",
        str(num_iters_warmup),
        "--num-iters",
        str(num_iters),
        "--output-json",
        str(out_json),
        *flags,
        *extra_flags,
    ]
    if arm in EAGER_ARMS:
        cmd.append("--enforce-eager")

    t0 = time.perf_counter()
    with out_log.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-20:])
        raise RuntimeError(
            f"{arm} repeat={repeat} failed rc={proc.returncode} "
            f"after {elapsed:.1f}s\n        {tail}"
        )
    avg = json.loads(out_json.read_text()).get("avg_latency")
    print(f"  [ok] r={repeat} {arm}: avg={avg:.4f}s ({elapsed:.1f}s)")
    return out_json


def avg_latency(path: Path) -> float:
    return float(json.loads(path.read_text())["avg_latency"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--num-iters-warmup", type=int, default=2)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--input-len", type=int, default=INPUT_LEN)
    parser.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--f-cpu-store", type=float, default=F_CPU_STORE)
    parser.add_argument("--cpu-threads", type=int, default=CPU_THREADS)
    parser.add_argument("--only-arms", nargs="*", default=None)
    parser.add_argument(
        "--extra-flag",
        action="append",
        default=[],
        help="Pass-through flag string for `vllm bench latency`.",
    )
    args = parser.parse_args()
    extra_flags = [tok for s in args.extra_flag for tok in shlex.split(s)]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    all_arms = arms(f_cpu_store=args.f_cpu_store, cpu_threads=args.cpu_threads)
    selected = (
        all_arms
        if args.only_arms is None
        else {name: all_arms[name] for name in args.only_arms}
    )

    print(
        f"[setup] model={args.model} input={args.input_len} "
        f"output={args.output_len} B={args.batch_size} "
        f"f={args.f_cpu_store} threads={args.cpu_threads} "
        f"repeat={args.repeat} iters={args.num_iters} "
        f"warmup={args.num_iters_warmup}"
    )
    print(f"[results] {args.results_dir}")

    paths: dict[str, list[Path]] = {name: [] for name in selected}
    for r in range(args.repeat):
        print(f"\n[repeat {r}]")
        for name, flags in selected.items():
            paths[name].append(
                run_cell(
                    arm=name,
                    flags=flags,
                    repeat=r,
                    results_dir=args.results_dir,
                    num_iters=args.num_iters,
                    num_iters_warmup=args.num_iters_warmup,
                    force=args.force,
                    extra_flags=extra_flags,
                    model=args.model,
                    dtype=args.dtype,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    batch_size=args.batch_size,
                )
            )

    rows: dict[str, dict[str, object]] = {}
    for name, ps in paths.items():
        values = [avg_latency(p) for p in ps]
        rows[name] = {
            "latencies": values,
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    eager = rows.get("native_eager_real", {}).get("mean")
    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "batch_size": args.batch_size,
        "f_cpu_store": args.f_cpu_store,
        "cpu_threads": args.cpu_threads,
        "num_iters": args.num_iters,
        "num_iters_warmup": args.num_iters_warmup,
        "repeat": args.repeat,
        "rows": rows,
        "decision": {},
    }
    if isinstance(eager, float):
        for candidate in (
            "capture_wait_kernel_real",
            "capture_wait_uva_real",
            "piecewise_host_callback_real",
            "piecewise_wait_kernel_real",
            "piecewise_cots_split_host_callback_real",
            "piecewise_cots_split_wait_kernel_real",
            "piecewise_cots_split_wait_uva_real",
        ):
            cand = rows.get(candidate, {}).get("mean")
            if isinstance(cand, float):
                summary["decision"][candidate] = {
                    "delta_vs_native_eager_ms": (cand - eager) * 1000.0,
                    "beats_native_eager": cand < eager,
                    "beats_native_eager_by_50ms": (eager - cand) >= 0.050,
                }

    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 92)
    print(f"{'arm':<34} {'mean(s)':>10} {'min(s)':>10} {'max(s)':>10}")
    print("-" * 92)
    for name, row in rows.items():
        print(
            f"{name:<34} {row['mean']:>10.4f} "
            f"{row['min']:>10.4f} {row['max']:>10.4f}"
        )
    print("=" * 92)
    if summary["decision"]:
        print("\nDecision deltas vs native_eager_real:")
        for name, decision in summary["decision"].items():
            print(
                f"  {name}: {decision['delta_vs_native_eager_ms']:+.1f} ms, "
                f"beats eager={decision['beats_native_eager']}, "
                f"beats by 50ms={decision['beats_native_eager_by_50ms']}"
            )
    print(f"\n[summary] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
