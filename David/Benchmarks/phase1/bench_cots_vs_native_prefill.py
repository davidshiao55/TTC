#!/usr/bin/env python3
"""Phase 1a §1.13a — COTS vs vLLM native PrefetchOffloader (prefill-heavy).

At matched offload depth, how does the COTS Phase 1a backend
(`--offload-backend cots`) compare to vLLM's native `prefetch` on RTX 4090
+ Qwen2.5-7B BF16, in a **prefill-dominated** workload?

Setup follows Phase 0 §0.10 verbatim (input_len=256, output_len=32 ⇒
prefill:decode CPU work ratio ≈ 8:1). This regime is NOT what COTS is
designed for — COTS pays a linear-in-num_tokens CPU GEMM cost on prefill
forwards, so a 256-token prefill puts CPU well outside the §0.3.3 "free
regime". Reported anyway as the worst-case data point and to document
where COTS loses. The COTS-target regime (decode-heavy short prompt) is
measured by the companion script `bench_cots_vs_native_decode.py`
(§1.13b).

Sweep
-----
COTS:                       f_cpu_store ∈ {0.09, 0.22, 0.50}
Prefetch (N=1, G varies):   G ∈ {2, 4, 7, 14, 28} — densest-spread baselines
                            from Phase 0 §0.10.2(d) (the strongest possible
                            prefetch reference at each depth).
Reference:                  `none` (no offload, also reused from §0.10).

UVA omitted per Phase 0 §0.10 finding 3 ("never the right baseline above
B=1"). All prefetch + none JSONs are reused from Phase 0 — methodology is
identical (same input/output lengths, batches, warmup + bench iters).

Outputs go to ``David/Benchmarks/phase1/results/cots_vs_native_prefill/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

PHASE1_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE1_DIR / "results" / "cots_vs_native_prefill"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 256
OUTPUT_LEN = 32
WARMUP_ITERS = 2
BENCH_ITERS = 3
DEFAULT_BATCHES = [1, 16, 64]

# Qwen2.5-7B layer arithmetic (BF16) — used to compute exact offloaded bytes
# for each arm so they share a common GiB axis.
HIDDEN = 3584
QKV_OUT = (28 + 2 * 4) * 128  # = 4608
INTERMEDIATE = 18944
NUM_LAYERS = 28
WO_BYTES_PER_LAYER = HIDDEN * HIDDEN * 2  # WO is NOT offloaded by COTS
COTS_OFFLOADABLE_PER_LAYER = (
    HIDDEN * QKV_OUT * 2  # WQKV
    + HIDDEN * 2 * INTERMEDIATE * 2  # MLP1 (gate+up)
    + INTERMEDIATE * HIDDEN * 2  # MLP2
)
PREFETCH_PER_LAYER_BYTES = COTS_OFFLOADABLE_PER_LAYER + WO_BYTES_PER_LAYER  # incl WO


def offloaded_layer_count(G: int, N: int, num_layers: int = NUM_LAYERS) -> int:
    count = 0
    for start in range(0, num_layers, G):
        end = min(start + G, num_layers)
        count += min(N, end - start)
    return count


def gib(b: float) -> float:
    return b / (1024 ** 3)


def build_arms() -> dict[str, dict]:
    """Return arm spec dict.

    Each spec: {"flags": [...], "offloaded_gib": float, "family": "cots"|"prefetch"|"uva"|"none"}
    """
    arms: dict[str, dict] = {
        "none": {"flags": [], "offloaded_gib": 0.0, "family": "none"},
    }

    # COTS arms — offloaded GiB = f × (12.15 GiB - WO_total_bytes / GiB).
    # WO is GPU-resident in COTS; native prefetch offloads whole layers
    # (WO included). For shared-axis comparison the COTS x-value is the
    # actual COTS-offloaded bytes (excl WO). The offloader's startup log
    # validates the analytic figure within ~1%.
    for f in (0.09, 0.22, 0.50):
        bytes_offloaded = (
            COTS_OFFLOADABLE_PER_LAYER * NUM_LAYERS * f
        )
        arms[f"cots_{int(f * 100):03d}"] = {
            "flags": [
                "--offload-backend", "cots",
                "--cots-f-cpu-store", str(f),
            ],
            "offloaded_gib": round(gib(bytes_offloaded), 3),
            "family": "cots",
        }

    # Prefetch arms: N=1, G ∈ {2, 4, 7, 14, 28} — densest-spread baselines
    # per Phase 0 §0.10.2(d), strongest possible prefetch reference at each
    # depth. JSONs reused from §0.10.2(d) (same input/output/batches setup).
    for G in [2, 4, 7, 14, 28]:
        n_layers = offloaded_layer_count(G=G, N=1)
        arms[f"prefetch_{G}x1"] = {
            "flags": [
                "--offload-backend", "prefetch",
                "--offload-group-size", str(G),
                "--offload-num-in-group", "1",
                "--offload-prefetch-step", "1",
            ],
            "offloaded_gib": round(gib(n_layers * PREFETCH_PER_LAYER_BYTES), 3),
            "family": "prefetch",
        }

    # UVA arms intentionally omitted: phase 0 §0.10 finding 3 established
    # UVA is "never the right baseline above B=1". Cost without information.
    return arms


def run_cell(arm: str, flags: list[str], batch: int) -> Path:
    out_json = RESULTS_DIR / f"{arm}_b{batch}.json"
    out_log = RESULTS_DIR / f"{arm}_b{batch}.log"
    if out_json.exists():
        print(f"  [skip] {arm} b={batch} (cached)")
        return out_json

    # Use the same Python interpreter the driver was launched with, so the
    # subprocess always finds the right vllm install (avoids PATH issues
    # with conda envs not propagating to subprocess). vllm exposes its CLI
    # via vllm.entrypoints.cli.main rather than a top-level `vllm` module.
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", MODEL,
        "--dtype", DTYPE,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--batch-size", str(batch),
        "--num-iters-warmup", str(WARMUP_ITERS),
        "--num-iters", str(BENCH_ITERS),
        "--enforce-eager",
        "--output-json", str(out_json),
        *flags,
    ]
    t0 = time.perf_counter()
    with open(out_log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    dur = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-15:])
        print(f"  [FAIL] {arm} b={batch} rc={proc.returncode} ({dur:.1f}s)\n        {tail}")
    else:
        print(f"  [ok]  {arm} b={batch} ({dur:.1f}s)")
    return out_json


def parse_avg(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("avg_latency")
    except (json.JSONDecodeError, OSError):
        return None


def summarize(arms: dict, batches: list[int]) -> dict:
    out = {}
    for arm, spec in arms.items():
        per_b = {}
        for B in batches:
            avg = parse_avg(RESULTS_DIR / f"{arm}_b{B}.json")
            per_b[B] = (
                None if avg is None
                else {
                    "avg_latency_s": round(avg, 4),
                    "tokens_per_s": round(B * OUTPUT_LEN / avg, 1),
                }
            )
        out[arm] = {
            "flags": spec["flags"],
            "family": spec["family"],
            "offloaded_gib": spec["offloaded_gib"],
            "by_batch": per_b,
        }
    return out


def hardware_id() -> str:
    try:
        import torch
        gpu = torch.cuda.get_device_name(0).replace(" ", "").replace("/", "")
    except Exception:
        gpu = "unknown-gpu"
    return f"{gpu}_{platform.processor() or 'unknown-cpu'}".lower()[:64]


def env_info() -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0]}
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        })
    except Exception:
        pass
    return info


def plot(summary: dict, batches: list[int], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable.")
        return

    cots = sorted([a for a in summary if summary[a]["family"] == "cots"],
                  key=lambda a: summary[a]["offloaded_gib"])
    pf = sorted([a for a in summary if summary[a]["family"] == "prefetch"],
                key=lambda a: summary[a]["offloaded_gib"])

    fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4.5),
                             sharey=False)
    if len(batches) == 1:
        axes = [axes]

    for ax, B in zip(axes, batches):
        cell0 = summary["none"]["by_batch"][B]
        none_lat = cell0["avg_latency_s"] if cell0 else None

        def y(arm: str) -> float | None:
            c = summary[arm]["by_batch"][B]
            return c["avg_latency_s"] if c else None

        ax.plot([summary[a]["offloaded_gib"] for a in pf], [y(a) for a in pf],
                "o-", color="tab:blue",
                label="Prefetch (N=1, max spread) [§0.10.2d]")
        ax.plot([summary[a]["offloaded_gib"] for a in cots], [y(a) for a in cots],
                "D-", color="tab:red", label="COTS", linewidth=2)
        if none_lat is not None:
            ax.axhline(none_lat, color="black", linestyle=":", alpha=0.5,
                       label="none (no offload)")
        ax.set_xlabel("Offloaded weights (GiB)")
        ax.set_ylabel("Avg latency (s)")
        ax.set_title(f"B = {B}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"COTS vs Prefetch (prefill-heavy) — {MODEL.split('/')[-1]} BF16, "
        f"input={INPUT_LEN}, output={OUTPUT_LEN}",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only-arms", nargs="*", default=None,
                    help="Run only the specified arms")
    ap.add_argument("--batches", type=int, nargs="*", default=DEFAULT_BATCHES)
    ap.add_argument("--exp", action="store_true",
                    help="Run measurements (default: just summarize cached)")
    ap.add_argument("--plot", action="store_true",
                    help="Generate the comparison PDF")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = build_arms()
    run_arms = arms if not args.only_arms else {
        name: arms[name] for name in args.only_arms if name in arms
    }
    if args.only_arms and len(run_arms) < len(args.only_arms):
        unknown = set(args.only_arms) - set(arms)
        print(f"[warn] unknown arms: {sorted(unknown)}")

    print(f"[setup] hardware={hardware_id()}")
    print(f"  arms: {len(run_arms)}, batches: {args.batches}")

    if args.exp:
        for arm, spec in run_arms.items():
            for B in args.batches:
                run_cell(arm, spec["flags"], B)

    summary = summarize(arms, args.batches)
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(
        {"env": env_info(), "model": MODEL, "input_len": INPUT_LEN,
         "output_len": OUTPUT_LEN, "summary": summary},
        indent=2,
    ))
    print(f"[summary] wrote {summary_path}")

    # Headline table to stdout.
    print("\n" + "=" * 78)
    print(f"{'arm':<22} {'GiB':>6}  " + "  ".join(
        f"{f'B={B} (s)':>10}" for B in args.batches
    ) + f"  {'tok/s @ B=' + str(args.batches[-1]):>16}")
    print("-" * 78)
    for arm, spec in summary.items():
        row = f"{arm:<22} {spec['offloaded_gib']:>6.2f}  "
        for B in args.batches:
            cell = spec["by_batch"][B]
            row += f"{cell['avg_latency_s']:>10.3f}  " if cell else f"{'—':>10}  "
        last = spec["by_batch"][args.batches[-1]]
        tok_s = f"{last['tokens_per_s']:>16.1f}" if last else f"{'—':>16}"
        print(row + tok_s)
    print("=" * 78)

    if args.plot:
        plot(summary, args.batches, RESULTS_DIR / "cots_vs_native_prefill.pdf")


if __name__ == "__main__":
    main()
