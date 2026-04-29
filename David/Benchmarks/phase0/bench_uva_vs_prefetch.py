#!/usr/bin/env python3
"""Phase 0.10.1 — UVA vs Prefetch (head-to-head)

At varying offload depth, which native vLLM weight offloader is faster on
RTX 4090 + Qwen2.5-7B BF16?

Sweep
-----
Prefetch (G=14, K=1):  N ∈ {1, 2, 4, 7, 10}   (G chosen so num_layers=28 divides
                                              evenly → exact coverage at every N)
UVA:                   cpu_offload_gb ∈ {1, 2, 4, 6, 8, 10, 12}
Reference:             `none` (no offload).

Both curves on a shared x-axis (offloaded GiB), computed exactly from the
model structure. PrefetchOffloader's offloaded bytes = (#offloaded layers)
× (per-layer decoder weight bytes); UVAOffloader's ≈ `cpu_offload_gb` (vLLM
walks named_parameters() until that many bytes are placed).

Each cell shells out to ``vllm bench latency``. Outputs go to
``results/0.10_uva_vs_prefetch/``.

Usage
-----
    cd /TTC/FastTTS-thesis
    python /TTC/David/Benchmarks/phase0/bench_uva_vs_prefetch.py --exp --plot
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

PHASE0_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE0_DIR / "results" / "0.10_uva_vs_prefetch"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 256
OUTPUT_LEN = 32
WARMUP_ITERS = 2
BENCH_ITERS = 3
DEFAULT_BATCHES = [1, 16, 64]

# Qwen2.5-7B layer arithmetic (BF16) — used to compute exact offloaded bytes
# for prefetch arms so they share UVA's GiB axis.
HIDDEN = 3584
QKV_OUT = (28 + 2 * 4) * 128             # = 4608
INTERMEDIATE = 18944
NUM_LAYERS = 28
PER_LAYER_BYTES = (
    HIDDEN * QKV_OUT * 2                 # WQKV
    + HIDDEN * HIDDEN * 2                # WO
    + HIDDEN * 2 * INTERMEDIATE * 2      # MLP1 (gate+up merged)
    + INTERMEDIATE * HIDDEN * 2          # MLP2
)


def offloaded_layer_count(G: int, N: int, num_layers: int = NUM_LAYERS) -> int:
    """Mirror PrefetchOffloader's grouping: last N of every G-sized window."""
    count = 0
    for start in range(0, num_layers, G):
        end = min(start + G, num_layers)
        count += min(N, end - start)
    return count


def gib(b: float) -> float:
    return b / (1024 ** 3)


def build_arms() -> dict[str, dict]:
    arms: dict[str, dict] = {"none": {"flags": [], "offloaded_gib": 0.0}}
    # G=14 divides 28 evenly into 2 groups, so coverage = N/G is exact.
    for N in [1, 2, 4, 7, 10]:
        n_layers = offloaded_layer_count(G=14, N=N)
        arms[f"prefetch_14x{N}"] = {
            "flags": [
                "--offload-group-size", "14",
                "--offload-num-in-group", str(N),
                "--offload-prefetch-step", "1",
            ],
            "offloaded_gib": round(gib(n_layers * PER_LAYER_BYTES), 3),
        }
    for X in [1, 2, 4, 6, 8, 10, 12]:
        arms[f"uva_{X}"] = {
            "flags": ["--cpu-offload-gb", str(X)],
            "offloaded_gib": float(X),
        }
    return arms


def run_cell(arm: str, flags: list[str], batch: int) -> Path:
    out_json = RESULTS_DIR / f"{arm}_b{batch}.json"
    out_log = RESULTS_DIR / f"{arm}_b{batch}.log"
    if out_json.exists():
        print(f"  [skip] {arm} b={batch} (cached)")
        return out_json

    cmd = [
        "vllm", "bench", "latency",
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
                else {"avg_latency_s": round(avg, 4),
                      "tokens_per_s": round(B * OUTPUT_LEN / avg, 1)}
            )
        out[arm] = {
            "flags": spec["flags"],
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
        info.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0)})
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

    pf = sorted([a for a in summary if a.startswith("prefetch_")],
                key=lambda a: summary[a]["offloaded_gib"])
    uv = sorted([a for a in summary if a.startswith("uva_")],
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
                "o-", color="tab:blue", label="Prefetch (G=14, K=1)")
        ax.plot([summary[a]["offloaded_gib"] for a in uv], [y(a) for a in uv],
                "s-", color="tab:orange", label="UVA")
        if none_lat is not None:
            ax.axhline(none_lat, color="k", linestyle="--", linewidth=0.8,
                       label=f"none = {none_lat:.2f}s")
        ax.set_xlabel("Offloaded GiB")
        ax.set_ylabel("avg latency (s)")
        ax.set_title(f"B={B}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("UVA vs Prefetch (Qwen2.5-7B BF16, in=256 out=32, eager)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--only-arms", nargs="+", default=None)
    ap.add_argument("--only-batches", type=int, nargs="+", default=None)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = build_arms()
    run_arms = arms if not args.only_arms else {
        a: arms[a] for a in args.only_arms if a in arms
    }
    if args.only_arms and len(run_arms) < len(args.only_arms):
        unknown = set(args.only_arms) - set(arms)
        ap.error(f"unknown arms: {sorted(unknown)}. choices: {list(arms)}")
    batches = args.only_batches or DEFAULT_BATCHES

    print("§0.10.1 — UVA vs Prefetch")
    print(f"  per-layer decoder bytes: {gib(PER_LAYER_BYTES):.3f} GiB")
    print(f"  arms: {len(run_arms)}, batches: {batches}\n")

    if args.exp:
        for arm, spec in run_arms.items():
            for B in batches:
                run_cell(arm, spec["flags"], B)

    summary = summarize(arms, batches)
    profile = {
        "schema_version": 1,
        "section": "0.10.1",
        "hardware_id": hardware_id(),
        "model": {"name": MODEL, "dtype": DTYPE},
        "env": env_info(),
        "config": {
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "warmup_iters": WARMUP_ITERS,
            "bench_iters": BENCH_ITERS,
            "batches": batches,
            "enforce_eager": True,
            "per_layer_decoder_bytes": PER_LAYER_BYTES,
            "num_layers": NUM_LAYERS,
        },
        "arms": summary,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(profile, indent=2))
    print(f"\n[summary] {RESULTS_DIR / 'summary.json'}")

    if args.plot:
        plot(summary, batches, RESULTS_DIR / "uva_vs_prefetch.pdf")


if __name__ == "__main__":
    main()
