#!/usr/bin/env python3
"""Phase 0.10.2 — PrefetchOffloader knob sweep

Explores each of PrefetchOffloader's three numerical knobs on Qwen2.5-7B BF16
across batch ∈ {1, 16, 64}:

    G  (offload_group_size)    — group every G consecutive layers
    N  (offload_num_in_group)  — offload last N of every group; coverage = N/G
    K  (offload_prefetch_step) — prefetch K layers ahead

All G choices are divisors of 28 (Qwen2.5-7B's decoder layer count) so coverage
is exact and the same for every G in the sweep — no partial-group artifacts.

Three independent sub-sweeps:

  (a) G at fixed 50% coverage, K=1
        G ∈ {2, 4, 14, 28} with N = G // 2
        All four offload exactly 14 layers (50% of 28). Question: at byte-equal
        coverage, does spacing between offloaded layers matter? Larger G ⇒ more
        compute time between offloaded layers ⇒ more H2D-hiding budget.

  (b) N at fixed G=14, K=1
        N ∈ {1, 2, 4, 7, 10}  (coverage = 7%, 14%, 29%, 50%, 71%)
        Question: how does latency scale with offloaded coverage?

  (c) K at fixed G=14, N=4
        K ∈ {1, 2, 3, 4}      (memory permitting; K=4 may OOM)
        Question: does multi-layer-ahead help, or is layer-ahead enough?

Outputs go to ``results/0.10_prefetch_knobs/``.

Usage
-----
    cd /TTC/FastTTS-thesis
    python /TTC/David/Benchmarks/phase0/bench_prefetch_knobs.py --exp --plot
    python ... --only-batches 64
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
RESULTS_DIR = PHASE0_DIR / "results" / "0.10_prefetch_knobs"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 256
OUTPUT_LEN = 32
WARMUP_ITERS = 2
BENCH_ITERS = 3
DEFAULT_BATCHES = [1, 16, 64]

# Decoder-layer bytes (BF16) — used for offloaded-bytes annotation.
HIDDEN, INTERMEDIATE, NUM_LAYERS = 3584, 18944, 28
QKV_OUT = (28 + 2 * 4) * 128
PER_LAYER_BYTES = (
    HIDDEN * QKV_OUT * 2
    + HIDDEN * HIDDEN * 2
    + HIDDEN * 2 * INTERMEDIATE * 2
    + INTERMEDIATE * HIDDEN * 2
)


def offloaded_layer_count(G: int, N: int, num_layers: int = NUM_LAYERS) -> int:
    count = 0
    for start in range(0, num_layers, G):
        count += min(N, min(start + G, num_layers) - start)
    return count


def gib(b: float) -> float:
    return b / (1024 ** 3)


# ---------------------------------------------------------------------------
# Sub-sweep arm definitions
# ---------------------------------------------------------------------------
def sweep_a() -> list[tuple[str, int, int, int]]:
    """G at fixed 50% coverage. All offload exactly 14 layers."""
    return [(f"prefetch_{G}x{G // 2}", G, G // 2, 1) for G in [2, 4, 14, 28]]


def sweep_b() -> list[tuple[str, int, int, int]]:
    """N at fixed G=14, K=1. G=14 divides 28 → exact coverage at every N."""
    return [(f"prefetch_14x{N}", 14, N, 1) for N in [1, 2, 4, 7, 10]]


def sweep_c() -> list[tuple[str, int, int, int]]:
    """K at fixed G=14, N=4. K=4 may OOM — attempt and record."""
    out: list[tuple[str, int, int, int]] = []
    for K in [1, 2, 3, 4]:
        suffix = "" if K == 1 else f"_k{K}"
        out.append((f"prefetch_14x4{suffix}", 14, 4, K))
    return out


def all_arms() -> dict[str, dict]:
    arms: dict[str, dict] = {}
    for name, G, N, K in sweep_a() + sweep_b() + sweep_c():
        if name in arms:
            continue
        arms[name] = {
            "flags": [
                "--offload-group-size", str(G),
                "--offload-num-in-group", str(N),
                "--offload-prefetch-step", str(K),
            ],
            "G": G, "N": N, "K": K,
            "offloaded_gib": round(gib(offloaded_layer_count(G, N) * PER_LAYER_BYTES), 3),
        }
    return arms


# ---------------------------------------------------------------------------
# Cell run + parse
# ---------------------------------------------------------------------------
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
    for name, spec in arms.items():
        per_b = {}
        for B in batches:
            avg = parse_avg(RESULTS_DIR / f"{name}_b{B}.json")
            per_b[B] = (
                None if avg is None
                else {"avg_latency_s": round(avg, 4),
                      "tokens_per_s": round(B * OUTPUT_LEN / avg, 1)}
            )
        out[name] = {
            "G": spec["G"], "N": spec["N"], "K": spec["K"],
            "offloaded_gib": spec["offloaded_gib"],
            "by_batch": per_b,
        }
    return out


# ---------------------------------------------------------------------------
# Plot — three sub-sweeps × len(batches) panels
# ---------------------------------------------------------------------------
def plot(summary: dict, batches: list[int], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable.")
        return

    fig, axes = plt.subplots(3, len(batches),
                             figsize=(5 * len(batches), 12),
                             sharey=False)
    if len(batches) == 1:
        axes = axes.reshape(3, 1)

    def y(arm: str, B: int) -> float | None:
        cell = summary[arm]["by_batch"][B]
        return cell["avg_latency_s"] if cell else None

    sweeps = [
        ("(a) G at fixed 50% coverage (14 layers)",
         "group_size G  (with N = G/2, K=1)",
         [a for a, *_ in sweep_a()],
         lambda a: summary[a]["G"], "o-", "tab:blue"),
        ("(b) N at fixed G=14",
         "num_in_group N  (G=14, K=1)",
         [a for a, *_ in sweep_b()],
         lambda a: summary[a]["N"], "s-", "tab:green"),
        ("(c) K at fixed G=14, N=4",
         "prefetch_step K  (G=14, N=4)",
         [a for a, *_ in sweep_c()],
         lambda a: summary[a]["K"], "^-", "tab:red"),
    ]

    for row, (title, xlabel, arms, xfn, style, color) in enumerate(sweeps):
        for col, B in enumerate(batches):
            ax = axes[row, col]
            xs = [xfn(a) for a in arms]
            ys = [y(a, B) for a in arms]
            ax.plot(xs, ys, style, color=color)
            for a, x_, y_ in zip(arms, xs, ys):
                if y_ is not None:
                    ax.annotate(f"{summary[a]['offloaded_gib']:.1f} GiB",
                                (x_, y_), fontsize=7,
                                xytext=(4, 4), textcoords="offset points")
            if row == 2:  # K sub-sweep — show all four ticks; OOM cells become gaps
                ax.set_xticks([1, 2, 3, 4])
                if any(v is None for v in ys):
                    valid_ys = [v for v in ys if v is not None]
                    if valid_ys:
                        ax.annotate("OOM", (xs[ys.index(None)],
                                            max(valid_ys) * 0.95),
                                    fontsize=8, color="black")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"avg latency @ B={B} (s)")
            ax.set_title(f"{title}  —  B={B}")
            ax.grid(alpha=0.3)

    fig.suptitle("PrefetchOffloader knob sweep (Qwen2.5-7B BF16, in=256 out=32, eager)",
                 y=1.005, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--only-arms", nargs="+", default=None)
    ap.add_argument("--only-batches", type=int, nargs="+", default=None)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = all_arms()
    run_arms = arms if not args.only_arms else {
        a: arms[a] for a in args.only_arms if a in arms
    }
    if args.only_arms and len(run_arms) < len(args.only_arms):
        unknown = set(args.only_arms) - set(arms)
        ap.error(f"unknown arms: {sorted(unknown)}. choices: {list(arms)}")
    batches = args.only_batches or DEFAULT_BATCHES

    print("§0.10.2 — PrefetchOffloader knob sweep")
    print(f"  per-layer decoder bytes: {gib(PER_LAYER_BYTES):.3f} GiB")
    print(f"  arms: {len(run_arms)}, batches: {batches}\n")

    if args.exp:
        for arm, spec in run_arms.items():
            for B in batches:
                run_cell(arm, spec["flags"], B)

    summary = summarize(arms, batches)
    profile = {
        "schema_version": 1,
        "section": "0.10.2",
        "hardware_id": hardware_id(),
        "model": {"name": MODEL, "dtype": DTYPE},
        "env": env_info(),
        "config": {
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "batches": batches,
            "warmup_iters": WARMUP_ITERS,
            "bench_iters": BENCH_ITERS,
            "enforce_eager": True,
            "per_layer_decoder_bytes": PER_LAYER_BYTES,
            "num_layers": NUM_LAYERS,
        },
        "sweeps": {
            "a_group_size_at_25pct": [a for a, *_ in sweep_a()],
            "b_coverage_at_G8":       [a for a, *_ in sweep_b()],
            "c_distance_at_G8_N2":    [a for a, *_ in sweep_c()],
        },
        "buffer_note": "Buffer pool grows linearly with K — large K may OOM at this model size on a 24 GiB GPU.",
        "arms": summary,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(profile, indent=2))
    print(f"\n[summary] {RESULTS_DIR / 'summary.json'}")

    if args.plot:
        plot(summary, batches, RESULTS_DIR / "prefetch_knobs.pdf")


if __name__ == "__main__":
    main()
