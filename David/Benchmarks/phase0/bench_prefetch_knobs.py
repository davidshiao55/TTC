#!/usr/bin/env python3
"""Phase 0.10.1 — PrefetchOffloader knob sweep

Explores each of PrefetchOffloader's three numerical knobs on Qwen2.5-7B BF16
across batch ∈ {1, 16, 64}:

    G  (offload_group_size)    — group every G consecutive layers
    N  (offload_num_in_group)  — offload last N of every group; coverage = N/G
    K  (offload_prefetch_step) — prefetch K layers ahead

All G choices are divisors of 28 (Qwen2.5-7B's decoder layer count) so coverage
is exact and the same for every G in the sweep — no partial-group artifacts.

Four sub-sweeps, ordered to follow the goal-oriented narrative
"establish best knob → measure best config" rather than alphabetical knob order:

  (a) G at fixed 50% coverage, K=1
        G ∈ {2, 4, 14, 28} with N = G // 2
        All four offload exactly 14 layers (50% of 28). Question: at byte-equal
        coverage, does spacing between offloaded layers matter? Larger G ⇒ more
        compute time between offloaded layers ⇒ more H2D-hiding budget.
        Result: smaller G (denser uniform spacing) wins.

  (b) N=1, G varies (canonical coverage sweep — densest possible spread at each depth)
        G ∈ {2, 4, 7, 14, 28} → 14/7/4/2/1 offloaded layers = 6.08/3.04/
        1.74/0.87/0.43 GiB. Generalizes (a)'s "denser-is-better" finding to
        all depths. Treat this as the empirically best prefetch baseline at
        any offload depth.

  (c) N at fixed G=14, K=1 (legacy clustered baseline, kept for matched-bytes
        comparison vs (b))
        N ∈ {1, 2, 4, 7, 10}  (coverage = 7%, 14%, 29%, 50%, 71%)
        Pre-restructure, this was the canonical coverage sweep. Now used only
        as the matched-bytes anchor in (b)'s "uniform vs clustered" comparison.

  (d) K at fixed G=4, N=1 (uniform-spread K sweep at the empirically best config)
        K ∈ {1, 2, 3, 4}      (memory permitting; K=4 may OOM)
        Question: does multi-layer-ahead help under the empirically best
        spacing config? K>1's potential upside is structurally smaller under
        uniform spread (each prefetch already gets G−1 = 3 layers of compute
        hiding from K=1 alone), so this is a K=1-conservative test.

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
    """N=1, G varies. One offloaded layer per group → maximally uniform
    spread at each depth. Canonical coverage sweep (post-restructure)."""
    return [(f"prefetch_{G}x1", G, 1, 1) for G in [2, 4, 7, 14, 28]]


def sweep_c() -> list[tuple[str, int, int, int]]:
    """N at fixed G=14, K=1 — legacy clustered baseline. Kept for matched-
    bytes comparison vs sweep_b's uniform-spread arms (the comparison that
    establishes "uniform > clustered at all depths"). G=14 divides 28
    → exact coverage at every N."""
    return [(f"prefetch_14x{N}", 14, N, 1) for N in [1, 2, 4, 7, 10]]


def sweep_d() -> list[tuple[str, int, int, int]]:
    """K at fixed G=4, N=1 — uniform-spread at ~25% coverage (7 layers).

    Picks G=4 N=1 instead of the prior G=14 N=4 because §0.10.1's "uniform
    spacing dominates clustering" finding makes uniform N=1 the empirically
    best knob configuration; the K sweep should be measured at that
    configuration rather than the suboptimal clustered one. K>1's potential
    upside is structurally smaller under uniform spread (each prefetch
    already gets G−1 = 3 layers of compute hiding from K=1 alone), so this
    K=1-conservative test is a stronger statement that K=1 suffices.

    K=4 with G=4 N=1 (7 offloaded layers × K=4 buffer slots) may OOM —
    attempt and record. Compared with the legacy K-sweep at (G=14, N=4),
    this config has ~13% less per-buffer GiB but K-multiplied buffer count
    is in the same range; the OOM behavior is empirically open.
    """
    out: list[tuple[str, int, int, int]] = []
    for K in [1, 2, 3, 4]:
        suffix = "" if K == 1 else f"_k{K}"
        out.append((f"prefetch_4x1{suffix}", 4, 1, K))
    return out


def all_arms() -> dict[str, dict]:
    arms: dict[str, dict] = {}
    for name, G, N, K in sweep_a() + sweep_b() + sweep_c() + sweep_d():
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

    # Use the same Python interpreter the driver was launched with so the
    # subprocess always finds the right vllm install (PATH may not include
    # the conda env when called from non-interactive shells).
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

    fig, axes = plt.subplots(5, len(batches),
                             figsize=(5 * len(batches), 20),
                             sharey=False)
    if len(batches) == 1:
        axes = axes.reshape(5, 1)

    def y(arm: str, B: int) -> float | None:
        cell = summary[arm]["by_batch"][B]
        return cell["avg_latency_s"] if cell else None

    # Plot ordering follows the post-restructure sub-sweep labels:
    #   row 0 = (a) G at 50%, row 1 = (b) N=1 G varies, row 2 = (c) clustered
    #   legacy, row 3 = (d) K sweep at (G=4, N=1).
    sweeps = [
        ("(a) G at fixed 50% coverage (14 layers)",
         "group_size G  (with N = G/2, K=1)",
         [a for a, *_ in sweep_a()],
         lambda a: summary[a]["G"], "o-", "tab:blue"),
        ("(b) N=1, G varies (canonical coverage sweep)",
         "group_size G  (N=1, K=1)",
         [a for a, *_ in sweep_b()],
         lambda a: summary[a]["G"], "D-", "tab:purple"),
        ("(c) N at fixed G=14 (legacy clustered baseline)",
         "num_in_group N  (G=14, K=1)",
         [a for a, *_ in sweep_c()],
         lambda a: summary[a]["N"], "s--", "tab:green"),
        ("(d) K at fixed G=4, N=1",
         "prefetch_step K  (G=4, N=1)",
         [a for a, *_ in sweep_d()],
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
            if row == 3:  # K sub-sweep (now (d)) — show all four ticks; OOM cells become gaps
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

    # Row 4: matched-bytes comparison between (b) uniform N=1 G-varying and (c)
    # legacy clustered G=14 N-varying. Empirical demonstration that (b)'s
    # densest-spread arms beat (c)'s clustered arms at every matched depth —
    # generalizes (a)'s "denser-is-better-at-fixed-bytes" finding from the
    # single 50% point to all depths.
    sweep_b_arms = sorted([a for a, *_ in sweep_b()],
                          key=lambda a: summary[a]["offloaded_gib"])
    sweep_c_arms = sorted([a for a, *_ in sweep_c()],
                          key=lambda a: summary[a]["offloaded_gib"])
    for col, B in enumerate(batches):
        ax = axes[4, col]
        xs_c = [summary[a]["offloaded_gib"] for a in sweep_c_arms]
        ys_c = [y(a, B) for a in sweep_c_arms]
        xs_b = [summary[a]["offloaded_gib"] for a in sweep_b_arms]
        ys_b = [y(a, B) for a in sweep_b_arms]
        ax.plot(xs_c, ys_c, "s--", color="tab:green",
                label="(c) G=14, N varies (clustered, legacy)")
        ax.plot(xs_b, ys_b, "D-", color="tab:purple", linewidth=2,
                label="(b) N=1, G varies (canonical)")
        for a, x_, y_ in zip(sweep_b_arms, xs_b, ys_b):
            if y_ is not None:
                ax.annotate(f"G={summary[a]['G']}", (x_, y_), fontsize=7,
                            xytext=(4, -10), textcoords="offset points",
                            color="tab:purple")
        ax.set_xlabel("Offloaded weights (GiB)")
        ax.set_ylabel(f"avg latency @ B={B} (s)")
        ax.set_title(f"(b) uniform vs (c) clustered at matched bytes  —  B={B}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

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
        "section": "0.10.1",
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
            "a_G_at_50pct":           [a for a, *_ in sweep_a()],
            "b_coverage_uniform_N1":  [a for a, *_ in sweep_b()],
            "c_coverage_clustered_G14_legacy": [a for a, *_ in sweep_c()],
            "d_K_at_G4_N1":           [a for a, *_ in sweep_d()],
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
