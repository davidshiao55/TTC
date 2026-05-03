#!/usr/bin/env python3
"""Phase 1a §1.13b — COTS vs Prefetch in COTS's target regime (decode-heavy).

Companion to ``bench_cots_vs_native_prefill.py`` (§1.13a). The latter ran
at the §0.10 setup (input=256, output=32 ⇒ prefill:decode CPU ratio 8:1)
which is **prefill-dominated** — well outside the §0.3.3 "free regime" and
the worst case for COTS.

This script runs the **decode-heavy** setup that COTS is designed for, per
`thesis_proposal.md` (FastTTS / TTC: many parallel beams, small max_tokens
per beam, decode-dominated execution):

    input_len  = 8     (short prompt → small prefill)
    output_len = 128   (decode dominates: 128 forwards per generate)
    batches    = {1, 4, 16}

At each batch, prefill:decode CPU ratio is 8 / (128·B) — even at B=16 that's
8 / 2048 ≈ 0.4% prefill (vs §1.13a's 88% prefill). This isolates COTS's
target regime.

Prefetch baselines use the densest-spread arms from Phase 0 §0.10.2(d)
(N=1, G varies — the strongest possible prefetch baseline at each depth).
UVA omitted per §0.10 finding 3.

Outputs go to ``David/Benchmarks/phase1/results/cots_vs_native_decode/``.
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
RESULTS_DIR = PHASE1_DIR / "results" / "cots_vs_native_decode"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
WARMUP_ITERS = 2
BENCH_ITERS = 3
DEFAULT_BATCHES = [1, 4, 16]

# Qwen2.5-7B layer arithmetic (BF16) — same as bench_cots_vs_native_prefill.py.
HIDDEN = 3584
QKV_OUT = (28 + 2 * 4) * 128  # = 4608
INTERMEDIATE = 18944
NUM_LAYERS = 28
WO_BYTES_PER_LAYER = HIDDEN * HIDDEN * 2  # NOT offloaded by COTS
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
    """COTS at three f values + densest-spread prefetch baselines (N=1, G
    varies, per Phase 0 §0.10.2(d)). UVA omitted per §0.10 finding 3.
    """
    arms: dict[str, dict] = {
        "none": {"flags": [], "offloaded_gib": 0.0, "family": "none"},
    }

    for f in (0.05, 0.09, 0.22, 0.50):
        bytes_offloaded = COTS_OFFLOADABLE_PER_LAYER * NUM_LAYERS * f
        arms[f"cots_{int(f * 100):03d}"] = {
            "flags": [
                "--offload-backend", "cots",
                "--cots-f-cpu-store", str(f),
            ],
            "offloaded_gib": round(gib(bytes_offloaded), 3),
            "family": "cots",
        }

    # Prefetch arms: N=1, G ∈ {2, 4, 7, 14, 28} → max-spread baselines per
    # §0.10.2(d), strongest possible prefetch reference at each depth.
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
    return arms


def run_cell(arm: str, flags: list[str], batch: int) -> Path:
    out_json = RESULTS_DIR / f"{arm}_b{batch}.json"
    out_log = RESULTS_DIR / f"{arm}_b{batch}.log"
    if out_json.exists():
        print(f"  [skip] {arm} b={batch} (cached)")
        return out_json

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
        f"COTS vs Prefetch (decode-heavy) — {MODEL.split('/')[-1]} BF16, "
        f"input={INPUT_LEN}, output={OUTPUT_LEN}",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only-arms", nargs="*", default=None)
    ap.add_argument("--batches", type=int, nargs="*", default=DEFAULT_BATCHES)
    ap.add_argument("--exp", action="store_true")
    ap.add_argument("--plot", action="store_true")
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
    print(f"  arms: {len(run_arms)}, batches: {args.batches}, "
          f"input={INPUT_LEN}, output={OUTPUT_LEN}")

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
        plot(summary, args.batches, RESULTS_DIR / "cots_vs_native_decode.pdf")


if __name__ == "__main__":
    main()
