#!/usr/bin/env python3
"""Phase 0.10.2 — Stock vLLM PrefetchOffloader broad parameter search.

Sweeps the (G, N) parameter space at K=1 to find the best stock config at
each offloaded-layer count. The point: the canonical divisor-G arms in
§0.10.3 (G ∈ {1, 2, 4, 7, 14, 28}, N=1) ALWAYS include the final decoder
layer in their offloaded set, which puts the wraparound prefetch in the
worst possible spot. Non-divisor G values can place the last offloaded
layer earlier in the model, leaving tail compute to hide the wraparound
H2D — and a search over (G, N) finds these configs naturally without any
patches.

Strategy: smart subsample. For each target offloaded layer count L,
enumerate all (G, N) pairs producing exactly L layers via the stock
picker `idx % G >= G − N`, cap at ~6 distinct (G, N) per L (stratified
on G to cover small/mid/large group sizes).

Workload: Qwen2.5-7B BF16, input=256 output=32, --enforce-eager, stock
`prefetch` backend (defer_wraparound=False). B=1 for the broad search;
top-3 per L revalidated at B ∈ {1, 16, 64}; K-sweep at the chosen
overall-best (G, N).

All cells run with **stock vLLM**, no thesis instrumentation patches.

Outputs to ``results/0.10_full_sweep/``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

PHASE0_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE0_DIR / "results" / "0.10_full_sweep"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
NUM_LAYERS = 28  # Qwen2.5-7B
INPUT_LEN = 256
OUTPUT_LEN = 32
WARMUP_ITERS = 2
BENCH_ITERS = 3
DEFAULT_BATCHES = [1, 16, 64]

HIDDEN, INTERMEDIATE = 3584, 18944
QKV_OUT = (28 + 2 * 4) * 128
PER_LAYER_BYTES = (
    HIDDEN * QKV_OUT * 2
    + HIDDEN * HIDDEN * 2
    + HIDDEN * 2 * INTERMEDIATE * 2
    + INTERMEDIATE * HIDDEN * 2
)


# ---------------------------------------------------------------------------
# Picker math — matches stock PrefetchOffloader exactly
# ---------------------------------------------------------------------------
def picker_layers(G: int, N: int, num_layers: int = NUM_LAYERS) -> list[int]:
    """Stock vLLM picker: offload module_index where idx % G >= G − N."""
    return [i for i in range(num_layers) if i % G >= G - N]


def buffer_pool_bytes(layers_per_offloaded_module: int, K: int = 1) -> int:
    """K × per-layer-bytes (the static GPU buffer pool)."""
    return K * PER_LAYER_BYTES


def gib(b: int) -> float:
    return b / (1024 ** 3)


# ---------------------------------------------------------------------------
# Smart subsample
# ---------------------------------------------------------------------------
TARGET_L_VALUES = [1, 2, 3, 4, 5, 7, 10, 14, 21, 28]
MAX_PER_L = 6


def all_configs_for_L(L: int, num_layers: int = NUM_LAYERS) -> list[tuple[int, int]]:
    """All (G, N) pairs producing exactly L offloaded layers via the picker."""
    out: list[tuple[int, int]] = []
    for G in range(1, num_layers + 1):
        for N in range(1, G + 1):
            if len(picker_layers(G, N, num_layers)) == L:
                out.append((G, N))
    return out


def subsample(configs: list[tuple[int, int]], cap: int) -> list[tuple[int, int]]:
    """Stratified subsample on G to cover small/mid/large group sizes."""
    if len(configs) <= cap:
        return configs
    # Sort by G; pick `cap` evenly-spaced indices.
    s = sorted(configs, key=lambda c: c[0])
    indices = [int(round(i * (len(s) - 1) / (cap - 1))) for i in range(cap)]
    return [s[i] for i in dict.fromkeys(indices)]  # dedup while preserving order


def build_arm_list() -> list[dict]:
    """Build the full list of cells to run. Each cell has metadata."""
    arms: list[dict] = []
    for L in TARGET_L_VALUES:
        configs = all_configs_for_L(L)
        if not configs:
            continue
        chosen = subsample(configs, MAX_PER_L)
        for G, N in chosen:
            layers = picker_layers(G, N)
            arms.append({
                "name": f"L{L}_G{G}_N{N}_K1",
                "G": G, "N": N, "K": 1,
                "L": L,
                "offloaded_layers": layers,
                "includes_final_layer": (NUM_LAYERS - 1) in layers,
                "saved_GiB": round(gib(L * PER_LAYER_BYTES), 4),
                "buffer_pool_GiB": round(gib(buffer_pool_bytes(L, 1)), 4),
            })
    return arms


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------
def cell_path(arm_name: str, batch: int) -> Path:
    return RESULTS_DIR / f"{arm_name}_b{batch}.json"


def run_cell(arm: dict, batch: int, K: int | None = None) -> Path:
    K = K if K is not None else arm["K"]
    suffix = f"_k{K}" if K != 1 else ""
    name = f"{arm['name']}{suffix}"
    out_json = cell_path(name, batch)
    out_log = out_json.with_suffix(".log")
    if out_json.exists():
        print(f"  [skip] {name} b={batch} (cached)")
        return out_json

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", MODEL, "--dtype", DTYPE,
        "--input-len", str(INPUT_LEN),
        "--output-len", str(OUTPUT_LEN),
        "--batch-size", str(batch),
        "--num-iters-warmup", str(WARMUP_ITERS),
        "--num-iters", str(BENCH_ITERS),
        "--enforce-eager",
        "--offload-backend", "prefetch",  # stock factory; no defer
        "--offload-group-size", str(arm["G"]),
        "--offload-num-in-group", str(arm["N"]),
        "--offload-prefetch-step", str(K),
        "--output-json", str(out_json),
    ]
    env = {**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}

    t0 = time.perf_counter()
    with open(out_log, "w") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)
            print(f"  [TIMEOUT] {name} b={batch}")
            return out_json
    dur = time.perf_counter() - t0
    if rc != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-12:])
        print(f"  [FAIL] {name} b={batch} rc={rc} ({dur:.1f}s)\n        {tail}")
    else:
        try:
            avg = json.loads(out_json.read_text()).get("avg_latency")
            print(f"  [ok]  {name} b={batch}: avg={avg:.4f}s ({dur:.1f}s)")
        except (json.JSONDecodeError, OSError):
            print(f"  [ok?] {name} b={batch}: ({dur:.1f}s) — JSON unparseable")
    return out_json


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def parse_avg(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("avg_latency")
    except (json.JSONDecodeError, OSError):
        return None


def best_per_L(arms: list[dict], batch: int) -> dict[int, list[tuple[dict, float]]]:
    """Per-L, ranked list of (arm, latency)."""
    grouped: dict[int, list[tuple[dict, float]]] = {}
    for arm in arms:
        avg = parse_avg(cell_path(arm["name"], batch))
        if avg is None:
            continue
        grouped.setdefault(arm["L"], []).append((arm, avg))
    for L in grouped:
        grouped[L].sort(key=lambda x: x[1])
    return grouped


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
    ap.add_argument("--exp", action="store_true",
                    help="Run the broad B=1 sweep (Phase A2).")
    ap.add_argument("--top3-batch-validate", action="store_true",
                    help="Re-run top-3 (G,N) per L at B ∈ {1, 16, 64} (Phase A3).")
    ap.add_argument("--ksweep-best", type=str, default=None,
                    help="K-sweep at the chosen best (G, N). Pass arm name "
                         "from broad search, e.g. 'L1_G28_N1_K1'.")
    ap.add_argument("--ksweep-batches", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--ksweep-Ks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--report", action="store_true",
                    help="Just print the summary table from cached JSONs; no runs.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    arms = build_arm_list()
    print(f"[setup] broad sweep: {len(arms)} cells across L ∈ {TARGET_L_VALUES}")
    print(f"[setup] workload: input={INPUT_LEN} output={OUTPUT_LEN}, "
          f"warmup={WARMUP_ITERS} iters={BENCH_ITERS}")

    # Phase A2: broad B=1 sweep
    if args.exp:
        print("\n=== A2: broad sweep at B=1 ===")
        for arm in arms:
            run_cell(arm, batch=1)

    # Phase A3: top-3 per L at B ∈ {1, 16, 64}
    if args.top3_batch_validate:
        print("\n=== A3: top-3 per L at B ∈ {1, 16, 64} ===")
        ranked = best_per_L(arms, batch=1)
        top3_arms: list[dict] = []
        for L, entries in ranked.items():
            for arm, _ in entries[:3]:
                if arm not in top3_arms:
                    top3_arms.append(arm)
        print(f"  picked {len(top3_arms)} top-3 cells across {len(ranked)} L values")
        for batch in [1, 16, 64]:
            for arm in top3_arms:
                run_cell(arm, batch)

    # Phase A4: K-sweep at chosen best
    if args.ksweep_best is not None:
        print(f"\n=== A4: K-sweep at {args.ksweep_best} ===")
        target = next((a for a in arms if a["name"] == args.ksweep_best), None)
        if target is None:
            sys.exit(f"unknown arm name {args.ksweep_best!r}; choices: "
                     f"{[a['name'] for a in arms]}")
        for K in args.ksweep_Ks:
            for batch in args.ksweep_batches:
                run_cell(target, batch, K=K)

    # Always print the summary table
    print("\n" + "=" * 100)
    print("BROAD SWEEP SUMMARY (B=1)")
    print("=" * 100)
    print(f"{'L':>3} {'arm':<22} {'G':>3} {'N':>3} "
          f"{'fin?':>5} {'GiB':>6} {'buf':>5} {'avg_lat (s)':>12}")
    print("-" * 100)
    for L in TARGET_L_VALUES:
        L_arms = [a for a in arms if a["L"] == L]
        rows = []
        for arm in L_arms:
            avg = parse_avg(cell_path(arm["name"], 1))
            rows.append((arm, avg))
        rows.sort(key=lambda r: (r[1] if r[1] is not None else float("inf")))
        for arm, avg in rows:
            avg_s = f"{avg:.4f}" if avg is not None else "—"
            fin = "YES" if arm["includes_final_layer"] else "no"
            print(f"{L:>3} {arm['name']:<22} {arm['G']:>3} {arm['N']:>3} "
                  f"{fin:>5} {arm['saved_GiB']:>6.2f} "
                  f"{arm['buffer_pool_GiB']:>5.2f} {avg_s:>12}")
        if L_arms:
            print()  # blank line between L groups

    # Best per L
    print("\n=== BEST PER L (B=1) ===")
    ranked = best_per_L(arms, batch=1)
    for L in sorted(ranked):
        arm, avg = ranked[L][0]
        fin = "YES" if arm["includes_final_layer"] else "no"
        print(f"  L={L:>2}: {arm['name']:<22} G={arm['G']:>2} N={arm['N']:>2} "
              f"fin={fin:>3} avg={avg:.4f}s")

    # Top-3 batch validation summary
    if any((cell_path(a["name"], 16).exists() or cell_path(a["name"], 64).exists())
           for a in arms):
        print("\n=== TOP-3 BATCH VALIDATION ===")
        for L in sorted(ranked):
            print(f"L={L}:")
            for arm, _ in ranked[L][:3]:
                row = []
                for batch in [1, 16, 64]:
                    avg = parse_avg(cell_path(arm["name"], batch))
                    row.append(f"B={batch}: {avg:.4f}s" if avg is not None else f"B={batch}: —")
                print(f"  {arm['name']:<22}  {'  '.join(row)}")

    # K-sweep summary
    if args.ksweep_best is not None:
        target_name = args.ksweep_best
        print(f"\n=== K-SWEEP AT {target_name} ===")
        for K in args.ksweep_Ks:
            suffix = f"_k{K}" if K != 1 else ""
            row = []
            for batch in args.ksweep_batches:
                avg = parse_avg(cell_path(f"{target_name}{suffix}", batch))
                row.append(f"B={batch}: {avg:.4f}s" if avg is not None else f"B={batch}: —")
            print(f"  K={K}  {'  '.join(row)}")

    # Persist summary
    summary = {
        "schema_version": 1,
        "section": "0.10.2",
        "hardware_id": hardware_id(),
        "model": {"name": MODEL, "dtype": DTYPE},
        "env": env_info(),
        "config": {
            "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
            "warmup_iters": WARMUP_ITERS, "bench_iters": BENCH_ITERS,
            "per_layer_decoder_bytes": PER_LAYER_BYTES,
            "num_layers": NUM_LAYERS,
        },
        "arms": [
            {**arm, "by_batch": {
                str(B): {"avg_latency_s": parse_avg(cell_path(arm["name"], B))}
                for B in [1, 16, 64]
            }}
            for arm in arms
        ],
    }
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
