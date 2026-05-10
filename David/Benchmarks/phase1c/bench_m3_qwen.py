#!/usr/bin/env python3
"""§1c.29 real-model A/B — Qwen2.5-7B M3 wait kernel on/off.

The synthetic stub at `bench_m3_wait_kernel_ab.py` confirmed M3 is
substrate-positive. THIS bench runs the same A/B on the real model
with FastTTS-equivalent decode (input=8, output=128) so the
acceptance gate can be evaluated against the §1c.29 commit-3
revision:

  1. Real-mode wall delta ≥ +50 ms/generate.
  2. Spin-cost budget: spin_iters_total × ~100 ns ≤ 10 % of
     recovered sync_cb_wait_total_ns (estimate; nsys trace
     replaces this if margin is tight).
  3. No correctness regression (parity test already covers).

Seven arms (apples-to-apples per the commit-3-real review):
  * none_capture                — graph-mode no-offload baseline.
  * cots_native_eager_dryrun    — substrate gate, no capture.
  * cots_native_eager_real      — substrate + real CPU GEMM, no
                                  capture; the production
                                  alternative M3 must beat to
                                  justify a default flip.
  * cots_m3_off_capture_dryrun  — baseline substrate, M3 off.
  * cots_m3_on_capture_dryrun   — substrate A/B, M3 on.
  * cots_m3_off_capture_real    — production baseline, M3 off.
  * cots_m3_on_capture_real     — M3 candidate.

Two env vars are required for the spin-budget gate to be
populated:
  * VLLM_COTS_DIAG=1 — enables the diag wait kernel that
    increments `m3_immediate_resume_count`,
    `m3_lagging_wait_count`, `m3_wait_spin_iters_total`. Without
    this the production wait kernel runs and the M3 counters
    stay zero.
  * VLLM_COTS_DUMP_COUNTERS=1 — registers the atexit dump in
    `cots_ops._dump_counters_at_exit` (cots_ops.py:451) that
    writes per-runner counters to stderr at EngineCore process
    exit. The harness parses these from the .log file next to
    the .json. Without this env, the .log has no counter dump
    and the spin-budget computation is skipped.

Counter reset caveat (commit-3-real review note): the front-end
side `_diag_pre` reset in latency.py runs in the bench process,
NOT in the EngineCore subprocess that owns the CotsCpuInfer
counters. So under multiproc=spawn the dumped counters span
ALL warmup + measured iters, not just the measured window. The
spin-budget RATIO (spin / sync_cb_wait_total_ns saved) is
scale-invariant as long as both arms ran the same warmup + iter
count, but absolute per-generate numbers should be divided by
the total generate count if they're reported.

Outputs go to `David/Benchmarks/phase1c/results/m3_qwen/`.

Default grid: batches [1], 1 warmup + 2 iters (the §1c.27 bench
shape — fast enough for repeat runs while letting the wall delta
stabilize). First run loads weights ~30s.

Run:
    VLLM_COTS_DIAG=1 /opt/conda/envs/thesis/bin/python \\
        David/Benchmarks/phase1c/bench_m3_qwen.py
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

PHASE1C_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE1C_DIR / "results" / "m3_qwen"

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
INPUT_LEN = 8
OUTPUT_LEN = 128
DEFAULT_BATCHES = [1]
DEFAULT_F = 0.05
DEFAULT_THREADS = 16


def arms_for(threads: int, f_cpu_store: float = DEFAULT_F) -> dict[str, list[str]]:
    """Return arm name -> CLI flag list.

    Seven-arm grid (apples-to-apples per the commit-3-real review):
      * `none_capture`              — graph-mode no-offload baseline
      * `cots_native_eager_dryrun`  — substrate gate, no capture
      * `cots_native_eager_real`    — substrate + real CPU GEMM, no
                                      capture; the reference M3 must
                                      beat to justify a default flip
      * `cots_m3_off_capture_dryrun`/`_on_*`  — captured, no GEMM
      * `cots_m3_off_capture_real` / `_on_*` — captured, real GEMM

    Native+capture+M3 is the only path that uses the wait kernel
    (§1c.29 gate 1 rejects M3 under cpu_runner='python'; gate 2
    rejects M3 under enforce_eager=True). Eager arms run on the
    same native substrate without graph capture, providing the
    "should we just use native_eager_real instead of M3?" reference.
    """
    cots_base = [
        "--offload-backend",
        "cots",
        "--cots-f-cpu-store",
        str(f_cpu_store),
        "--cots-cpu-runner",
        "native",
    ]
    if threads != DEFAULT_THREADS:
        cots_base += ["--cots-cpu-num-threads", str(threads)]
    return {
        "none_capture": [],
        "cots_native_eager_dryrun": cots_base + ["--cots-dry-run"],
        "cots_native_eager_real": cots_base,
        "cots_m3_off_capture_dryrun": cots_base + ["--cots-dry-run"],
        "cots_m3_on_capture_dryrun": cots_base
        + ["--cots-dry-run", "--cots-m3-wait-kernel"],
        "cots_m3_off_capture_real": cots_base,
        "cots_m3_on_capture_real": cots_base + ["--cots-m3-wait-kernel"],
    }


# Arms that need --enforce-eager. Eager arms include the explicit
# native_eager_{dryrun,real} reference cells; other arms run under
# graph capture (the M3 path requires it; none_capture explicitly
# tests it).
_EAGER_ARMS: frozenset[str] = frozenset({
    "cots_native_eager_dryrun",
    "cots_native_eager_real",
})


def cell_path(
    arm: str,
    batch: int,
    threads: int,
    *,
    output_len: int = OUTPUT_LEN,
    f_cpu_store: float = DEFAULT_F,
) -> Path:
    """Per-cell JSON path. Default-valued (output_len, f) are
    elided from the filename so the existing committed grid at
    output=128, f=0.05 is preserved bit-for-bit. Non-default
    workload points get a suffix `_o<output>_f<f×100>` so a wider
    grid can coexist with the prior cells under the same dir."""
    name = f"{arm}_b{batch}"
    if threads != DEFAULT_THREADS:
        name += f"_t{threads}"
    if output_len != OUTPUT_LEN:
        name += f"_o{output_len}"
    if f_cpu_store != DEFAULT_F:
        # 0.10 → "f10", 0.05 → elided, 0.20 → "f20" (avoids
        # decimal points in filenames).
        name += f"_f{int(round(f_cpu_store * 100))}"
    return RESULTS_DIR / f"{name}.json"


def run_cell(
    arm: str,
    flags: list[str],
    batch: int,
    threads: int,
    *,
    num_iters: int,
    num_iters_warmup: int,
    extra_flags: list[str],
    output_len: int = OUTPUT_LEN,
    f_cpu_store: float = DEFAULT_F,
) -> Path:
    out_json = cell_path(
        arm, batch, threads,
        output_len=output_len, f_cpu_store=f_cpu_store,
    )
    out_log = out_json.with_suffix(".log")
    if out_json.exists():
        print(f"  [skip] {arm} b={batch} (cached)")
        return out_json
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model",
        MODEL,
        "--dtype",
        DTYPE,
        "--input-len",
        str(INPUT_LEN),
        "--output-len",
        str(output_len),
        "--batch-size",
        str(batch),
        "--num-iters-warmup",
        str(num_iters_warmup),
        "--num-iters",
        str(num_iters),
        "--output-json",
        str(out_json),
        *flags,
        *extra_flags,
    ]
    if arm in _EAGER_ARMS:
        cmd.append("--enforce-eager")
    t0 = time.perf_counter()
    with open(out_log, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    dur = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n        ".join(out_log.read_text().splitlines()[-20:])
        print(f"  [FAIL] {arm} b={batch} rc={proc.returncode} ({dur:.1f}s)\n        {tail}")
    else:
        avg = json.loads(out_json.read_text()).get("avg_latency")
        print(f"  [ok]  {arm} b={batch}: avg={avg:.4f}s ({dur:.1f}s)")
    return out_json


def parse_avg(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("avg_latency")
    except (json.JSONDecodeError, OSError):
        return None


def parse_cots_counters(path: Path) -> dict[str, int]:
    """Best-effort extract `[cots]` counter dump from the bench log
    (printed at process exit when VLLM_COTS_DUMP_COUNTERS=1 — see
    cots_ops.py:_dump_counters_at_exit at cots_ops.py:451). The bench
    launches `vllm bench latency` as a subprocess that exits cleanly;
    the dump lands in the .log next to the .json. Returns {} when the
    dump isn't found.

    NB: VLLM_COTS_DIAG=1 controls whether the diag wait kernel runs
    (gating m3_immediate_resume_count / m3_lagging_wait_count /
    m3_wait_spin_iters_total at increment time);
    VLLM_COTS_DUMP_COUNTERS=1 controls whether the atexit dump fires
    (gating whether we can READ the counters). Both are needed for
    the M3 spin-budget gate."""
    log_path = path.with_suffix(".log")
    if not log_path.exists():
        return {}
    counters: dict[str, int] = {}
    keys = (
        "m3_immediate_resume_count",
        "m3_lagging_wait_count",
        "m3_wait_spin_iters_total",
        "sync_cb_count",
        "sync_cb_wait_total_ns",
        "dispatch_cb_count",
        "worker_run_count",
        "worker_busy_total_ns",
    )
    try:
        for line in log_path.read_text().splitlines():
            # The dump from cots_ops.py:425 prints lines like
            #   "(EngineCore pid=12345)     m3_lagging_wait_count: 1234"
            # under VLLM_WORKER_MULTIPROC_METHOD=spawn. Strip the
            # engine-prefix and split on ":".
            stripped = line.strip()
            for key in keys:
                tag = key + ":"
                idx = stripped.find(tag)
                if idx == -1:
                    continue
                try:
                    counters[key] = int(stripped[idx + len(tag):].strip().split()[0])
                except (ValueError, IndexError):
                    pass
                break
    except OSError:
        pass
    return counters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, nargs="*", default=DEFAULT_BATCHES)
    ap.add_argument("--threads", type=int, nargs="*", default=[DEFAULT_THREADS])
    ap.add_argument(
        "--only-arms",
        nargs="*",
        default=None,
        help="Subset of arms to run; default = all 7 (none_capture + "
        "native_eager × {dryrun,real} + m3 × {off,on} × {dryrun,real})",
    )
    ap.add_argument("--num-iters", type=int, default=2)
    ap.add_argument("--num-iters-warmup", type=int, default=1)
    ap.add_argument(
        "--output-len", type=int, default=OUTPUT_LEN,
        help="Generated tokens per sequence (default 128).",
    )
    ap.add_argument(
        "--f-cpu-store", type=float, default=DEFAULT_F,
        help="CotsOffloadConfig.f_cpu_store (default 0.05).",
    )
    ap.add_argument(
        "--extra-flag",
        action="append",
        default=[],
        help="Pass-through flag string for `vllm bench latency` (shlex-split).",
    )
    args = ap.parse_args()
    args.extra_flag = [tok for s in args.extra_flag for tok in shlex.split(s)]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] threads={args.threads}, batches={args.batches}, "
        f"f={args.f_cpu_store}, input={INPUT_LEN}, output={args.output_len}, "
        f"iters={args.num_iters} (warmup {args.num_iters_warmup})"
    )

    rows: dict = {}
    for t in args.threads:
        arms = arms_for(t, f_cpu_store=args.f_cpu_store)
        run_arms = (
            arms if not args.only_arms
            else {n: arms[n] for n in args.only_arms if n in arms}
        )
        print(f"\n[t={t}] arms={list(run_arms)}")
        for arm, flags in run_arms.items():
            for B in args.batches:
                run_cell(
                    arm,
                    flags,
                    B,
                    t,
                    num_iters=args.num_iters,
                    num_iters_warmup=args.num_iters_warmup,
                    extra_flags=args.extra_flag,
                    output_len=args.output_len,
                    f_cpu_store=args.f_cpu_store,
                )

        for arm in arms_for(t, f_cpu_store=args.f_cpu_store):
            row = {
                B: parse_avg(cell_path(
                    arm, B, t,
                    output_len=args.output_len,
                    f_cpu_store=args.f_cpu_store,
                ))
                for B in args.batches
            }
            rows[(arm, t)] = row

    print("\n" + "=" * 88)
    header = f"{'arm':<32} {'t':>3}  " + "  ".join(
        f"{f'B={B} (s)':>11}" for B in args.batches
    )
    print(header)
    print("-" * 88)
    for (arm, t), row in rows.items():
        cells = "  ".join(
            f"{row[B]:>11.4f}" if row[B] is not None else f"{'—':>11}"
            for B in args.batches
        )
        print(f"{arm:<32} {t:>3}  {cells}")
    print("=" * 88)

    print("\n=== §1c.29 acceptance gate evaluation ===")
    for t in args.threads:
        for B in args.batches:
            off_dry = rows.get(("cots_m3_off_capture_dryrun", t), {}).get(B)
            on_dry = rows.get(("cots_m3_on_capture_dryrun", t), {}).get(B)
            off_real = rows.get(("cots_m3_off_capture_real", t), {}).get(B)
            on_real = rows.get(("cots_m3_on_capture_real", t), {}).get(B)
            none_cap = rows.get(("none_capture", t), {}).get(B)
            eager_dry = rows.get(("cots_native_eager_dryrun", t), {}).get(B)
            eager_real = rows.get(("cots_native_eager_real", t), {}).get(B)
            print(f"\n  t={t} B={B}:")
            if off_dry is not None and on_dry is not None:
                d_dry = (off_dry - on_dry) * 1000  # ms
                print(
                    f"    dryrun: M3 off {off_dry:.4f}s, on {on_dry:.4f}s  "
                    f"Δ {d_dry:+.1f} ms/gen"
                )
            if off_real is not None and on_real is not None:
                d_real_ms = (off_real - on_real) * 1000
                target = "PASS" if d_real_ms >= 50.0 else "FAIL"
                print(
                    f"    real:   M3 off {off_real:.4f}s, on {on_real:.4f}s  "
                    f"Δ {d_real_ms:+.1f} ms/gen   "
                    f"§1c.29 gate1 ≥ +50 ms: {target}"
                )
            # Reviewer's apples-to-apples gate: native_capture_real
            # + M3 must beat native_eager_real by a meaningful
            # margin to justify the captured-graph + wait-kernel
            # path over plain eager. The +50 ms on M3-off is a
            # substrate trade; what matters for production is
            # whether the production candidate (M3 on, captured)
            # is faster than just running eager.
            if eager_real is not None and on_real is not None:
                d_vs_eager_ms = (eager_real - on_real) * 1000
                target = "PASS" if d_vs_eager_ms > 0 else "FAIL"
                print(
                    f"    vs eager: native_eager_real "
                    f"{eager_real:.4f}s, m3_on_capture_real "
                    f"{on_real:.4f}s  Δ {d_vs_eager_ms:+.1f} ms/gen   "
                    f"M3-vs-eager (must be > 0 to justify default flip): "
                    f"{target}"
                )
            # Capture-mode COTS overhead vs none_capture (the §1.14
            # equivalent for M3-on). Lower is better; informational.
            if none_cap is not None and on_real is not None:
                cap_overhead = (on_real - none_cap) * 1000
                print(
                    f"    capture overhead: m3_on_real - none_capture = "
                    f"{cap_overhead:+.1f} ms/gen"
                )
            if none_cap is not None and off_real is not None:
                cap_overhead_off = (off_real - none_cap) * 1000
                print(
                    f"                      m3_off_real - none_capture = "
                    f"{cap_overhead_off:+.1f} ms/gen"
                )
            if eager_dry is not None and eager_real is not None:
                d_eager_cpu = (eager_real - eager_dry) * 1000
                print(
                    f"    eager cpu work: native_eager_real - "
                    f"native_eager_dryrun = {d_eager_cpu:+.1f} ms/gen"
                )
                # Spin budget (estimate) — relies on counter dump
                # being captured in the .log. With DIAG=1 the dump
                # is enabled.
                on_counters = parse_cots_counters(
                    cell_path("cots_m3_on_capture_real", B, t)
                )
                off_counters = parse_cots_counters(
                    cell_path("cots_m3_off_capture_real", B, t)
                )
                if on_counters and off_counters:
                    spin = on_counters.get("m3_wait_spin_iters_total", 0)
                    saved = off_counters.get("sync_cb_wait_total_ns", 0)
                    spin_ns_est = spin * 100  # PTX nanosleep.u32 100 hint
                    if saved > 0:
                        ratio = spin_ns_est / saved
                        bud = "PASS" if ratio <= 0.10 else "FAIL"
                        print(
                            f"    spin budget (est): "
                            f"{spin_ns_est/1e6:.2f} ms / {saved/1e6:.2f} ms "
                            f"= {ratio*100:.1f}%   ≤ 10%: {bud}"
                        )
                        imm = on_counters.get("m3_immediate_resume_count", 0)
                        lag = on_counters.get("m3_lagging_wait_count", 0)
                        tot = imm + lag
                        if tot:
                            print(
                                f"    diag canary: immediate={imm}, "
                                f"lagging={lag} ({lag/tot*100:.1f}%)"
                            )

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "input_len": INPUT_LEN,
                "output_len": OUTPUT_LEN,
                "f": DEFAULT_F,
                "batches": args.batches,
                "threads": args.threads,
                "num_iters": args.num_iters,
                "rows": {
                    f"{arm}_t{t}": {str(B): v for B, v in rows[(arm, t)].items()}
                    for (arm, t) in rows
                },
            },
            indent=2,
        )
    )
    print(f"\n  results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
