"""Full-dataset comparison: AE (V0) vs Thesis (V1) — FastTTS with SBE.

Runs FastTTS beam search (with Speculative Beam Extension) on the full
AIME 2024 dataset (30 problems) through both environments and compares
accuracy and performance across multiple n values.

Baseline (no SBE) is already validated byte-for-byte identical in
compare.py Test 5, so only the SBE path is tested here.

Run from /TTC/FastTTS-thesis with ``conda activate thesis``:

    python migration_verification/compare_full.py
    python migration_verification/compare_full.py --n 8 16    # specific n values
    python migration_verification/compare_full.py --resume     # skip completed runs

Results are saved to migration_verification/full_results/ and can be
re-analyzed without re-running:

    python migration_verification/compare_full.py --analyze-only
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "full_results"

GEN_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
PRM_MODEL = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

BEAM_WIDTH = 4
NUM_ITERATIONS = 10
DEFAULT_N_VALUES = [8, 16, 32, 64]

SPEC_BEAM_EXTENSION = True
GEN_GPU_MEM = 0.25
VER_GPU_MEM = 0.15


def load_dataset():
    """Load AIME 2024 dataset."""
    from datasets import load_dataset as hf_load
    ds = hf_load("HuggingFaceH4/aime_2024", split="train")

    problems = []
    for i, ex in enumerate(ds):
        prompt_field = "problem" if "problem" in ex else "prompt"
        answer_field = "answer" if "answer" in ex else "reference_answer"
        problems.append({
            "id": ex.get("id", f"aime2024-{i}"),
            "prompt": ex[prompt_field],
            "reference_answer": str(ex[answer_field]),
        })
    return problems


def kill_gpu_processes():
    """Kill any orphaned processes holding GPU memory."""
    import glob
    killed = []
    my_pid = os.getpid()
    for maps_path in glob.glob("/proc/*/maps"):
        try:
            pid = int(maps_path.split("/")[2])
            if pid == my_pid:
                continue
            with open(maps_path, "r") as f:
                if "nvidia" in f.read():
                    os.kill(pid, 9)
                    killed.append(pid)
        except (ValueError, OSError, PermissionError):
            continue
    if killed:
        time.sleep(2)
        print(f"  Cleaned up {len(killed)} orphaned GPU process(es): {killed}")


def run_worker(env_name, worker_script, test_cases, timeout=7200):
    """Run a worker script in the given conda env via subprocess."""
    # Clean up any leftover GPU processes from previous runs
    kill_gpu_processes()

    worker_path = str(SCRIPT_DIR / worker_script)
    input_file = str(RESULTS_DIR / f".tmp_input_{env_name}.json")
    output_file = str(RESULTS_DIR / f".tmp_output_{env_name}.json")

    with open(input_file, "w") as f:
        json.dump(test_cases, f)

    print(f"  Running {worker_script} in '{env_name}' env "
          f"(n={test_cases['n']}, {len(test_cases['problems'])} problems)...")

    t0 = time.time()
    result = subprocess.run(
        ["conda", "run", "-n", env_name, "python", worker_path,
         input_file, output_file],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  STDERR (last 40 lines):")
        for line in result.stderr.strip().split("\n")[-40:]:
            print(f"    {line}")
        raise RuntimeError(
            f"{worker_script} failed in {env_name} env (exit {result.returncode})"
        )

    try:
        with open(output_file, "r") as f:
            data = json.load(f)
        print(f"  Done in {elapsed:.0f}s")
        return data
    except FileNotFoundError:
        print(f"  STDOUT (last 20 lines):\n{result.stdout[-2000:]}")
        raise RuntimeError(f"No output file from {worker_script}")
    finally:
        for path in (input_file, output_file):
            if os.path.exists(path):
                os.remove(path)


def result_path(env, n):
    return RESULTS_DIR / f"{env}_fasttts_n{n}.json"


def run_single(env_name, worker_script, n, problems, resume=False):
    """Run one (env, n) combination. Returns results dict."""
    out_path = result_path(env_name, n)
    if resume and out_path.exists():
        print(f"  [skip] {env_name}/n={n} — already exists")
        with open(out_path) as f:
            return json.load(f)

    test_cases = {
        "gen_model": GEN_MODEL,
        "prm_model": PRM_MODEL,
        "beam_width": BEAM_WIDTH,
        "n": n,
        "num_iterations": NUM_ITERATIONS,
        "spec_beam_extension": SPEC_BEAM_EXTENSION,
        "gen_gpu_mem": GEN_GPU_MEM,
        "ver_gpu_mem": VER_GPU_MEM,
        "problems": problems,
    }

    results = run_worker(env_name, worker_script, test_cases)

    # Save results
    with open(out_path, "w") as f:
        json.dump(results, f)

    return results


# ===================================================================
# Accuracy evaluation (reuses the project's grading infrastructure)
# ===================================================================

def extract_answer(completion):
    """Extract \\boxed{...} answer from completion text."""
    import re
    matches = re.findall(r'\\boxed\{([^}]+)\}', completion)
    if matches:
        return matches[-1].strip()
    return None


def grade_simple(predicted, reference):
    """Simple string-match grading after normalization."""
    if predicted is None:
        return False
    pred = predicted.strip().replace(",", "").replace(" ", "")
    ref = reference.strip().replace(",", "").replace(" ", "")
    return pred == ref


def grade_with_evaluator(results_dict, problems):
    """Grade using the project's math_equal grader for robust evaluation.

    Falls back to simple grading if the evaluator is unavailable.
    """
    eval_dir = SCRIPT_DIR.parent / "accuracy_evaluation" / "evaluation"

    try:
        sys.path.insert(0, str(eval_dir))
        from grader import math_equal_process
        from parser import extract_answer as eval_extract_answer
        use_evaluator = True
    except ImportError:
        use_evaluator = False

    correct = 0
    total = len(problems)
    per_problem = []

    for i in range(total):
        key = str(i)
        if key not in results_dict:
            per_problem.append({"correct": False, "pred": None})
            continue

        r = results_dict[key]
        ref = r["reference_answer"]

        # Use top-1 by PRM score (same as the existing evaluate.py)
        scores = r["scores"]
        completions = r["completions"]

        # scores structure: [[step_scores_per_beam, ...]]
        if scores and scores[0]:
            prod_scores = [np.prod(s) for s in scores[0]]
            best_idx = int(np.argmax(prod_scores))
            best_completion = completions[0][best_idx]
        else:
            best_completion = r["pred"][0] if r.get("pred") else ""

        if use_evaluator:
            pred = eval_extract_answer(best_completion, "math")
            try:
                is_correct = math_equal_process((0, pred, str(ref)))
            except Exception:
                is_correct = grade_simple(pred, str(ref))
        else:
            pred = extract_answer(best_completion)
            is_correct = grade_simple(pred, str(ref))

        if is_correct:
            correct += 1
        per_problem.append({"correct": bool(is_correct), "pred": pred, "ref": str(ref)})

    return correct, total, per_problem


# ===================================================================
# Performance analysis
# ===================================================================

def compute_perf_metrics(results_dict, n_problems):
    """Compute aggregate performance metrics from a results dict."""
    gen_lats = []
    ver_lats = []
    wall_times = []
    comp_tokens = []

    for i in range(n_problems):
        r = results_dict.get(str(i))
        if r is None:
            continue
        gen_lats.append(r.get("n_generator_latency_s", 0) or r.get("total_generator_latency_s", 0))
        ver_lats.append(r.get("n_verifier_latency_s", 0) or r.get("total_verifier_latency_s", 0))
        wall_times.append(r.get("wall_time_s", 0))
        comp_tokens.append(r.get("n_completion_tokens", 0))

    total_gen = sum(gen_lats)
    total_ver = sum(ver_lats)
    total_wall = sum(wall_times)
    total_tokens = sum(comp_tokens)
    n = len(gen_lats)

    return {
        "n_problems": n,
        "total_gen_s": total_gen,
        "total_ver_s": total_ver,
        "total_wall_s": total_wall,
        "avg_gen_s": total_gen / n if n else 0,
        "avg_ver_s": total_ver / n if n else 0,
        "avg_wall_s": total_wall / n if n else 0,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens / n if n else 0,
        "goodput": total_tokens / total_wall if total_wall > 0 else 0,
    }


# ===================================================================
# Comparison and reporting
# ===================================================================

def compare_accuracy(ae_results, th_results, problems, n_val):
    """Compare accuracy between AE and Thesis for a given n."""
    ae_correct, ae_total, ae_per = grade_with_evaluator(ae_results, problems)
    th_correct, th_total, th_per = grade_with_evaluator(th_results, problems)

    ae_acc = ae_correct / ae_total * 100 if ae_total else 0
    th_acc = th_correct / th_total * 100 if th_total else 0

    # Per-problem agreement
    agree = 0
    ae_only = 0
    th_only = 0
    for a, t in zip(ae_per, th_per):
        if a["correct"] == t["correct"]:
            agree += 1
        elif a["correct"]:
            ae_only += 1
        else:
            th_only += 1

    return {
        "n": n_val,
        "ae_correct": ae_correct,
        "th_correct": th_correct,
        "total": ae_total,
        "ae_acc": ae_acc,
        "th_acc": th_acc,
        "agree": agree,
        "ae_only": ae_only,
        "th_only": th_only,
        "ae_per_problem": ae_per,
        "th_per_problem": th_per,
    }


def print_summary(all_accuracy, all_perf):
    """Print final comparison tables."""
    print("\n" + "=" * 80)
    print("FULL DATASET COMPARISON — AIME 2024 (30 problems, FastTTS + SBE)")
    print("=" * 80)

    # Accuracy table
    if all_accuracy:
        print(f"\n  Accuracy")
        print(f"  {'n':>5}  {'AE acc':>12}  {'TH acc':>12}  {'Agree':>7}  {'AE only':>8}  {'TH only':>8}")
        print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*7}  {'─'*8}  {'─'*8}")
        for a in sorted(all_accuracy, key=lambda x: x["n"]):
            print(f"  {a['n']:>5}  "
                  f"{a['ae_correct']:>2}/{a['total']} ({a['ae_acc']:>5.1f}%)  "
                  f"{a['th_correct']:>2}/{a['total']} ({a['th_acc']:>5.1f}%)  "
                  f"{a['agree']:>5}/{a['total']}  "
                  f"{a['ae_only']:>6}  "
                  f"{a['th_only']:>6}")

    # Performance table
    if all_perf:
        print(f"\n  Latency (avg per problem)")
        print(f"  {'n':>5}  {'AE gen':>9}  {'TH gen':>9}  {'AE ver':>9}  {'TH ver':>9}  "
              f"{'AE wall':>9}  {'TH wall':>9}  {'Wall ratio':>10}")
        print(f"  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*10}")
        for p in sorted(all_perf, key=lambda x: x["n"]):
            ae = p["ae"]
            th = p["th"]
            ratio = th["avg_wall_s"] / ae["avg_wall_s"] if ae["avg_wall_s"] > 0 else float("inf")
            print(f"  {p['n']:>5}  "
                  f"{ae['avg_gen_s']:>8.2f}s  {th['avg_gen_s']:>8.2f}s  "
                  f"{ae['avg_ver_s']:>8.2f}s  {th['avg_ver_s']:>8.2f}s  "
                  f"{ae['avg_wall_s']:>8.2f}s  {th['avg_wall_s']:>8.2f}s  "
                  f"{ratio:>9.2f}x")

        # Goodput comparison
        print(f"\n  Goodput")
        print(f"  {'n':>5}  {'AE tok/s':>10}  {'TH tok/s':>10}  {'Speedup':>9}")
        print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*9}")
        for p in sorted(all_perf, key=lambda x: x["n"]):
            ae = p["ae"]
            th = p["th"]
            speedup = th["goodput"] / ae["goodput"] if ae["goodput"] > 0 else float("inf")
            print(f"  {p['n']:>5}  {ae['goodput']:>9.1f}  {th['goodput']:>9.1f}  {speedup:>8.2f}x")


def save_summary_json(all_accuracy, all_perf):
    """Save summary as JSON for later analysis."""
    summary = {
        "dataset": "AIME 2024",
        "method": "fasttts (SBE)",
        "accuracy": [],
        "performance": [],
    }
    for a in all_accuracy:
        summary["accuracy"].append({
            "n": a["n"],
            "ae_correct": a["ae_correct"],
            "th_correct": a["th_correct"],
            "total": a["total"],
            "ae_acc": a["ae_acc"],
            "th_acc": a["th_acc"],
            "agree": a["agree"],
            "ae_only": a["ae_only"],
            "th_only": a["th_only"],
        })
    for p in all_perf:
        summary["performance"].append({
            "n": p["n"],
            "ae": p["ae"],
            "th": p["th"],
        })

    out = RESULTS_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full AIME 2024 comparison: AE (V0) vs Thesis (V1) — FastTTS + SBE")
    parser.add_argument("--n", type=int, nargs="+", default=DEFAULT_N_VALUES,
                        help=f"n values to test (default: {DEFAULT_N_VALUES})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs that already have saved results")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results, don't run anything")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_values = sorted(args.n)

    print("=" * 80)
    print("Full Dataset Comparison: AE (V0) vs Thesis (V1) — FastTTS + SBE")
    print(f"  Dataset: AIME 2024")
    print(f"  n values: {n_values}")
    print(f"  beam_width={BEAM_WIDTH}, iterations={NUM_ITERATIONS}")
    if args.analyze_only:
        print(f"  Mode: analyze-only (no new runs)")
    elif args.resume:
        print(f"  Mode: resume (skip completed)")
    print("=" * 80)

    # Load dataset
    print("\nLoading AIME 2024 dataset...")
    problems = load_dataset()
    print(f"  {len(problems)} problems loaded")

    if not args.analyze_only:
        for n in n_values:
            print(f"\n{'─'*60}")
            print(f"  n={n}")
            print(f"{'─'*60}")

            # AE
            print(f"\n  --- AE (baseline env) ---")
            run_single("baseline", "worker_full_ae.py", n,
                       problems, resume=args.resume)

            # Thesis
            print(f"\n  --- Thesis (thesis env) ---")
            run_single("thesis", "worker_full_thesis.py", n,
                       problems, resume=args.resume)

    # Analyze
    print("\n" + "=" * 80)
    print("ANALYZING RESULTS")
    print("=" * 80)

    all_accuracy = []
    all_perf = []

    for n in n_values:
        ae_path = result_path("baseline", n)
        th_path = result_path("thesis", n)

        if not ae_path.exists() or not th_path.exists():
            missing = []
            if not ae_path.exists():
                missing.append(f"AE ({ae_path.name})")
            if not th_path.exists():
                missing.append(f"Thesis ({th_path.name})")
            print(f"  [skip] n={n} — missing: {', '.join(missing)}")
            continue

        with open(ae_path) as f:
            ae_results = json.load(f)
        with open(th_path) as f:
            th_results = json.load(f)

        # Accuracy
        acc = compare_accuracy(ae_results, th_results, problems, n)
        all_accuracy.append(acc)
        print(f"  n={n}: AE={acc['ae_correct']}/{acc['total']} "
              f"({acc['ae_acc']:.1f}%)  TH={acc['th_correct']}/{acc['total']} "
              f"({acc['th_acc']:.1f}%)  agree={acc['agree']}/{acc['total']}")

        # Performance
        ae_perf = compute_perf_metrics(ae_results, len(problems))
        th_perf = compute_perf_metrics(th_results, len(problems))
        all_perf.append({"n": n, "ae": ae_perf, "th": th_perf})

    print_summary(all_accuracy, all_perf)
    save_summary_json(all_accuracy, all_perf)

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
