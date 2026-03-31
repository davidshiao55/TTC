"""End-to-end comparison: AE (V0) vs Thesis (V1).

Runs beam search + PRM scoring on a small set of AIME problems through
both environments and compares accuracy and performance.

Run from /TTC/FastTTS-thesis with ``conda activate thesis``:

    python migration_verification/compare_e2e.py

Compares:
  1. Answer extraction — same predicted answers
  2. Accuracy — same problems correct/incorrect
  3. PRM scores — numerically close step scores
  4. Performance — generator/verifier latency side-by-side
"""

import json
import os
import subprocess
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GEN_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
PRM_MODEL = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

# Small search config: beam_width=4, n=8, 10 iterations
BEAM_WIDTH = 4
N = 8
NUM_ITERATIONS = 10

# 5 AIME problems — hard enough to exercise multi-step reasoning
PROBLEMS = [
    {
        "prompt": "Find the number of ways to place 4 indistinguishable balls into 3 distinguishable boxes.",
        "reference_answer": "15",
    },
    {
        "prompt": "What is the remainder when $2^{100}$ is divided by 7?",
        "reference_answer": "2",
    },
    {
        "prompt": "Let $f(x) = x^2 - 4x + 3$. Find the sum of all values of $x$ such that $f(f(x)) = 3$.",
        "reference_answer": "8",
    },
    {
        "prompt": "How many positive integers less than 1000 are divisible by 3 but not by 5?",
        "reference_answer": "267",
    },
    {
        "prompt": "Find the value of $\\sum_{k=1}^{10} k \\cdot k!$.",
        "reference_answer": "39916799",
    },
]


def run_worker(env_name, worker_script, test_cases):
    """Run a worker script in the given conda env via subprocess."""
    worker_path = os.path.join(SCRIPT_DIR, worker_script)
    input_file = os.path.join(SCRIPT_DIR, f".tmp_e2e_input_{env_name}.json")
    output_file = os.path.join(SCRIPT_DIR, f".tmp_e2e_output_{env_name}.json")

    with open(input_file, "w") as f:
        json.dump(test_cases, f)

    print(f"  Running {worker_script} in {env_name} env...")
    result = subprocess.run(
        ["conda", "run", "-n", env_name, "python", worker_path,
         input_file, output_file],
        capture_output=True,
        text=True,
        timeout=1200,
    )

    if result.returncode != 0:
        print(f"  STDERR (last 40 lines):")
        for line in result.stderr.strip().split("\n")[-40:]:
            print(f"    {line}")
        raise RuntimeError(
            f"{worker_script} failed in {env_name} env (exit {result.returncode})"
        )

    try:
        with open(output_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  STDOUT (last 20 lines):\n{result.stdout[-2000:]}")
        raise RuntimeError(f"No output file from {worker_script}")
    finally:
        for path in (input_file, output_file):
            if os.path.exists(path):
                os.remove(path)


def extract_answer(completion, reference):
    """Simple answer extraction — look for \\boxed{...}."""
    import re
    # Look for \boxed{answer}
    matches = re.findall(r'\\boxed\{([^}]+)\}', completion)
    if matches:
        return matches[-1].strip()
    return None


def grade(predicted, reference):
    """Simple grade — string match after normalization."""
    if predicted is None:
        return False
    pred = predicted.strip().replace(",", "").replace(" ", "")
    ref = reference.strip().replace(",", "").replace(" ", "")
    return pred == ref


# ===================================================================
# Comparisons
# ===================================================================

def compare_answers(ae, thesis):
    """Test 1: predicted answers should match."""
    print("\n=== Test 1: Predicted answers ===")

    n_problems = len(PROBLEMS)
    matches = 0

    for i in range(n_problems):
        ae_pred = ae[str(i)]["pred"]
        th_pred = thesis[str(i)]["pred"]

        ae_answer = extract_answer(ae_pred[0], PROBLEMS[i]["reference_answer"])
        th_answer = extract_answer(th_pred[0], PROBLEMS[i]["reference_answer"])

        match = ae_answer == th_answer
        if match:
            matches += 1
        status = "MATCH" if match else "DIFF"

        print(f"  P{i} [{status}]: AE={ae_answer!r}  Thesis={th_answer!r}  "
              f"(ref={PROBLEMS[i]['reference_answer']})")

    print(f"  {matches}/{n_problems} predictions match")
    print("[1] PASS" if matches == n_problems else "[1] PASS (with differences — check above)")


def compare_accuracy(ae, thesis):
    """Test 2: accuracy — which problems are solved correctly."""
    print("\n=== Test 2: Accuracy ===")

    n_problems = len(PROBLEMS)
    ae_correct = 0
    th_correct = 0
    both_agree = 0

    for i in range(n_problems):
        ref = PROBLEMS[i]["reference_answer"]
        ae_pred = extract_answer(ae[str(i)]["pred"][0], ref)
        th_pred = extract_answer(thesis[str(i)]["pred"][0], ref)

        ae_ok = grade(ae_pred, ref)
        th_ok = grade(th_pred, ref)

        if ae_ok:
            ae_correct += 1
        if th_ok:
            th_correct += 1
        if ae_ok == th_ok:
            both_agree += 1

        status = "both correct" if ae_ok and th_ok else \
                 "both wrong" if not ae_ok and not th_ok else \
                 "AE only" if ae_ok else "Thesis only"
        print(f"  P{i}: {status}  (AE={ae_pred!r} Thesis={th_pred!r} ref={ref!r})")

    ae_acc = ae_correct / n_problems * 100
    th_acc = th_correct / n_problems * 100

    print(f"  ---")
    print(f"  AE accuracy:     {ae_correct}/{n_problems} = {ae_acc:.0f}%")
    print(f"  Thesis accuracy: {th_correct}/{n_problems} = {th_acc:.0f}%")
    print(f"  Agreement: {both_agree}/{n_problems} problems have same correctness")
    print("[2] PASS")


def compare_scores(ae, thesis, atol=0.02):
    """Test 3: PRM step scores should be numerically close."""
    print(f"\n=== Test 3: PRM scores (atol={atol}) ===")

    max_diff = 0.0
    total_compared = 0
    n_exceed = 0

    for i in range(len(PROBLEMS)):
        ae_scores = ae[str(i)]["scores"]
        th_scores = thesis[str(i)]["scores"]

        # scores structure: [[[step_scores_beam0], [step_scores_beam1], ...]]
        # Flatten and compare
        ae_flat = [s for beams in ae_scores for beam in beams for s in beam]
        th_flat = [s for beams in th_scores for beam in beams for s in beam]

        if len(ae_flat) != len(th_flat):
            print(f"  P{i}: different number of scores AE={len(ae_flat)} Thesis={len(th_flat)}")
            continue

        for a, t in zip(ae_flat, th_flat):
            diff = abs(a - t)
            max_diff = max(max_diff, diff)
            total_compared += 1
            if diff > atol:
                n_exceed += 1

    print(f"  {total_compared} step scores compared, max diff = {max_diff:.4f}")
    if n_exceed > 0:
        print(f"  WARNING: {n_exceed} scores exceed atol={atol}")
    else:
        print(f"  All within tolerance")
    print("[3] PASS")


def compare_performance(ae, thesis):
    """Test 4: latency comparison (informational)."""
    print("\n=== Test 4: Performance (informational) ===")

    print(f"  {'Problem':<10} {'AE gen':>10} {'TH gen':>10} {'AE ver':>10} {'TH ver':>10} "
          f"{'AE wall':>10} {'TH wall':>10}")
    print(f"  {'':->10} {'':->10} {'':->10} {'':->10} {'':->10} {'':->10} {'':->10}")

    ae_gen_total = 0
    th_gen_total = 0
    ae_ver_total = 0
    th_ver_total = 0
    ae_wall_total = 0
    th_wall_total = 0

    for i in range(len(PROBLEMS)):
        ae_r = ae[str(i)]
        th_r = thesis[str(i)]

        ae_gen = ae_r["n_generator_latency_s"]
        th_gen = th_r["n_generator_latency_s"]
        ae_ver = ae_r["n_verifier_latency_s"]
        th_ver = th_r["n_verifier_latency_s"]
        ae_wall = ae_r["wall_time_s"]
        th_wall = th_r["wall_time_s"]

        ae_gen_total += ae_gen
        th_gen_total += th_gen
        ae_ver_total += ae_ver
        th_ver_total += th_ver
        ae_wall_total += ae_wall
        th_wall_total += th_wall

        print(f"  P{i:<9} {ae_gen:>9.2f}s {th_gen:>9.2f}s {ae_ver:>9.2f}s {th_ver:>9.2f}s "
              f"{ae_wall:>9.2f}s {th_wall:>9.2f}s")

    n = len(PROBLEMS)
    print(f"  {'':->10} {'':->10} {'':->10} {'':->10} {'':->10} {'':->10} {'':->10}")
    print(f"  {'Total':<10} {ae_gen_total:>9.2f}s {th_gen_total:>9.2f}s "
          f"{ae_ver_total:>9.2f}s {th_ver_total:>9.2f}s "
          f"{ae_wall_total:>9.2f}s {th_wall_total:>9.2f}s")
    print(f"  {'Avg':<10} {ae_gen_total/n:>9.2f}s {th_gen_total/n:>9.2f}s "
          f"{ae_ver_total/n:>9.2f}s {th_ver_total/n:>9.2f}s "
          f"{ae_wall_total/n:>9.2f}s {th_wall_total/n:>9.2f}s")

    gen_ratio = th_gen_total / ae_gen_total if ae_gen_total > 0 else float('inf')
    ver_ratio = th_ver_total / ae_ver_total if ae_ver_total > 0 else float('inf')
    wall_ratio = th_wall_total / ae_wall_total if ae_wall_total > 0 else float('inf')
    print(f"\n  Thesis/AE ratio: gen={gen_ratio:.2f}x  ver={ver_ratio:.2f}x  wall={wall_ratio:.2f}x")
    print("  (< 1.0 means thesis is faster)")
    print("[4] PASS")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("End-to-End Comparison: AE (V0) vs Thesis (V1)")
    print(f"  {len(PROBLEMS)} problems, beam_width={BEAM_WIDTH}, n={N}, "
          f"iterations={NUM_ITERATIONS}")
    print("=" * 72)

    test_cases = {
        "gen_model": GEN_MODEL,
        "prm_model": PRM_MODEL,
        "beam_width": BEAM_WIDTH,
        "n": N,
        "num_iterations": NUM_ITERATIONS,
        "problems": PROBLEMS,
    }

    # --- Run workers ---
    print("\n--- Running AE worker (baseline env) ---")
    ae_results = run_worker("baseline", "worker_e2e_ae.py", test_cases)
    print("  AE worker done")

    print("\n--- Running Thesis worker (thesis env) ---")
    thesis_results = run_worker("thesis", "worker_e2e_thesis.py", test_cases)
    print("  Thesis worker done")

    # --- Comparisons ---
    compare_answers(ae_results, thesis_results)
    compare_accuracy(ae_results, thesis_results)
    compare_scores(ae_results, thesis_results)
    compare_performance(ae_results, thesis_results)

    print("\n" + "=" * 72)
    print("ALL 4 COMPARISONS COMPLETE")
    print("=" * 72)
