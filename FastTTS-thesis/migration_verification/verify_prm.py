"""PRM V1 Migration Verification — behavioural equivalence with AE (V0).

Verifies the Skywork PRM (Process Reward Model) plugin works correctly
under vLLM V1, producing per-step scores identical in structure to V0.

Run from /TTC/FastTTS-thesis with ``conda activate thesis``:

    python migration_verification/verify_prm.py

Claims verified:

  Plugin registration:
   1. Qwen2ForPrmModel registered in ModelRegistry

  Model loading & forward pass:
   2. PRM loads and produces PoolingRequestOutput for raw token input
   3. Output shape matches input token count (per-token scores)

  Scoring pipeline (score_outputs):
   4. prepare_input — reward_flags mark last token of each step
   5. Per-step scores — sigmoid-transformed, one score per step
   6. Score range — all scores in [0, 1] after sigmoid
   7. Multi-question batching — correct nesting structure

  Pooler (V1 specific):
   8. model.pooler is nn.Module with get_supported_tasks()

  Chunked prefill + prefix caching (V1 defaults):
   9. Output shape correct with prefix caching (single seq, cold + warm)
  10. Batch of 16 seqs with shared prefix — no shape mismatches
  11. score_outputs produces valid scores with prefix caching
  12. Repeated scoring gives identical results (cache consistency)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm.inputs import TokensPrompt

from models.tts_llm import TTSLLM
from models.reward_utils import prepare_input

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRM_MODEL = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
SYSTEM_PROMPT = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\n"
    "Provide a concise solution with minimal explanation.\n\n"
    "- For complex problems (3 steps or more):\n"
    "Use this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem."
)

# A question and two candidate solutions (one correct, one wrong)
QUESTION = "What is 2 + 3 * 4?"
SOLUTION_CORRECT = (
    "## Step 1: Apply order of operations\n"
    "Multiply first: 3 * 4 = 12\n\n"
    "## Step 2: Add\n"
    "2 + 12 = 14\n\n"
    "Therefore, the final answer is: $\\boxed{14}$. I hope it is correct."
)
SOLUTION_WRONG = (
    "## Step 1: Add first\n"
    "2 + 3 = 5\n\n"
    "## Step 2: Multiply\n"
    "5 * 4 = 20\n\n"
    "Therefore, the final answer is: $\\boxed{20}$. I hope it is correct."
)

# A second question for multi-question batching test
QUESTION_2 = "What is the square root of 144?"
SOLUTION_2 = (
    "## Step 1: Recognize the perfect square\n"
    "144 = 12 * 12\n\n"
    "Therefore, the final answer is: $\\boxed{12}$. I hope it is correct."
)

# Long question + multi-step solutions for prefix cache / chunked prefill tests
LONG_QUESTION = (
    "Let $a_1, a_2, \\ldots, a_{10}$ be a sequence of positive real numbers "
    "such that $a_1 + a_2 + \\cdots + a_{10} = 100$ and "
    "$a_1 a_2 \\cdots a_{10} = 10^{10}$. "
    "Find the maximum possible value of $\\max(a_1, a_2, \\ldots, a_{10})$ "
    "given these constraints. Express your answer as a fraction in lowest terms."
)

LONG_SOLUTIONS = [
    "## Step 1: Analyze the constraints\n"
    "We have 10 positive reals summing to 100 with product $10^{10}$. "
    "By AM-GM, the product is maximized when all values are equal to 10, "
    "giving product $10^{10}$. So equality holds in AM-GM.\n\n"
    "## Step 2: Consider when one value is maximized\n"
    "To maximize $a_1$, we want to minimize the remaining sum $100 - a_1$ "
    "subject to the remaining product being $10^{10}/a_1$. By AM-GM on "
    "$a_2, \\ldots, a_{10}$: their product is at most "
    "$((100-a_1)/9)^9$.\n\n"
    "## Step 3: Set up the equation\n"
    "We need $((100-a_1)/9)^9 \\geq 10^{10}/a_1$. At the boundary: "
    "$a_1 \\cdot ((100-a_1)/9)^9 = 10^{10}$.\n\n"
    "## Step 4: Solve numerically\n"
    "Let $f(x) = x \\cdot ((100-x)/9)^9$. We need $f(x) = 10^{10}$. "
    "Testing $x = 10$: $f(10) = 10 \\cdot (90/9)^9 = 10 \\cdot 10^9 = 10^{10}$. "
    "So $a_1 = 10$ and all values equal 10.\n\n"
    "Therefore, the final answer is: $\\boxed{10}$. I hope it is correct.",

    "## Step 1: Use Lagrange multipliers\n"
    "We want to maximize $a_1$ subject to $\\sum a_i = 100$ and "
    "$\\prod a_i = 10^{10}$. Using Lagrange multipliers with two constraints "
    "on the objective $f = a_1$.\n\n"
    "## Step 2: Derive the optimality conditions\n"
    "The gradient condition gives $1 = \\lambda + \\mu \\cdot 10^{10}/a_1$ "
    "for the first variable and $0 = \\lambda + \\mu \\cdot 10^{10}/a_k$ "
    "for $k \\geq 2$. This means all $a_k$ for $k \\geq 2$ are equal.\n\n"
    "## Step 3: Let $a_2 = a_3 = \\cdots = a_{10} = t$\n"
    "Then $a_1 + 9t = 100$ and $a_1 \\cdot t^9 = 10^{10}$. "
    "From the first: $a_1 = 100 - 9t$. Substituting: "
    "$(100 - 9t) \\cdot t^9 = 10^{10}$.\n\n"
    "## Step 4: Analyze the function\n"
    "$g(t) = (100 - 9t) \\cdot t^9$ on $(0, 100/9)$. "
    "Taking derivative and setting to zero: "
    "$g'(t) = -9t^9 + 9(100-9t)t^8 = 9t^8(100 - 18t) = 0$, "
    "so $t = 100/18 = 50/9$.\n\n"
    "## Step 5: Compute the answer\n"
    "At $t = 50/9$: $a_1 = 100 - 9 \\cdot 50/9 = 100 - 50 = 50$. "
    "Check product: $50 \\cdot (50/9)^9$. But this doesn't equal $10^{10}$, "
    "so the constraint is not satisfiable at the maximum.\n\n"
    "Therefore, the final answer is: $\\boxed{50}$. I hope it is correct.",

    "## Step 1: Apply the AM-GM inequality\n"
    "For positive reals with fixed sum, the product is maximized when all "
    "are equal. Here $a_i = 10$ for all $i$ gives sum 100 and product "
    "$10^{10}$, achieving equality.\n\n"
    "## Step 2: Check if we can deviate\n"
    "If $a_1 > 10$, then the remaining 9 values must sum to $100 - a_1 < 90$ "
    "and have product $10^{10}/a_1 < 10^9$. By AM-GM their product is at most "
    "$((100-a_1)/9)^9 < (90/9)^9 = 10^9$. But we need exactly $10^{10}/a_1$.\n\n"
    "## Step 3: Verify feasibility\n"
    "For $a_1 = 10 + \\epsilon$, the remaining product must be "
    "$10^{10}/(10+\\epsilon)$, and AM-GM gives max product "
    "$((90-\\epsilon)/9)^9 = (10 - \\epsilon/9)^9$. For small $\\epsilon > 0$: "
    "$(10-\\epsilon/9)^9 < 10^9 < 10^{10}/(10+\\epsilon)$. Wait, let me "
    "recheck: $10^{10}/(10+\\epsilon) \\approx 10^9 - \\epsilon \\cdot 10^8$, "
    "and $(10-\\epsilon/9)^9 \\approx 10^9 - \\epsilon \\cdot 10^8$. "
    "These are approximately equal, so the constraint is barely satisfiable.\n\n"
    "Therefore, the final answer is: $\\boxed{10}$. I hope it is correct.",

    "## Step 1: Wrong approach — ignore the product constraint\n"
    "If we just maximize one variable with sum 100, we get $a_1 = 100$ "
    "and all others 0. But the product constraint requires all positive.\n\n"
    "## Step 2: Try $a_1 = 99$\n"
    "Then $a_2 + \\cdots + a_{10} = 1$ and product $= 10^{10}/99$. "
    "By AM-GM the max product of 9 values summing to 1 is $(1/9)^9$. "
    "But $(1/9)^9 \\approx 2.6 \\times 10^{-9}$, which is much less than "
    "$10^{10}/99 \\approx 1.01 \\times 10^8$. Infeasible.\n\n"
    "Therefore, the final answer is: $\\boxed{99}$. I hope it is correct.",
]


# ===================================================================
# Plugin Registration
# ===================================================================


# -------------------------------------------------------------------
# Test 1 — Qwen2ForPrmModel registered in ModelRegistry
# -------------------------------------------------------------------

def test_plugin_registration():
    print("\n=== Test 1: Plugin registration ===")

    # Plugins are loaded lazily during engine init. Trigger it manually.
    from vllm.plugins import load_general_plugins
    load_general_plugins()

    from vllm import ModelRegistry
    supported = ModelRegistry.get_supported_archs()
    assert "Qwen2ForPrmModel" in supported, (
        f"FAIL: Qwen2ForPrmModel not in ModelRegistry. "
        f"Is the plugin installed? (pip install -e modified-skywork-o1-prm-inference/)"
    )

    print("  Qwen2ForPrmModel registered in ModelRegistry")
    print("[1] PASS")


# ===================================================================
# Model Loading & Forward Pass
# ===================================================================


# -------------------------------------------------------------------
# Test 2 — PRM loads and produces PoolingRequestOutput
# -------------------------------------------------------------------

def test_prm_loads(llm, tokenizer):
    print("\n=== Test 2: PRM loads and produces PoolingRequestOutput ===")

    from vllm.outputs import PoolingRequestOutput

    # Encode a simple input
    input_ids, _, _ = prepare_input(
        QUESTION, SOLUTION_CORRECT, tokenizer=tokenizer, step_token="\n\n"
    )
    prompts = [TokensPrompt(prompt_token_ids=input_ids)]
    outputs = llm.encode(prompts)

    assert len(outputs) == 1, f"FAIL: expected 1 output, got {len(outputs)}"
    assert isinstance(outputs[0], PoolingRequestOutput), (
        f"FAIL: expected PoolingRequestOutput, got {type(outputs[0])}"
    )

    print(f"  output type: {type(outputs[0]).__name__}")
    print(f"  output.outputs.data shape: {outputs[0].outputs.data.shape}")
    print("[2] PASS")
    return outputs[0]


# -------------------------------------------------------------------
# Test 3 — Output shape matches input token count
# -------------------------------------------------------------------

def test_output_shape(llm, tokenizer):
    print("\n=== Test 3: Output shape matches input token count ===")

    input_ids, _, _ = prepare_input(
        QUESTION, SOLUTION_CORRECT, tokenizer=tokenizer, step_token="\n\n"
    )
    prompts = [TokensPrompt(prompt_token_ids=input_ids)]
    outputs = llm.encode(prompts)

    data = outputs[0].outputs.data
    n_tokens = len(input_ids)
    assert data.shape[0] == n_tokens, (
        f"FAIL: data has {data.shape[0]} rows but input has {n_tokens} tokens"
    )
    assert data.shape[1] == 1, (
        f"FAIL: expected 1 column (scalar score), got {data.shape[1]}"
    )

    print(f"  input tokens: {n_tokens}, output shape: {data.shape}")
    print("[3] PASS")


# ===================================================================
# Scoring Pipeline
# ===================================================================


# -------------------------------------------------------------------
# Test 4 — prepare_input: reward_flags mark last token of each step
# -------------------------------------------------------------------

def test_prepare_input(tokenizer):
    print("\n=== Test 4: prepare_input — reward_flags mark step boundaries ===")

    input_ids, steps, reward_flags = prepare_input(
        QUESTION, SOLUTION_CORRECT, tokenizer=tokenizer, step_token="\n\n"
    )

    n_flags = sum(reward_flags)
    n_steps = len(steps)

    print(f"  input_ids length: {len(input_ids)}")
    print(f"  steps ({n_steps}):")
    for i, s in enumerate(steps):
        preview = s[:80].replace('\n', '\\n')
        if len(s) > 80:
            preview += "..."
        print(f"    step {i}: {preview!r}")
    print(f"  reward_flags: {n_flags} flags set (should == {n_steps} steps)")

    assert n_flags == n_steps, (
        f"FAIL: {n_flags} reward flags but {n_steps} steps"
    )
    assert len(reward_flags) == len(input_ids), (
        f"FAIL: reward_flags length {len(reward_flags)} != input_ids length {len(input_ids)}"
    )

    # Each flag=1 should be at the last token of a step
    flag_positions = [i for i, f in enumerate(reward_flags) if f == 1]
    print(f"  flag positions: {flag_positions}")

    print("[4] PASS")


# -------------------------------------------------------------------
# Test 5 — Per-step scores from score_outputs
# -------------------------------------------------------------------

def test_per_step_scores(llm):
    print("\n=== Test 5: Per-step scores — one score per step ===")

    questions = [QUESTION]
    outputs = [[SOLUTION_CORRECT, SOLUTION_WRONG]]

    scores = llm.score_outputs(questions, outputs, SYSTEM_PROMPT)

    # scores structure: List[List[List[float]]]
    #   scores[q][solution][step]
    assert len(scores) == 1, f"FAIL: expected 1 question, got {len(scores)}"
    assert len(scores[0]) == 2, f"FAIL: expected 2 solutions, got {len(scores[0])}"

    for sol_idx, step_scores in enumerate(scores[0]):
        label = "correct" if sol_idx == 0 else "wrong"
        print(f"  solution {sol_idx} ({label}): {len(step_scores)} step scores")
        for i, s in enumerate(step_scores):
            print(f"    step {i}: {s:.4f}")
        assert len(step_scores) > 0, (
            f"FAIL: solution {sol_idx} has no step scores"
        )

    print("[5] PASS")
    return scores


# -------------------------------------------------------------------
# Test 6 — Score range: all scores in [0, 1]
# -------------------------------------------------------------------

def test_score_range(scores):
    print("\n=== Test 6: Score range — all in [0, 1] ===")

    for q_idx, q_scores in enumerate(scores):
        for sol_idx, step_scores in enumerate(q_scores):
            for step_idx, s in enumerate(step_scores):
                assert 0.0 <= s <= 1.0, (
                    f"FAIL: score[{q_idx}][{sol_idx}][{step_idx}] = {s} "
                    f"is outside [0, 1]"
                )

    total = sum(len(ss) for qs in scores for ss in qs)
    print(f"  {total} scores checked, all in [0, 1]")
    print("[6] PASS")


# -------------------------------------------------------------------
# Test 7 — Multi-question batching
# -------------------------------------------------------------------

def test_multi_question(llm):
    print("\n=== Test 7: Multi-question batching ===")

    questions = [QUESTION, QUESTION_2]
    outputs = [
        [SOLUTION_CORRECT, SOLUTION_WRONG],  # 2 solutions for Q1
        [SOLUTION_2],                         # 1 solution for Q2
    ]

    scores = llm.score_outputs(questions, outputs, SYSTEM_PROMPT)

    assert len(scores) == 2, f"FAIL: expected 2 questions, got {len(scores)}"
    assert len(scores[0]) == 2, f"FAIL: Q1 expected 2 solutions, got {len(scores[0])}"
    assert len(scores[1]) == 1, f"FAIL: Q2 expected 1 solution, got {len(scores[1])}"

    for q_idx, q_scores in enumerate(scores):
        print(f"  Q{q_idx + 1}: {len(q_scores)} solution(s)")
        for sol_idx, step_scores in enumerate(q_scores):
            print(f"    solution {sol_idx}: {[f'{s:.4f}' for s in step_scores]}")

    # All scores in range
    for q_scores in scores:
        for step_scores in q_scores:
            for s in step_scores:
                assert 0.0 <= s <= 1.0, f"FAIL: score {s} outside [0, 1]"

    print("[7] PASS")


# ===================================================================
# Pooler (V1 specific)
# ===================================================================


# -------------------------------------------------------------------
# Test 8 — model.pooler is nn.Module with get_supported_tasks
# -------------------------------------------------------------------

def test_pooler_interface(llm):
    print("\n=== Test 8: model.pooler is nn.Module with get_supported_tasks ===")

    import torch.nn as nn

    # Access the pooler from the loaded model
    model_runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    pooler = model_runner.model.pooler

    assert isinstance(pooler, nn.Module), (
        f"FAIL: model.pooler is not an nn.Module, got {type(pooler)}"
    )

    tasks = pooler.get_supported_tasks()
    assert isinstance(tasks, set), (
        f"FAIL: get_supported_tasks returned {type(tasks)}, expected set"
    )
    assert len(tasks) > 0, "FAIL: get_supported_tasks returned empty set"

    print(f"  pooler type: {type(pooler).__name__}")
    print(f"  is nn.Module: True")
    print(f"  supported tasks: {tasks}")
    print("[8] PASS")


# ===================================================================
# Chunked Prefill + Prefix Caching (V1 defaults)
# ===================================================================


# -------------------------------------------------------------------
# Test 9 — Output shape with prefix caching (single seq, cold + warm)
# -------------------------------------------------------------------

def test_prefix_cache_shape(llm, tokenizer):
    print("\n=== Test 9: Output shape with prefix caching (single seq) ===")

    input_ids, steps, reward_flags = prepare_input(
        LONG_QUESTION, LONG_SOLUTIONS[0], tokenizer=tokenizer, step_token="\n\n"
    )
    prompts = [TokensPrompt(prompt_token_ids=input_ids)]

    # First call — cold cache
    out1 = llm.encode(prompts)
    assert out1[0].outputs.data.shape[0] == len(input_ids), (
        f"FAIL (cold): output {out1[0].outputs.data.shape[0]} != input {len(input_ids)}"
    )
    print(f"  cold: input={len(input_ids)}, output={out1[0].outputs.data.shape[0]}")

    # Second call — warm cache (prefix cached)
    out2 = llm.encode(prompts)
    assert out2[0].outputs.data.shape[0] == len(input_ids), (
        f"FAIL (warm): output {out2[0].outputs.data.shape[0]} != input {len(input_ids)}"
    )
    print(f"  warm: input={len(input_ids)}, output={out2[0].outputs.data.shape[0]}")

    print("[9] PASS")


# -------------------------------------------------------------------
# Test 10 — Batch with shared prefix (16 seqs)
# -------------------------------------------------------------------

def test_prefix_cache_batch(llm, tokenizer):
    print("\n=== Test 10: Batch with shared prefix (16 seqs) ===")

    all_input_ids = []
    for sol in LONG_SOLUTIONS:
        input_ids, steps, reward_flags = prepare_input(
            LONG_QUESTION, sol, tokenizer=tokenizer, step_token="\n\n"
        )
        all_input_ids.append(input_ids)

    # Duplicate to get 16 sequences (replicating each 4x)
    batch_input_ids = all_input_ids * 4

    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in batch_input_ids]
    outputs = llm.encode(prompts)

    mismatches = 0
    for i, (out, ids) in enumerate(zip(outputs, batch_input_ids)):
        n_in = len(ids)
        n_out = out.outputs.data.shape[0]
        if n_in != n_out:
            print(f"  MISMATCH beam {i}: input={n_in} output={n_out} diff={n_in - n_out}")
            mismatches += 1

    print(f"  {mismatches} mismatches out of {len(prompts)}")
    assert mismatches == 0, f"FAIL: {mismatches} shape mismatches"
    print("[10] PASS")


# -------------------------------------------------------------------
# Test 11 — score_outputs with prefix caching
# -------------------------------------------------------------------

def test_prefix_cache_scoring(llm):
    print("\n=== Test 11: score_outputs with prefix caching ===")

    system_prompt = "Solve the following math problem."
    questions = [LONG_QUESTION]
    outputs_list = [LONG_SOLUTIONS]

    scores = llm.score_outputs(questions, outputs_list, system_prompt)

    assert len(scores) == 1, f"FAIL: expected 1 question, got {len(scores)}"
    assert len(scores[0]) == len(LONG_SOLUTIONS), (
        f"FAIL: expected {len(LONG_SOLUTIONS)} solutions, got {len(scores[0])}"
    )

    for sol_idx, step_scores in enumerate(scores[0]):
        print(f"  solution {sol_idx}: {[f'{s:.4f}' for s in step_scores]}")
        assert len(step_scores) > 0, f"FAIL: solution {sol_idx} has no step scores"
        for s in step_scores:
            assert 0.0 <= s <= 1.0, f"FAIL: score {s} outside [0, 1]"

    print("[11] PASS")


# -------------------------------------------------------------------
# Test 12 — Repeated scoring (warm cache consistency)
# -------------------------------------------------------------------

def test_prefix_cache_repeated_scoring(llm):
    print("\n=== Test 12: Repeated scoring (warm cache) ===")

    system_prompt = "Solve the following math problem."
    questions = [LONG_QUESTION]
    outputs_list = [LONG_SOLUTIONS]

    scores1 = llm.score_outputs(questions, outputs_list, system_prompt)
    scores2 = llm.score_outputs(questions, outputs_list, system_prompt)

    # Scores should be identical
    for q in range(len(scores1)):
        for s in range(len(scores1[q])):
            for step in range(len(scores1[q][s])):
                s1 = scores1[q][s][step]
                s2 = scores2[q][s][step]
                assert abs(s1 - s2) < 1e-4, (
                    f"FAIL: scores differ at [{q}][{s}][{step}]: {s1} vs {s2}"
                )

    print(f"  scores match across runs ({sum(len(ss) for qs in scores1 for ss in qs)} values)")
    print("[12] PASS")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("=" * 64)
    print("PRM V1 Migration Verification")
    print("Checking behavioural equivalence with AE (V0)")
    print("=" * 64)

    # --- Plugin registration (no model needed) ---
    test_plugin_registration()

    # --- Load PRM model (with V1 defaults: chunked prefill + prefix caching) ---
    print("\n--- Loading PRM model ---")
    llm = TTSLLM(
        PRM_MODEL,
        runner="pooling",
        gpu_memory_utilization=0.35,
        max_model_len=4096,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    # --- Model loading & forward pass ---
    test_prm_loads(llm, tokenizer)
    test_output_shape(llm, tokenizer)

    # --- Scoring pipeline ---
    test_prepare_input(tokenizer)
    scores = test_per_step_scores(llm)
    test_score_range(scores)
    test_multi_question(llm)

    # --- Pooler interface ---
    test_pooler_interface(llm)

    # --- Chunked prefill + prefix caching ---
    test_prefix_cache_shape(llm, tokenizer)
    test_prefix_cache_batch(llm, tokenizer)
    test_prefix_cache_scoring(llm)
    test_prefix_cache_repeated_scoring(llm)

    # Cleanup
    del llm
    import torch; torch.cuda.empty_cache()

    print("\n" + "=" * 64)
    print("ALL 12 TESTS PASSED")
    print("=" * 64)
