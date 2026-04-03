"""
Verify V0 vs V1 behavior when prompt >= max_model_len.

V0 (AE): returns gracefully with stop_reason="length"
V1 (thesis): raises VLLMValidationError

This confirms the generator overflow crash is a V0→V1 behavioral change.

Usage:
    conda activate baseline && cd /TTC/FastTTS-AE && python /TTC/FastTTS-thesis/migration_verification/verify_gen_overflow.py
    conda activate thesis   && cd /TTC/FastTTS-thesis && python migration_verification/verify_gen_overflow.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import vllm
from vllm import LLM, SamplingParams


def build_prompt_at_length(tokenizer, target_tokens):
    """Build a prompt with exactly target_tokens tokens."""
    base = "Solve this problem step by step.\n\n"
    filler = "Apply mathematical reasoning here. "
    prompt = base
    while len(tokenizer.encode(prompt)) < target_tokens - 10:
        prompt += filler
    # Fine-tune character by character
    while len(tokenizer.encode(prompt)) < target_tokens:
        prompt += "x"
    # Trim if overshot
    while len(tokenizer.encode(prompt)) > target_tokens:
        prompt = prompt[:-1]
    return prompt


def run_test(llm, tokenizer, prompt_tokens, max_model_len, label):
    """Run a single generate test and report the result."""
    prompt = build_prompt_at_length(tokenizer, prompt_tokens)
    actual_tokens = len(tokenizer.encode(prompt))
    print(f"\n--- {label}: {actual_tokens} tokens (limit={max_model_len}) ---")
    try:
        outputs = llm.generate([prompt], SamplingParams(max_tokens=2048, temperature=0.0))
        out = outputs[0].outputs[0]
        print(f"  OK: finish_reason={out.finish_reason}, "
              f"stop_reason={out.stop_reason}, "
              f"output_tokens={len(out.token_ids)}")
        return "ok"
    except Exception as e:
        ename = type(e).__name__
        # Truncate long error messages
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "..."
        print(f"  ERROR: {ename}: {msg}")
        return "error"


def main():
    MAX_MODEL_LEN = 4096
    model = "Qwen/Qwen2.5-Math-7B-Instruct"

    print("=" * 60)
    print(f"Generator overflow test — vLLM {vllm.__version__}")
    print(f"Model: {model}, max_model_len={MAX_MODEL_LEN}")
    print("=" * 60)

    print(f"\nLoading model...")
    llm = LLM(
        model=model,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.73,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    results = {}
    results["under"]  = run_test(llm, tokenizer, MAX_MODEL_LEN - 10, MAX_MODEL_LEN,
                                 "Test 1: prompt under limit")
    results["at-1"]   = run_test(llm, tokenizer, MAX_MODEL_LEN - 1,  MAX_MODEL_LEN,
                                 "Test 2: prompt at limit-1")
    results["at"]     = run_test(llm, tokenizer, MAX_MODEL_LEN,      MAX_MODEL_LEN,
                                 "Test 3: prompt at limit")
    results["over+1"] = run_test(llm, tokenizer, MAX_MODEL_LEN + 1,  MAX_MODEL_LEN,
                                 "Test 4: prompt at limit+1")
    results["over+20"]= run_test(llm, tokenizer, MAX_MODEL_LEN + 20, MAX_MODEL_LEN,
                                 "Test 5: prompt at limit+20")

    print(f"\n{'='*60}")
    print(f"Summary (vLLM {vllm.__version__}):")
    for label, result in results.items():
        print(f"  {label:>10}: {result}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
