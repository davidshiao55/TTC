"""
End-to-end test: does the AE's beam search crash when the 7B generator
produces responses that approach max_model_len=4096?

Run from both envs:
    conda activate baseline && cd /TTC/FastTTS-AE && python /TTC/FastTTS-thesis/migration_verification/verify_gen_overflow_e2e.py
    conda activate thesis   && cd /TTC/FastTTS-thesis && python migration_verification/verify_gen_overflow_e2e.py
"""

import sys
import os
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the cwd (FastTTS-AE or FastTTS-thesis), not from this script's dir
sys.path.insert(0, os.getcwd())

import vllm
from fasttts import FastTTS
from config import FastTTSConfig, SearchConfig

# Normal AIME problem, but with reduced max_model_len to force the overflow.
PROBLEM = (
    "Let $p$ be the least prime number for which there exists a positive integer $n$ "
    "such that $n^{4}+1$ is divisible by $p^{2}$. Find the least positive integer $m$ "
    "such that $m^{4}+1$ is divisible by $p^{2}$."
)
MAX_MODEL_LEN = 600  # room for 2-3 steps, should overflow at iteration 3-4


def run_test():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct", trust_remote_code=True)
    prompt_tokens = len(tokenizer.encode(PROBLEM))

    print("=" * 60)
    print(f"Generator overflow E2E test — vLLM {vllm.__version__}")
    print(f"Generator: Qwen/Qwen2.5-Math-7B-Instruct")
    print(f"Verifier:  Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B")
    print(f"Config: n=8, beam_width=4, iterations=15")
    print(f"Prompt tokens: {prompt_tokens} (max_model_len={MAX_MODEL_LEN})")
    print(f"Remaining budget for generation: {MAX_MODEL_LEN - prompt_tokens} tokens")
    print("=" * 60)

    config = FastTTSConfig(
        generator_vllm_config={
            "model": "Qwen/Qwen2.5-Math-7B-Instruct",
            "gpu_memory_utilization": 0.73,
            "max_model_len": MAX_MODEL_LEN,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "seed": 42,
        },
        verifier_vllm_config={
            "model": "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
            "gpu_memory_utilization": 0.16,
            "max_model_len": MAX_MODEL_LEN,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "seed": 42,
        },
        search_config=SearchConfig(
            approach="beam_search",
            beam_width=4,
            n=8,
            num_iterations=15,
            temperature=0.8,  # stochastic — needed to trigger the bug
        ),
        spec_beam_extension=True,
    )

    print("\nInitializing FastTTS...")
    tts = FastTTS(config)

    # Try 5 different AIME problems to find one that triggers the bug
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    test_problems = [ds[i]["problem"] for i in range(10)]

    for idx, problem in enumerate(test_problems):
        print(f"\n--- Problem {idx} ({problem[:60]}...) ---")
        try:
            results = tts.search([problem])
            print(f"  OK")
        except Exception as e:
            ename = type(e).__name__
            msg = str(e)[:200]
            print(f"  CRASHED: {ename}: {msg}")
            break
    else:
        print(f"\nAll 5 problems completed without crash.")


if __name__ == "__main__":
    run_test()
