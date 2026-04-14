"""Thesis end-to-end worker — beam search + PRM scoring on a small problem set.

Called by compare_e2e.py via subprocess. Do NOT run directly.
Usage: python worker_e2e_thesis.py <input.json> <output.json>
"""

import json
import os
import sys
import time

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
sys.path.insert(0, "/TTC/FastTTS-thesis")

from fasttts import FastTTS
from config import FastTTSConfig, SearchConfig


def run(test_cases):
    config = FastTTSConfig(
        generator_vllm_config={
            "model": test_cases["gen_model"],
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.20,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "seed": 42,
            "enforce_eager": True,
        },
        verifier_vllm_config={
            "model": test_cases["prm_model"],
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.45,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "seed": 42,
            "enforce_eager": True,
        },
        spec_beam_extension=False,
        offload_enabled=False,
    )

    search_config = SearchConfig(
        approach="beam_search",
        beam_width=test_cases["beam_width"],
        n=test_cases["n"],
        num_iterations=test_cases["num_iterations"],
        temperature=0.8,
        max_tokens=2048,
    )

    fasttts = FastTTS(config)
    fasttts.initialize()

    results = {}
    problems = test_cases["problems"]

    for i, prob in enumerate(problems):
        t0 = time.time()
        search_results = fasttts.search([prob["prompt"]], search_config=search_config)
        wall_time = time.time() - t0

        results[str(i)] = {
            "prompt": prob["prompt"],
            "reference_answer": prob["reference_answer"],
            "completions": search_results.completions,
            "scores": search_results.scores,
            "pred": search_results.pred,
            # After the early-exit fix, n_*_latency_s == total_*_latency_s.
            # Keep the AE-side field names so compare_e2e.py stays unchanged.
            "n_generator_latency_s": search_results.total_generator_latency_s,
            "n_verifier_latency_s": search_results.total_verifier_latency_s,
            "n_completion_tokens": search_results.n_completion_tokens,
            "wall_time_s": wall_time,
        }

    fasttts.shutdown()
    return results


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file) as f:
        test_cases = json.load(f)
    results = run(test_cases)
    with open(output_file, "w") as f:
        json.dump(results, f)
