"""AE full-dataset worker — runs beam search on all AIME problems.

Called by compare_full.py via subprocess. Do NOT run directly.
Usage: python worker_full_ae.py <input.json> <output.json>
"""

import json
import os
import sys
import time

os.environ["VLLM_USE_V1"] = "0"
sys.path.insert(0, "/TTC/FastTTS-AE")

from fasttts import FastTTS
from config import FastTTSConfig, SearchConfig


def run(test_cases):
    config = FastTTSConfig(
        generator_vllm_config={
            "model": test_cases["gen_model"],
            "max_model_len": 4096,
            "gpu_memory_utilization": test_cases["gen_gpu_mem"],
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "seed": 42,
            "enforce_eager": False,
        },
        verifier_vllm_config={
            "model": test_cases["prm_model"],
            "max_model_len": 4096,
            "gpu_memory_utilization": test_cases["ver_gpu_mem"],
            "tensor_parallel_size": 1,
            "enable_prefix_caching": True,
            "seed": 42,
            "enforce_eager": False,
        },
        spec_beam_extension=test_cases["spec_beam_extension"],
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
            "id": prob["id"],
            "prompt": prob["prompt"],
            "reference_answer": prob["reference_answer"],
            "completions": search_results["completions"],
            "scores": search_results["scores"],
            "pred": search_results["pred"],
            "n_generator_latency_s": search_results["n_generator_latency_s"],
            "n_verifier_latency_s": search_results["n_verifier_latency_s"],
            "n_completion_tokens": search_results["n_completion_tokens"],
            "total_generator_latency_s": search_results.get("total_generator_latency_s", 0),
            "total_verifier_latency_s": search_results.get("total_verifier_latency_s", 0),
            "effective_num_tokens": search_results.get("effective_num_tokens", []),
            "total_num_tokens": search_results.get("total_num_tokens", 0),
            "completion_time": search_results.get("completion_time", []),
            "wall_time_s": wall_time,
        }
        print(f"  AE problem {i+1}/{len(problems)} done ({wall_time:.1f}s)", flush=True)

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
