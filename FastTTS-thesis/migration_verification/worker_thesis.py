"""Thesis worker — runs in the thesis conda env, dumps results as JSON.

Called by compare.py via subprocess. Do NOT run directly.
"""

import json
import os
import sys

os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

sys.path.insert(0, "/TTC/FastTTS-thesis")

from vllm.inputs import TokensPrompt
from models.tts_llm import TTSLLM
from models.reward_utils import prepare_input


def run(test_cases):
    """Run PRM scoring and baseline generation, return results dict."""
    results = {}

    # ---- PRM scoring ----
    prm = TTSLLM(
        test_cases["prm_model"],
        runner="pooling",
        gpu_memory_utilization=0.35,
        max_model_len=4096,
        enforce_eager=True,
        seed=42,
    )
    tokenizer = prm.get_tokenizer()

    scores = prm.score_outputs(
        test_cases["questions"],
        test_cases["outputs"],
        test_cases["system_prompt"],
    )
    results["prm_scores"] = scores

    # Also dump raw per-token rewards for one input to compare numerics
    input_ids, _, reward_flags = prepare_input(
        test_cases["questions"][0],
        test_cases["outputs"][0][0],
        tokenizer=tokenizer,
        step_token="\n\n",
    )
    prompts = [TokensPrompt(prompt_token_ids=input_ids)]
    raw_output = prm.encode(prompts)
    results["raw_rewards"] = raw_output[0].outputs.data.tolist()
    results["reward_flags"] = reward_flags
    results["input_ids"] = input_ids

    del prm
    import torch; torch.cuda.empty_cache()

    # ---- Baseline generation (no SBE, temp=0) ----
    from vllm import SamplingParams

    gen = TTSLLM(
        test_cases["gen_model"],
        runner="generate",
        gpu_memory_utilization=0.35,
        max_model_len=1024,
        enforce_eager=True,
        seed=42,
    )

    params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    gen_outputs = gen.generate(
        [test_cases["gen_prompt"]] * test_cases["n_beams"],
        sampling_params=params,
    )

    results["gen_texts"] = [o.outputs[0].text for o in gen_outputs]
    results["gen_finish_reasons"] = [o.outputs[0].finish_reason for o in gen_outputs]
    results["gen_stop_reasons"] = [o.outputs[0].stop_reason for o in gen_outputs]

    del gen
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file) as f:
        test_cases = json.load(f)
    results = run(test_cases)
    with open(output_file, "w") as f:
        json.dump(results, f)
