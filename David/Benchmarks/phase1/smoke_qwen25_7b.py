"""Phase 1a §6 — Qwen2.5-7B end-to-end smoke test.

Loads Qwen2.5-7B-Instruct with --offload-backend cots --cots-f-cpu-store 0.09
and runs a deterministic greedy decode. Optionally compares to a baseline run
without offload (use two separate invocations — see note below).

This lives under Benchmarks/ rather than Tests/ because it loads ~14 GB of
weights and takes a couple minutes. Run manually:

    cd /TTC/FastTTS-thesis  # avoid the /TTC python-path gotcha
    # cots-only smoke:
    python /TTC/David/Benchmarks/phase1/smoke_qwen25_7b.py --skip-baseline
    # baseline-only smoke (separate invocation):
    python /TTC/David/Benchmarks/phase1/smoke_qwen25_7b.py --baseline-only

Pass criteria:
  1. The cots run completes without crashing.
  2. Offloader log shows N=84 linears registered, ~1.1 GB GPU saved at f=0.09.
  3. Generated text is semantically reasonable (e.g., "Paris" given a France
     prompt). BF16 + cuBLAS-kernel-selection differences may flip the
     lowest-margin token; we don't require bitwise equality vs. baseline.

Note on the baseline comparison: vLLM V1 spawns an engine SUBPROCESS for each
LLM(). When that subprocess tears down (on `del llm`), GPU memory isn't
released until the process actually exits — which can be longer than gc + the
following LLM() construction expects. So back-to-back baseline+cots in one
Python invocation OOMs. Run them as separate invocations and visually compare.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess

import torch


def _gpu_mem_mb() -> float:
    """Free GPU memory snapshot via nvidia-smi (vLLM hides allocator state)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def _run(model: str, offload_backend: str, f_cpu_store: float, prompt: str,
         max_tokens: int, max_model_len: int, gpu_mem_util: float) -> dict:
    from vllm import LLM, SamplingParams

    engine_kwargs = {
        "model": model,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_mem_util,
    }
    if offload_backend == "cots":
        engine_kwargs["offload_backend"] = "cots"
        engine_kwargs["cots_f_cpu_store"] = f_cpu_store

    print(f"\n=== Loading: backend={offload_backend} ===")
    pre_load_mb = _gpu_mem_mb()
    llm = LLM(**engine_kwargs)
    post_load_mb = _gpu_mem_mb()

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    out = llm.generate(prompt, sp, use_tqdm=False)
    text = out[0].outputs[0].text
    token_ids = list(out[0].outputs[0].token_ids)

    # Free the engine before next run.
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "backend": offload_backend,
        "f_cpu_store": f_cpu_store if offload_backend == "cots" else 0.0,
        "pre_load_mb": pre_load_mb,
        "post_load_mb": post_load_mb,
        "weights_mb": post_load_mb - pre_load_mb,
        "text": text,
        "token_ids": token_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--f-cpu-store", type=float, default=0.09)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Deterministic prompt for the smoke test.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the noop baseline run (default behavior; see module docstring).",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the noop baseline (skip the cots run).",
    )
    args = parser.parse_args()

    if os.path.abspath(os.getcwd()) == "/TTC":
        raise RuntimeError(
            "Don't run from /TTC — vllm import resolves to the namespace "
            "package not the editable install. cd elsewhere first."
        )

    results = []
    if args.baseline_only:
        results.append(
            _run(
                args.model,
                "noop",
                0.0,
                args.prompt,
                args.max_tokens,
                args.max_model_len,
                args.gpu_memory_utilization,
            )
        )
    else:
        results.append(
            _run(
                args.model,
                "cots",
                args.f_cpu_store,
                args.prompt,
                args.max_tokens,
                args.max_model_len,
                args.gpu_memory_utilization,
            )
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\n[{r['backend']}] f_cpu_store={r['f_cpu_store']}")
        print(f"  GPU mem after load: {r['post_load_mb']:.0f} MB "
              f"(weights = {r['weights_mb']:.0f} MB)")
        print(f"  Generated tokens:   {r['token_ids']}")
        print(f"  Generated text:     {r['text']!r}")

    print("\nDone. To compare against baseline, run a separate invocation "
          "with --baseline-only and visually compare the generated text.")


if __name__ == "__main__":
    main()
