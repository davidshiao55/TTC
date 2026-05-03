"""Phase 1a §8 — Nsight overlap probe for the cots backend.

Loads Qwen2.5-7B-Instruct with --offload-backend cots --cots-f-cpu-store 0.09
and runs a single greedy decode step under NVTX annotations. The trace is
intended to be opened in Nsight Systems GUI to visually verify per-sub-module
GPU/CPU concurrent execution: a `cutlass_*_gemm` kernel on the GPU stream
overlapping with an `mkldnn_*` (oneDNN) call on a CPU thread, followed by the
Triton UVA copy kernel reading pinned memory and writing to the GPU
destination buffer.

Run from anywhere:

    nsys profile -o David/Benchmarks/phase1/results/cots_overlap.nsys-rep \\
        --trace=cuda,nvtx,osrt --force-overwrite=true \\
        python David/Benchmarks/phase1/probe_cots_overlap.py

Then open in Nsight Systems GUI.

Pass criteria (visual): under each NVTX `decode_step` range, find one of the
WQKV/MLP1/MLP2 sub-modules and confirm two concurrent bars during its CPU
GEMM. Phase 0 §0.5.5 expects fg_s2c (UVA copy completion) ~30 μs.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.cuda.nvtx as nvtx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--f-cpu-store", type=float, default=0.09)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Skip cots offload; profile the GPU-only baseline for comparison.",
    )
    args = parser.parse_args()

    # Avoid the python-path gotcha (CLAUDE.md): never run from /TTC.
    cwd = os.getcwd()
    if os.path.abspath(cwd) == "/TTC":
        raise RuntimeError(
            "Don't run vllm-using scripts from /TTC — vllm import resolves to "
            "the namespace package, not the editable install. cd elsewhere."
        )

    from vllm import LLM, SamplingParams

    engine_kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.85,
    }
    if not args.baseline:
        engine_kwargs.update(
            {
                "offload_backend": "cots",
                "cots_f_cpu_store": args.f_cpu_store,
            }
        )

    print(f"[probe_cots_overlap] engine_kwargs={engine_kwargs}")
    llm = LLM(**engine_kwargs)

    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    prompt = "The capital of France is"
    print(f"[probe_cots_overlap] Warmup forward...")
    _ = llm.generate(prompt, sampling, use_tqdm=False)

    # Profiled run: tag each decode step with NVTX so the trace is navigable.
    print(f"[probe_cots_overlap] Profiled forward (max_tokens={args.max_tokens})...")
    nvtx.range_push("profiled_generate")
    out = llm.generate(prompt, sampling, use_tqdm=False)
    nvtx.range_pop()

    torch.cuda.synchronize()

    print("\n[probe_cots_overlap] Output:")
    for o in out:
        print(f"  prompt: {o.prompt!r}")
        print(f"  generated tokens: {o.outputs[0].token_ids}")
        print(f"  generated text: {o.outputs[0].text!r}")


if __name__ == "__main__":
    main()
