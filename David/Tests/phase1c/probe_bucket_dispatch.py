"""Probe: diagnose B=1 decode → unexpected bucket size.

Usage (run as separate invocations to avoid V1 engine-subprocess teardown OOMs):
    cd /TTC/David/Tests/phase1c   # NOT /TTC (python-path gotcha)
    python probe_bucket_dispatch.py eager
    python probe_bucket_dispatch.py graph

What this exercises:
    - Loads a small model with `offload_backend=cots`, runs one tiny decode.
    - The instrumented `[BUCKET-PROBE]` log lines in cots.py print:
        * `_capture_buckets` at offloader init
        * live `actual_num_tokens` + prior `_current_bucket` before each forward
        * post-forward `_current_bucket` (the bucket the forward actually used)
    - VLLM_LOGGING_LEVEL=DEBUG also surfaces the
      `cudagraph_dispatcher.dispatch` decision (mode + batch_descriptor)
      that gpu_model_runner logs at gpu_model_runner.py:3905.

Verdict table (B=1 decode):
    enforce_eager=True  + bucket=1     → normal
    enforce_eager=False + bucket=large → H1 (FULL-graph slab pad, expected)
    eager+bucket=large+capture_buckets[0]==large → H2 (config trimmed)
    eager+bucket=large+capture_buckets[0]==1     → H3 (real bug)
"""

from __future__ import annotations

import gc
import os
import sys

import torch


def _run(enforce_eager: bool) -> None:
    from vllm import LLM, SamplingParams

    print(f"\n{'='*70}\n[probe] enforce_eager={enforce_eager}\n{'='*70}",
          flush=True)

    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B-Instruct",
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        max_model_len=512,
        gpu_memory_utilization=0.80,
        offload_backend="cots",
        cots_f_cpu_store=0.09,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=4)
    print("[probe] >>> generate() begin", flush=True)
    out = llm.generate("The capital of France is", sp, use_tqdm=False)
    print("[probe] <<< generate() end", flush=True)
    print(f"[probe] generated: {out[0].outputs[0].text!r}", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    if os.path.abspath(os.getcwd()) == "/TTC":
        raise RuntimeError(
            "Don't run from /TTC — vllm import resolves to the namespace "
            "package not the editable install. cd to /TTC/David/Tests/phase1c "
            "or similar first."
        )

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

    mode = sys.argv[1] if len(sys.argv) > 1 else "eager"
    if mode == "eager":
        _run(enforce_eager=True)
    elif mode == "graph":
        _run(enforce_eager=False)
    else:
        raise SystemExit(f"unknown mode {mode!r}; use 'eager' or 'graph'")


if __name__ == "__main__":
    main()
