#!/usr/bin/env python3
"""Greedy-output parity smoke for Phase 1c capture/piecewise arms.

This is intentionally a benchmark-side smoke rather than a pytest: it loads
Qwen2.5-7B and should be run manually from `/TTC/FastTTS-thesis` in the thesis
environment.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
F_CPU_STORE = 0.05
CPU_THREADS = 16
PROMPT = "Question: What is 17 + 25?\nAnswer:"

DEFAULT_PIECEWISE_SPLITTING_OPS = [
    "vllm::unified_attention",
    "vllm::unified_attention_with_output",
    "vllm::unified_mla_attention",
    "vllm::unified_mla_attention_with_output",
    "vllm::mamba_mixer2",
    "vllm::mamba_mixer",
    "vllm::short_conv",
    "vllm::linear_attention",
    "vllm::plamo2_mamba_mixer",
    "vllm::gdn_attention_core",
    "vllm::olmo_hybrid_gdn_full_forward",
    "vllm::kda_attention",
    "vllm::sparse_attn_indexer",
    "vllm::rocm_aiter_sparse_attn_indexer",
    "vllm::unified_kv_cache_update",
    "vllm::unified_mla_kv_cache_update",
]
COTS_SPLITTING_OPS = DEFAULT_PIECEWISE_SPLITTING_OPS + [
    "vllm::cots_submit_gemm",
    "vllm::cots_sync_then_uva",
]


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/phase1c_capture_gap") / f"piecewise_parity_{stamp}"


def arm_kwargs(name: str, max_model_len: int) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": MODEL,
        "dtype": DTYPE,
        "seed": 0,
        "max_model_len": max_model_len,
        "enable_prefix_caching": False,
    }
    cots: dict[str, Any] = {
        "offload_backend": "cots",
        "cots_f_cpu_store": F_CPU_STORE,
        "cots_cpu_runner": "native",
        "cots_cpu_num_threads": CPU_THREADS,
    }
    legacy_cots: dict[str, Any] = {**cots, "cots_auto_graph_split": False}
    if name == "native_eager_real":
        return {**base, **cots, "enforce_eager": True}
    if name == "cots_default_real":
        return {**base, **cots}
    if name == "capture_wait_kernel_real":
        return {**base, **legacy_cots, "cots_capture_sync_mode": "wait_kernel"}
    if name == "piecewise_wait_kernel_real":
        return {
            **base,
            **legacy_cots,
            "cots_capture_sync_mode": "wait_kernel",
            "compilation_config": {"cudagraph_mode": "PIECEWISE"},
        }
    if name == "piecewise_host_callback_real":
        return {
            **base,
            **legacy_cots,
            "compilation_config": {"cudagraph_mode": "PIECEWISE"},
        }
    if name == "piecewise_cots_split_host_callback_real":
        return {
            **base,
            **legacy_cots,
            "compilation_config": {
                "cudagraph_mode": "PIECEWISE",
                "splitting_ops": COTS_SPLITTING_OPS,
            },
        }
    if name == "piecewise_cots_split_wait_kernel_real":
        return {
            **base,
            **legacy_cots,
            "cots_capture_sync_mode": "wait_kernel",
            "compilation_config": {
                "cudagraph_mode": "PIECEWISE",
                "splitting_ops": COTS_SPLITTING_OPS,
            },
        }
    if name == "piecewise_cots_split_wait_uva_real":
        return {
            **base,
            **legacy_cots,
            "cots_capture_sync_mode": "wait_uva_kernel",
            "compilation_config": {
                "cudagraph_mode": "PIECEWISE",
                "splitting_ops": COTS_SPLITTING_OPS,
            },
        }
    if name == "piecewise_cots_split_inductor_host_callback_real":
        return {
            **base,
            **legacy_cots,
            "compilation_config": {
                "cudagraph_mode": "PIECEWISE",
                "use_inductor_graph_partition": True,
                "splitting_ops": COTS_SPLITTING_OPS,
            },
        }
    if name == "piecewise_cots_split_inductor_wait_kernel_real":
        return {
            **base,
            **legacy_cots,
            "cots_capture_sync_mode": "wait_kernel",
            "compilation_config": {
                "cudagraph_mode": "PIECEWISE",
                "use_inductor_graph_partition": True,
                "splitting_ops": COTS_SPLITTING_OPS,
            },
        }
    raise KeyError(f"unknown arm {name!r}")


def run_one_arm(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams

    kwargs = arm_kwargs(args.run_arm, args.max_model_len)
    llm = LLM(**kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    t0 = time.perf_counter()
    output = llm.generate(args.prompt, sampling, use_tqdm=False)[0].outputs[0]
    elapsed = time.perf_counter() - t0
    result = {
        "arm": args.run_arm,
        "model": MODEL,
        "dtype": DTYPE,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "elapsed_s": elapsed,
        "text": output.text,
        "token_ids": list(output.token_ids),
    }
    args.output_json.write_text(json.dumps(result, indent=2))
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def run_parent(args: argparse.Namespace) -> int:
    if os.path.abspath(os.getcwd()) == "/TTC":
        raise RuntimeError(
            "Run from /TTC/FastTTS-thesis to avoid importing /TTC/vllm as a "
            "namespace package before the editable install."
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for arm in args.arms:
        out_json = args.results_dir / f"{arm}.json"
        out_log = args.results_dir / f"{arm}.log"
        if out_json.exists() and not args.force:
            print(f"[skip] {arm}")
        else:
            cmd = [
                sys.executable,
                __file__,
                "--run-arm",
                arm,
                "--output-json",
                str(out_json),
                "--prompt",
                args.prompt,
                "--max-tokens",
                str(args.max_tokens),
                "--max-model-len",
                str(args.max_model_len),
            ]
            print(f"[run] {arm}")
            with out_log.open("w") as fh:
                proc = subprocess.run(
                    cmd, stdout=fh, stderr=subprocess.STDOUT, check=False
                )
            if proc.returncode != 0:
                tail = "\n        ".join(out_log.read_text().splitlines()[-30:])
                raise RuntimeError(f"{arm} failed rc={proc.returncode}\n        {tail}")
        records.append(json.loads(out_json.read_text()))

    ref = records[0]
    ref_tokens = ref["token_ids"]
    summary = {
        "reference_arm": ref["arm"],
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "records": records,
        "parity": {},
    }
    for rec in records[1:]:
        summary["parity"][rec["arm"]] = {
            "token_ids_match_reference": rec["token_ids"] == ref_tokens,
            "text_matches_reference": rec["text"] == ref["text"],
        }

    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 84)
    print(f"reference: {ref['arm']}")
    for rec in records:
        print(f"{rec['arm']:<34} tokens={rec['token_ids']}")
    for arm, parity in summary["parity"].items():
        print(
            f"{arm:<34} token_parity={parity['token_ids_match_reference']} "
            f"text_parity={parity['text_matches_reference']}"
        )
    print(f"[summary] {summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "native_eager_real",
            "cots_default_real",
            "capture_wait_kernel_real",
            "piecewise_cots_split_wait_kernel_real",
            "piecewise_cots_split_wait_uva_real",
        ],
    )
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--run-arm", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.run_arm is not None:
        if args.output_json is None:
            parser.error("--run-arm requires --output-json")
        return run_one_arm(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
