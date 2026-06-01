#!/usr/bin/env python3
"""Small Phase 2 task-level quality probe.

This is not a replacement for full FastTTS accuracy. It is a targeted check
for the current split672 candidate: pad simple math prompts near the split,
decode past the split, then compare extracted answer correctness between
weight-only COTS and hybrid KV + weight COTS.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

TTC_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = TTC_ROOT / "FastTTS-thesis" / "accuracy_evaluation" / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from grader import math_equal_process  # noqa: E402
from parser import extract_answer  # noqa: E402


TASKS: tuple[tuple[str, str], ...] = (
    ("Compute 17 times 23.", "391"),
    ("If 3x + 5 = 20, what is x?", "5"),
    ("What is the area of a rectangle with side lengths 12 and 7?", "84"),
    ("Simplify 2/3 + 5/6.", "3/2"),
)


@dataclass
class TaskResult:
    mode: str
    task_idx: int
    reference: str
    prompt_tokens: int
    split_step: int
    generated_tokens: int
    post_split_tokens: int
    extracted_answer: str
    correct: bool
    text: str


def _encode_chat(
    tokenizer: Any,
    problem: str,
    pad_words: int,
    reasoning_steps: int,
) -> list[int]:
    padding = " ".join(["padding"] * pad_words)
    user = (
        "The repeated word before the problem is padding; ignore it.\n"
        f"{padding}\n\n"
        f"Problem: {problem}\n"
        f"Write at least {reasoning_steps} numbered reasoning steps before "
        "the final answer. "
        "Put the final answer in \\boxed{}."
    )
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step and put the final answer in \\boxed{}.",
        },
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )


def _prompt_ids_for_target(
    tokenizer: Any,
    problem: str,
    target_prompt_tokens: int,
    reasoning_steps: int,
) -> list[int]:
    base = _encode_chat(tokenizer, problem, 0, reasoning_steps)
    if len(base) > target_prompt_tokens:
        raise ValueError(
            f"Base prompt already has {len(base)} tokens, "
            f"target={target_prompt_tokens}"
        )

    lo, hi = 0, 1
    while (
        len(_encode_chat(tokenizer, problem, hi, reasoning_steps))
        <= target_prompt_tokens
    ):
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if (
            len(_encode_chat(tokenizer, problem, mid, reasoning_steps))
            <= target_prompt_tokens
        ):
            lo = mid
        else:
            hi = mid
    return _encode_chat(tokenizer, problem, lo, reasoning_steps)


def _engine_kwargs(args: argparse.Namespace, *, hybrid: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": "bfloat16",
        "max_model_len": args.total_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "trust_remote_code": True,
        "disable_log_stats": True,
        "disable_hybrid_kv_cache_manager": True,
        "async_scheduling": False,
        "max_num_seqs": args.batch,
        "offload_backend": "cots",
        "cots_f_cpu_store": args.cots_f_cpu_store,
        "cots_f_prefetch": args.cots_f_prefetch,
    }
    if args.cots_weight_modules is not None:
        kwargs["cots_weight_modules"] = args.cots_weight_modules
    if hybrid:
        kwargs.update(
            cots_kv_split_blocks=args.split_tokens // 16,
            cots_kv_cpu_pool_bytes=int(args.cpu_pool_gb * (1 << 30)),
        )
    return kwargs


def _run_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    hybrid: bool,
    prompts: list[TokensPrompt],
    prompt_lengths: list[int],
) -> list[TaskResult]:
    llm = LLM(**_engine_kwargs(args, hybrid=hybrid))
    sampling = SamplingParams(
        max_tokens=args.decode_tokens,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=args.ignore_eos,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    results: list[TaskResult] = []
    for idx, (output, prompt_len) in enumerate(zip(outputs, prompt_lengths)):
        text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        generated = len(token_ids)
        split_step = max(args.split_tokens - prompt_len, 0)
        extracted = extract_answer(text, "math")
        correct = bool(math_equal_process((idx, extracted, TASKS[idx][1])))
        results.append(
            TaskResult(
                mode=mode,
                task_idx=idx,
                reference=TASKS[idx][1],
                prompt_tokens=prompt_len,
                split_step=split_step,
                generated_tokens=generated,
                post_split_tokens=max(generated - split_step, 0),
                extracted_answer=extracted,
                correct=correct,
                text=text,
            )
        )
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--target-prompt-tokens", type=int, default=608)
    parser.add_argument("--split-tokens", type=int, default=672)
    parser.add_argument("--total-tokens", type=int, default=768)
    parser.add_argument("--decode-tokens", type=int, default=160)
    parser.add_argument("--reasoning-steps", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.68)
    parser.add_argument("--cpu-pool-gb", type=float, default=4.0)
    parser.add_argument("--cots-f-cpu-store", type=float, default=0.02)
    parser.add_argument("--cots-f-prefetch", type=float, default=0.02)
    parser.add_argument(
        "--cots-weight-modules",
        nargs="+",
        default=None,
        help="COTS weight modules to offload, e.g. qkv mlp or qkv mlp wo.",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--second-weight-control", action="store_true")
    parser.add_argument("--output", default="/tmp/phase2_task_quality.json")
    args = parser.parse_args()
    if args.split_tokens % 16 != 0:
        raise SystemExit("split_tokens must be block-aligned")
    if args.target_prompt_tokens + args.decode_tokens > args.total_tokens:
        raise SystemExit("target_prompt_tokens + decode_tokens must fit total_tokens")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = [
        _prompt_ids_for_target(
            tokenizer,
            problem,
            args.target_prompt_tokens,
            args.reasoning_steps,
        )
        for problem, _ in TASKS
    ]
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids]
    prompt_lengths = [len(ids) for ids in prompt_ids]
    args.batch = len(prompts)

    all_results: list[TaskResult] = []
    all_results.extend(
        _run_mode(
            args,
            mode="weight_only",
            hybrid=False,
            prompts=prompts,
            prompt_lengths=prompt_lengths,
        )
    )
    if args.second_weight_control:
        all_results.extend(
            _run_mode(
                args,
                mode="weight_only_b",
                hybrid=False,
                prompts=prompts,
                prompt_lengths=prompt_lengths,
            )
        )
        mode_names = ("weight_only", "weight_only_b")
    else:
        all_results.extend(
            _run_mode(
                args,
                mode="hybrid",
                hybrid=True,
                prompts=prompts,
                prompt_lengths=prompt_lengths,
            )
        )
        mode_names = ("weight_only", "hybrid")

    by_mode = {
        mode: [r for r in all_results if r.mode == mode]
        for mode in mode_names
    }
    summary = {
        "settings": vars(args),
        "accuracy": {
            mode: sum(r.correct for r in rows) / len(rows)
            for mode, rows in by_mode.items()
        },
        "post_split_coverage": {
            mode: {
                "min_post_split_tokens": min(r.post_split_tokens for r in rows),
                "max_post_split_tokens": max(r.post_split_tokens for r in rows),
                "num_crossed_split": sum(r.post_split_tokens > 0 for r in rows),
            }
            for mode, rows in by_mode.items()
        },
        "results": [asdict(r) for r in all_results],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    for row in all_results:
        print(
            f"{row.mode} task={row.task_idx} correct={int(row.correct)} "
            f"prompt={row.prompt_tokens} gen={row.generated_tokens} "
            f"post_split={row.post_split_tokens} extracted={row.extracted_answer!r}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
