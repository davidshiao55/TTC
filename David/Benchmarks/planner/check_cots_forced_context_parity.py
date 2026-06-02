#!/usr/bin/env python3
"""Forced-context logprob parity check for COTS weight dispatch.

Free greedy generation is a brittle oracle for COTS dispatch because tiny
numeric drift can change the sampled continuation and then every later token is
conditioned on a different context. This probe instead:

1. records a no-offload reference continuation;
2. forces every arm through that exact continuation;
3. compares raw logprobs and top-k sets before the forcing mask is applied.

Run from /TTC/FastTTS-thesis in the thesis environment.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

TTC_ROOT = Path(__file__).resolve().parents[3]
PHASE2_DIR = TTC_ROOT / "David/Benchmarks/phase2"
sys.path.insert(0, str(PHASE2_DIR))
os.environ["PYTHONPATH"] = (
    str(PHASE2_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")
)


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DTYPE = "bfloat16"
CAPTURE_BUCKETS = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
    216,
    224,
    232,
    240,
    248,
    256,
    272,
    288,
    304,
    320,
    336,
    352,
    368,
    384,
    400,
    416,
    432,
    448,
    464,
    480,
    496,
    512,
    768,
    1024,
    1536,
    2048,
    4096,
    8192,
)


def default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/TTC/results/planner/cots_forced_context_parity") / stamp


def bucket_for(n: int) -> int:
    for bucket in CAPTURE_BUCKETS:
        if n <= bucket:
            return int(bucket)
    return int(CAPTURE_BUCKETS[-1])


def make_prompts(batch: int, prompt_tokens: int):
    from vllm import TokensPrompt

    if prompt_tokens < 1:
        raise ValueError("--prompt-tokens must be positive")
    shared = [100] * (prompt_tokens - 1)
    return [
        TokensPrompt(prompt_token_ids=shared + [200 + idx])
        for idx in range(batch)
    ]


def dispatch_table(
    *,
    batch: int,
    f_cpu_store: float,
    f_cpu: float,
    f_prefetch: float,
    layout: str,
) -> dict[int, tuple[float, float]]:
    if layout == "uniform":
        return {
            int(bucket): (float(f_cpu), float(f_prefetch))
            for bucket in CAPTURE_BUCKETS
        }
    if layout == "decode-only":
        table = {
            int(bucket): (0.0, float(f_cpu_store))
            for bucket in CAPTURE_BUCKETS
        }
        table[bucket_for(batch)] = (float(f_cpu), float(f_prefetch))
        return table
    raise ValueError(f"unknown dispatch layout: {layout}")


def thread_table(batch: int, cpu_threads: int) -> dict[int, int]:
    return {int(bucket): 4 for bucket in CAPTURE_BUCKETS if bucket <= 512} | {
        bucket_for(batch): int(cpu_threads)
    }


def route_pair(args: argparse.Namespace, arm: str) -> tuple[float, float] | None:
    if arm.startswith("none_"):
        return None
    if "prefetch" in arm:
        return (0.0, float(args.f_cpu_store))
    if "cpu" in arm:
        return (float(args.f_cpu_store), 0.0)
    if "hybrid" in arm:
        return (float(args.f_cpu), float(args.f_prefetch))
    raise ValueError(f"unknown COTS route arm: {arm}")


def is_graph_arm(arm: str) -> bool:
    return arm.endswith("_graph")


def is_eager_arm(arm: str) -> bool:
    return arm.endswith("_eager")


def arm_kwargs(
    args: argparse.Namespace,
    *,
    arm: str,
    forced: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": 0,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "trust_remote_code": True,
        "max_num_seqs": args.max_num_seqs or args.batch,
    }
    if forced:
        kwargs["logits_processors"] = [
            "phase2_forced_logits_proc:CaptureForceLogitsProcessor"
        ]
    if is_eager_arm(arm):
        kwargs["enforce_eager"] = True
    pair = route_pair(args, arm)
    if pair is not None:
        f_cpu, f_prefetch = pair
        kwargs.update(
            offload_backend="cots",
            cots_f_cpu_store=args.f_cpu_store,
            cots_f_prefetch=0.0,
            cots_dispatch_table=dispatch_table(
                batch=args.batch,
                f_cpu_store=args.f_cpu_store,
                f_cpu=f_cpu,
                f_prefetch=f_prefetch,
                layout=args.dispatch_layout,
            ),
            cots_cpu_runner="native",
            cots_cpu_num_threads=args.cpu_threads,
        )
        if is_graph_arm(arm):
            kwargs["cots_cpu_num_threads_by_bucket"] = thread_table(
                args.batch, args.cpu_threads
            )
    return kwargs


def run_reference(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams

    prompts = make_prompts(args.batch, args.prompt_tokens)
    llm = LLM(**arm_kwargs(args, arm=args.reference_arm, forced=False))
    sampling = SamplingParams(
        max_tokens=args.decode_tokens,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    forced: dict[str, list[int]] = {}
    records = []
    for prompt, output in zip(prompts, outputs):
        prompt_ids = prompt["prompt_token_ids"]
        tail = str(prompt_ids[-1])
        token_ids = [int(token) for token in output.outputs[0].token_ids]
        forced[tail] = token_ids
        records.append(
            {
                "prompt_tail": int(tail),
                "prompt_token_ids": [int(token) for token in prompt_ids],
                "token_ids": token_ids,
            }
        )
    args.output_json.write_text(
        json.dumps(
            {
                "arm": args.reference_arm,
                "forced_by_prompt_tail": forced,
                "records": records,
            },
            indent=2,
        )
    )
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def run_forced_arm(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams

    forced_payload = json.loads(args.forced_json.read_text())
    forced = forced_payload["forced_by_prompt_tail"]
    prompts = make_prompts(args.batch, args.prompt_tokens)
    os.environ["COTS_FORCE_LOGITS_MODE"] = args.run_arm
    os.environ["COTS_FORCE_LOGITS_OUT"] = str(args.records_jsonl)
    os.environ["COTS_FORCE_LOGITS_TOPK"] = str(args.topk)
    llm = LLM(**arm_kwargs(args, arm=args.run_arm, forced=True))
    sampling = SamplingParams(
        max_tokens=args.decode_tokens,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
        extra_args={"forced_by_prompt_tail": forced},
    )
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    records = []
    for prompt, output in zip(prompts, outputs):
        tail = int(prompt["prompt_token_ids"][-1])
        records.append(
            {
                "prompt_tail": tail,
                "token_ids": [int(token) for token in output.outputs[0].token_ids],
            }
        )
    args.output_json.write_text(
        json.dumps({"arm": args.run_arm, "records": records}, indent=2)
    )
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def load_logprob_records(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        key = (str(rec["mode"]), int(rec["prompt_tail"]), int(rec["step"]))
        records[key] = rec
    return records


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"max": None, "mean": None, "p95": None}
    ordered = sorted(values)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "max": max(values),
        "mean": sum(values) / len(values),
        "p95": p95,
    }


def compare_arm(
    *,
    arm: str,
    reference_arm: str,
    forced: dict[str, list[int]],
    logprob_records: dict[tuple[str, int, int], dict[str, Any]],
    topk: int,
) -> dict[str, Any]:
    positions = 0
    missing: list[dict[str, int]] = []
    forced_deltas: list[float] = []
    top1_deltas: list[float] = []
    jaccards: list[float] = []
    top1_same = 0
    top1_mismatches: list[dict[str, Any]] = []

    for tail_s, tokens in forced.items():
        tail = int(tail_s)
        for step, token in enumerate(tokens):
            ref = logprob_records.get((reference_arm, tail, step))
            got = logprob_records.get((arm, tail, step))
            if ref is None or got is None:
                if len(missing) < 20:
                    missing.append({"tail": tail, "step": step})
                continue
            positions += 1
            forced_deltas.append(
                abs(float(ref["forced_logprob"]) - float(got["forced_logprob"]))
            )
            ref_top = int(ref["top_ids"][0])
            got_top = int(got["top_ids"][0])
            if ref_top == got_top:
                top1_same += 1
                top1_deltas.append(
                    abs(
                        float(ref["top_logprobs"][0])
                        - float(got["top_logprobs"][0])
                    )
                )
            elif len(top1_mismatches) < 20:
                top1_mismatches.append(
                    {
                        "tail": tail,
                        "step": step,
                        "forced_token": int(token),
                        "reference_top1": ref_top,
                        "candidate_top1": got_top,
                        "reference_top1_logprob": float(ref["top_logprobs"][0]),
                        "candidate_top1_logprob": float(got["top_logprobs"][0]),
                    }
                )
            ref_set = set(int(token_id) for token_id in ref["top_ids"])
            got_set = set(int(token_id) for token_id in got["top_ids"])
            jaccards.append(len(ref_set & got_set) / len(ref_set | got_set))

    expected = sum(len(tokens) for tokens in forced.values())
    return {
        "arm": arm,
        "positions_expected": expected,
        "positions_compared": positions,
        "missing_records": missing,
        "top1_same": top1_same,
        "top1_match_rate": (top1_same / positions) if positions else None,
        "forced_token_logprob_delta": stats(forced_deltas),
        "top1_logprob_delta_when_same": stats(top1_deltas),
        f"top{topk}_jaccard": stats(jaccards),
        "top1_mismatches": top1_mismatches,
    }


def forced_output_failures(
    *, forced: dict[str, list[int]], arm_json: Path
) -> list[dict[str, Any]]:
    payload = json.loads(arm_json.read_text())
    failures = []
    for record in payload["records"]:
        tail = str(record["prompt_tail"])
        expected = forced[tail]
        got = [int(token) for token in record["token_ids"]]
        if got != expected:
            failures.append(
                {
                    "tail": int(tail),
                    "expected_prefix": expected[:8],
                    "got_prefix": got[:8],
                }
            )
    return failures


def run_child(
    *,
    args: argparse.Namespace,
    mode: str,
    arm: str,
    output_json: Path,
    forced_json: Path | None,
    records_jsonl: Path,
) -> None:
    cmd = [
        sys.executable,
        __file__,
        "--child-mode",
        mode,
        "--run-arm",
        arm,
        "--output-json",
        str(output_json),
        "--records-jsonl",
        str(records_jsonl),
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--batch",
        str(args.batch),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--decode-tokens",
        str(args.decode_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--f-cpu-store",
        str(args.f_cpu_store),
        "--f-cpu",
        str(args.f_cpu),
        "--f-prefetch",
        str(args.f_prefetch),
        "--cpu-threads",
        str(args.cpu_threads),
        "--dispatch-layout",
        args.dispatch_layout,
        "--topk",
        str(args.topk),
    ]
    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if forced_json is not None:
        cmd.extend(["--forced-json", str(forced_json)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PHASE2_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    if args.enable_counters:
        env.update(
            {
                "VLLM_COTS_COUNTERS": "1",
                "VLLM_COTS_DUMP_COUNTERS_ON_SHUTDOWN": "1",
                "VLLM_COTS_RESET_COUNTERS_AFTER_CUDAGRAPH_CAPTURE": "1",
            }
        )
    if args.disable_compile_cache:
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    env["VLLM_CACHE_ROOT"] = f"/tmp/ttc-cots-forced-parity/{args.run_stamp}/{arm}"
    log_path = output_json.with_suffix(".log")
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"{arm} failed rc={proc.returncode}\n{tail}")


def run_parent(args: argparse.Namespace) -> int:
    if os.path.abspath(os.getcwd()) == "/TTC":
        raise RuntimeError("Run from /TTC/FastTTS-thesis, not /TTC")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records_jsonl = args.results_dir / "logprob_records.jsonl"
    records_jsonl.unlink(missing_ok=True)
    reference_json = args.results_dir / f"{args.reference_arm}.reference.json"

    print(f"[reference] {args.reference_arm}", flush=True)
    run_child(
        args=args,
        mode="reference",
        arm=args.reference_arm,
        output_json=reference_json,
        forced_json=None,
        records_jsonl=records_jsonl,
    )
    forced = json.loads(reference_json.read_text())["forced_by_prompt_tail"]

    forced_arms = [args.reference_arm, *args.candidate_arms]
    arm_jsons: dict[str, Path] = {}
    for arm in forced_arms:
        out_json = args.results_dir / f"{arm}.forced.json"
        arm_jsons[arm] = out_json
        print(f"[forced] {arm}", flush=True)
        run_child(
            args=args,
            mode="forced",
            arm=arm,
            output_json=out_json,
            forced_json=reference_json,
            records_jsonl=records_jsonl,
        )

    logprob_records = load_logprob_records(records_jsonl)
    comparisons = {}
    forced_failures = {}
    for arm in forced_arms:
        failures = forced_output_failures(forced=forced, arm_json=arm_jsons[arm])
        if failures:
            forced_failures[arm] = failures
    for arm in args.candidate_arms:
        comparisons[arm] = compare_arm(
            arm=arm,
            reference_arm=args.reference_arm,
            forced=forced,
            logprob_records=logprob_records,
            topk=args.topk,
        )
    direct_comparisons = {}
    for prefix in ("graph", "eager"):
        prefetch_arm = f"cots_prefetch_{prefix}"
        cpu_arm = f"cots_decode_cpu_{prefix}"
        if prefetch_arm in forced_arms and cpu_arm in forced_arms:
            direct_comparisons[f"{cpu_arm}_vs_{prefetch_arm}"] = compare_arm(
                arm=cpu_arm,
                reference_arm=prefetch_arm,
                forced=forced,
                logprob_records=logprob_records,
                topk=args.topk,
            )

    summary = {
        "reference_arm": args.reference_arm,
        "candidate_arms": args.candidate_arms,
        "batch": args.batch,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "f_cpu_store": args.f_cpu_store,
        "f_cpu": args.f_cpu,
        "f_prefetch": args.f_prefetch,
        "dispatch_layout": args.dispatch_layout,
        "topk": args.topk,
        "forced_output_failures": forced_failures,
        "comparisons": comparisons,
        "direct_comparisons": direct_comparisons,
    }
    summary["all_forced_outputs_match"] = not forced_failures
    summary["all_records_present"] = all(
        comp["positions_compared"] == comp["positions_expected"]
        for comp in comparisons.values()
    )
    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["all_forced_outputs_match"] and summary["all_records_present"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=default_results_dir())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dtype", default=DTYPE)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--f-cpu-store", type=float, default=0.30)
    parser.add_argument("--f-cpu", type=float, default=0.15)
    parser.add_argument("--f-prefetch", type=float, default=0.15)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument(
        "--dispatch-layout",
        choices=("decode-only", "uniform"),
        default="decode-only",
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--reference-arm", default="none_eager")
    parser.add_argument(
        "--candidate-arms",
        nargs="+",
        default=[
            "none_graph",
            "cots_prefetch_graph",
            "cots_decode_cpu_graph",
            "cots_decode_cpu_eager",
        ],
    )
    parser.add_argument("--enable-counters", action="store_true")
    parser.add_argument("--disable-compile-cache", action="store_true")
    parser.add_argument("--child-mode", choices=("reference", "forced"))
    parser.add_argument("--run-arm")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--forced-json", type=Path)
    parser.add_argument("--records-jsonl", type=Path)
    args = parser.parse_args()

    if args.child_mode is not None:
        if args.run_arm is None or args.output_json is None or args.records_jsonl is None:
            parser.error("--child-mode requires --run-arm, --output-json, --records-jsonl")
        if args.child_mode == "forced" and args.forced_json is None:
            parser.error("--child-mode forced requires --forced-json")
    arms = [args.reference_arm, *(args.candidate_arms or [])]
    if any("hybrid" in arm for arm in arms) and (
        args.f_cpu + args.f_prefetch > args.f_cpu_store + 1e-9
    ):
        parser.error("--f-cpu + --f-prefetch must be <= --f-cpu-store")
    return args


def main() -> int:
    args = parse_args()
    if args.child_mode == "reference":
        return run_reference(args)
    if args.child_mode == "forced":
        return run_forced_arm(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
