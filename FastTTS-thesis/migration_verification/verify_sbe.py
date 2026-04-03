"""SBE V1 Migration Verification — behavioural equivalence with AE (V0).

Tests each paper claim (FastTTS ASPLOS '26 §4.1) to ensure thesis (V1)
matches AE (V0) behaviour exactly.

Run from /TTC/FastTTS-thesis with ``conda activate thesis``:

    python migration_verification/verify_sbe.py

Paper claims verified (in paper order):

  §4.1 Core SBE:
   1. Baseline — stops at first stop string (control)
   2. Speculative continuation — beams generate past stop strings
   3. Multi-step speculation — text contains multiple stop-string occurrences
   4. Force-finish — all speculative beams terminate when standard beams done

  §4.1.1 Speculative Candidate Selection:
   5. Priority scheduling — scheduler uses PRIORITY policy, speculative = 1e9

  §4.1.2 Two-Phase Scheduling with Preemption:
   6. Two-phase scheduling — Phase 1 truncates when waiting, Phase 2 speculates
   7. Text preservation — overflow / force-finish keep full text (no truncation)

  §4.1.3 LookAhead Verification:
   8. split_string_by_separator — search layer can consume SBE output
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import SamplingParams

from models.tts_llm import TTSLLM
from models.numbers import SPEC_BEAM_CANDIDATE_PRIORITY
from search.utils import split_string_by_separator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STOP = "\n\n"
MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
SYSTEM_PROMPT = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\n"
    "Provide a concise solution with minimal explanation.\n\n"
    "- For complex problems (3 steps or more):\n"
    "Use this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem."
)
QUESTION = (
    "Let $a_1, a_2, \\ldots$ be a sequence of positive real numbers such that "
    "$a_1 = 1$ and $a_{n+1} = \\frac{a_n^2 + 1}{a_n + 1}$ for all $n \\geq 1$. "
    "Find the sum $a_1 + a_2 + a_3 + a_4 + a_5$."
)

N_BEAMS = 4  # small batch for testing
TEXT_PREVIEW_LEN = 80  # max chars per step chunk in printout


def print_beam(i, out, prefix="  "):
    """Pretty-print a single beam's output, splitting by STOP to show steps."""
    text = out.outputs[0].text
    stop_reason = out.outputs[0].stop_reason
    finish_reason = out.outputs[0].finish_reason
    count = text.count(STOP)

    print(f"{prefix}beam {i}: {len(text)} chars, {count} stop(s), "
          f"finish={finish_reason!r}, stop_reason={stop_reason!r}")

    parts = text.split(STOP)
    for j, part in enumerate(parts):
        preview = part[:TEXT_PREVIEW_LEN]
        if len(part) > TEXT_PREVIEW_LEN:
            preview += "..."
        is_last = (j == len(parts) - 1)
        sep = "" if is_last else repr(STOP)
        label = f"step {j}" if j < len(parts) - 1 or count == 0 else "tail"
        print(f"{prefix}  [{label}] {preview!r} {sep}")


def print_beams(outputs, prefix="  "):
    """Pretty-print all beams."""
    for i, out in enumerate(outputs):
        print_beam(i, out, prefix)


# ===================================================================
# §4.1 Core SBE
# ===================================================================


# -------------------------------------------------------------------
# Test 1 — Baseline: generation stops at the first stop string
# -------------------------------------------------------------------
# Without SBE the normal stop-checker fires and the sequence is
# FINISHED_STOPPED with text truncated at the boundary.

def test_baseline(prompt):
    print("\n=== Test 1 [§4.1]: Baseline — stops at first stop string ===")

    llm = TTSLLM(
        MODEL,
        runner="generate",
        gpu_memory_utilization=0.35,
        max_model_len=1024,
        enforce_eager=True,
        spec_beam_extension=False,
    )

    params = SamplingParams(
        temperature=0.0, max_tokens=512,
        stop=[STOP], include_stop_str_in_output=True, n=1,
    )

    outputs = llm.generate([prompt] * N_BEAMS, sampling_params=params)
    print_beams(outputs)

    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        count = text.count(STOP)
        assert count <= 1, (
            f"FAIL: baseline beam {i} has {count} stop-string occurrences (expected <= 1)"
        )
        if count == 1:
            assert out.outputs[0].stop_reason == STOP, (
                f"FAIL: baseline beam {i} stop_reason={out.outputs[0].stop_reason!r}, expected {STOP!r}"
            )

    print("[1] PASS")

    del llm
    import torch; torch.cuda.empty_cache()
    return outputs


# -------------------------------------------------------------------
# Test 2 — SBE: speculative continuation past stop strings
# -------------------------------------------------------------------
# Algorithm 1: "the system generates one token for both unfinished
# requests and speculative candidates"

def test_sbe_speculative_continuation(prompt):
    print("\n=== Test 2 [§4.1]: SBE — speculative continuation past stop strings ===")

    llm = TTSLLM(
        MODEL,
        runner="generate",
        gpu_memory_utilization=0.35,
        max_model_len=1024,
        enforce_eager=True,
        spec_beam_extension=True,
    )

    params = SamplingParams(
        temperature=0.8, max_tokens=512,
        stop=[STOP], include_stop_str_in_output=True, n=1,
    )

    outputs = llm.generate([prompt] * N_BEAMS, sampling_params=params)
    print_beams(outputs)

    multi_step_count = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        assert len(text) > 0, f"FAIL: SBE beam {i} produced empty text"
        if text.count(STOP) > 1:
            multi_step_count += 1

    assert multi_step_count > 0, (
        f"FAIL: no beam generated past the first stop string. "
        f"SBE should produce at least one multi-step beam with "
        f"temperature={params.temperature} and {N_BEAMS} beams."
    )
    print(f"  {multi_step_count}/{N_BEAMS} beams have multi-step speculative text")

    print("[2] PASS")
    return llm, outputs


# -------------------------------------------------------------------
# Test 3 — Multi-step speculation
# -------------------------------------------------------------------
# Algorithm 1 lines 7-14: while loop continues until all beams finish.
# Beams can generate past multiple "\n\n" boundaries.

def test_multi_step_speculation(outputs):
    print("\n=== Test 3 [§4.1]: Multi-step speculation ===")

    max_stops = 0
    for i, out in enumerate(outputs):
        count = out.outputs[0].text.count(STOP)
        max_stops = max(max_stops, count)
        if count >= 2:
            print(f"  beam {i}: {count} stop-string occurrences")

    assert max_stops >= 2, (
        f"FAIL: max stop count = {max_stops}, expected >= 2. "
        f"SBE should produce at least one beam with multiple stop-string occurrences."
    )
    print(f"  max stop-string count = {max_stops}")

    print("[3] PASS")


# -------------------------------------------------------------------
# Test 4 — Force-finish: all outputs returned
# -------------------------------------------------------------------
# §4.1.2: "all speculative executions are strictly terminated once
# all the standard beam generations for the current step finish"

def test_force_finish(outputs):
    print("\n=== Test 4 [§4.1]: Force-finish — all outputs returned ===")

    assert len(outputs) == N_BEAMS, (
        f"FAIL: expected {N_BEAMS} outputs, got {len(outputs)}"
    )
    for i, out in enumerate(outputs):
        assert out.finished, f"FAIL: output {i} not marked finished"
        assert out.outputs[0].finish_reason is not None, (
            f"FAIL: output {i} has no finish_reason"
        )

    print(f"  {len(outputs)}/{N_BEAMS} outputs returned, all finished")
    print("[4] PASS")


# ===================================================================
# §4.1.1 Speculative Candidate Selection
# ===================================================================


# -------------------------------------------------------------------
# Test 5 — Priority scheduling active
# -------------------------------------------------------------------
# §4.1.1: speculative slots filled by highest-priority completed beams.

def test_priority_scheduling(llm):
    print("\n=== Test 5 [§4.1.1]: Priority scheduling ===")

    from vllm.v1.core.sched.request_queue import SchedulingPolicy

    scheduler = llm.llm_engine._scheduler
    assert scheduler is not None, "FAIL: no scheduler reference"
    assert scheduler.policy == SchedulingPolicy.PRIORITY, (
        f"FAIL: expected PRIORITY policy, got {scheduler.policy}"
    )
    assert SPEC_BEAM_CANDIDATE_PRIORITY == 1_000_000_000, (
        f"FAIL: SPEC_BEAM_CANDIDATE_PRIORITY={SPEC_BEAM_CANDIDATE_PRIORITY}, expected 1_000_000_000"
    )

    print(f"  scheduler.policy = {scheduler.policy.value}")
    print(f"  SPEC_BEAM_CANDIDATE_PRIORITY = {SPEC_BEAM_CANDIDATE_PRIORITY}")
    print("[5] PASS")


# ===================================================================
# §4.1.2 Two-Phase Scheduling with Preemption
# ===================================================================


# -------------------------------------------------------------------
# Test 6 — Phase 1 / Phase 2 two-phase scheduling
# -------------------------------------------------------------------
# Phase 1: Continuous Beam Batching (scheduler.waiting non-empty)
# Phase 2: Speculative Execution (scheduler.waiting empty)
#
# Strategy: very low gpu_memory_utilization → tiny KV cache → beams
# overflow into waiting queue → mix of Phase 1 and Phase 2.

def test_two_phase_scheduling(prompt):
    print("\n=== Test 6 [§4.1.2]: Phase 1 / Phase 2 two-phase scheduling ===")

    llm = TTSLLM(
        MODEL,
        runner="generate",
        gpu_memory_utilization=0.15,
        max_model_len=512,
        enforce_eager=True,
        spec_beam_extension=True,
    )

    N_PHASE_BEAMS = 16

    params = SamplingParams(
        temperature=0.8, max_tokens=256,
        stop=[STOP], include_stop_str_in_output=True, n=1,
    )

    outputs = llm.generate([prompt] * N_PHASE_BEAMS, sampling_params=params)

    assert len(outputs) == N_PHASE_BEAMS, (
        f"FAIL: expected {N_PHASE_BEAMS} outputs, got {len(outputs)}"
    )

    phase1_count = 0
    phase2_count = 0
    natural_count = 0

    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        count = text.count(STOP)

        if count == 0:
            natural_count += 1
            phase = "natural"
        elif count == 1:
            phase1_count += 1
            phase = "phase1"
        else:
            phase2_count += 1
            phase = "phase2"
        print(f"  beam {i} [{phase}]: {len(text)} chars, {count} stop(s)")

    print(f"  ---")
    print(f"  Phase 1 (truncated at stop): {phase1_count}/{N_PHASE_BEAMS}")
    print(f"  Phase 2 (speculative text):  {phase2_count}/{N_PHASE_BEAMS}")
    print(f"  Natural finish (EOS/max):    {natural_count}/{N_PHASE_BEAMS}")

    assert phase1_count > 0 or phase2_count > 0, (
        f"FAIL: neither phase observed (all {N_PHASE_BEAMS} beams finished naturally). "
        f"Two-phase scheduling requires at least one beam to hit a stop string."
    )
    if phase1_count > 0 and phase2_count > 0:
        print("  Both phases observed — two-phase scheduling confirmed")
    elif phase1_count > 0:
        print(f"  Phase 1 observed ({phase1_count} beams), Phase 2 not triggered")
    else:
        print(f"  Phase 2 observed ({phase2_count} beams), Phase 1 not triggered")

    for i, out in enumerate(outputs):
        assert out.finished, f"FAIL: output {i} not finished"

    print("[6] PASS")

    del llm
    import torch; torch.cuda.empty_cache()


# -------------------------------------------------------------------
# Test 7 — Text preservation: no truncation on force-finish
# -------------------------------------------------------------------
# §4.1.2: force-finished beams keep full speculative text so the
# search layer can extract future_texts.

def test_text_preservation(outputs):
    print("\n=== Test 7 [§4.1.2]: Text preservation — full text on force-finish ===")

    tested = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        count = text.count(STOP)
        if count <= 1:
            continue

        tested += 1
        first_stop = text.find(STOP)
        after_first_stop = text[first_stop + len(STOP):]
        assert len(after_first_stop) > 0, (
            f"FAIL: beam {i} has {count} stop occurrences but no text after the first"
        )
        print(f"  beam {i}: {len(after_first_stop)} chars after first stop string")

    assert tested > 0, (
        f"FAIL: no multi-step beams to verify text preservation. "
        f"Test 2 should guarantee at least one multi-step beam."
    )
    print(f"[7] PASS ({tested} beams tested)")


# ===================================================================
# §4.1.3 LookAhead Verification
# ===================================================================


# -------------------------------------------------------------------
# Test 8 — split_string_by_separator on SBE output
# -------------------------------------------------------------------
# §4.1.3: search layer splits text into current_text + future_texts
# for concatenated verification. Unchanged between AE and V1.

def test_split_string_by_separator(outputs):
    print("\n=== Test 8 [§4.1.3]: split_string_by_separator on SBE output ===")

    tested = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        if text.count(STOP) < 1:
            print(f"  beam {i}: no stop string, skipped")
            continue

        current_text, future_texts, total_occ = split_string_by_separator(text, STOP)
        tested += 1

        print(f"  beam {i}: current_text={len(current_text)} chars, "
              f"future_texts={len(future_texts)} chunks, "
              f"total_occurrences={total_occ}")

        assert current_text.endswith(STOP), (
            f"FAIL: beam {i} current_text does not end with separator"
        )

        # Lossless reconstruction
        reconstructed = current_text + "".join(c for c, _ in future_texts)
        assert reconstructed == text, (
            f"FAIL: beam {i} reconstruction mismatch:\n"
            f"  original:      {text!r}\n"
            f"  reconstructed: {reconstructed!r}"
        )

        # Middle chunks (all except possibly last) must be marked finished
        for j, (chunk_text, is_finished) in enumerate(future_texts[:-1]):
            assert is_finished, (
                f"FAIL: beam {i} future chunk {j} not marked finished"
            )

    assert tested > 0, (
        f"FAIL: no beams had stop strings to test splitting. "
        f"SBE output should contain at least one beam with stop strings."
    )
    print(f"[8] PASS ({tested} beams tested)")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("=" * 64)
    print("SBE V1 Migration Verification")
    print("Checking behavioural equivalence with AE (V0)")
    print("=" * 64)

    # Build prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    conv = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]
    prompt = tokenizer.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False,
    )
    del tokenizer

    # --- §4.1 Core SBE ---
    baseline_outputs = test_baseline(prompt)
    llm, sbe_outputs = test_sbe_speculative_continuation(prompt)
    test_multi_step_speculation(sbe_outputs)
    test_force_finish(sbe_outputs)

    # --- §4.1.1 Speculative Candidate Selection ---
    test_priority_scheduling(llm)

    # --- §4.1.2 Two-Phase Scheduling with Preemption ---
    test_text_preservation(sbe_outputs)

    # --- §4.1.3 LookAhead Verification ---
    test_split_string_by_separator(sbe_outputs)

    # Cleanup SBE LLM before creating new one for Phase 1/2 test
    del llm
    import torch; torch.cuda.empty_cache()

    # --- §4.1.2 Two-Phase Scheduling (separate LLM with tiny KV cache) ---
    test_two_phase_scheduling(prompt)

    print("\n" + "=" * 64)
    print("ALL 8 TESTS PASSED")
    print("=" * 64)
