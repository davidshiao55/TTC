"""Score propagation for PRM prefix caching.

When prefix caching is enabled for the PRM (skip_reading_prefix_cache=False),
cached prefix tokens produce no hidden states.  Step boundaries within the
cached prefix get None placeholders instead of scores.  These three layers
fill those Nones from available sources.

Dependency on SBE:

    Layer 1 (lock_prev_scores): Always needed with prefix caching.
        Earlier step boundaries fall in cached prefix -> Nones.
        prev_scores always has them (scored when they were the "last" step
        in a prior iteration).

    Layers 2-3: Only triggered when SBE is active.
        SBE lookahead places step boundaries in the middle of scored text
        (not at the end).  Duplication trims lookahead scores
        (all_scores[:i]) but the parent's KV cache persists -> Nones that
        prev_scores can't fill.
        Layer 2 fills from the skipped beam bank (parent beam context).
        Layer 3 catches the unfillable case (pruned beam KV reuse).

    Without SBE, Layer 1 fills all Nones.  Layers 2-3 naturally no-op
    because the has_nones check is False after Layer 1.
"""

import logging
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def lock_prev_scores(
    flat_step_rewards: List[List[Optional[float]]],
    flat_prev_scores: List[List[float]],
) -> None:
    """Layer 1: Lock in prev_scores over fresh computation.

    Once a step is scored, its value is locked -- fresh recomputation is
    ignored to ensure consistent scores across iterations and duplicates
    (avoids BF16 noise from different computation contexts).

    Mutates flat_step_rewards in place.
    """
    for idx, step_reward in enumerate(flat_step_rewards):
        prev = flat_prev_scores[idx]
        for j in range(min(len(step_reward), len(prev))):
            if prev[j] is not None:
                step_reward[j] = prev[j]


def propagate_within_batch(
    flat_step_rewards: List[List[Optional[float]]],
    flat_input_ids: List[List[int]],
    flat_reward_flags: List[List[int]],
    skipped_beam_context: Optional[List[Tuple]] = None,
    tokenizer=None,
    prepare_input_fn: Optional[Callable] = None,
) -> None:
    """Layer 2: Fill Nones from batch neighbors and skipped beams.

    If beam A and beam B share the same token prefix up to a step boundary,
    then the PRM score at that boundary is identical (same tokens -> same
    hidden state).  Copy scores from whichever beam computed them.

    Skipped beams (SBE skip: len(all_scores) >= i+1) are not in the current
    scoring batch but their scores are available via skipped_beam_context.

    Only runs when Nones exist after Layer 1 (not on the critical path).
    Without SBE this is never triggered.

    Mutates flat_step_rewards in place.
    """
    bank = {}

    # Add entries from skipped beams (not in current batch but have scores)
    if skipped_beam_context and tokenizer and prepare_input_fn:
        for question, completion, all_scores in skipped_beam_context:
            input_ids, _, reward_flags = prepare_input_fn(
                question, completion, tokenizer=tokenizer, step_token="\n\n"
            )
            flag_positions = [i for i, f in enumerate(reward_flags) if f == 1]
            for j, pos in enumerate(flag_positions):
                if j < len(all_scores) and all_scores[j] is not None:
                    prefix_key = tuple(input_ids[:pos + 1])
                    bank[prefix_key] = all_scores[j]

    # Add entries from current batch
    for idx, step_scores in enumerate(flat_step_rewards):
        flags = flat_reward_flags[idx]
        ids = flat_input_ids[idx]
        flag_positions = [i for i, f in enumerate(flags) if f == 1]
        for j, pos in enumerate(flag_positions):
            if j < len(step_scores) and step_scores[j] is not None:
                prefix_key = tuple(ids[:pos + 1])
                bank[prefix_key] = step_scores[j]

    # Fill Nones from bank
    for idx, step_scores in enumerate(flat_step_rewards):
        if None not in step_scores:
            continue
        flags = flat_reward_flags[idx]
        ids = flat_input_ids[idx]
        flag_positions = [i for i, f in enumerate(flags) if f == 1]
        for j, pos in enumerate(flag_positions):
            if j < len(step_scores) and step_scores[j] is None:
                prefix_key = tuple(ids[:pos + 1])
                if prefix_key in bank:
                    step_scores[j] = bank[prefix_key]


def validate_no_missing(
    flat_step_rewards: List[List[Optional[float]]],
    flat_input_ids: List[List[int]],
    flat_reward_flags: List[List[int]],
    flat_prev_scores: List[List[float]],
    rewards,
    tokenizer,
) -> None:
    """Layer 3: Raise RuntimeError with full diagnostics if Nones remain.

    Remaining Nones indicate a pruned beam's KV cache was reused by a new
    beam with matching tokens -- the pruned beam's scores are permanently
    lost.  This is documented as 'unsolvable' in vllm_v1_migration.md.

    Without SBE this is never triggered (Layer 1 fills all Nones).
    """
    for idx, step_reward in enumerate(flat_step_rewards):
        if any(s is None for s in step_reward):
            n_missing = sum(1 for s in step_reward if s is None)
            reward_flag = flat_reward_flags[idx]
            reward_embedding = rewards[idx].outputs.data.tolist()
            offset = len(reward_flag) - len(reward_embedding)
            flag_positions = [i for i, f in enumerate(reward_flag) if f == 1]
            none_positions = [j for j, s in enumerate(step_reward) if s is None]
            prev = flat_prev_scores[idx] if idx < len(flat_prev_scores) else []
            problem_text = tokenizer.decode(flat_input_ids[idx])
            raise RuntimeError(
                f"PRM score propagation failed: "
                f"{n_missing}/{len(step_reward)} step scores still None "
                f"(prompt idx={idx}).\n"
                f"  offset={offset}, input_ids len={len(flat_input_ids[idx])}\n"
                f"  flag_positions={flag_positions}\n"
                f"  none_positions={none_positions}\n"
                f"  prev_scores len={len(prev)}\n"
                f"\nFULL TEXT:\n{problem_text}"
            )
