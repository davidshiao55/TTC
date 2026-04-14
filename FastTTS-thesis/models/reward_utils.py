"""PRM-input preparation helpers.

Converts a ``(problem, response)`` pair into ``(input_ids, steps,
reward_flags)`` for the process-reward model. Each ``\\n\\n``-separated step
in ``response`` is tokenized exactly once to avoid boundary drift between
budget accounting and output building (see migration doc §6a).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_STEP_TOKEN = "\n\n"


def prepare_input(problem, response, tokenizer, step_token=DEFAULT_STEP_TOKEN, max_model_len=4096):
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + "\n")
    max_response_tokens = max_model_len - len(prompt_ids) - 1

    # Split into steps and tokenize each exactly once.
    # Using a single tokenization for both budget checking and output
    # avoids token-count mismatches from tokenizer boundary effects.
    raw_steps = response.split(step_token)
    step_entries = []  # list of (text, token_ids)
    for i, step in enumerate(raw_steps):
        if step == "" and not step_entries:
            continue  # skip leading empty
        is_last = (i == len(raw_steps) - 1)
        text = step if is_last else step + step_token
        ids = tokenizer.encode(text)
        step_entries.append((text, ids))

    # If total exceeds budget, keep complete steps from the end
    # (later steps are more informative for PRM scoring).
    total_tokens = sum(len(ids) for _, ids in step_entries)
    if total_tokens > max_response_tokens:
        kept = []
        budget = 0
        for text, ids in reversed(step_entries):
            if budget + len(ids) > max_response_tokens:
                # Extreme edge case : Newest step alone exceeds budget — tail-truncate its
                # tokens so the boundary marker (last token) is kept.
                if not kept:
                    remaining = max_response_tokens - budget
                    if remaining > 0:
                        kept.append((text, ids[-remaining:]))
                        budget += remaining
                        logger.warning(
                            f"Tail-truncated runaway newest step: kept last "
                            f"{remaining} of {len(ids)} tokens "
                            f"(budget {max_response_tokens})"
                        )
                break
            kept.append((text, ids))
            budget += len(ids)
        kept.reverse()
        step_entries = kept
        logger.warning(
            f"Response truncated at step boundary: kept {len(kept)}/{len(raw_steps)} "
            f"steps ({budget}/{max_response_tokens} token budget)"
        )

    # Build output arrays from the pre-tokenized steps
    steps = []
    reward_flags = [0] * len(prompt_ids)
    response_ids = []
    for text, ids in step_entries:
        response_ids.extend(ids)
        steps.append(text)
        flag = [0] * len(ids)
        if flag:
            flag[-1] = 1
        reward_flags.extend(flag)
    input_ids = prompt_ids + response_ids
    return input_ids, steps, reward_flags

def sigmoid(x):
    result = 1/(np.exp(-x) + 1)
    if hasattr(result, 'item'):
        return result.item()
    return float(result)
