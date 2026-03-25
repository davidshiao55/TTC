import numpy as np
import logging

logger = logging.getLogger(__name__)

def prepare_input(problem, response, tokenizer, step_token):
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + "\n")
    # Truncate response at the token level
    tokenizer.truncation_side = "left"
    tokenized_response = tokenizer.encode(response, truncation=True, max_length=3072-len(prompt_ids)-1)
    decoded_response = tokenizer.decode(tokenized_response)
    if decoded_response != response:
        logger.warning(f"Reponse truncated when preparing input for reward function")
    steps = []
    reward_flags = [0] * len(prompt_ids)
    response_ids = []
    char_idx = 0
    for step in decoded_response.split(step_token):
        if step == "" and len(steps) == 0:
            continue  # skip leading empty
        # Add the step
        step_text = step
        if char_idx + len(step) < len(decoded_response):
            # Not the last step, so add the step_token back
            step_text += step_token
        step_ids = tokenizer.encode(step_text)
        response_ids.extend(step_ids)
        steps.append(step_text)
        # Mark the last token of this step for reward
        flag = [0] * len(step_ids)
        if flag:
            flag[-1] = 1
        reward_flags.extend(flag)
        char_idx += len(step) + len(step_token)
    input_ids = prompt_ids + response_ids
    return input_ids, steps, reward_flags

def sigmoid(x):
    result = 1/(np.exp(-x) + 1)
    if hasattr(result, 'item'):
        return result.item()
    return float(result)