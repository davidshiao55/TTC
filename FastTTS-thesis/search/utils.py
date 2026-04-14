#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import numpy as np


def split_string_by_separator(s: str, z: str) -> tuple[str, list[str], int]:
    """
    Splits a string `s` by a separator `z`.

    Args:
        s: The input string to split.
        z: The separator string.

    Returns:
        A tuple containing:
        - The first part of the string up to and including the first separator (if present).
        - The rest of the string as a list of strings, each including the separator (except possibly the last one).
        - The total count of the separator `z` in the string `s`.
    """
    parts = s.split(z)
    total_occurrences = len(parts) - 1
    if total_occurrences == 0:
        return s, [], total_occurrences

    first_chunk = parts[0] + z
    chunks = []
    for i, part in enumerate(parts[1:-1], 1):
        # Only add non-empty chunks
        if part or (i < len(parts) - 1):
            chunks.append((part + z, True))
    if not s.endswith(z):
        chunks.append((parts[-1], False))
    return first_chunk, chunks, total_occurrences


def build_conversation(
    prompt: str, 
    response: Optional[str], 
    system_prompt: str
) -> List[dict[str, str]]:
    """Build a conversation format for the model."""
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if response and response.strip():
        conversation.append({"role": "assistant", "content": response})

    return conversation


def aggregate_scores(scores: List[float], strategy: str = "last") -> float:
    """Aggregate step scores using the specified strategy."""
    if not scores:
        return 0.0
        
    if strategy == "last":
        return scores[-1]
    elif strategy == "min":
        return min(scores)
    elif strategy == "prod":
        return np.prod(scores)
    elif strategy == "mean":
        return np.mean(scores)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


_rng = np.random.default_rng(42)
def truncate_sentence_by_tokens(
    sentence: str, 
    tokenizer,
    mean_ratio: float = 0.85, 
    std_ratio: float = 0.1,
    min_words: int = 1,
) -> str:
    """
    Randomly truncate words in a sentence according to a normal distribution.
    
    Args:
        sentence: The input sentence to truncate.
        tokenizer: Tokenizer to use for tokenization and detokenization.
        mean_ratio: Mean ratio of words to keep (0.0 to 1.0). Default is 0.85.
        std_ratio: Standard deviation of the ratio (0.0 to 1.0). Default is 0.1.
        min_words: Minimum number of words to keep. Default is 1.
    
    Returns:
        The truncated first half of the sentence as a string.
        
    Raises:
        ValueError: If mean_ratio or std_ratio are not between 0 and 1.
    """
    if not (0 <= mean_ratio <= 1):
        raise ValueError("mean_ratio must be between 0 and 1")
    if not (0 <= std_ratio <= 1):
        raise ValueError("std_ratio must be between 0 and 1")
    
    # Tokenize the sentence
    token_ids = tokenizer.encode(sentence)
    total_tokens = len(token_ids)
    
    if total_tokens == 0:
        return ""
    
    # Calculate target number of tokens to keep using normal distribution
    ratio = _rng.normal(mean_ratio, std_ratio)
    ratio = np.clip(ratio, 0.0, 1.0)  # Ensure ratio is between 0 and 1
    
    target_tokens = int(ratio * total_tokens)
    
    # Apply min/max constraints
    target_tokens = max(min_words, target_tokens)
    
    # Ensure we don't exceed total tokens and trim at least 1 token
    target_tokens = min(target_tokens, total_tokens - 1)
    
    # Get the truncated token IDs
    truncated_token_ids = token_ids[:target_tokens]
    
    # Convert back to string
    return tokenizer.decode(truncated_token_ids)