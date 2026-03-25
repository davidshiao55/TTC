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


def assign_prefix_priorities(tokenized_seqs: list[list[int]]) -> list[int]:
    """
    Assign priorities to each sequence based on the longest shared prefix grouping.
    Returns a list of priorities (lower = higher priority), one per input sequence.
    """
    n = len(tokenized_seqs)
    priorities = [None] * n
    remaining = set(range(n))
    current_priority = 0

    # Helper: find the largest group with the longest shared prefix
    def find_largest_prefix_group(indices):
        if not indices:
            return set(), []
        # Sort indices by their tokenized sequence
        sorted_indices = sorted(indices, key=lambda i: tokenized_seqs[i])
        max_prefix_len = 0
        best_group = set()
        for i in range(len(sorted_indices)):
            for j in range(i+1, len(sorted_indices)):
                # Find common prefix length
                a, b = tokenized_seqs[sorted_indices[i]], tokenized_seqs[sorted_indices[j]]
                prefix_len = 0
                while prefix_len < min(len(a), len(b)) and a[prefix_len] == b[prefix_len]:
                    prefix_len += 1
                if prefix_len > max_prefix_len and prefix_len > 0:
                    # Find all with this prefix
                    group = set(k for k in sorted_indices if tokenized_seqs[k][:prefix_len] == a[:prefix_len])
                    if len(group) > 1:
                        max_prefix_len = prefix_len
                        best_group = group
        if not best_group:
            # No group found, pick one remaining
            return {sorted_indices[0]}, []
        return best_group, tokenized_seqs[next(iter(best_group))][:max_prefix_len]

    while remaining:
        group, prefix = find_largest_prefix_group(remaining)
        for idx in group:
            priorities[idx] = current_priority
        remaining -= group
        current_priority += 1
    return priorities

np.random.seed(42)
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
    ratio = np.random.normal(mean_ratio, std_ratio)
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