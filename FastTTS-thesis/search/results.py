"""Typed result container for FastTTS search strategies.

All search strategies and ``FastTTS.search`` return a :class:`SearchResults`
instance. JSONL serialization is preserved via :meth:`SearchResults.to_dict`,
so benchmark artifacts keep the same schema as before this dataclass was
introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

# Field names are repeated at runtime for append_batch; keep them in sync
# with the dataclass field list below.
_PER_PROBLEM_FIELDS = (
    "completions",
    "pred",
    "scores",
    "effective_num_tokens",
    "completion_time",
)
_SCALAR_FIELDS = (
    "total_num_tokens",
    "n_completion_tokens",
    "total_generator_latency_s",
    "total_verifier_latency_s",
)


@dataclass
class SearchResults:
    """Canonical return type for every search strategy.

    Shape conventions:
      - Per-problem lists are outer-indexed by problem (one slot per item
        in the input ``problems`` list passed to ``FastTTS.search``).
      - Per-beam lists inside those are indexed by beam.
      - Scalar fields aggregate over the whole call; multi-batch merging
        sums them via :meth:`append_batch`.
    """

    completions: List[List[str]] = field(default_factory=list)
    pred: List[str] = field(default_factory=list)
    scores: List[List[List[float]]] = field(default_factory=list)
    effective_num_tokens: List[List[int]] = field(default_factory=list)
    completion_time: List[List[float]] = field(default_factory=list)

    total_num_tokens: int = 0
    n_completion_tokens: int = 0
    total_generator_latency_s: float = 0.0
    total_verifier_latency_s: float = 0.0

    def append_batch(self, other: "SearchResults") -> None:
        """Merge ``other`` in-place: concat per-problem lists, sum scalars."""
        for name in _PER_PROBLEM_FIELDS:
            getattr(self, name).extend(getattr(other, name))
        for name in _SCALAR_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict matching the pre-dataclass JSONL schema."""
        return asdict(self)
