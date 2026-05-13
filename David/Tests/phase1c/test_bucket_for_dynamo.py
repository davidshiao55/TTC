# SPDX-License-Identifier: Apache-2.0
"""§1c.18 fix verification — `_bucket_for` is Dynamo-traceable.

Two parts:

1. **Parity** — the Dynamo-friendly linear-scan `_bucket_for`
   produces identical results to the original `bisect_left`-based
   implementation for all interesting boundary classes (below first,
   exact match, between buckets, equal to largest, above largest).
   `lookup_dispatch` shares semantics by construction so it gets the
   same coverage.

2. **Dynamo trace** — a tiny module replicating the Phase 1c
   first-decoder-pre-hook shape (forward pre-hook calls a method that
   does the bucket lookup) compiles cleanly under
   `torch.compile(fullgraph=True)`. The original `bisect_left`-based
   path raised `torch._dynamo.exc.Unsupported: ... _bisect.bisect_left
   ...` here; the fix removes that failure mode.

If THIS test passes, the §1c.18 blocker is closed at the unit level.
The real-model anchors are the retained capture-gap and piecewise
parity benches in `David/Benchmarks/phase1c/`.
"""

from __future__ import annotations

from bisect import bisect_left

import pytest
import torch


# --- 1. Parity ---


def _bucket_for_old(buckets: tuple[int, ...], num_tokens: int) -> int:
    """The original `bisect_left`-based implementation. Used here as
    the parity oracle — kept locally so this test does not depend on
    the old method living anywhere in the tree."""
    i = bisect_left(buckets, num_tokens)
    if i >= len(buckets):
        return buckets[-1]
    return buckets[i]


def _bucket_for_new(buckets: tuple[int, ...], num_tokens: int) -> int:
    """The Dynamo-friendly linear scan. Matches the implementation in
    `cots.py:CotsOffloader._bucket_for`. Repeated locally so the test
    is self-contained and survives renames."""
    for bucket in buckets:
        if num_tokens <= bucket:
            return bucket
    return buckets[-1]


_BUCKET_TUPLES = [
    (1,),
    (1, 4, 16),
    (1, 2, 4, 8, 16, 32),
    (8, 32, 128, 512),
]


@pytest.mark.parametrize("buckets", _BUCKET_TUPLES)
def test_parity_below_first_bucket(buckets: tuple[int, ...]) -> None:
    """Below the smallest bucket → smallest bucket."""
    n = max(0, buckets[0] - 1)
    assert _bucket_for_new(buckets, n) == _bucket_for_old(buckets, n)
    assert _bucket_for_new(buckets, n) == buckets[0]


@pytest.mark.parametrize("buckets", _BUCKET_TUPLES)
def test_parity_exact_bucket_match(buckets: tuple[int, ...]) -> None:
    """num_tokens == bucket value → that bucket."""
    for b in buckets:
        assert _bucket_for_new(buckets, b) == _bucket_for_old(buckets, b)
        assert _bucket_for_new(buckets, b) == b


@pytest.mark.parametrize("buckets", _BUCKET_TUPLES)
def test_parity_between_buckets(buckets: tuple[int, ...]) -> None:
    """num_tokens strictly between buckets[i] and buckets[i+1] →
    buckets[i+1] (round up)."""
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        if hi - lo < 2:
            continue
        n = lo + 1
        assert _bucket_for_new(buckets, n) == _bucket_for_old(buckets, n)
        assert _bucket_for_new(buckets, n) == hi


@pytest.mark.parametrize("buckets", _BUCKET_TUPLES)
def test_parity_above_largest(buckets: tuple[int, ...]) -> None:
    """Above largest bucket → clamps to largest bucket
    (`_bucket_for` semantics; `lookup_dispatch` falls back to
    eager_fallback_entry separately, see §1c.13)."""
    n = buckets[-1] + 1
    assert _bucket_for_new(buckets, n) == _bucket_for_old(buckets, n)
    assert _bucket_for_new(buckets, n) == buckets[-1]


def test_parity_offloader_method_matches_oracle() -> None:
    """The actual `CotsOffloader._bucket_for` method (not just a copy
    of the implementation) matches the bisect oracle. Reaches into
    the class to bind `_capture_buckets` directly so we don't have to
    construct a full offloader for a pure-Python lookup test."""
    from vllm.model_executor.offloader.cots import CotsOffloader

    class _Stub:
        _capture_buckets = (1, 4, 16, 64)

    bound = CotsOffloader._bucket_for.__get__(_Stub())
    for n in [0, 1, 2, 3, 4, 5, 7, 16, 17, 64, 100]:
        assert bound(n) == _bucket_for_old((1, 4, 16, 64), n), (
            f"mismatch at num_tokens={n}"
        )


# --- 2. Dynamo trace ---


class _PreHookModule(torch.nn.Module):
    """Replicates the structural shape of the Phase 1c
    `_first_decoder_pre_hook` issue: a forward pre-hook that calls a
    method which does a bucket lookup, registered via
    `register_forward_pre_hook`. The pre-hook stashes the resolved
    bucket on `self.current_bucket` (mirroring `_current_bucket` on
    the offloader)."""

    def __init__(self, capture_buckets: tuple[int, ...]) -> None:
        super().__init__()
        self.capture_buckets = capture_buckets
        self.current_bucket: int | None = None
        self.linear = torch.nn.Linear(4, 4)
        # Match the offloader's pre-hook registration shape: positional
        # `(module, args)`, no kwargs. See `cots.py:_install_bucket_prehook`.
        self.register_forward_pre_hook(_PreHookModule._pre_hook)

    @staticmethod
    def _pre_hook(self, args):  # noqa: ANN001 — torch hook signature
        x = args[0]
        self.current_bucket = self._bucket_for(int(x.shape[0]))
        return None

    def _bucket_for(self, num_tokens: int) -> int:
        for bucket in self.capture_buckets:
            if num_tokens <= bucket:
                return bucket
        return self.capture_buckets[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_dynamo_trace_pre_hook_with_linear_scan() -> None:
    """The fix-shape pre-hook compiles under `fullgraph=True`. With
    the old `bisect_left`-based lookup this raised
    `torch._dynamo.exc.Unsupported: ... _bisect.bisect_left ...`."""
    if not torch.cuda.is_available():
        pytest.skip("torch.compile path needs a real device")
    device = "cuda"
    mod = _PreHookModule((1, 4, 16, 64)).to(device)
    x = torch.randn(4, 4, device=device)
    compiled = torch.compile(mod, fullgraph=True)
    y = compiled(x)
    assert y.shape == (4, 4)
    assert mod.current_bucket == 4, (
        f"pre-hook should have resolved bucket=4 for num_tokens=4, "
        f"got {mod.current_bucket}"
    )


def test_dynamo_compile_regresses_old_bisect_left() -> None:
    """Sanity check the negative case: a bisect_left-based bucket
    lookup STILL raises under fullgraph=True. Locks in the §1c.18
    diagnosis so a future Dynamo update that silently makes
    bisect_left traceable doesn't make the linear-scan fix look
    redundant — and so a regression to bisect_left in production
    code would be caught by some test, even if this one is informational
    rather than a blocking gate."""
    if not torch.cuda.is_available():
        pytest.skip("torch.compile path needs a real device")
    device = "cuda"

    class _OldStyleModule(torch.nn.Module):
        def __init__(self, capture_buckets: tuple[int, ...]) -> None:
            super().__init__()
            self.capture_buckets = capture_buckets
            self.current_bucket: int | None = None
            self.linear = torch.nn.Linear(4, 4)
            self.register_forward_pre_hook(_OldStyleModule._pre_hook)

        @staticmethod
        def _pre_hook(self, args):  # noqa: ANN001
            from bisect import bisect_left

            x = args[0]
            n = int(x.shape[0])
            i = bisect_left(self.capture_buckets, n)
            if i >= len(self.capture_buckets):
                self.current_bucket = self.capture_buckets[-1]
            else:
                self.current_bucket = self.capture_buckets[i]
            return None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    mod = _OldStyleModule((1, 4, 16, 64)).to(device)
    x = torch.randn(4, 4, device=device)
    compiled = torch.compile(mod, fullgraph=True)
    with pytest.raises(Exception, match="bisect_left|skipped|Unsupported"):
        compiled(x)
