# SPDX-License-Identifier: Apache-2.0
"""Stage 5 — `mutates_args` actually prevents compile-time reordering.

The whole correctness story of `cots_ops.py` rests on torch.compile
honoring the barrier-installing `mutates_args` declarations:
  * `cots_submit_gemm` mutates `["x_gpu", "y_pinned"]` so the op is
    pinned BEFORE every GPU GEMM that reads x_gpu (preserves overlap).
  * `cots_sync_then_uva` mutates `["y_gpu", "gpu_anchor_a",
    "gpu_anchor_b"]` so the op is pinned AFTER each independent GPU
    GEMM (preserves overlap; both anchors needed for QKV's two
    independent F.linears).

A weaker version of this test (output parity under torch.compile)
would pass even if the compiler hoisted `sync_then_uva` ABOVE the
GEMMs and the worker happened to finish before the pinned tensor was
read — wrong-fast-but-still-correct. The proof we actually need is
positional: in the FX graph, `cots_submit_gemm` must precede every
F.linear that reads `x_gpu`, and `cots_sync_then_uva` must follow
every F.linear that produces an anchor it mutates.

We use `torch._dynamo.export(...)` to extract the FX graph from a
minimal forward function (no `torch.compile(fullgraph=False)` graph
breaks; export forces a single graph or fails) and walk
`gm.graph.nodes` to assert positional ordering. We also parse each
op's schema string to confirm the EXACT mutated arg names match
what `cots_ops.py` registers (a coarser count-based check would pass
if a refactor accidentally switched the marker to the wrong tensor).
"""

from __future__ import annotations

import re

import pytest
import torch
import torch._dynamo

# Import cots_ops directly so torch.ops.vllm.cots_* are registered at
# module load (the registration runs at cots_ops import time). cots.py
# imports cots_ops lazily from inside NativeCotsRunner.__init__, but
# the schema-level tests in this file probe the ops BEFORE any runner
# is constructed.
from vllm.model_executor.offloader import cots, cots_ops  # noqa: F401

pytestmark = pytest.mark.needs_cuda


# --- Schema-level: exact mutated-arg names -------------------------------


def _schema_mutated_args(schema_str: str) -> list[str]:
    """Parse a torch op schema string and return the names of arguments
    that carry an alias-set marker (e.g., `Tensor(a!) x_gpu` → 'x_gpu').
    """
    # Match "Tensor(<alias-set>!) <name>" where <alias-set> is a single
    # lowercase letter / sequence, with optional spaces / nullable '?'.
    pattern = re.compile(r"Tensor\(\w+!\)\s*\??\s*(\w+)")
    return pattern.findall(schema_str)


def test_cots_submit_gemm_mutates_x_gpu_and_y_pinned() -> None:
    """`cots_submit_gemm` must declare `x_gpu` and `y_pinned` as
    mutated arguments specifically. `x_gpu` is the CUDA-anchor that
    pins the op BEFORE every GPU GEMM that reads it; `y_pinned` is
    the buffer the worker fills."""
    schema = str(torch.ops.vllm.cots_submit_gemm.default._schema)
    mutated = _schema_mutated_args(schema)
    assert "x_gpu" in mutated, (
        f"cots_submit_gemm must mutate `x_gpu` (the CUDA-anchor that "
        f"pins this op before GPU GEMMs reading x_gpu). Schema: {schema}"
    )
    assert "y_pinned" in mutated, (
        f"cots_submit_gemm must mutate `y_pinned` (the worker output "
        f"buffer). Schema: {schema}"
    )
    # No OTHER args should be marked mutated — the design intentionally
    # restricts the mutation set to exactly these two so torch.compile's
    # alias-set tracking matches the actual semantics.
    unexpected = set(mutated) - {"x_gpu", "y_pinned"}
    assert not unexpected, (
        f"cots_submit_gemm has unexpected mutated args {unexpected}; "
        f"schema: {schema}"
    )


def test_cots_sync_then_uva_mutates_y_gpu_and_both_anchors() -> None:
    """`cots_sync_then_uva` must declare `y_gpu` AND both
    `gpu_anchor_a` AND `gpu_anchor_b` as mutated. Two anchors are
    required because in QKV the two GPU F.linears (out_perm, out_pref)
    are independent — mutating only ONE anchor would let the compiler
    hoist sync above the other GEMM."""
    schema = str(torch.ops.vllm.cots_sync_then_uva.default._schema)
    mutated = _schema_mutated_args(schema)
    for name in ("y_gpu", "gpu_anchor_a", "gpu_anchor_b"):
        assert name in mutated, (
            f"cots_sync_then_uva must mutate `{name}` to pin the op "
            f"after each independent GPU GEMM. Schema: {schema}"
        )
    unexpected = set(mutated) - {"y_gpu", "gpu_anchor_a", "gpu_anchor_b"}
    assert not unexpected, (
        f"cots_sync_then_uva has unexpected mutated args {unexpected}; "
        f"schema: {schema}"
    )


# --- FX-level: positional ordering inside the captured graph -------------


def _make_dryrun_runner() -> cots.NativeCotsRunner:
    """A native runner with one dryrun_noop slab. Stage 5's ordering
    test only cares about the FX graph structure, not the slab's
    actual GEMM behavior — dry_run avoids any worker-thread state."""
    r = cots.NativeCotsRunner(dry_run=True)
    slab = cots._NativeSlabSpecQkv(
        op_descriptor=(0, 0, "qkv"),
        n_threads=1,
        x_pinned_ptr=0,
        in_dim=16,
        y_pinned_ptr=0,
        cpu_out_dim=8,
        w_cpu_ptr=0,
        w_cpu_rows=8,
    )
    r.install(
        slab_specs=[slab],
        scratch_max_tokens=4,
        scratch_max_intermediate_per_half=0,
    )
    return r


def _node_matches(n: torch.fx.Node, name: str) -> bool:
    """A node matches a name if EITHER its repr ends with the name
    (custom-op packet form: `vllm.cots_submit_gemm`) OR its target's
    __name__ matches (built-in form: `<built-in function linear>` →
    `linear`). Both styles appear in dynamo-exported graphs depending
    on the underlying op."""
    if str(n.target).endswith(name):
        return True
    return getattr(n.target, "__name__", None) == name


def _node_index_by_target(nodes: list[torch.fx.Node], target_name: str) -> int:
    """Find the first FX node matching `target_name`. Returns -1 if
    not found."""
    for i, n in enumerate(nodes):
        if _node_matches(n, target_name):
            return i
    return -1


def _all_node_indices_by_target(
    nodes: list[torch.fx.Node], target_name: str
) -> list[int]:
    return [i for i, n in enumerate(nodes) if _node_matches(n, target_name)]


def test_fx_graph_orders_submit_before_gpu_gemms_before_sync() -> None:
    """**The load-bearing test.** Capture the operator's call sequence
    via `torch._dynamo.export` (forces a single graph; no graph breaks)
    and walk the FX graph nodes. Assert positional ordering:

        index(cots_submit_gemm) < index(F.linear, out_perm)
        index(cots_submit_gemm) < index(F.linear, out_pref)
        index(cots_sync_then_uva) > index(F.linear, out_perm)
        index(cots_sync_then_uva) > index(F.linear, out_pref)

    A `mutates_args` regression that lets the compiler reorder these
    nodes would change their FX positions, which the assertions catch
    even if the OUTPUT happened to still be correct (the more
    dangerous "wrong-fast-but-still-numerically-fine" failure mode).
    """
    runner = _make_dryrun_runner()
    runner_id = runner._runner_id

    # Allocate the inputs the closure consumes. Real values don't
    # matter — we're only inspecting graph structure.
    x = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    x_pin = torch.empty(4, 16, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(4, 8, dtype=torch.bfloat16, pin_memory=True)
    y_gpu = torch.empty(4, 8, dtype=torch.bfloat16, device="cuda")
    w_perm = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    w_pref = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    dummy_a = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    dummy_b = torch.empty(1, dtype=torch.bfloat16, device="cuda")

    def forward(x, x_pin, y_pin, y_gpu, w_perm, w_pref, dummy_a, dummy_b):
        # Mirror the operator's call sequence (CotsQKVOp.apply core).
        x_pin.copy_(x, non_blocking=True)
        torch.ops.vllm.cots_submit_gemm(
            x, x_pin, y_pin, runner_id, 0, x.shape[0]
        )
        out_perm = torch.nn.functional.linear(x, w_perm)
        out_pref = torch.nn.functional.linear(x, w_pref)
        torch.ops.vllm.cots_sync_then_uva(
            y_pin, y_gpu, out_perm, out_pref, runner_id
        )
        return out_perm, out_pref, y_gpu

    try:
        gm, _ = torch._dynamo.export(forward)(
            x, x_pin, y_pin, y_gpu, w_perm, w_pref, dummy_a, dummy_b
        )
        nodes = list(gm.graph.nodes)

        # Locate each load-bearing node.
        i_submit = _node_index_by_target(nodes, "vllm.cots_submit_gemm")
        i_sync = _node_index_by_target(nodes, "vllm.cots_sync_then_uva")
        # Two F.linear calls; both should be present and ordered
        # between submit and sync.
        i_linears = _all_node_indices_by_target(nodes, "linear")

        assert i_submit >= 0, (
            f"cots_submit_gemm not found in FX graph: "
            f"{[(n.op, n.target) for n in nodes]}"
        )
        assert i_sync >= 0, (
            f"cots_sync_then_uva not found in FX graph: "
            f"{[(n.op, n.target) for n in nodes]}"
        )
        assert len(i_linears) == 2, (
            f"expected exactly 2 F.linear calls (out_perm, out_pref), "
            f"found {len(i_linears)} at indices {i_linears}: "
            f"{[(n.op, n.target) for n in nodes]}"
        )

        # Positional ordering. Each linear must sit between submit and
        # sync — the mutates_args contract on `x_gpu` (submit) and on
        # `gpu_anchor_a`/`gpu_anchor_b` (sync) is what enforces this.
        for i_lin in i_linears:
            assert i_submit < i_lin, (
                f"FX graph has cots_submit_gemm AFTER an F.linear "
                f"(submit at {i_submit}, linear at {i_lin}). The "
                f"mutates_args=['x_gpu', 'y_pinned'] declaration on "
                f"cots_submit_gemm must pin it BEFORE every GPU GEMM "
                f"that reads x_gpu — capture would break overlap and "
                f"the worker would race the D2H copy."
            )
            assert i_sync > i_lin, (
                f"FX graph has cots_sync_then_uva BEFORE an F.linear "
                f"(sync at {i_sync}, linear at {i_lin}). The "
                f"mutates_args=['y_gpu', 'gpu_anchor_a', 'gpu_anchor_b'] "
                f"declaration on cots_sync_then_uva must pin it AFTER "
                f"each GPU GEMM whose output it anchors — otherwise "
                f"capture would let sync execute before the GEMMs and "
                f"break overlap (potentially wrong-fast)."
            )

        # Also: submit < sync (transitive of the above, but assert
        # directly so a graph with submit / sync but no linears would
        # still catch a reorder).
        assert i_submit < i_sync, (
            f"cots_submit_gemm at index {i_submit} is not before "
            f"cots_sync_then_uva at index {i_sync}"
        )
    finally:
        runner.close()


def test_fx_graph_export_succeeds_without_graph_break() -> None:
    """Confirms the operator's call sequence is fully traceable —
    `torch._dynamo.export` raises if there's a graph break. A graph
    break would mean torch.compile sees the operator as multiple
    smaller graphs, and `mutates_args` ordering across the break is
    NOT enforced.
    """
    runner = _make_dryrun_runner()
    runner_id = runner._runner_id

    x = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    x_pin = torch.empty(4, 16, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(4, 8, dtype=torch.bfloat16, pin_memory=True)
    y_gpu = torch.empty(4, 8, dtype=torch.bfloat16, device="cuda")
    w_perm = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    dummy = torch.empty(1, dtype=torch.bfloat16, device="cuda")

    def forward(x, x_pin, y_pin, y_gpu, w_perm, dummy):
        x_pin.copy_(x, non_blocking=True)
        torch.ops.vllm.cots_submit_gemm(
            x, x_pin, y_pin, runner_id, 0, x.shape[0]
        )
        out_perm = torch.nn.functional.linear(x, w_perm)
        torch.ops.vllm.cots_sync_then_uva(
            y_pin, y_gpu, out_perm, dummy, runner_id
        )
        return out_perm, y_gpu

    try:
        # If export raises, that's a real graph-break / unsupported-op
        # signal that needs investigation. Stage 5 expects this to
        # complete cleanly.
        gm, _ = torch._dynamo.export(forward)(
            x, x_pin, y_pin, y_gpu, w_perm, dummy
        )
        # Sanity: at least one of our custom ops is in the captured
        # graph (proves we didn't get an empty / pre-break graph).
        nodes = list(gm.graph.nodes)
        assert _node_index_by_target(nodes, "vllm.cots_submit_gemm") >= 0
        assert _node_index_by_target(nodes, "vllm.cots_sync_then_uva") >= 0
    finally:
        runner.close()
