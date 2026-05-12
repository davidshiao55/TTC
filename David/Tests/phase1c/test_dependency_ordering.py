# SPDX-License-Identifier: Apache-2.0
"""Stage 5 — `mutates_args` actually prevents compile-time reordering.

The whole correctness story of `cots_ops.py` rests on torch.compile
honoring the barrier-installing `mutates_args` declarations. Final
schemas after the §1c.20 simplification:
  * `cots_submit_gemm` mutates `["x_gpu"]` (only) — pins this op
    BEFORE every GPU GEMM that reads x_gpu, AND provides the
    (submit → sync) ordering edge consumed by sync's
    `submit_anchor` read. y_pinned and x_pinned are NOT in the op
    signature; the C++ side owns the pinned-buffer addresses via
    slab pointers populated at install time.
  * `cots_sync_then_uva` mutates `["y_gpu", "gpu_anchor_a",
    "gpu_anchor_b"]` so the op is pinned AFTER each independent GPU
    GEMM (preserves overlap; both anchors needed for QKV's two
    independent F.linears). It also reads `submit_anchor` (==
    x_gpu) for the cross-op data dep, and resolves the pinned
    output via the active dispatch bucket and C++ slab pointer.

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

import inspect
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


def test_cots_submit_gemm_mutates_only_x_gpu() -> None:
    """`cots_submit_gemm` must declare ONLY `x_gpu` as mutated.

    §1c.20: `y_pinned` was previously declared mutated, which caused
    Inductor's functionalization to clone the pinned slice into a
    fresh pageable CPU buffer post-mutation. The new schema removes
    `y_pinned` from the op signature entirely; the worker writes to
    the pinned output via the slab-side pointer populated at install
    time, and `cots_sync_then_uva` reads `submit_anchor` (= x_gpu)
    to enforce the (submit → sync) data dependency.
    """
    schema = str(torch.ops.vllm.cots_submit_gemm.default._schema)
    mutated = _schema_mutated_args(schema)
    assert "x_gpu" in mutated, (
        f"cots_submit_gemm must mutate `x_gpu` (the CUDA-anchor that "
        f"pins this op before GPU GEMMs reading x_gpu, AND the read by "
        f"cots_sync_then_uva that pins sync after submit). Schema: {schema}"
    )
    assert "y_pinned" not in mutated, (
        f"§1c.20: `y_pinned` MUST NOT be in cots_submit_gemm's "
        f"mutates_args. Inductor's functionalization clones the pinned "
        f"slice into a fresh pageable CPU buffer when y_pinned is "
        f"declared mutated, breaking uva_copy_into_gpu's page-locked-"
        f"storage requirement. The worker writes via the slab pointer; "
        f"the (submit → sync) dep rides on x_gpu's submit_anchor read. "
        f"Schema: {schema}"
    )
    unexpected = set(mutated) - {"x_gpu"}
    assert not unexpected, (
        f"cots_submit_gemm has unexpected mutated args {unexpected}; "
        f"schema: {schema}"
    )


def test_cots_submit_gemm_does_not_take_y_pinned() -> None:
    """§1c.20: `y_pinned` is not just removed from `mutates_args` —
    it's removed from the op's argument list entirely. This is what
    ensures Inductor never sees the pinned slice as a custom-op
    input that could be functionalized into a pageable clone."""
    schema = str(torch.ops.vllm.cots_submit_gemm.default._schema)
    assert "y_pinned" not in schema, (
        f"§1c.20: cots_submit_gemm's signature MUST NOT contain "
        f"y_pinned. The previous schema's `y_pinned` arg with "
        f"`mutates_args=[..., 'y_pinned']` triggered Inductor's "
        f"functionalization clone. Schema: {schema}"
    )
    assert "task_id" not in schema and "num_tokens" not in schema, (
        f"§1c.35: native submit must not take compile-visible task_id "
        f"or num_tokens scalars. Schema: {schema}"
    )
    assert "layer_idx" in schema and "op_kind_code" in schema, (
        f"§1c.35: native submit should carry only stable call-site "
        f"identity plus runner_id. Schema: {schema}"
    )


def test_cots_sync_then_uva_takes_submit_anchor() -> None:
    """§1c.20: `cots_sync_then_uva` must take a `submit_anchor`
    argument (read-only) so torch.compile can chain (submit → sync)
    via the x_gpu data dependency without y_pinned being in submit's
    mutates_args. submit_anchor is NOT in mutates_args — only y_gpu
    and the two gpu anchors are."""
    schema = str(torch.ops.vllm.cots_sync_then_uva.default._schema)
    assert "submit_anchor" in schema, (
        f"§1c.20: cots_sync_then_uva must take a `submit_anchor` arg "
        f"(read-only) so sync stays after submit via the x_gpu read. "
        f"Schema: {schema}"
    )
    mutated = _schema_mutated_args(schema)
    assert "submit_anchor" not in mutated, (
        f"submit_anchor must be read-only — mutating it would force "
        f"y_gpu / gpu_anchors / x_gpu to alias-share with submit's "
        f"barrier, defeating the design. Schema: {schema}"
    )


def test_cots_sync_then_uva_does_not_take_y_pinned() -> None:
    """§1c.20: `cots_sync_then_uva`'s signature MUST NOT include
    `y_pinned`. Inductor's functionalization on captured graphs
    materializes any CPU tensor arg by allocating a fresh pageable
    CPU buffer (worst case via a GPU intermediate + blocking
    GPU→CPU copy that CUDA Graph capture rejects). The sync impl
    reaches the pinned output via the slab pointer through
    `CotsCpuInfer.y_pinned_view(task_id, bucket)`. The op's
    Python-visible args are CUDA tensors + scalar ids only."""
    schema = str(torch.ops.vllm.cots_sync_then_uva.default._schema)
    assert "y_pinned" not in schema, (
        f"§1c.20: cots_sync_then_uva's signature MUST NOT contain "
        f"y_pinned. Inductor would materialize it. Schema: {schema}"
    )
    assert "task_id" not in schema and "num_tokens" not in schema, (
        f"§1c.35: native sync must not take compile-visible task_id "
        f"or num_tokens scalars. Schema: {schema}"
    )
    assert "layer_idx" in schema and "op_kind_code" in schema, (
        f"§1c.35: native sync should carry only stable call-site "
        f"identity plus runner_id. Schema: {schema}"
    )


def test_cots_sync_then_uva_mutates_only_gpu_args() -> None:
    """§1c.20: under the new schema the mutated set is exactly
    {y_gpu, gpu_anchor_a, gpu_anchor_b} — submit_anchor is read-only,
    runner_id / layer_idx / op_kind_code are scalars."""
    schema = str(torch.ops.vllm.cots_sync_then_uva.default._schema)
    mutated = _schema_mutated_args(schema)
    expected = {"y_gpu", "gpu_anchor_a", "gpu_anchor_b"}
    assert set(mutated) == expected, (
        f"cots_sync_then_uva mutated set is {set(mutated)}; expected "
        f"{expected}. Schema: {schema}"
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


def test_native_dispatch_state_is_required_and_per_runner() -> None:
    """§1c.35: task resolution uses OOG dispatch state keyed by runner_id."""
    rid_a = 1_000_001
    rid_b = 1_000_002
    qkv = cots_ops.op_kind_code("qkv")
    try:
        cots_ops._COTS_TASK_ID_FOR[rid_a] = {(0, 1, "qkv"): 11}
        cots_ops._COTS_TASK_ID_FOR[rid_b] = {(0, 4, "qkv"): 44}

        with pytest.raises(RuntimeError, match="no active COTS dispatch state"):
            cots_ops._resolve_task_for_dispatch(rid_a, 0, qkv, "test")

        cots_ops.set_active_dispatch_state(rid_a, bucket=1, live_num_tokens=1)
        cots_ops.set_active_dispatch_state(rid_b, bucket=4, live_num_tokens=2)

        assert cots_ops._resolve_task_for_dispatch(rid_a, 0, qkv, "test") == (
            11,
            1,
            1,
        )
        assert cots_ops._resolve_task_for_dispatch(rid_b, 0, qkv, "test") == (
            44,
            4,
            2,
        )
    finally:
        cots_ops._COTS_TASK_ID_FOR.pop(rid_a, None)
        cots_ops._COTS_TASK_ID_FOR.pop(rid_b, None)
        cots_ops._COTS_ACTIVE_DISPATCH.pop(rid_a, None)
        cots_ops._COTS_ACTIVE_DISPATCH.pop(rid_b, None)


def test_native_transfer_rows_are_bounded_by_bucket_and_tensor_rows() -> None:
    """Task bucket is capacity; transfer rows must not overread tensors."""
    assert cots_ops._bounded_transfer_rows(1, 8192, "test") == 1
    assert cots_ops._bounded_transfer_rows(8192, 256, "test") == 256
    assert cots_ops._bounded_transfer_rows(512, 512, "test") == 512
    with pytest.raises(AssertionError, match="transfer row count"):
        cots_ops._bounded_transfer_rows(8, 0, "test")


def test_gpu_model_runner_publishes_dispatch_for_dummy_forwards() -> None:
    """Profile/warmup/capture forwards must publish OOG dispatch state too."""
    from vllm.v1.worker import gpu_model_runner as gmr

    source = inspect.getsource(gmr.GPUModelRunner._dummy_run)
    publish = "self._publish_offloader_dispatch(batch_desc, num_tokens_unpadded)"
    assert publish in source, (
        "_dummy_run must publish the active BatchDescriptor before model "
        "execution. Native COTS custom ops intentionally refuse to infer a "
        "bucket from tensor shapes when profile/capture forwards bypass "
        "execute_model."
    )
    assert source.index(publish) < source.index("set_forward_context("), (
        "_dummy_run must publish offloader dispatch state before entering "
        "the forward context and invoking the model."
    )


def test_gpu_model_runner_dispatch_helper_calls_offloader(monkeypatch) -> None:
    """The shared helper should pass a ForwardDispatchInfo object through."""
    from vllm.forward_context import BatchDescriptor
    from vllm.v1.worker import gpu_model_runner as gmr

    seen = []

    class Recorder:
        def on_dispatch(self, info):
            seen.append(info)

    monkeypatch.setattr(gmr, "get_offloader", lambda: Recorder())
    runner = gmr.GPUModelRunner.__new__(gmr.GPUModelRunner)
    desc = BatchDescriptor(num_tokens=8, uniform=True)

    runner._publish_offloader_dispatch(desc, 3)

    assert len(seen) == 1
    assert isinstance(seen[0], gmr.ForwardDispatchInfo)
    assert seen[0].batch_descriptor is desc
    assert seen[0].num_tokens_unpadded == 3


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
        max_num_tokens=4,
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
    op_kind = cots_ops.op_kind_code("qkv")

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
        # §1c.20 schema: y_pinned is no longer a custom-op arg on
        # either op. Submit gets x_gpu/x_pinned + ids; sync gets only
        # CUDA tensors + ids and resolves the pinned view via the
        # slab pointer in C++ (`y_pinned_view(task_id, bucket)`).
        # §1c.20 final schema: BOTH x_pinned and y_pinned are gone
        # from the custom op signatures. The C++ side bundles the
        # x_gpu → slab.x_pinned_ptr D2H into submit_on_stream, and
        # the sync impl reaches the pinned output via
        # y_pinned_view(task_id, bucket). Both ops take only
        # CUDA tensors + scalar ids. No CPU view is built in this
        # forward — Inductor would materialize it into a pageable
        # buffer otherwise.
        del x_pin, y_pin
        torch.ops.vllm.cots_submit_gemm(x, runner_id, 0, op_kind)
        out_perm = torch.nn.functional.linear(x, w_perm)
        out_pref = torch.nn.functional.linear(x, w_pref)
        torch.ops.vllm.cots_sync_then_uva(
            y_gpu, out_perm, out_pref, x, runner_id, 0, op_kind
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
                f"mutates_args=['x_gpu'] declaration on "
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
    op_kind = cots_ops.op_kind_code("qkv")

    x = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    x_pin = torch.empty(4, 16, dtype=torch.bfloat16, pin_memory=True)
    y_pin = torch.empty(4, 8, dtype=torch.bfloat16, pin_memory=True)
    y_gpu = torch.empty(4, 8, dtype=torch.bfloat16, device="cuda")
    w_perm = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
    dummy = torch.empty(1, dtype=torch.bfloat16, device="cuda")

    def forward(x, x_pin, y_pin, y_gpu, w_perm, dummy):
        # §1c.20 final schema: both x_pinned and y_pinned dropped.
        del x_pin, y_pin
        torch.ops.vllm.cots_submit_gemm(x, runner_id, 0, op_kind)
        out_perm = torch.nn.functional.linear(x, w_perm)
        torch.ops.vllm.cots_sync_then_uva(
            y_gpu, out_perm, dummy, x, runner_id, 0, op_kind
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
