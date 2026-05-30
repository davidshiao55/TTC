#!/usr/bin/env python3
"""Summarize measured Phase 2 KV cells for policy work.

This is an analysis helper, not a planner input. It keeps the current
measurement artifact honest by showing which exact workload/runtime cells are
throughput-positive and which feature patterns break simple global rules.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DirectCell:
    label: str
    model: str
    mode: str
    gpu_memory_utilization: float | None
    total_tokens: int | None
    prompt_tokens: int | None
    split_tokens: int | None
    active_suffix_tokens: int | None
    active_suffix_blocks: int | None
    active_suffix_fraction: float | None
    gpu_effective_capacity: float | None
    hybrid_effective_capacity: float | None
    capacity_gain_pct: float | None
    gpu_only_out_tok_s: float
    hybrid_out_tok_s: float

    @property
    def ratio(self) -> float:
        return self.hybrid_out_tok_s / self.gpu_only_out_tok_s

    @property
    def delta_pct(self) -> float:
        return (self.ratio - 1.0) * 100.0

    @property
    def capacity_gain_per_suffix_block(self) -> float | None:
        if self.capacity_gain_pct is None:
            return None
        if self.active_suffix_blocks is None or self.active_suffix_blocks <= 0:
            return None
        return self.capacity_gain_pct / self.active_suffix_blocks


def _model_family(model: str) -> str:
    if "Llama" in model:
        return "llama"
    if "Qwen" in model:
        return "qwen"
    return model.split("/")[-1].lower()


def _mode(cell: dict[str, Any]) -> str:
    return "eager" if cell.get("enforce_eager") else "graph"


def _split_tokens(cell: dict[str, Any]) -> int | None:
    if cell.get("split_tokens") is not None:
        return int(cell["split_tokens"])
    if cell.get("split_blocks") is not None:
        return int(cell["split_blocks"]) * int(cell.get("block_size", 16))
    return None


def _capacity_gain_pct(cell: dict[str, Any]) -> float | None:
    gpu = cell.get("gpu_only_effective_capacity")
    hybrid = cell.get("hybrid_effective_capacity")
    if gpu is None or hybrid is None or float(gpu) <= 0:
        return None
    return (float(hybrid) / float(gpu) - 1.0) * 100.0


def _direct_cells(data: dict[str, Any]) -> list[DirectCell]:
    rows: list[DirectCell] = []
    for cell in data.get("cells", []):
        if "gpu_only_out_tok_s" not in cell or "hybrid_out_tok_s" not in cell:
            continue
        split = _split_tokens(cell)
        total = cell.get("total_tokens")
        prompt = cell.get("prompt_tokens")
        active_suffix_tokens = None
        active_suffix_blocks = None
        active_suffix_fraction = None
        if total is not None and split is not None:
            active_suffix_tokens = max(int(total) - int(split), 0)
            active_suffix_blocks = math.ceil(active_suffix_tokens / 16)
        if active_suffix_tokens is not None and total is not None and prompt is not None:
            generated = max(int(total) - int(prompt), 1)
            active_suffix_fraction = active_suffix_tokens / generated
        rows.append(
            DirectCell(
                label=str(cell["label"]),
                model=_model_family(str(cell.get("model", ""))),
                mode=_mode(cell),
                gpu_memory_utilization=cell.get("gpu_memory_utilization"),
                total_tokens=None if total is None else int(total),
                prompt_tokens=None if prompt is None else int(prompt),
                split_tokens=split,
                active_suffix_tokens=active_suffix_tokens,
                active_suffix_blocks=active_suffix_blocks,
                active_suffix_fraction=active_suffix_fraction,
                gpu_effective_capacity=(
                    None
                    if cell.get("gpu_only_effective_capacity") is None
                    else float(cell["gpu_only_effective_capacity"])
                ),
                hybrid_effective_capacity=(
                    None
                    if cell.get("hybrid_effective_capacity") is None
                    else float(cell["hybrid_effective_capacity"])
                ),
                capacity_gain_pct=_capacity_gain_pct(cell),
                gpu_only_out_tok_s=float(cell["gpu_only_out_tok_s"]),
                hybrid_out_tok_s=float(cell["hybrid_out_tok_s"]),
            )
        )
    return rows


def _fmt(value: object, width: int, precision: int = 1) -> str:
    if value is None:
        return "na".rjust(width)
    if isinstance(value, float):
        return f"{value:.{precision}f}".rjust(width)
    return str(value).rjust(width)


def _print_table(rows: list[DirectCell]) -> None:
    print(
        "model mode mem total split suf_blk suf_frac gpu_cap hyb_cap cap_gain% cap/block ratio delta% gpu_tok_s hybrid_tok_s label"
    )
    for row in rows:
        print(
            f"{row.model:5} {row.mode:5} "
            f"{_fmt(row.gpu_memory_utilization, 4, 2)} "
            f"{_fmt(row.total_tokens, 5)} "
            f"{_fmt(row.split_tokens, 5)} "
            f"{_fmt(row.active_suffix_blocks, 7)} "
            f"{_fmt(row.active_suffix_fraction, 8, 2)} "
            f"{_fmt(row.gpu_effective_capacity, 7, 2)} "
            f"{_fmt(row.hybrid_effective_capacity, 7, 2)} "
            f"{_fmt(row.capacity_gain_pct, 9, 1)} "
            f"{_fmt(row.capacity_gain_per_suffix_block, 9, 2)} "
            f"{row.ratio:5.3f} "
            f"{row.delta_pct:6.2f} "
            f"{row.gpu_only_out_tok_s:9.1f} "
            f"{row.hybrid_out_tok_s:12.1f} "
            f"{row.label}"
        )


def _print_policy_notes(rows: list[DirectCell], *, win_margin: float) -> None:
    promoted = [row for row in rows if row.ratio >= 1.0 + win_margin]
    near = [row for row in rows if 1.0 < row.ratio < 1.0 + win_margin]
    rejected = [row for row in rows if row.ratio <= 1.0]

    print()
    print(f"Profile-gated candidates at win_margin={win_margin:.3f}:")
    if promoted:
        for row in promoted:
            print(f"  keep hybrid: {row.label} ({row.ratio:.3f}x)")
    else:
        print("  none")

    print("Near ties below the promotion margin:")
    if near:
        for row in near:
            print(f"  measure more or keep GPU-only: {row.label} ({row.ratio:.3f}x)")
    else:
        print("  none")

    high_capacity_losses = [
        row
        for row in rejected
        if row.capacity_gain_pct is not None and row.capacity_gain_pct >= 4.0
    ]
    print("Capacity-gain-alone failures (cap_gain >= 4% but hybrid <= GPU-only):")
    if high_capacity_losses:
        for row in high_capacity_losses:
            print(
                "  "
                f"{row.label}: cap_gain={row.capacity_gain_pct:.1f}%, "
                f"suffix_blocks={row.active_suffix_blocks}, ratio={row.ratio:.3f}x"
            )
    else:
        print("  none")

    same_shape_flips = [
        row
        for row in rows
        if row.model == "llama"
        and row.mode == "eager"
        and row.total_tokens == 768
        and row.split_tokens == 736
    ]
    if len(same_shape_flips) > 1:
        print("Same shape can flip with memory pressure:")
        for row in same_shape_flips:
            print(
                "  "
                f"mem={row.gpu_memory_utilization}: cap_gain="
                f"{row.capacity_gain_pct:.1f}% ratio={row.ratio:.3f}x"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(__file__).with_name("phase2_kv_measurement_cells.json"),
    )
    parser.add_argument("--win-margin", type=float, default=0.01)
    args = parser.parse_args()

    data = json.loads(args.artifact.read_text())
    rows = _direct_cells(data)
    rows.sort(
        key=lambda r: (
            r.model,
            r.mode,
            -1.0 if r.gpu_memory_utilization is None else r.gpu_memory_utilization,
            -1 if r.total_tokens is None else r.total_tokens,
            -1 if r.split_tokens is None else r.split_tokens,
            r.label,
        )
    )
    _print_table(rows)
    _print_policy_notes(rows, win_margin=args.win_margin)


if __name__ == "__main__":
    main()
