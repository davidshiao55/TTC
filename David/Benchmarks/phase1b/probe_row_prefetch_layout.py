"""Phase 1b row-prefetch contention microprobe.

Direct measurement of the H2D bandwidth gap between the pre-fix
(strided/pitched) and post-fix (contiguous transposed) prefetch source
layouts for kind=row handles (MLP2 / down_proj). Independent of the
full-model bench — pinpoints the layout cost on this specific GPU.

Pre-fix source: pinned (out_dim, n_cpu), narrow(1, 0, n_pref).
  When n_pref < n_cpu, narrow returns a strided view with stride
  (n_cpu, 1) over (out_dim, n_pref) → pitched H2D.
  When n_pref == n_cpu, narrow degenerates to the full tensor →
  single contiguous H2D (this is why bench-2 prefetch-only B arm
  doesn't see the slowdown).

Post-fix source: pinned (max_n_prefetch, out_dim), narrow(0, 0, n_pref).
  Always contiguous, regardless of n_pref.

Default Qwen2.5-7B down_proj shape (out_dim=3584, in_dim=18944),
f_cpu_store=0.30. Run with:

    python David/Benchmarks/phase1b/probe_row_prefetch_layout.py
"""

import time

import torch


def bench(src, dst, axis, n, iters=200, warmup=20):
    for _ in range(warmup):
        dst.narrow(axis, 0, n).copy_(src.narrow(axis, 0, n), non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.narrow(axis, 0, n).copy_(src.narrow(axis, 0, n), non_blocking=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    out_dim, in_dim = 3584, 18944  # Qwen2.5-7B down_proj
    n_cpu = int(0.30 * in_dim)  # 5683
    dt = torch.bfloat16

    print(f"# Qwen2.5-7B down_proj: out_dim={out_dim}, in_dim={in_dim}")
    print(f"# n_cpu = {n_cpu} (f_cpu_store=0.30)")
    print(f"{'case':<48} {'ms':>8} {'GB/s':>8} {'ratio':>7}")
    print("-" * 75)

    src_strided = torch.empty(
        (out_dim, n_cpu), dtype=dt, pin_memory=True
    ).normal_()
    dst_strided = torch.empty((out_dim, n_cpu), dtype=dt, device="cuda")

    rows = []
    for f_pref_frac in (0.05, 0.15, 0.25, 0.30):
        n_pref = int(f_pref_frac * in_dim)
        bytes_moved = n_pref * out_dim * 2  # bf16

        t_strided = bench(src_strided, dst_strided, axis=1, n=n_pref)

        src_contig = torch.empty(
            (n_pref, out_dim), dtype=dt, pin_memory=True
        ).normal_()
        dst_contig = torch.empty(
            (n_pref, out_dim), dtype=dt, device="cuda"
        )
        t_contig = bench(src_contig, dst_contig, axis=0, n=n_pref)

        ratio = t_strided / t_contig if t_contig else float("nan")
        gbps_s = bytes_moved / t_strided / 1e9 if t_strided else 0
        gbps_c = bytes_moved / t_contig / 1e9 if t_contig else 0

        rows.append(
            (f_pref_frac, n_pref, t_strided, gbps_s, t_contig, gbps_c, ratio)
        )

        print(
            f"f_pref={f_pref_frac:.2f}  n_pref={n_pref:>5}  strided    "
            f"{t_strided*1e3:>8.3f} {gbps_s:>8.2f} {'(ref)':>7}"
        )
        print(
            f"f_pref={f_pref_frac:.2f}  n_pref={n_pref:>5}  contiguous "
            f"{t_contig*1e3:>8.3f} {gbps_c:>8.2f} {ratio:>6.2f}x"
        )

    print()
    print("# Per-forward extrapolation (28 row handles, B=1):")
    n_pref_15 = int(0.15 * in_dim)
    src_contig_15 = torch.empty(
        (n_pref_15, out_dim), dtype=dt, pin_memory=True
    ).normal_()
    dst_contig_15 = torch.empty((n_pref_15, out_dim), dtype=dt, device="cuda")
    t_str = bench(src_strided, dst_strided, axis=1, n=n_pref_15)
    t_con = bench(src_contig_15, dst_contig_15, axis=0, n=n_pref_15)
    pf_str = t_str * 28
    pf_con = t_con * 28
    print(f"  strided  : {pf_str*1e3:.3f} ms / forward")
    print(f"  contig   : {pf_con*1e3:.3f} ms / forward")
    print(f"  delta    : {(pf_str - pf_con)*1e3:.3f} ms / forward")
    print(
        f"  x 128 fwd: {(pf_str - pf_con) * 128 * 1e3:.0f} ms saved over a "
        f"128-step decode (row-handle H2D alone)"
    )


if __name__ == "__main__":
    main()
