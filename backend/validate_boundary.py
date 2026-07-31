"""Plain script to check compute_boundary() against the thesis baseline
before anything is wired to the API.

Run from backend/:

    python validate_boundary.py --k-sub 100
    python validate_boundary.py --k-sub 100 --save data/boundary_check.npz

No framework, no defaults borrowed from compute_boundary -- every baseline
value is read from swing_option.config and passed explicitly, exactly as an
API caller would have to.
"""

import argparse
import time

import numpy as np

from app.boundary import compute_boundary
from swing_option import config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k-sub", type=int, default=100, help="Sub-stepping factor (default: 100)")
    p.add_argument("--save", type=str, default=None, help="Write result to this .npz path")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Solving baseline config at K_sub={args.k_sub} ...")
    started = time.time()
    result = compute_boundary(
        T=config.T,
        N=config.N,
        sigma=config.sigma,
        kappa=config.kappa,
        r=config.r,
        K=config.K,
        S_0=config.S_0,
        alpha_func=config.alpha,
        C_max=config.C_max,
        C_min=config.C_min,
        q_max=config.q_max,
        K_sub=args.k_sub,
    )
    wall_clock = time.time() - started

    boundary = result["boundary"]
    stats = result["stats"]
    meta = result["meta"]

    finite = np.isfinite(boundary)
    n_total = boundary.size
    n_nan = n_total - int(finite.sum())

    print()
    print("=== price ===")
    print(f"  option price          : {stats['price']:.4f}")
    print(f"  thesis production ref : ~1573.74")

    print()
    print("=== boundary ===")
    print(f"  shape                  : {boundary.shape}  (dates, C+1)")
    print(f"  min / max (finite)     : {np.nanmin(boundary):.4f} / {np.nanmax(boundary):.4f}")
    print(f"  NaN count / share      : {n_nan} / {n_total}  ({100.0 * n_nan / n_total:.2f}%)")

    print()
    print("=== stats ===")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")

    print()
    print("=== meta ===")
    for k, v in meta.items():
        print(f"  {k:<22}: {v}")

    # -- monotonicity check: boundary must be non-decreasing in C ----------
    # Per row, look only at the finite (thresholded) cells in column order.
    # Rows with fewer than 2 finite cells say nothing about monotonicity
    # and are excluded from the fraction, but counted separately below.
    #
    # interpolate_boundary() solves a linear zero-crossing per cell, which
    # leaves genuine sub-grid-step floating point noise on the result (observed
    # up to ~1e-4 in one run, against a typical step of ~1e-2 and a delta_X of
    # ~1e-1) -- an absolute tolerance near machine epsilon flags that noise as
    # "decreasing". Scale the tolerance off delta_X, the resolution's own
    # natural price unit, instead of a fixed constant.
    tol = max(1e-6, 0.01 * meta["delta_X"])
    n_checkable = 0
    n_monotonic = 0
    n_skipped = 0
    for row in boundary:
        vals = row[np.isfinite(row)]
        if vals.size < 2:
            n_skipped += 1
            continue
        n_checkable += 1
        if np.all(np.diff(vals) >= -tol):
            n_monotonic += 1

    frac = (n_monotonic / n_checkable) if n_checkable else float("nan")
    print()
    print("=== monotonicity (boundary must rise in C) ===")
    print(f"  tolerance               : {tol:.2e}  (1% of delta_X)")
    print(f"  rows checked            : {n_checkable} (skipped {n_skipped} with <2 finite cells)")
    print(f"  non-decreasing fraction : {frac:.4f}")
    if n_checkable and frac < 0.5:
        print("  WARNING: fraction is far from 1 -- the volume axis looks mirrored.")

    # -- forced region should sit near the end of the contract, not the start
    forced_onset = stats.get("forced_onset_mean_t")
    print()
    print("=== forced region timing ===")
    if forced_onset is None:
        print("  no forced-exercise states in this config")
    else:
        frac_of_T = forced_onset / config.T
        print(f"  forced_onset_mean_t     : {forced_onset:.4f}  ({frac_of_T:.1%} of T)")
        if frac_of_T < 0.5:
            print("  WARNING: mean forced date is in the first half of the contract.")

    print()
    print(f"wall clock: {wall_clock:.1f}s")

    if args.save:
        np.savez(
            args.save,
            boundary=boundary,
            alpha=result["alpha"],
            time_grid=result["time_grid"],
            **{f"stats_{k}": v for k, v in stats.items() if v is not None},
            **{f"meta_{k}": v for k, v in meta.items()},
        )
        print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
