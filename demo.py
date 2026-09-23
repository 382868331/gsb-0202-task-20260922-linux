"""Deterministic TPS demo: fit a surface deformation, warp a grid, and
trigger one input rejection.  Pure standard library; runs in well under
8 seconds.  Run with:  python demo.py
"""

import math

from tps import TPSInputError, fit_tps


def make_control_points():
    """12 deterministic, non-collinear control points."""
    pts = []
    for i in range(12):
        angle = i * 2.399963229728653  # golden-angle spiral
        radius = 1.0 + 0.35 * i
        pts.append((radius * math.cos(angle), radius * math.sin(angle)))
    return pts


def displace(p):
    """Smooth deterministic displacement field (a mild swirl + bump)."""
    x, y = p
    r2 = x * x + y * y
    bump = 0.6 * math.exp(-r2 / 8.0)
    return (x + 0.15 * y + bump * math.cos(0.5 * x),
            y - 0.10 * x + bump * math.sin(0.5 * y))


def main():
    print("=== TPS demo: surface deformation fit ===")
    src = make_control_points()
    tgt = [displace(p) for p in src]

    tps = fit_tps(src, tgt, lambda_=1e-4)
    print(f"fitted {len(tps.control_points)} control points, "
          f"lambda={tps.lambda_}")
    print(f"rms residual   = {tps.rms_error:.6e}")
    print(f"max residual   = {tps.max_error:.6e}")
    print(f"bending energy = {tps.bending_energy:.6f} "
          f"(x: {tps.bending_energy_per_axis[0]:.6f}, "
          f"y: {tps.bending_energy_per_axis[1]:.6f})")

    # Warp a 5x5 measurement grid through the deformation.
    print("\nwarped 5x5 grid (corners and center):")
    grid = [(2.0 * i - 4.0, 2.0 * j - 4.0) for j in range(5) for i in range(5)]
    warped = tps.warp(grid)
    for idx in (0, 4, 12, 20, 24):
        gx, gy = grid[idx]
        wx, wy = warped[idx]
        print(f"  ({gx:+.1f}, {gy:+.1f}) -> ({wx:+.4f}, {wy:+.4f})")

    probe = (0.5, -1.0)
    j = tps.jacobian(probe)
    print(f"\nJacobian at {probe}:")
    print(f"  dfx/dx={j[0][0]:+.4f}  dfx/dy={j[0][1]:+.4f}")
    print(f"  dfy/dx={j[1][0]:+.4f}  dfy/dy={j[1][1]:+.4f}")

    # Boundary case: duplicate source points with conflicting targets
    # must be rejected with a locatable error.
    print("\n=== rejection boundary: conflicting duplicate points ===")
    bad_src = src[:4] + [src[1]]
    bad_tgt = tgt[:4] + [(tgt[1][0] + 0.25, tgt[1][1])]
    try:
        fit_tps(bad_src, bad_tgt)
    except TPSInputError as exc:
        print(f"rejected as expected: {type(exc).__name__}: {exc}")
    else:
        raise SystemExit("ERROR: conflicting duplicates were not rejected")

    print("\ndemo finished.")


if __name__ == "__main__":
    main()
