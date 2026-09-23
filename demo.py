"""Fixed-input demonstration of tps2d.

Run with:  python demo.py

It really fits a thin-plate spline to a deterministic 4x4 grid (no network,
no sleep, no pre-recorded output), prints a normal deformation result and
then triggers a genuine contract rejection (duplicate source point mapping to
two conflicting targets).
"""

import math
import time

from tps2d import fit, TPSInputError


def build_problem():
    """Deterministic 16 control points on a regular grid with smooth warp."""
    sources = []
    targets = []
    for j in range(4):
        for i in range(4):
            x, y = 3.0 * i, 3.0 * j
            sources.append([x, y])
            targets.append([
                x + 1.2 * math.sin(0.35 * y),
                y + 1.2 * math.cos(0.30 * x),
            ])
    return sources, targets


def main():
    started = time.perf_counter()

    sources, targets = build_problem()
    result = fit(sources, targets, lam=0.0)
    warp = result.warp

    print("=== tps2d demo: thin-plate spline on a deterministic 4x4 grid ===")
    print(f"control points: {len(sources)} (limit 40), lambda = 0.0")
    print(f"affine design rank: {result.rank}")
    print(f"max control-point residual: {max(result.residuals):.3e}")
    print(f"RMS residual: {result.rms_residual:.3e}")
    print(f"bending energy w^T K w (both coords): {result.bending_energy:.6f}")

    # Warp interior/edge grid points that are NOT control points.
    probes = [[1.5, 1.5], [4.5, 4.5], [7.5, 1.5], [4.5, 7.5], [0.75, 6.0]]
    print("\nwarped probe points:")
    print(f"  {'source':>16} -> {'target':>22}   det(J)")
    for p in probes:
        q = warp.evaluate(p)
        J = warp.jacobian(p)
        det_j = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        print(
            f"  ({p[0]:5.2f},{p[1]:5.2f})"
            f" -> ({q[0]:8.4f},{q[1]:8.4f})   {det_j: .5f}"
        )

    # Analytic Jacobian vs central finite differences, computed live.
    p = probes[0]
    h = 1e-6
    jx_p = warp.evaluate([p[0] + h, p[1]])
    jx_m = warp.evaluate([p[0] - h, p[1]])
    jy_p = warp.evaluate([p[0], p[1] + h])
    jy_m = warp.evaluate([p[0], p[1] - h])
    j_num = [
        [(jx_p[0] - jx_m[0]) / (2 * h), (jy_p[0] - jy_m[0]) / (2 * h)],
        [(jx_p[1] - jx_m[1]) / (2 * h), (jy_p[1] - jy_m[1]) / (2 * h)],
    ]
    j_ana = warp.jacobian(p)
    err = max(
        abs(j_ana[r][c] - j_num[r][c]) for r in range(2) for c in range(2)
    )
    print(f"\nmax |J_analytic - J_finite-diff| at {p}: {err:.3e}")

    # ---- a boundary that actually fires -------------------------------
    print("\n=== contract boundary: conflicting duplicate source ===")
    bad_sources = [[0, 0], [3, 0], [0, 3], [3, 0]]
    bad_targets = [[0, 0], [3, 1], [1, 3], [3, 2]]  # point (3,0) twice, different
    try:
        fit(bad_sources, bad_targets)
    except TPSInputError as exc:
        print(f"TPSInputError raised as required: {exc}")
    else:
        raise SystemExit("ERROR: expected TPSInputError was not raised")

    print(f"\nelapsed: {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
