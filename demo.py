"""Fixed-input demonstration of tps2d.

Run with:  python demo.py

It really fits thin-plate splines to deterministic control points (no
network, no sleep, no pre-recorded output):

  1. a normal lam = 0 interpolation on a 4x4 grid, with analytic-vs-finite-
     difference Jacobian check;
  2. the new per-point confidence feature: a biased anchor is down-weighted
     and the trusted region stays near identity (boundary: zero-confidence
     anchor with lam = 0 is rejected);
  3. a genuine contract rejection: a duplicate source point mapping to two
     conflicting targets.
"""

import math
import time

from tps2d import fit, TPSInputError, DegeneratePointsError


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


def section_normal_interpolation():
    sources, targets = build_problem()
    result = fit(sources, targets, lam=0.0)
    warp = result.warp

    print("=== 1. normal interpolation: TPS on a deterministic 4x4 grid ===")
    print(f"control points: {len(sources)} (limit 40), lambda = 0.0")
    print(f"affine design rank: {result.rank}, merged: {result.merged_count}")
    print(f"max control-point residual: {max(result.residuals):.3e}")
    print(f"RMS residual: {result.rms_residual:.3e}")
    print(f"bending energy w^T K w (both coords): {result.bending_energy:.6f}")

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


def section_confidence_outlier():
    print("\n=== 2. confidence weights: one biased anchor on a 3x3 grid ===")
    # Identity mapping everywhere, except one corner anchor that claims a
    # gross displacement (+4, -3) no genuine point supports.
    sources = [[x, y] for y in (0.0, 2.0, 4.0) for x in (0.0, 2.0, 4.0)]
    targets = [[x, y] for x, y in sources]
    outlier = 8  # point (4, 4) claims (8, 1)
    targets[outlier] = [8.0, 1.0]
    lam = 0.5

    full = fit(sources, targets, lam=lam)  # every point equally trusted
    down = [1.0] * 9
    down[outlier] = 0.01
    weighted = fit(sources, targets, lam=lam, confidence=down)

    print(f"regularisation lambda = {lam}, biased anchor at {sources[outlier]}")
    print(f"  equal weights   -> effective={full.effective_count}, "
          f"max trusted residual={max(full.residuals[:8]):.4f}, "
          f"outlier residual={full.residuals[outlier]:.4f}")
    print(f"  weight 0.01     -> effective={weighted.effective_count}, "
          f"max trusted residual={max(weighted.residuals[:8]):.4f}, "
          f"outlier residual={weighted.residuals[outlier]:.4f}")
    for p in ([2.0, 2.0], [3.0, 3.0]):
        qf = full.warp.evaluate(p)
        qw = weighted.warp.evaluate(p)
        print(f"  interior {p}: equal-weight -> ({qf[0]:.3f},{qf[1]:.3f}), "
              f"down-weighted -> ({qw[0]:.3f},{qw[1]:.3f}) [identity expected]")

    # Boundary that actually fires: zero confidence is incompatible with
    # exact interpolation.
    print("\n-- boundary: confidence[i] = 0 together with lam = 0 --")
    try:
        fit(sources, targets, lam=0.0, confidence=[1.0] * 8 + [0.0])
    except TPSInputError as exc:
        print(f"TPSInputError raised as required: {exc}")
    else:
        raise SystemExit("ERROR: expected TPSInputError was not raised")

    # And a degenerate *active* set: only collinear points carry weight.
    print("-- boundary: positive-weight points collinear -> degenerate --")
    try:
        fit(
            [[0, 0], [1, 0], [2, 0], [1, 1]],
            [[0, 0], [1, 0], [2, 0], [1, 1]],
            lam=1.0,
            confidence=[1.0, 1.0, 1.0, 0.0],
        )
    except DegeneratePointsError as exc:
        print(f"DegeneratePointsError raised as required: {exc}")
    else:
        raise SystemExit("ERROR: expected DegeneratePointsError was not raised")


def section_conflicting_duplicate():
    print("\n=== 3. contract boundary: conflicting duplicate source ===")
    bad_sources = [[0, 0], [3, 0], [0, 3], [3, 0]]
    bad_targets = [[0, 0], [3, 1], [1, 3], [3, 2]]  # (3,0) twice, different
    try:
        fit(bad_sources, bad_targets, lam=1.0, confidence=[1, 1, 1, 0])
    except TPSInputError as exc:
        # Compatibility is checked before weighting: zero confidence does not
        # make contradictory data compatible.
        print(f"TPSInputError raised as required: {exc}")
    else:
        raise SystemExit("ERROR: expected TPSInputError was not raised")


def main():
    started = time.perf_counter()
    section_normal_interpolation()
    section_confidence_outlier()
    section_conflicting_duplicate()
    print(f"\nelapsed: {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
