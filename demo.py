"""Fixed-input demonstration of tps2d.

Run with:  python demo.py

It really fits thin-plate splines to deterministic inputs (no network, no
sleep, no pre-recorded output):

  1. a normal non-affine deformation on a 4x4 grid (lam = 0), with analytic
     Jacobian checked against live finite differences;
  2. the confidence-weighted smoothing spline: four trusted identity corners
     plus one gross outlier, compared at full / low / zero confidence and at
     lam = 0 (where confidence is degenerate and every point is hit);
  3. contract boundaries that actually fire: a conflicting duplicate source,
     an invalid (negative) confidence weight, and a degenerate positive
     subset (only collinear points are trusted with lam > 0).
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


def section_grid():
    sources, targets = build_problem()
    result = fit(sources, targets, lam=0.0)
    warp = result.warp

    print("=== 1. normal fit: thin-plate spline on a deterministic 4x4 grid ===")
    print(f"control points: {len(sources)} (limit 40), lambda = 0.0")
    print(f"affine design rank: {result.rank}")
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


def section_confidence():
    print("\n=== 2. confidence-weighted smoothing (lam = 0.1) ===")
    # Four trusted corners map by identity; the centre claims y' = 2.0.
    src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
    tgt = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 2.0]]
    print("trusted corners (identity) + one outlier (0.5,0.5) -> (0.5,2.0)")
    print(f"  {'confidence of outlier':>22}   {'y at centre':>11}  "
          f"{'max corner resid':>17}")
    for sigma in (1.0, 1e-3, 0.0):
        res = fit(src, tgt, lam=0.1,
                  confidence=[1.0, 1.0, 1.0, 1.0, sigma])
        y_centre = res.warp.evaluate(src[4])[1]
        corner_resid = max(res.residuals[:4])
        tag = "  (zero: point dropped)" if sigma == 0.0 else ""
        print(f"  {sigma:22g}   {y_centre:11.6f}  {corner_resid:17.2e}{tag}")

    dropped = fit(src, tgt, lam=0.1,
                  confidence=[1.0, 1.0, 1.0, 1.0, 0.0])
    print(f"  dropped-point reported residual ||f(c)-v|| = "
          f"{dropped.residuals[4]:.6f} (analytic: 1.5)")
    print(f"  reported merged confidence vector: {dropped.confidence}")

    # Boundary: at lam = 0 confidence is degenerate -- exact interpolation.
    exact = fit(src, tgt, lam=0.0,
                confidence=[0.0, 9.0, 0.0, 1e-6, 0.25])
    print(f"\n  lam = 0 with arbitrary confidence: max residual = "
          f"{max(exact.residuals):.3e} (exact interpolation, weights ignored)")

    # Duplicate compatible observations accumulate their confidence.
    dup_src = [[0, 0], [1, 0], [0, 1], [1, 0]]
    dup_tgt = [[0, 0], [1, 0], [0, 1], [1, 0]]
    dup_sig = [1.0, 0.4, 2.0, 0.6]
    merged = fit(dup_src, dup_tgt, lam=0.5, confidence=dup_sig)
    print(f"  duplicate (1,0) weights 0.4+0.6 merge -> confidence "
          f"{merged.confidence}, merged_count = {merged.merged_count}")


def section_errors():
    print("\n=== 3. contract boundaries (errors actually raised) ===")

    print("[a] conflicting duplicate source point:")
    bad_sources = [[0, 0], [3, 0], [0, 3], [3, 0]]
    bad_targets = [[0, 0], [3, 1], [1, 3], [3, 2]]
    try:
        fit(bad_sources, bad_targets, confidence=[1, 1, 1, 0.1])
    except TPSInputError as exc:
        print(f"    TPSInputError: {exc}")
    else:
        raise SystemExit("ERROR: expected TPSInputError was not raised")

    print("[b] negative confidence weight:")
    try:
        fit([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]],
            confidence=[1.0, -0.2, 1.0])
    except TPSInputError as exc:
        print(f"    TPSInputError: {exc}")
    else:
        raise SystemExit("ERROR: expected TPSInputError was not raised")

    print("[c] lam > 0 but only collinear points are trusted:")
    src = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]]
    tgt = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]]
    try:
        fit(src, tgt, lam=0.1, confidence=[1.0, 1.0, 1.0, 0.0, 0.0])
    except DegeneratePointsError as exc:
        print(f"    DegeneratePointsError: {exc}")
    else:
        raise SystemExit("ERROR: expected DegeneratePointsError was not raised")


def main():
    started = time.perf_counter()
    section_grid()
    section_confidence()
    section_errors()
    print(f"\nelapsed: {time.perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
