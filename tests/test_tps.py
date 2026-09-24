"""Necessary tests for the tps2d package.

Expected values come from closed-form construction (e.g. a known affine map)
or from ``_reference`` (an independent from-scratch implementation).  The
package under test is never used to manufacture its own expectations;
round-trip assertions are only used as supplementary checks.

Error accounting (see README "Numerical accuracy"):
  * exact interpolation / affine recovery tolerance : 1e-9
  * package-vs-independent-reference tolerance      : 1e-9
  * finite-difference Jacobian tolerance            : 5e-5
"""

import math
import unittest

import tps2d
from tps2d import fit, TPSInputError, DegeneratePointsError, MAX_CONTROL_POINTS

from _reference import (
    reference_fit,
    reference_warp,
    reference_fit_weighted,
    finite_diff_jacobian,
    kernel,
)


def grid_points(nx=4, ny=4, scale=3.0):
    return [
        [scale * i, scale * j]
        for j in range(ny)
        for i in range(nx)
    ]


class AffineRecoveryTests(unittest.TestCase):
    """Pure affine data must give w = 0 analytically."""

    A0 = (2.0, -1.0)
    A1 = (3.0, 0.5)
    A2 = (-1.0, 2.0)

    def affine_map(self, p):
        x, y = p
        return [
            self.A0[0] + self.A1[0] * x + self.A2[0] * y,
            self.A0[1] + self.A1[1] * x + self.A2[1] * y,
        ]

    def setUp(self):
        self.src = [[0, 0], [1, 0], [0, 1], [2, 3], [-2, 1.5], [3.5, -2]]
        self.tgt = [self.affine_map(p) for p in self.src]
        self.res = fit(self.src, self.tgt)

    def test_weights_zero(self):
        for i, (wx, wy) in enumerate(self.res.weights):
            self.assertLess(abs(wx), 1e-9, f"w_x[{i}] = {wx}")
            self.assertLess(abs(wy), 1e-9, f"w_y[{i}] = {wy}")

    def test_affine_coefficients(self):
        a = self.res.affine
        self.assertAlmostEqual(a[0][0], self.A0[0], delta=1e-9)
        self.assertAlmostEqual(a[0][1], self.A0[1], delta=1e-9)
        self.assertAlmostEqual(a[1][0], self.A1[0], delta=1e-9)
        self.assertAlmostEqual(a[1][1], self.A1[1], delta=1e-9)
        self.assertAlmostEqual(a[2][0], self.A2[0], delta=1e-9)
        self.assertAlmostEqual(a[2][1], self.A2[1], delta=1e-9)

    def test_bending_energy_zero(self):
        self.assertLess(abs(self.res.bending_energy), 1e-9)

    def test_residuals_zero(self):
        self.assertLess(max(self.res.residuals), 1e-9)
        self.assertLess(self.res.rms_residual, 1e-9)

    def test_regularised_affine_still_affine(self):
        res = fit(self.src, self.tgt, lam=10.0)
        self.assertLess(max(abs(w) for row in res.weights for w in row), 1e-9)
        self.assertLess(abs(res.bending_energy), 1e-9)


class ExactInterpolationTests(unittest.TestCase):
    """lam = 0 must recover every control point exactly."""

    def setUp(self):
        self.src = grid_points(4, 4)
        # Smooth, non-affine, deterministic displacement field.
        self.tgt = [
            [x + 0.4 * math.sin(0.5 * y), y + 0.3 * math.cos(0.4 * x)]
            for x, y in self.src
        ]
        self.res = fit(self.src, self.tgt, lam=0.0)

    def test_control_points_exact(self):
        for s, t in zip(self.src, self.tgt):
            got = self.res.warp.evaluate(s)
            self.assertLess(abs(got[0] - t[0]), 1e-9, (s, got, t))
            self.assertLess(abs(got[1] - t[1]), 1e-9)

    def test_residuals_reported_zero(self):
        self.assertLess(max(self.res.residuals), 1e-9)

    def test_ptw_constraint(self):
        # P^T w = 0 for both coordinates.
        for c in (0, 1):
            s0 = sum(w[c] for w in self.res.weights)
            sx = sum(x * w[c] for (x, y), w in zip(self.src, self.res.weights))
            sy = sum(y * w[c] for (x, y), w in zip(self.src, self.res.weights))
            self.assertLess(abs(s0), 1e-8)
            self.assertLess(abs(sx), 1e-8)
            self.assertLess(abs(sy), 1e-8)

    def test_bending_energy_positive_and_matches_reference(self):
        self.assertGreater(self.res.bending_energy, 0.0)
        params = reference_fit(self.src, self.tgt, 0.0)
        wx, wy, _, _, pts = params
        expected = 0.0
        for wv in (wx, wy):
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    expected += 2.0 * wv[i] * wv[j] * kernel(
                        pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]
                    )
        self.assertAlmostEqual(
            self.res.bending_energy, expected, delta=1e-8 * max(1.0, abs(expected))
        )

    def test_jacobian_matches_finite_differences(self):
        # Generic probe points (none coincides with a control point).
        probes = [[0.5, 0.7], [3.3, 6.1], [7.9, 2.2], [-1.2, 4.4]]
        for p in probes:
            ana = self.res.warp.jacobian(p)
            num = finite_diff_jacobian(
                lambda x, y: self.res.warp.evaluate([x, y]), p[0], p[1], h=1e-6
            )
            for r in range(2):
                for c in range(2):
                    self.assertLess(
                        abs(ana[r][c] - num[r][c]),
                        5e-5,
                        f"J[{r}][{c}] at {p}: {ana} vs {num}",
                    )

    def test_jacobian_at_control_point_finite_and_consistent(self):
        # The kernel gradient is 0 at r = 0; Jacobian stays finite there.
        p = self.src[5]
        ana = self.res.warp.jacobian(p)
        num = finite_diff_jacobian(
            lambda x, y: self.res.warp.evaluate([x, y]), p[0], p[1], h=1e-5
        )
        for entry_ana, entry_num in zip(sum(ana, []), sum(num, [])):
            self.assertTrue(math.isfinite(entry_ana))
            self.assertLess(abs(entry_ana - entry_num), 2e-4)

    def test_matches_independent_reference(self):
        params = reference_fit(self.src, self.tgt, 0.0)
        for p in [[0.3, 1.1], [4.7, 5.9], [-3.0, 8.0], [9.2, -1.5]]:
            got = self.res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 1e-9)
            self.assertLess(abs(got[1] - ry), 1e-9)


class DegenerateAndDuplicateTests(unittest.TestCase):
    def test_collinear_rejected(self):
        src = [[0, 0], [1, 1], [2, 2], [3, 3]]
        tgt = [[0, 0], [1, 0], [2, 0], [3, 0]]
        with self.assertRaises(DegeneratePointsError):
            fit(src, tgt)

    def test_two_points_rejected(self):
        with self.assertRaises(DegeneratePointsError):
            fit([[0, 0], [1, 0]], [[0, 0], [1, 1]])

    def test_single_point_rejected(self):
        with self.assertRaises(DegeneratePointsError):
            fit([[0, 0]], [[1, 1]])

    def test_conflicting_duplicate_rejected(self):
        src = [[0, 0], [1, 0], [0, 1], [1, 0]]
        tgt = [[0, 0], [2, 0], [0, 2], [2.5, 0]]
        with self.assertRaises(TPSInputError):
            fit(src, tgt)

    def test_identical_duplicate_merged(self):
        src = [[0, 0], [1, 0], [0, 1], [1, 0]]
        tgt = [[0, 0], [2, 0], [0, 2], [2, 0]]
        res = fit(src, tgt)
        self.assertEqual(res.merged_count, 3)
        self.assertEqual(len(res.weights), 3)
        got = res.warp.evaluate([1, 0])
        self.assertAlmostEqual(got[0], 2.0, delta=1e-9)
        self.assertAlmostEqual(got[1], 0.0, delta=1e-9)

    def test_near_degenerate_report(self):
        # Collinear within the fixed numerical tolerance is still rejected.
        src = [[0, 0], [1, 0], [2, 1e-14]]
        tgt = [[0, 0], [1, 0], [2, 0]]
        with self.assertRaises(DegeneratePointsError):
            fit(src, tgt)


class RegularizationTests(unittest.TestCase):
    def setUp(self):
        # Smooth underlying field plus deterministic jitter.
        self.src = grid_points(4, 4)
        self.tgt = [
            [
                x + 0.3 * math.sin(0.6 * x + 0.4 * y) + 0.25 * math.sin(2.3 * i),
                y + 0.3 * math.cos(0.5 * y - 0.3 * x) + 0.25 * math.cos(1.7 * i),
            ]
            for i, (x, y) in enumerate(self.src)
        ]

    def test_zero_lambda_interpolates(self):
        res = fit(self.src, self.tgt, lam=0.0)
        self.assertLess(max(res.residuals), 1e-9)

    def test_smoothing_trades_fit_for_energy(self):
        r0 = fit(self.src, self.tgt, lam=0.0)
        r1 = fit(self.src, self.tgt, lam=20.0)
        self.assertGreater(r1.rms_residual, r0.rms_residual)
        self.assertGreater(r1.rms_residual, 1e-3)
        self.assertLess(r1.bending_energy, r0.bending_energy)
        n0 = sum(w * w for row in r0.weights for w in row)
        n1 = sum(w * w for row in r1.weights for w in row)
        self.assertLess(n1, n0)

    def test_regularised_solution_matches_reference(self):
        params = reference_fit(self.src, self.tgt, lam=7.5)
        res = fit(self.src, self.tgt, lam=7.5)
        for p in [[0.4, 2.6], [5.5, 5.5], [8.8, 0.2]]:
            got = res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 1e-9)
            self.assertLess(abs(got[1] - ry), 1e-9)
        # Regularised fit does not exactly hit the control points.
        self.assertGreater(max(res.residuals), 1e-4)


class InputValidationTests(unittest.TestCase):
    GOOD = [[0, 0], [1, 0], [0, 1]]

    def test_nan_rejected(self):
        with self.assertRaises(TPSInputError):
            fit([[0, 0], [float("nan"), 0], [0, 1]], self.GOOD)

    def test_infinity_rejected(self):
        with self.assertRaises(TPSInputError):
            fit([[0, 0], [1, float("inf")], [0, 1]], self.GOOD)
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, lam=float("-inf"))

    def test_bool_rejected_as_number(self):
        with self.assertRaises(TPSInputError):
            fit([[0, 0], [True, 0], [0, 1]], self.GOOD)
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, lam=True)

    def test_negative_lambda_rejected(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, lam=-1e-12)

    def test_zero_lambda_accepted(self):
        res = fit(self.GOOD, self.GOOD, lam=0)
        self.assertEqual(res.merged_count, 3)

    def test_length_mismatch(self):
        with self.assertRaises(TPSInputError):
            fit([[0, 0], [1, 0]], self.GOOD)

    def test_empty_points(self):
        with self.assertRaises(TPSInputError):
            fit([], [])

    def test_bad_point_shape(self):
        with self.assertRaises(TPSInputError):
            fit([[0, 0], [1], [0, 1]], self.GOOD)
        with self.assertRaises(TPSInputError):
            fit([[0, 0], "x", [0, 1]], self.GOOD)

    def test_non_numeric_coordinate(self):
        with self.assertRaises(TPSInputError):
            fit([[0, 0], [1, None], [0, 1]], self.GOOD)

    def test_too_many_points(self):
        src = [[float(i), float((i * 3) % 11)] for i in range(MAX_CONTROL_POINTS + 1)]
        with self.assertRaises(TPSInputError):
            fit(src, src)

    def test_boundary_40_points_ok(self):
        src = [[float(i), float((i * 7) % 13)] for i in range(40)]
        res = fit(src, src)
        self.assertEqual(res.merged_count, 40)
        self.assertEqual(res.rank, 3)

    def test_evaluate_rejects_bad_input(self):
        res = fit(self.GOOD, self.GOOD)
        with self.assertRaises(TPSInputError):
            res.warp.evaluate([1.0, float("nan")])
        with self.assertRaises(TPSInputError):
            res.warp.evaluate([1.0])
        with self.assertRaises(TPSInputError):
            res.warp.jacobian([False, 1.0])

    def test_batch_all_or_none(self):
        res = fit(self.GOOD, self.GOOD)
        with self.assertRaises(TPSInputError):
            res.warp.warp_points([[0, 0], [1, 1], [float("inf"), 0]])
        # warp object remains usable after the failed batch call.
        self.assertEqual(res.warp.evaluate([0, 0]), [0.0, 0.0])


class ScaleRobustnessTests(unittest.TestCase):
    """Internal normalisation must make results scale independent.

    Pixel coordinates in medical images are commonly O(1e3..1e6); an
    un-normalised assembly of U = r^2 log r breaks down there.
    """

    def setUp(self):
        base = grid_points(4, 4, scale=1.0)
        self.t0 = [
            [x + 0.4 * math.sin(0.5 * y + 0.2), y + 0.3 * math.cos(0.4 * x - 0.1)]
            for x, y in base
        ]
        self.s0 = base

    def test_pixel_scale_interpolation(self):
        for s in (1000.0, 1.0e6):
            src = [[x * s, y * s] for x, y in self.s0]
            tgt = [[vx * s, vy * s] for vx, vy in self.t0]
            res = fit(src, tgt, lam=0.0)
            # residual relative to the coordinate scale
            self.assertLess(max(res.residuals) / s, 3e-9, f"scale {s}")
            # warp output at scaled probe points is s times the unit output
            r_unit = fit(self.s0, self.t0, lam=0.0)
            for p in ([[0.5, 0.7], [3.3, 2.0]]):
                q = res.warp.evaluate([p[0] * s, p[1] * s])
                q0 = r_unit.warp.evaluate(p)
                self.assertLess(abs(q[0] / s - q0[0]), 1e-9)
                self.assertLess(abs(q[1] / s - q0[1]), 1e-9)

    def test_energy_and_lambda_scaling_law(self):
        # Bending energy is dimensionless under uniform scaling; regularising
        # at lam = lam0 * s^2 reproduces the same normalised problem.
        r_unit = fit(self.s0, self.t0, lam=0.01)
        for s in (1.0, 1000.0):
            src = [[x * s, y * s] for x, y in self.s0]
            tgt = [[vx * s, vy * s] for vx, vy in self.t0]
            res = fit(src, tgt, lam=0.01 * s * s)
            self.assertAlmostEqual(res.bending_energy, r_unit.bending_energy,
                                   delta=1e-10)
            self.assertAlmostEqual(res.rms_residual / s, r_unit.rms_residual,
                                   delta=1e-10)

    def test_affine_exact_at_large_scale(self):
        s = 1.0e5
        # Exact affine map: x' = 2s + 3x - y, y' = -s + 0.5x + 2y
        src = [[0.0, 0.0], [s, 0.0], [0.0, s], [2.0 * s, 3.0 * s]]
        tgt = [[2.0 * s, -s], [5.0 * s, -0.5 * s], [s, s], [5.0 * s, 6.0 * s]]
        res = fit(src, tgt)
        self.assertLess(max(abs(w) for row in res.weights for w in row), 1e-6)
        self.assertLess(max(res.residuals), 1e-4)


class KernelConventionTests(unittest.TestCase):
    def test_U_zero_at_origin(self):
        # A single coincident evaluation must be exact and finite regardless
        # of 0*log(0) convention.
        res = fit([[0, 0], [1, 0], [0, 1]], [[0.5, -0.5], [1, 0], [0, 1]])
        out = res.warp.evaluate([0, 0])
        self.assertTrue(all(math.isfinite(v) for v in out))
        self.assertAlmostEqual(out[0], 0.5, delta=1e-9)
        self.assertAlmostEqual(out[1], -0.5, delta=1e-9)


# ---------------------------------------------------------------------------
# Iteration 2: per-point confidence weights, joint affine + bending solve
# ---------------------------------------------------------------------------

def weighted_problem():
    """Deterministic 4x4 grid: smooth field + fixed jitter (shared data)."""
    src = grid_points(4, 4)
    tgt = [
        [
            x + 0.3 * math.sin(0.6 * x + 0.4 * y) + 0.25 * math.sin(2.3 * i),
            y + 0.3 * math.cos(0.5 * y - 0.3 * x) + 0.25 * math.cos(1.7 * i),
        ]
        for i, (x, y) in enumerate(src)
    ]
    return src, tgt


class WeightedFitReferenceTests(unittest.TestCase):
    """Weighted smoothing path vs the independent from-scratch reference."""

    def setUp(self):
        self.src, self.tgt = weighted_problem()
        # Deterministic, widely spread non-negative weights (no zeros here).
        self.sig = [0.5 + 0.37 * ((i * 13) % 7) for i in range(len(self.src))]
        self.lam = 3.0

    def test_matches_independent_weighted_reference(self):
        res = fit(self.src, self.tgt, lam=self.lam, confidence=self.sig)
        params = reference_fit_weighted(self.src, self.tgt, self.sig, self.lam)
        for p in [[0.4, 2.6], [5.5, 5.5], [8.8, 0.2], [3.0, 6.0], [-1.0, 9.0]]:
            got = res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 1e-9, (p, got, (rx, ry)))
            self.assertLess(abs(got[1] - ry), 1e-9, (p, got, (rx, ry)))

    def test_unit_confidence_matches_default_path(self):
        # Explicit all-one weights must recover the classical system exactly,
        # both interpolating and smoothing.
        for lam in (0.0, 3.0):
            r_default = fit(self.src, self.tgt, lam=lam)
            r_ones = fit(self.src, self.tgt, lam=lam, confidence=[1.0] * 16)
            self.assertIsNone(r_default.confidence)
            self.assertEqual(r_ones.confidence, [1.0] * 16)
            for i in range(16):
                for c in range(2):
                    self.assertEqual(
                        r_default.weights[i][c], r_ones.weights[i][c]
                    )
            self.assertEqual(r_default.affine, r_ones.affine)
            self.assertEqual(r_default.effective_count, r_ones.effective_count)

    def test_stationarity_relations_in_user_units(self):
        # Analytic first-order conditions, evaluated independently of the
        # solver: P^T w = 0 and, in user units as in normalised units,
        # w_i = sigma_i r_i / lam where r_i = v_i - f(c_i).
        res = fit(self.src, self.tgt, lam=self.lam, confidence=self.sig)
        for c in range(2):
            s0 = sum(w[c] for w in res.weights)
            sx = sum(p[0] * w[c] for p, w in zip(self.src, res.weights))
            sy = sum(p[1] * w[c] for p, w in zip(self.src, res.weights))
            self.assertLess(abs(s0), 1e-8)
            self.assertLess(abs(sx), 1e-7)
            self.assertLess(abs(sy), 1e-7)
        for i, (s, t, sigma) in enumerate(
            zip(self.src, self.tgt, self.sig)
        ):
            fx, fy = res.warp.evaluate(s)
            rx, ry = t[0] - fx, t[1] - fy
            self.assertLess(
                abs(res.weights[i][0] - sigma * rx / self.lam), 1e-8, i
            )
            self.assertLess(
                abs(res.weights[i][1] - sigma * ry / self.lam), 1e-8, i
            )

    def test_weight_scaling_pulls_fit_toward_observation(self):
        # Multiplying every confidence by a constant is equivalent to dividing
        # lam by the same constant: the same spline must result.  Expected
        # values again come from the independent reference, not the package.
        k = 5.0
        res = fit(
            self.src, self.tgt, lam=self.lam * k,
            confidence=[sigma * k for sigma in self.sig],
        )
        params = reference_fit_weighted(self.src, self.tgt, self.sig, self.lam)
        for p in [[0.4, 2.6], [5.5, 5.5], [8.8, 0.2]]:
            got = res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 1e-9)
            self.assertLess(abs(got[1] - ry), 1e-9)


class ZeroRegularizationDegeneracyTests(unittest.TestCase):
    """lam = 0 behaviour: exact interpolation; untrusted anchors invalid."""

    def test_zero_lambda_with_unit_weights_interpolates(self):
        src, tgt = weighted_problem()
        res = fit(src, tgt, lam=0.0, confidence=[1.0] * len(src))
        self.assertLess(max(res.residuals), 1e-9)
        self.assertEqual(res.effective_count, 16)

    def test_zero_confidence_at_zero_lambda_rejected(self):
        src, tgt = weighted_problem()
        sig = [1.0] * len(src)
        sig[7] = 0.0
        with self.assertRaises(TPSInputError):
            fit(src, tgt, lam=0.0, confidence=sig)

    def test_tiny_lambda_with_zero_weight_converges_to_trusted_interpolant(self):
        # Degenerate-looking but valid case: one untrusted anchor, lam -> 0.
        # The augmented KKT system must stay solvable and converge to the
        # exact interpolant through the 15 trusted points, built here by the
        # independent reference with lam = 0 and the point removed.
        src, tgt = weighted_problem()
        bad = 7
        sig = [1.0] * len(src)
        sig[bad] = 0.0
        res = fit(src, tgt, lam=1e-9, confidence=sig)
        self.assertEqual(res.merged_count, 16)
        self.assertEqual(res.effective_count, 15)
        # Untrusted anchor contributes no bending coefficient.
        self.assertEqual(res.weights[bad], [0.0, 0.0])
        trusted_src = src[:bad] + src[bad + 1:]
        trusted_tgt = tgt[:bad] + tgt[bad + 1:]
        params = reference_fit(trusted_src, trusted_tgt, 0.0)
        for p in [[0.4, 2.6], [5.5, 5.5], [8.8, 0.2], [3.0, 7.0]]:
            got = res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 5e-7, p)
            self.assertLess(abs(got[1] - ry), 5e-7, p)
        # Trusted points are hit (almost) exactly; the untrusted one needn't be.
        for i in range(16):
            if i == bad:
                continue
            self.assertLess(res.residuals[i], 1e-6, i)
        self.assertGreater(res.residuals[bad], 1e-3)

    def test_all_zero_confidence_rejected(self):
        src, tgt = weighted_problem()
        with self.assertRaises(DegeneratePointsError):
            fit(src, tgt, lam=1.0, confidence=[0.0] * len(src))

    def test_zero_weight_offline_cannot_fix_collinear_active_set(self):
        # Off-line information carried only by a zero-confidence point does
        # not affinely span the plane: the active set itself must rank 3.
        src = [[0, 0], [1, 0], [2, 0], [1, 1]]
        tgt = [[0, 0], [1, 0], [2, 0], [1, 1]]
        sig = [1.0, 1.0, 1.0, 0.0]
        with self.assertRaises(DegeneratePointsError):
            fit(src, tgt, lam=1.0, confidence=sig)

    def test_zero_weight_extra_point_is_ignored(self):
        # Same geometry, but the active three are non-collinear; the extra
        # zero-confidence anchor anywhere must not perturb the fit.
        src = [[0, 0], [1, 0], [0, 1], [5, 5]]
        tgt = [[0, 0], [1, 0], [0, 1], [-9, -9]]  # wild target at index 3
        sig = [1.0, 1.0, 1.0, 0.0]
        res = fit(src, tgt, lam=1e-6, confidence=sig)
        self.assertEqual(res.effective_count, 3)
        self.assertEqual(res.weights[3], [0.0, 0.0])
        params = reference_fit(src[:3], tgt[:3], 0.0)
        for p in [[0.3, 0.3], [2.0, -1.0], [-1.0, 2.0], [5.0, 5.0]]:
            got = res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 1e-6, p)
            self.assertLess(abs(got[1] - ry), 1e-6, p)


class LowWeightOutlierTests(unittest.TestCase):
    """A biased anchor must be down-weighted away (hand-built scenario)."""

    def setUp(self):
        # Identity transform on a 3x3 grid; the corner anchor claims a gross
        # displacement (+4, -3) that no genuine point supports.
        self.src = [
            [x, y] for y in (0.0, 2.0, 4.0) for x in (0.0, 2.0, 4.0)
        ]
        self.tgt = [[x, y] for x, y in self.src]
        self.outlier = 8  # point (4, 4)
        self.tgt[self.outlier] = [8.0, 1.0]

    def test_zero_weight_outlier_equals_trusted_only_fit(self):
        sig = [1.0] * 9
        sig[self.outlier] = 0.0
        res = fit(self.src, self.tgt, lam=1e-7, confidence=sig)
        # Independent interpolant through the 8 trusted identity anchors.
        params = reference_fit(self.src[:8], self.tgt[:8], 0.0)
        for p in [[0.5, 0.5], [3.0, 3.0], [4.0, 4.0], [6.0, 6.0]]:
            got = res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 2e-6, p)
            self.assertLess(abs(got[1] - ry), 2e-6, p)
        # Identity at the outlier's location, not its claimed (8, 1).
        q = res.warp.evaluate(self.src[self.outlier])
        self.assertLess(abs(q[0] - 4.0), 1e-5)
        self.assertLess(abs(q[1] - 4.0), 1e-5)

    def test_low_weight_outlier_against_reference_and_trusted_points(self):
        lam = 0.5

        def fit_with(w):
            sig = [1.0] * 9
            sig[self.outlier] = w
            return fit(self.src, self.tgt, lam=lam, confidence=sig)

        sig = [1.0] * 9
        sig[self.outlier] = 0.01
        res = fit_with(0.01)
        # Coefficients/warp come from the independent weighted reference.
        params = reference_fit_weighted(self.src, self.tgt, sig, lam)
        for p in [[0.5, 0.5], [3.0, 3.0], [4.0, 4.0]]:
            got = res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 1e-9, p)
            self.assertLess(abs(got[1] - ry), 1e-9, p)
        # Down-weighting the biased anchor must clearly reduce how far the
        # trusted anchors and the interior are dragged, relative to giving it
        # the same weight as a genuine point (thresholds well below the
        # hand-computed actual margins: 0.036 vs 0.216 and 4.21 vs 0.25).
        full = fit_with(1.0)
        self.assertLess(
            max(res.residuals[i] for i in range(8)),
            0.5 * max(full.residuals[i] for i in range(8)),
        )
        self.assertGreater(res.residuals[self.outlier], 1.0)
        self.assertLess(full.residuals[self.outlier], 1.0)
        q_low = res.warp.evaluate([3.0, 3.0])
        q_full = full.warp.evaluate([3.0, 3.0])
        self.assertLess(math.hypot(q_low[0] - 3.0, q_low[1] - 3.0), 0.2)
        self.assertGreater(math.hypot(q_full[0] - 3.0, q_full[1] - 3.0), 0.5)

    def test_influence_grows_monotonically_with_weight(self):
        # Small enumeration (package-to-package *relationship* check, used
        # only as a supplement to the reference checks above): as the
        # outlier's confidence rises from nothing, the warp at its location
        # must move monotonically toward its claimed target and the trusted
        # anchors' residuals must not decrease.
        lam = 0.5
        weights_seq = [0.0, 0.01, 0.1, 1.0, 10.0]
        dist_to_claim = []
        trusted_rms = []
        for w in weights_seq:
            sig = [1.0] * 9
            sig[self.outlier] = w
            res = fit(self.src, self.tgt, lam=lam, confidence=sig)
            q = res.warp.evaluate(self.src[self.outlier])
            dist_to_claim.append(math.hypot(q[0] - 8.0, q[1] - 1.0))
            trusted_rms.append(
                math.sqrt(
                    sum(res.residuals[i] ** 2 for i in range(8)) / 8.0
                )
            )
        for a, b in zip(dist_to_claim, dist_to_claim[1:]):
            self.assertLessEqual(b, a + 1e-12)
        for a, b in zip(trusted_rms, trusted_rms[1:]):
            self.assertGreaterEqual(b, a - 1e-12)
        self.assertGreater(dist_to_claim[0] - dist_to_claim[-1], 1.0)


class TranslationReproductionTests(unittest.TestCase):
    """A pure translation must be reproduced exactly: w = 0, affine shift.

    Expected values are the closed-form answer (linear part identity), not
    anything produced by the package.  Holds for every weight pattern and
    every lam >= 0 because the translation sets the objective to its global
    minimum zero.
    """

    TX, TY = 1.75, -2.5

    def _check(self, res, n):
        for i in range(n):
            self.assertLess(abs(res.weights[i][0]), 1e-9, i)
            self.assertLess(abs(res.weights[i][1]), 1e-9, i)
        a = res.affine
        self.assertAlmostEqual(a[0][0], self.TX, delta=1e-9)
        self.assertAlmostEqual(a[0][1], self.TY, delta=1e-9)
        self.assertAlmostEqual(a[1][0], 1.0, delta=1e-9)
        self.assertAlmostEqual(a[1][1], 0.0, delta=1e-9)
        self.assertAlmostEqual(a[2][0], 0.0, delta=1e-9)
        self.assertAlmostEqual(a[2][1], 1.0, delta=1e-9)
        self.assertLess(max(res.residuals), 1e-9)
        self.assertLess(abs(res.bending_energy), 1e-9)
        for p in [[0.3, -1.1], [4.7, 5.9], [10.0, 0.0]]:
            q = res.warp.evaluate(p)
            self.assertLess(abs(q[0] - (p[0] + self.TX)), 1e-9)
            self.assertLess(abs(q[1] - (p[1] + self.TY)), 1e-9)

    def test_translation_lam_zero(self):
        src = grid_points(4, 4)
        tgt = [[x + self.TX, y + self.TY] for x, y in src]
        self._check(fit(src, tgt, lam=0.0), 16)

    def test_translation_uneven_weights_and_regularisation(self):
        src = grid_points(4, 4)
        tgt = [[x + self.TX, y + self.TY] for x, y in src]
        # Includes a zero-confidence anchor: still exact on the rest.
        sig = [0.5 + 0.37 * ((i * 13) % 7) for i in range(16)]
        sig[5] = 0.0
        res = fit(src, tgt, lam=4.0, confidence=sig)
        self.assertEqual(res.effective_count, 15)
        self._check(res, 16)

    def test_translation_pixel_scale(self):
        s = 1000.0
        src = [[s * i, s * j] for j in range(4) for i in range(4)]
        tgt = [[x + s * self.TX, y + s * self.TY] for x, y in src]
        sig = [1.0 + 0.1 * (i % 3) for i in range(16)]
        res = fit(src, tgt, lam=10.0 * s * s, confidence=sig)
        self.assertLess(max(abs(w) for row in res.weights for w in row), 1e-7)
        self.assertAlmostEqual(res.affine[0][0], s * self.TX, delta=1e-7)
        self.assertLess(max(res.residuals) / s, 2e-9)


class ConfidenceDuplicateTests(unittest.TestCase):
    """Duplicate anchors: compatibility first, then precision accumulation."""

    def test_identical_duplicate_weights_add(self):
        src = [[0, 0], [1, 0], [0, 1]]
        tgt = [[0.2, -0.2], [1, 0], [0, 1]]
        sig = [0.3, 1.0, 1.0]
        res = fit(src * 2, tgt * 2, lam=2.0, confidence=sig * 2)
        self.assertEqual(res.merged_count, 3)
        self.assertEqual(res.confidence[0], 0.6)
        self.assertEqual(res.confidence[1], 2.0)
        self.assertEqual(res.confidence[2], 2.0)
        # Equivalent to one anchor carrying the summed precision.
        params = reference_fit_weighted(
            src, tgt, [0.6, 2.0, 2.0], 2.0
        )
        for p in [[0.3, 0.3], [-0.5, 1.0]]:
            got = res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 1e-9, p)
            self.assertLess(abs(got[1] - ry), 1e-9, p)

    def test_conflicting_duplicate_rejected_even_when_untrusted(self):
        # Compatibility is judged before weighting: an anchor claiming two
        # different targets is contradictory data regardless of confidence.
        src = [[0, 0], [1, 0], [0, 1], [1, 0]]
        tgt = [[0, 0], [2, 0], [0, 2], [2.5, 0]]
        with self.assertRaises(TPSInputError):
            fit(src, tgt, lam=1.0, confidence=[1, 1, 1, 0])

    def test_identical_duplicate_default_path_still_merges(self):
        src = [[0, 0], [1, 0], [0, 1], [1, 0]]
        tgt = [[0, 0], [2, 0], [0, 2], [2, 0]]
        res = fit(src, tgt, lam=2.0)
        self.assertEqual(res.merged_count, 3)
        self.assertIsNone(res.confidence)
        self.assertEqual(res.effective_count, 3)


class ConfidenceValidationTests(unittest.TestCase):
    GOOD = [[0, 0], [1, 0], [0, 1]]

    def test_confidence_wrong_length(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, 1.0])

    def test_confidence_negative(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, -0.1, 1.0])

    def test_confidence_nan_inf_rejected(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, float("nan"), 1.0])
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, float("inf"), 1.0])

    def test_confidence_bool_rejected(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[True, 1.0, 1.0])

    def test_confidence_bad_container(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence="111")
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, None, 1.0])

    def test_reported_fields_shape(self):
        res = fit(
            [[0, 0], [1, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 0], [0, 1], [1, 1]],
            lam=1.5,
            confidence=[2.0, 1.0, 0.5, 0.0],
        )
        self.assertEqual(res.merged_count, 4)
        self.assertEqual(res.effective_count, 3)
        self.assertEqual(res.confidence, [2.0, 1.0, 0.5, 0.0])
        self.assertEqual(len(res.weights), 4)
        self.assertEqual(len(res.residuals), 4)
        self.assertEqual(res.weights[3], [0.0, 0.0])
        # Failed/valid calls leave the warp usable (no half-updated state).
        q0 = res.warp.evaluate([0, 0])
        self.assertLess(math.hypot(q0[0], q0[1]), 1e-12)
        with self.assertRaises(TPSInputError):
            fit(
                [[0, 0], [1, 0], [0, 1], [1, 1]],
                [[0, 0], [1, 0], [0, 1], [1, 1]],
                lam=1.5,
                confidence=[2.0, 1.0, 0.5, -1.0],
            )
        q1 = res.warp.evaluate([1, 1])
        self.assertLess(math.hypot(q1[0] - 1.0, q1[1] - 1.0), 1e-12)


if __name__ == "__main__":
    unittest.main()
