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


class ConfidenceBasicTests(unittest.TestCase):
    """Per-point confidence weights: normal behaviour and analytic relations."""

    def setUp(self):
        self.src = [[0, 0], [2, 0], [0, 2], [2, 2], [1, 1], [3, 1]]
        self.tgt = [
            [
                x + 0.3 * math.sin(0.5 * y + 0.2 * i),
                y + 0.25 * math.cos(0.4 * x - 0.1 * i),
            ]
            for i, (x, y) in enumerate(self.src)
        ]
        self.sig = [1.0, 0.5, 2.0, 1.0, 0.25, 3.0]

    def test_unit_confidence_equals_unweighted(self):
        a = fit(self.src, self.tgt, lam=0.7)
        b = fit(self.src, self.tgt, lam=0.7, confidence=[1.0] * len(self.src))
        self.assertEqual(b.confidence, [1.0] * len(self.src))
        self.assertEqual(a.weights, b.weights)
        self.assertEqual(a.affine, b.affine)
        for p in [[0.3, 1.1], [2.5, 0.4]]:
            self.assertEqual(a.warp.evaluate(p), b.warp.evaluate(p))

    def test_none_confidence_default(self):
        res = fit(self.src, self.tgt, lam=0.7, confidence=None)
        self.assertEqual(res.confidence, [1.0] * len(self.src))

    def test_weighted_matches_independent_reference(self):
        res = fit(self.src, self.tgt, lam=0.7, confidence=self.sig)
        params = reference_fit_weighted(self.src, self.tgt, 0.7, self.sig)
        for p in [[0.3, 1.1], [1.7, 1.7], [2.5, 0.4], [-0.5, 2.2]]:
            got = res.warp.evaluate(p)
            rx, ry = reference_warp(params, p[0], p[1])
            self.assertLess(abs(got[0] - rx), 1e-9, (p, got[0], rx))
            self.assertLess(abs(got[1] - ry), 1e-9, (p, got[1], ry))

    def test_global_confidence_rescale_equivalence(self):
        # Analytic scaling law: multiplying every sigma by c is equivalent
        # to dividing lambda by c (objective multiplied by 1/c).
        c = 4.0
        a = fit(self.src, self.tgt, lam=0.2, confidence=[c] * len(self.src))
        b = fit(self.src, self.tgt, lam=0.2 / c)
        for p in [[0.3, 1.1], [1.0, 1.0], [2.5, 0.4]]:
            qa, qb = a.warp.evaluate(p), b.warp.evaluate(p)
            self.assertLess(abs(qa[0] - qb[0]), 1e-12)
            self.assertLess(abs(qa[1] - qb[1]), 1e-12)

    def test_stationarity_residual_relation(self):
        # Joint stationarity gives, per coordinate and point with sigma > 0:
        # v_i - f(c_i) = (lam / sigma_i) * w_i  (exact analytic identity,
        # i.e. f(c_i) - v_i = -(lam / sigma_i) * w_i).
        lam = 0.37
        res = fit(self.src, self.tgt, lam=lam, confidence=self.sig)
        for i, (s, t) in enumerate(zip(self.src, self.tgt)):
            q = res.warp.evaluate(s)
            for c in range(2):
                expected = -(lam / self.sig[i]) * res.weights[i][c]
                self.assertLess(
                    abs((q[c] - t[c]) - expected), 1e-9,
                    f"i={i} coord={c}: {q[c] - t[c]} vs {expected}",
                )

    def test_ptw_constraint_with_weights(self):
        res = fit(self.src, self.tgt, lam=0.7, confidence=self.sig)
        for c in range(2):
            s0 = sum(w[c] for w in res.weights)
            sx = sum(x * w[c] for (x, y), w in zip(self.src, res.weights))
            sy = sum(y * w[c] for (x, y), w in zip(self.src, res.weights))
            self.assertLess(abs(s0), 1e-8)
            self.assertLess(abs(sx), 1e-8)
            self.assertLess(abs(sy), 1e-8)

    def test_duplicate_identical_weights_accumulate(self):
        # Point (1,0) appears twice with the same target and weights 0.4+0.6.
        src = [[0, 0], [1, 0], [0, 1], [1, 0]]
        tgt = [[0, 0], [1, 0], [0, 1], [1, 0]]
        sig = [1.0, 0.4, 2.0, 0.6]
        res = fit(src, tgt, lam=0.5, confidence=sig)
        self.assertEqual(res.merged_count, 3)
        self.assertEqual(res.confidence, [1.0, 1.0, 2.0])
        equiv = fit([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]],
                    lam=0.5, confidence=[1.0, 1.0, 2.0])
        for p in [[0.3, 0.7], [0.5, 0.5], [1.0, 1.0]]:
            qa, qb = res.warp.evaluate(p), equiv.warp.evaluate(p)
            self.assertLess(abs(qa[0] - qb[0]), 1e-12)
            self.assertLess(abs(qa[1] - qb[1]), 1e-12)


class LowWeightOutlierTests(unittest.TestCase):
    """A low-confidence gross outlier must barely bias the spline."""

    # Four corners map by the identity map (an exact affine map, hence
    # bending energy 0 and zero residual on their own); the centre point
    # claims a wild y' = 2.0 displacement.
    SRC = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]
    TGT = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 2.0]]

    def fit(self, sigma_out, lam=0.1):
        return fit(self.SRC, self.TGT, lam=lam,
                   confidence=[1.0, 1.0, 1.0, 1.0, sigma_out])

    def test_full_weight_outlier_biases(self):
        res = self.fit(1.0)
        # Trusted corners are pulled away from identity.
        self.assertGreater(max(res.residuals[:4]), 1e-2)
        # The centre is fitted between 0.5 and 2.0.
        self.assertTrue(0.5 < res.warp.evaluate(self.SRC[4])[1] < 2.0)

    def test_low_weight_shrinks_bias_monotonically(self):
        r_hi = self.fit(1.0)
        r_lo = self.fit(1e-3)
        y_hi = r_hi.warp.evaluate(self.SRC[4])[1]
        y_lo = r_lo.warp.evaluate(self.SRC[4])[1]
        self.assertGreater(max(r_hi.residuals[:4]), max(r_lo.residuals[:4]))
        self.assertGreater(y_hi, y_lo)
        self.assertLess(max(r_lo.residuals[:4]), 1e-2)
        self.assertLess(abs(y_lo - 0.5), 1e-2)

    def test_zero_confidence_equals_deleting_point(self):
        # sigma = 0 at lam > 0: the outlier equation drops out completely.
        res = self.fit(0.0)
        ref = fit(self.SRC[:4], self.TGT[:4], lam=0.1)
        for p in [[0.3, 0.7], [0.5, 0.5], [-0.5, 1.3], [1.2, 0.2]]:
            qa, qb = res.warp.evaluate(p), ref.warp.evaluate(p)
            self.assertAlmostEqual(qa[0], qb[0], delta=1e-12)
            self.assertAlmostEqual(qa[1], qb[1], delta=1e-12)
        # The ignored point has a zero kernel coefficient ...
        self.assertEqual(res.weights[4], [0.0, 0.0])
        # ... the four trusted identity corners are hit exactly (affine
        # solution, bending energy 0) ...
        self.assertLess(max(res.residuals[:4]), 1e-12)
        self.assertLess(abs(res.bending_energy), 1e-12)
        # ... and the analytically expected residual at the ignored point is
        # ||(0.5, 0.5) - (0.5, 2.0)|| = 1.5.
        self.assertAlmostEqual(res.residuals[4], 1.5, delta=1e-12)

    def test_tiny_positive_weight_limit(self):
        r = self.fit(1e-8)
        ref = self.fit(0.0)
        for p in [[0.5, 0.5], [0.3, 0.7]]:
            qa, qb = r.warp.evaluate(p), ref.warp.evaluate(p)
            self.assertLess(abs(qa[0] - qb[0]), 1e-6)
            self.assertLess(abs(qa[1] - qb[1]), 1e-6)


class ZeroRegularizationConfidenceTests(unittest.TestCase):
    """At lam = 0 confidence is degenerate: exact interpolation unchanged."""

    SRC = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]
    TGT = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 2.0]]

    def test_lam_zero_weights_do_not_matter(self):
        sigs = [
            [1.0] * 5,
            [0.0, 1.0, 0.0, 1.0, 0.25],
            [0.0] * 5,
            [1e3, 1e-9, 3.7, 0.0, 42.0],
        ]
        base = fit(self.SRC, self.TGT, lam=0.0)
        for sig in sigs:
            res = fit(self.SRC, self.TGT, lam=0.0, confidence=sig)
            self.assertLess(max(res.residuals), 1e-9)
            for i in range(5):
                for c in range(2):
                    self.assertEqual(res.weights[i][c], base.weights[i][c])
            self.assertEqual(res.affine, base.affine)

    def test_lam_zero_zero_confidence_but_still_rank_checked(self):
        # Weights are ignored at lam=0, but a genuinely collinear geometry
        # must still be rejected.
        with self.assertRaises(DegeneratePointsError):
            fit([[0, 0], [1, 1], [2, 2], [3, 3]],
                [[0, 0], [1, 0], [2, 0], [3, 0]],
                lam=0.0, confidence=[0.0, 1.0, 0.0, 1.0])

    def test_lam_positive_needs_three_positive_points(self):
        with self.assertRaises(DegeneratePointsError):
            fit(self.SRC, self.TGT, lam=0.1,
                confidence=[1.0, 0.0, 0.0, 0.0, 1.0])
        with self.assertRaises(DegeneratePointsError):
            fit(self.SRC, self.TGT, lam=0.1, confidence=[0.0] * 5)

    def test_lam_positive_positive_points_collinear_rejected(self):
        # Three positive points on a line plus two zero-weight points off it.
        src = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]]
        tgt = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]]
        with self.assertRaises(DegeneratePointsError):
            fit(src, tgt, lam=0.1, confidence=[1.0, 1.0, 1.0, 0.0, 0.0])
        # ... but a non-collinear positive subset fits, and the zero-weight
        # points keep zero kernel coefficients.
        res = fit(src, tgt, lam=0.1, confidence=[0.0, 0.0, 1.0, 1.0, 1.0])
        self.assertEqual(res.weights[0], [0.0, 0.0])
        self.assertEqual(res.weights[1], [0.0, 0.0])
        self.assertEqual(res.rank, 3)


class ConfidenceTranslationReproTests(unittest.TestCase):
    """Pure translation with arbitrary non-negative weights, lam > 0."""

    TX, TY = 2.5, -1.0

    def setUp(self):
        self.src = [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]
        self.tgt = [[x + self.TX, y + self.TY] for x, y in self.src]
        self.sig = [1.0, 2.0, 0.5, 3.0, 0.1]
        self.res = fit(self.src, self.tgt, lam=0.3, confidence=self.sig)

    def test_pure_translation_affine(self):
        a = self.res.affine
        self.assertAlmostEqual(a[0][0], self.TX, delta=1e-10)
        self.assertAlmostEqual(a[0][1], self.TY, delta=1e-10)
        self.assertAlmostEqual(a[1][0], 1.0, delta=1e-10)
        self.assertAlmostEqual(a[1][1], 0.0, delta=1e-10)
        self.assertAlmostEqual(a[2][0], 0.0, delta=1e-10)
        self.assertAlmostEqual(a[2][1], 1.0, delta=1e-10)

    def test_zero_bending_and_residuals(self):
        self.assertLess(max(abs(w) for row in self.res.weights for w in row),
                        1e-10)
        self.assertLess(abs(self.res.bending_energy), 1e-12)
        self.assertLess(max(self.res.residuals), 1e-10)

    def test_translation_reproduced_everywhere(self):
        for p in [[0.37, 0.63], [-2.5, 4.1], [3.0, -1.0]]:
            q = self.res.warp.evaluate(p)
            self.assertAlmostEqual(q[0], p[0] + self.TX, delta=1e-10)
            self.assertAlmostEqual(q[1], p[1] + self.TY, delta=1e-10)
        J = self.res.warp.jacobian([0.37, 0.63])
        self.assertAlmostEqual(J[0][0], 1.0, delta=1e-12)
        self.assertAlmostEqual(J[0][1], 0.0, delta=1e-12)
        self.assertAlmostEqual(J[1][0], 0.0, delta=1e-12)
        self.assertAlmostEqual(J[1][1], 1.0, delta=1e-12)


class ConfidenceValidationTests(unittest.TestCase):
    GOOD = [[0, 0], [1, 0], [0, 1]]
    SIG = [1.0, 1.0, 1.0]

    def test_wrong_length_rejected(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, 1.0])
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, 1.0, 1.0, 1.0])

    def test_scalar_confidence_rejected(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=1.0)
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence="111")

    def test_negative_weight_rejected(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, -0.01, 1.0])

    def test_nonfinite_weight_rejected(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, float("nan"), 1.0])
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[float("inf"), 1.0, 1.0])

    def test_bool_weight_rejected(self):
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[True, 1.0, 1.0])
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[0.0, False, 1.0])

    def test_conflicting_duplicate_still_rejected_with_weights(self):
        src = [[0, 0], [1, 0], [0, 1], [1, 0]]
        tgt = [[0, 0], [2, 0], [0, 2], [2.5, 0]]
        with self.assertRaises(TPSInputError):
            fit(src, tgt, confidence=[1.0, 0.5, 1.0, 0.5])
        # Contradictory observations at the same location are incompatible
        # at any weight, even when the conflicting copy carries zero weight.
        with self.assertRaises(TPSInputError):
            fit(src, tgt, confidence=[1.0, 1.0, 1.0, 0.0])

    def test_failed_fit_leaves_no_result_and_warp_stays_usable(self):
        # A failing fit raises before any object is handed out.
        with self.assertRaises(TPSInputError):
            fit(self.GOOD, self.GOOD, confidence=[1.0, -1.0, 1.0])
        # A warp from a successful weighted fit survives a bad batch call.
        res = fit(self.GOOD, self.GOOD, confidence=self.SIG)
        with self.assertRaises(TPSInputError):
            res.warp.warp_points([[0, 0], [float("nan"), 0], [0, 1]])
        self.assertEqual(res.warp.evaluate([0, 0]), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
