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

from _reference import reference_fit, reference_warp, finite_diff_jacobian, kernel


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


def point_major_covariance(blocks, cross_blocks=None):
    """Build a 2n x 2n covariance in [x0,y0,x1,y1,...] order."""
    n = len(blocks)
    mat = [[0.0] * (2 * n) for _ in range(2 * n)]
    for i, block in enumerate(blocks):
        for a in range(2):
            for b in range(2):
                mat[2 * i + a][2 * i + b] = block[a][b]
    if cross_blocks:
        for i, j, block in cross_blocks:
            for a in range(2):
                for b in range(2):
                    value = float(block[a][b])
                    mat[2 * i + a][2 * j + b] = value
                    mat[2 * j + b][2 * i + a] = value
    return mat


class CovariancePropagationTests(unittest.TestCase):
    def setUp(self):
        self.sources = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        self.targets = [list(p) for p in self.sources]
        self.result = fit(self.sources, self.targets, lam=0.0)
        self.point = [4.0, 5.0]

    def test_affine_degenerate_three_points_by_hand(self):
        # With three non-collinear controls TPS has no bending part.  The
        # affine cardinal weights at (4, 5) are (-8, 4, 5).
        prediction = self.result.predict(
            self.point,
            point_major_covariance([
                [[0.03, 0.0], [0.0, 0.07]],
                [[0.03, 0.0], [0.0, 0.07]],
                [[0.03, 0.0], [0.0, 0.07]],
            ]),
        )
        expected_h = [-8.0, 4.0, 5.0]
        for block, got in zip(expected_h, prediction.influence):
            self.assertAlmostEqual(got[0][0], block, delta=1e-10)
            self.assertAlmostEqual(got[1][1], block, delta=1e-10)
            self.assertAlmostEqual(got[0][1], 0.0, delta=1e-12)
            self.assertAlmostEqual(got[1][0], 0.0, delta=1e-12)

        variance_x = sum(h * h for h in expected_h) * 0.03
        variance_y = sum(h * h for h in expected_h) * 0.07
        self.assertEqual(prediction.mean, [4.0, 5.0])
        self.assertAlmostEqual(prediction.covariance[0][0], variance_x, delta=1e-9)
        self.assertAlmostEqual(prediction.covariance[1][1], variance_y, delta=1e-9)
        self.assertAlmostEqual(prediction.covariance[0][1], 0.0, delta=1e-12)

    def test_fully_correlated_translation_error(self):
        # Every target has exactly the same random translation vector.  A TPS
        # reproduces it as a pure translation at every query point.
        translation_block = [[0.09, 0.06], [0.06, 0.04]]
        covariance = point_major_covariance(
            [translation_block for _ in self.sources],
            [
                (0, 1, translation_block),
                (0, 2, translation_block),
                (1, 2, translation_block),
            ],
        )
        prediction = self.result.predict(self.point, covariance)
        self.assertEqual(sum(h[0][0] for h in prediction.influence), 1.0)
        self.assertEqual(sum(h[1][1] for h in prediction.influence), 1.0)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    prediction.covariance[i][j],
                    translation_block[i][j],
                    delta=1e-10,
                )

    def test_independent_noise_and_cross_coordinate_correlation(self):
        blocks = [
            [[0.10, 0.02], [0.02, 0.20]],
            [[0.30, -0.05], [-0.05, 0.40]],
            [[0.50, 0.08], [0.08, 0.60]],
        ]
        prediction = self.result.predict(
            self.point, point_major_covariance(blocks)
        )
        h = [-8.0, 4.0, 5.0]
        expected = [[0.0, 0.0], [0.0, 0.0]]
        for i, hi in enumerate(h):
            expected[0][0] += hi * blocks[i][0][0] * hi
            expected[0][1] += hi * blocks[i][0][1] * hi
            expected[1][1] += hi * blocks[i][1][1] * hi
        expected[1][0] = expected[0][1]
        self.assertTrue(all(math.isfinite(v) for row in prediction.covariance for v in row))
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    prediction.covariance[i][j], expected[i][j], delta=1e-9
                )

    def test_cross_control_correlation_uses_full_matrix(self):
        n = len(self.sources)
        covariance = point_major_covariance(
            [[[1.0, 0.25], [0.25, 1.0]] for _ in range(n)],
            [
                (0, 1, [[0.50, -0.20], [0.10, 0.40]]),
                (0, 2, [[-0.30, 0.15], [0.20, 0.10]]),
                (1, 2, [[0.20, 0.30], [-0.10, -0.40]]),
            ],
        )
        prediction = self.result.predict(self.point, covariance)
        h = [-8.0, 4.0, 5.0]
        expected = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(n):
            for j in range(n):
                for a in range(2):
                    for b in range(2):
                        weight_i = h[i]
                        weight_j = h[j]
                        expected[a][b] += (
                            weight_i * covariance[2 * i + a][2 * j + b] * weight_j
                        )
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    prediction.covariance[i][j], expected[i][j], delta=1e-9
                )

    def test_fixed_deterministic_perturbations_match_linear_model(self):
        perturbations = [
            [[0.10, -0.20], [-0.05, 0.15], [0.30, 0.05]],
            [[-0.20, 0.10], [0.25, -0.05], [0.10, 0.40]],
        ]
        for delta in perturbations:
            vector = [value for point_delta in delta for value in point_delta]
            covariance = [[vector[i] * vector[j] for j in range(6)] for i in range(6)]
            prediction = self.result.predict(self.point, covariance)

            shifted = [
                [base[0] + d[0], base[1] + d[1]]
                for base, d in zip(self.targets, delta)
            ]
            shifted_prediction = fit(self.sources, shifted).warp.evaluate(self.point)
            expected_delta = [
                sum(prediction.influence[i][a][a] * delta[i][a] for i in range(3))
                for a in range(2)
            ]
            for a in range(2):
                self.assertAlmostEqual(
                    shifted_prediction[a] - self.point[a],
                    expected_delta[a],
                    delta=1e-10,
                )
                self.assertAlmostEqual(
                    prediction.covariance[a][a],
                    expected_delta[a] * expected_delta[a],
                    delta=1e-11,
                )
            self.assertAlmostEqual(
                prediction.covariance[0][1],
                expected_delta[0] * expected_delta[1],
                delta=1e-11,
            )

    def test_zero_covariance_is_allowed_and_means_no_target_uncertainty(self):
        prediction = self.result.predict(
            self.point, [[0.0] * 6 for _ in range(6)]
        )
        self.assertEqual(prediction.covariance, [[0.0, 0.0], [0.0, 0.0]])

    def test_invalid_covariance_rejected_before_propagation(self):
        good_zero = [[0.0] * 6 for _ in range(6)]
        bad_cases = [
            [[0.0] * 6 for _ in range(5)],
            [[float("nan")] * 6 for _ in range(6)],
            point_major_covariance([
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[-1.0, 0.0], [0.0, 1.0]],
            ]),
        ]
        for bad in bad_cases:
            with self.assertRaises(TPSInputError):
                self.result.predict(self.point, bad)

        asymmetric = point_major_covariance([
            [[1.0, 0.0], [0.0, 1.0]] for _ in range(3)
        ])
        asymmetric[0][1] = 0.5
        with self.assertRaises(TPSInputError):
            self.result.predict(self.point, asymmetric)

        with self.assertRaises(TPSInputError):
            self.result.predict([float("nan"), 0.0], good_zero)

    def test_non_affine_design_reuses_fixed_factor_and_interpolating_weights(self):
        sources = grid_points(4, 4)
        targets = [
            [x + 0.2 * math.sin(y), y + 0.1 * math.cos(x)]
            for x, y in sources
        ]
        result = fit(sources, targets)
        prediction = result.predict(sources[5], [[0.0] * 32 for _ in range(32)])
        # At a control point, interpolation makes its influence a 2-D identity
        # and all other control-point influences vanish.
        for i, block in enumerate(prediction.influence):
            expected_diagonal = 1.0 if i == 5 else 0.0
            self.assertAlmostEqual(block[0][0], expected_diagonal, delta=1e-8)
            self.assertAlmostEqual(block[1][1], expected_diagonal, delta=1e-8)
        self.assertIsNotNone(result._fixed_design[0])


if __name__ == "__main__":
    unittest.main()
