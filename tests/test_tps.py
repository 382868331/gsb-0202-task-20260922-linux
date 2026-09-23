"""Unit tests for the pure-Python 2D thin-plate spline module (tps.py).

References for expected values are analytic (affine map applied by hand,
kernel values, finite differences of the fitted function as required by
the task); no third-party TPS implementation is used.
"""

import math
import unittest

from tps import (
    MAX_CONTROL_POINTS,
    TPS,
    TPSDegenerateError,
    TPSInputError,
    fit_tps,
    kernel,
)


def make_points(n):
    """Deterministic, non-collinear, well-scattered control points."""
    pts = []
    for i in range(n):
        x = math.cos(i * 2.399963) * (1.0 + 0.31 * i)
        y = math.sin(i * 1.7) * (0.5 + 0.23 * i)
        pts.append((x, y))
    return pts


# A fixed affine map used as the analytic reference.
AFFINE = ((1.5, 0.8, -0.3), (-2.0, 0.1, 1.2))


def affine_ref(p):
    x, y = p
    return (AFFINE[0][0] + AFFINE[0][1] * x + AFFINE[0][2] * y,
            AFFINE[1][0] + AFFINE[1][1] * x + AFFINE[1][2] * y)


def bumpy_ref(p):
    """Non-affine smooth displacement field (deterministic)."""
    x, y = p
    return (x + 0.35 * math.sin(1.3 * y) + 0.1 * x * y,
            y + 0.25 * math.cos(0.9 * x) - 0.05 * x * x)


class TestKernel(unittest.TestCase):
    def test_kernel_values(self):
        self.assertEqual(kernel(0.0), 0.0)
        self.assertEqual(kernel(1.0), 0.0)  # 1^2 * log(1) = 0
        self.assertAlmostEqual(kernel(math.e), math.e * math.e, places=12)
        self.assertAlmostEqual(kernel(2.0), 4.0 * math.log(2.0), places=12)

    def test_kernel_rejects_bad_input(self):
        for bad in (float("nan"), float("inf"), -1.0, True, "1"):
            with self.assertRaises(TPSInputError, msg=repr(bad)):
                kernel(bad)


class TestAffineExactness(unittest.TestCase):
    """Pure affine targets must give w = 0 and reproduce the affine map."""

    def setUp(self):
        self.src = make_points(9)
        self.tgt = [affine_ref(p) for p in self.src]
        self.tps = fit_tps(self.src, self.tgt, lambda_=0.0)

    def test_weights_vanish(self):
        for w in self.tps.weights:
            for wi in w:
                self.assertAlmostEqual(wi, 0.0, delta=1e-9)

    def test_bending_energy_vanishes(self):
        self.assertAlmostEqual(self.tps.bending_energy, 0.0, delta=1e-15)

    def test_affine_coefficients_recovered(self):
        for got, want in zip(self.tps.affine, AFFINE):
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, delta=1e-8)

    def test_warp_matches_analytic_affine_at_fresh_points(self):
        probe = [(0.13, -0.42), (2.5, 1.75), (-1.1, 0.9), (0.0, 0.0)]
        for got, p in zip(self.tps.warp(probe), probe):
            want = affine_ref(p)  # analytic reference, not the core
            self.assertAlmostEqual(got[0], want[0], delta=1e-8)
            self.assertAlmostEqual(got[1], want[1], delta=1e-8)

    def test_jacobian_is_the_affine_part(self):
        j = self.tps.jacobian((0.37, -0.81))
        want = ((AFFINE[0][1], AFFINE[0][2]), (AFFINE[1][1], AFFINE[1][2]))
        for grow, wrow in zip(j, want):
            for g, w in zip(grow, wrow):
                self.assertAlmostEqual(g, w, delta=1e-8)


class TestInterpolation(unittest.TestCase):
    """lambda = 0 must interpolate the control points exactly."""

    def setUp(self):
        self.src = make_points(12)
        self.tgt = [bumpy_ref(p) for p in self.src]
        self.tps = fit_tps(self.src, self.tgt, lambda_=0.0)

    def test_control_points_recovered(self):
        for p, t in zip(self.src, self.tgt):
            got = self.tps.warp_point(p)
            self.assertAlmostEqual(got[0], t[0], delta=1e-8)
            self.assertAlmostEqual(got[1], t[1], delta=1e-8)

    def test_residuals_vanish(self):
        self.assertLess(self.tps.rms_error, 1e-8)
        self.assertLess(self.tps.max_error, 1e-8)
        self.assertEqual(len(self.tps.residuals), len(self.src))

    def test_side_constraint_ptw_zero(self):
        wx, wy = self.tps.weights
        for w in (wx, wy):
            self.assertAlmostEqual(sum(w), 0.0, delta=1e-8)
            self.assertAlmostEqual(
                sum(wi * p[0] for wi, p in zip(w, self.src)), 0.0, delta=1e-8)
            self.assertAlmostEqual(
                sum(wi * p[1] for wi, p in zip(w, self.src)), 0.0, delta=1e-8)

    def test_bending_energy_positive_for_nonaffine(self):
        self.assertGreater(self.tps.bending_energy, 0.0)
        ex, ey = self.tps.bending_energy_per_axis
        self.assertAlmostEqual(ex + ey, self.tps.bending_energy, delta=1e-12)

    def test_residual_statistics_consistent(self):
        sq = [dx * dx + dy * dy for dx, dy in self.tps.residuals]
        self.assertAlmostEqual(
            self.tps.rms_error, math.sqrt(sum(sq) / len(sq)), delta=1e-15)
        self.assertAlmostEqual(
            self.tps.max_error, math.sqrt(max(sq)), delta=1e-15)


class TestJacobianFiniteDifference(unittest.TestCase):
    """Analytic Jacobian vs central finite differences of the warp."""

    def test_matches_finite_difference(self):
        src = make_points(10)
        tgt = [bumpy_ref(p) for p in src]
        tps = fit_tps(src, tgt, lambda_=1e-4)
        h = 1e-6
        for p in [(0.31, -0.27), (1.2, 0.85), (-0.6, -0.4), src[3]]:
            x, y = p
            j = tps.jacobian(p)
            fxp = tps.warp_point((x + h, y))
            fxm = tps.warp_point((x - h, y))
            fyp = tps.warp_point((x, y + h))
            fym = tps.warp_point((x, y - h))
            fd = (((fxp[0] - fxm[0]) / (2 * h), (fyp[0] - fym[0]) / (2 * h)),
                  ((fxp[1] - fxm[1]) / (2 * h), (fyp[1] - fym[1]) / (2 * h)))
            for jrow, frow in zip(j, fd):
                for g, w in zip(jrow, frow):
                    self.assertAlmostEqual(g, w, delta=1e-5)


class TestDegenerateAndDuplicates(unittest.TestCase):
    def test_collinear_points_rejected(self):
        src = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        tgt = [(x + 0.5, y) for x, y in src]
        with self.assertRaises(TPSDegenerateError):
            fit_tps(src, tgt)

    def test_all_identical_points_rejected(self):
        src = [(1.0, 1.0)] * 4
        tgt = [(2.0, 3.0)] * 4  # merge to a single distinct point
        with self.assertRaises(TPSInputError):
            fit_tps(src, tgt)

    def test_too_few_points_rejected(self):
        with self.assertRaises(TPSInputError):
            fit_tps([(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0), (1.0, 0.0)])

    def test_duplicate_same_target_merged(self):
        src = make_points(6)
        tgt = [bumpy_ref(p) for p in src]
        base = fit_tps(src, tgt)
        # Append an exact copy of pair 2; the fit must be unchanged.
        ext_src = src + [src[2]]
        ext_tgt = tgt + [tgt[2]]
        merged = fit_tps(ext_src, ext_tgt)
        self.assertEqual(len(merged.control_points), len(src))
        for wb, wm in zip(base.weights, merged.weights):
            for b, m in zip(wb, wm):
                self.assertAlmostEqual(b, m, delta=1e-10)
        self.assertAlmostEqual(
            base.bending_energy, merged.bending_energy, delta=1e-10)

    def test_duplicate_conflicting_target_rejected(self):
        src = make_points(5)
        tgt = [bumpy_ref(p) for p in src]
        bad_src = src + [src[1]]
        bad_tgt = tgt + [(tgt[1][0] + 0.5, tgt[1][1])]
        with self.assertRaises(TPSInputError):
            fit_tps(bad_src, bad_tgt)


class TestRegularization(unittest.TestCase):
    def test_lambda_smooths(self):
        src = make_points(14)
        # Targets with a deterministic high-frequency component.
        tgt = [(x + 0.4 * math.sin(3.1 * x) * math.cos(2.7 * y)
                + 0.03 * math.sin(11.0 * x + 7.0 * y),
                y + 0.3 * math.cos(2.3 * x + 1.9 * y)
                + 0.03 * math.cos(13.0 * y - 5.0 * x))
               for x, y in src]
        exact = fit_tps(src, tgt, lambda_=0.0)
        smooth = fit_tps(src, tgt, lambda_=1e-2)
        # Interpolating fit reproduces the data; smoothed fit trades
        # residual for a smaller bending energy.
        self.assertLess(exact.rms_error, 1e-8)
        self.assertGreater(smooth.rms_error, exact.rms_error)
        self.assertLess(smooth.bending_energy, exact.bending_energy)
        self.assertGreaterEqual(smooth.bending_energy, 0.0)

    def test_lambda_zero_is_default(self):
        src = make_points(5)
        tgt = [bumpy_ref(p) for p in src]
        tps = fit_tps(src, tgt)
        self.assertEqual(tps.lambda_, 0.0)
        self.assertLess(tps.rms_error, 1e-8)


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.src = make_points(5)
        self.tgt = [bumpy_ref(p) for p in self.src]

    def test_nan_and_inf_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(TPSInputError):
                fit_tps([(0.0, 0.0), (1.0, 0.0), (bad, 1.0)],
                        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
            with self.assertRaises(TPSInputError):
                fit_tps(self.src, self.tgt, lambda_=bad)

    def test_bool_rejected(self):
        with self.assertRaises(TPSInputError):
            fit_tps([(True, 0.0), (1.0, 0.0), (0.0, 1.0)],
                    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        with self.assertRaises(TPSInputError):
            fit_tps(self.src, self.tgt, lambda_=True)

    def test_non_numeric_rejected(self):
        with self.assertRaises(TPSInputError):
            fit_tps([("0", 0.0), (1.0, 0.0), (0.0, 1.0)],
                    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        with self.assertRaises(TPSInputError):
            fit_tps(self.src, self.tgt, lambda_="0.1")

    def test_negative_lambda_rejected(self):
        with self.assertRaises(TPSInputError):
            fit_tps(self.src, self.tgt, lambda_=-1e-9)

    def test_control_point_limit(self):
        src = make_points(MAX_CONTROL_POINTS + 1)
        tgt = [bumpy_ref(p) for p in src]
        with self.assertRaises(TPSInputError):
            fit_tps(src, tgt)
        # Exactly at the limit must work.
        ok = fit_tps(src[:-1], tgt[:-1])
        self.assertIsInstance(ok, TPS)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(TPSInputError):
            fit_tps(self.src, self.tgt[:-1])

    def test_malformed_points_rejected(self):
        with self.assertRaises(TPSInputError):
            fit_tps([(0.0, 0.0, 1.0), (1.0, 0.0), (0.0, 1.0)],
                    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        with self.assertRaises(TPSInputError):
            fit_tps("not-points", self.tgt)

    def test_warp_and_jacobian_validate_input(self):
        tps = fit_tps(self.src, self.tgt)
        with self.assertRaises(TPSInputError):
            tps.warp([(float("nan"), 0.0)])
        with self.assertRaises(TPSInputError):
            tps.warp_point((1.0, 2.0, 3.0))
        with self.assertRaises(TPSInputError):
            tps.jacobian((True, 0.0))

    def test_warp_returns_list_of_tuples(self):
        tps = fit_tps(self.src, self.tgt)
        out = tps.warp([(0.1, 0.2), (0.3, 0.4)])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        for p in out:
            self.assertEqual(len(p), 2)


if __name__ == "__main__":
    unittest.main()
