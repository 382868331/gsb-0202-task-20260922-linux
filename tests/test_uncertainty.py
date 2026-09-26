"""Tests for target-covariance propagation (predict_with_covariance).

Hand-computed references cover three regimes:
  * affine degenerate warp (w = 0): influence reduces to the affine part;
  * fully correlated translation error: prediction covariance is sigma^2 I
    everywhere, because the TPS reproduces constants exactly;
  * independent per-target noise: variance is sigma^2 * sum(h_i^2).

A fixed deterministic perturbation sample cross-checks the linear
propagation against an actual refit, and a deterministic pseudo-sample
cloud checks the propagated covariance empirically.  The package under
test never manufactures its own expectations: references above are
closed-form, and the perturbation check compares two independent uses
(fit vs. propagation) of the public contract.
"""

import math
import unittest

from tps2d import fit, TPSInputError, TPSPrediction


def grid_problem(nx=4, ny=4, scale=3.0):
    sources = []
    targets = []
    for j in range(ny):
        for i in range(nx):
            x, y = scale * i, scale * j
            sources.append([x, y])
            targets.append([
                x + 1.2 * math.sin(0.35 * y),
                y + 1.2 * math.cos(0.30 * x),
            ])
    return sources, targets


def zero_cov(dim):
    return [[0.0] * dim for _ in range(dim)]


def independent_cov(dim, sigma2):
    cov = zero_cov(dim)
    for i in range(dim):
        cov[i][i] = sigma2
    return cov


def fully_correlated_translation_cov(n, sigma2):
    # Same (dx, dy) error added to every target: all 2x2 blocks are sigma^2 I.
    cov = zero_cov(2 * n)
    for i in range(2 * n):
        for j in range(2 * n):
            if i % 2 == j % 2:
                cov[i][j] = sigma2
    return cov


class TestAffineDegenerate(unittest.TestCase):
    """Pure affine warp: non-affine weights vanish, influence is barycentric."""

    def setUp(self):
        # f(x, y) = (2x - y + 5, x + 3y - 2) sampled on a 4-point simplex grid.
        self.sources = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]]
        self.targets = [
            [2.0 * x - y + 5.0, x + 3.0 * y - 2.0] for (x, y) in self.sources
        ]
        self.result = fit(self.sources, self.targets)

    def test_influence_is_affine_only(self):
        n = self.result.merged_count
        pred = self.result.predict_with_covariance(
            [[1.0, 1.0]], independent_cov(2 * n, 1.0)
        )
        h = pred.influence[0][0]
        # With 4 points the affine reproduction system is underdetermined,
        # so the weights themselves are not unique; the closed-form
        # properties are exact reproduction of 1, x and y at the query.
        px, py = 1.0, 1.0
        self.assertAlmostEqual(sum(h), 1.0, places=12)
        self.assertAlmostEqual(
            sum(h[i] * self.sources[i][0] for i in range(n)), px, places=12
        )
        self.assertAlmostEqual(
            sum(h[i] * self.sources[i][1] for i in range(n)), py, places=12
        )
        # And the propagated mean of the affine targets is the affine map.
        fx = sum(h[i] * self.targets[i][0] for i in range(n))
        self.assertAlmostEqual(fx, 2.0 * px - py + 5.0, places=12)

    def test_mean_matches_closed_form(self):
        n = self.result.merged_count
        pred = self.result.predict_with_covariance(
            [[1.5, 2.5]], independent_cov(2 * n, 0.3)
        )
        self.assertAlmostEqual(pred.means[0][0], 2.0 * 1.5 - 2.5 + 5.0, places=12)
        self.assertAlmostEqual(pred.means[0][1], 1.5 + 3.0 * 2.5 - 2.0, places=12)


class TestFullyCorrelatedTranslation(unittest.TestCase):
    """A shared translation error passes through unchanged everywhere."""

    def test_covariance_is_sigma2_identity(self):
        sources, targets = grid_problem()
        result = fit(sources, targets)
        n = result.merged_count
        sigma2 = 0.7
        cov = fully_correlated_translation_cov(n, sigma2)
        queries = [[1.3, 2.7], [4.4, 5.5], [8.9, 0.2]]
        pred = result.predict_with_covariance(queries, cov)
        for q in range(len(queries)):
            c = pred.covariances[q]
            self.assertAlmostEqual(c[0][0], sigma2, places=9)
            self.assertAlmostEqual(c[1][1], sigma2, places=9)
            self.assertAlmostEqual(c[0][1], 0.0, places=12)
            self.assertAlmostEqual(c[1][0], 0.0, places=12)


class TestIndependentNoise(unittest.TestCase):
    def test_variance_is_sigma2_times_sum_of_squares(self):
        sources, targets = grid_problem()
        result = fit(sources, targets)
        n = result.merged_count
        sigma2 = 0.4
        pred = result.predict_with_covariance(
            [[2.2, 3.3]], independent_cov(2 * n, sigma2)
        )
        h = pred.influence[0][0]
        hand = sigma2 * sum(v * v for v in h)
        self.assertAlmostEqual(pred.covariances[0][0][0], hand, places=12)
        self.assertAlmostEqual(pred.covariances[0][1][1], hand, places=12)
        self.assertAlmostEqual(pred.covariances[0][0][1], 0.0, places=15)

    def test_correlated_cross_terms_matter(self):
        # Off-diagonal correlation between two control points must change
        # the result vs. summing diagonal variances only.
        sources, targets = grid_problem()
        result = fit(sources, targets)
        n = result.merged_count
        query = [[2.2, 3.3]]
        diag_only = result.predict_with_covariance(
            query, independent_cov(2 * n, 0.4)
        ).covariances[0][0][0]
        cov = independent_cov(2 * n, 0.4)
        cov[0][2] = cov[2][0] = 0.3  # correlate target 0.x and target 1.x
        with_corr = result.predict_with_covariance(query, cov).covariances[0][0][0]
        h = result.predict_with_covariance(query, independent_cov(2 * n, 0.4)).influence[0][0]
        hand = diag_only + 2.0 * 0.3 * h[0] * h[1]
        self.assertAlmostEqual(with_corr, hand, places=12)
        self.assertNotAlmostEqual(with_corr, diag_only, places=6)


class TestDeterministicPerturbation(unittest.TestCase):
    """Linear propagation vs. an actual refit on perturbed targets."""

    def test_refit_matches_linear_prediction(self):
        sources, targets = grid_problem()
        result = fit(sources, targets)
        n = result.merged_count
        # Fixed, deterministic perturbation (no randomness involved).
        delta = [
            [0.01 * math.sin(1.7 * k + 0.3), 0.01 * math.cos(1.1 * k - 0.8)]
            for k in range(n)
        ]
        cov = zero_cov(2 * n)  # covariance irrelevant for the mean path
        queries = [[1.1, 2.9], [6.3, 4.2]]
        pred = result.predict_with_covariance(queries, cov)
        perturbed_targets = [
            [targets[k][0] + delta[k][0], targets[k][1] + delta[k][1]]
            for k in range(n)
        ]
        refit = fit(sources, perturbed_targets)
        for q, pt in enumerate(queries):
            exact = refit.warp.evaluate(pt)
            # Linear propagation: f_new(p) = f(p) + H(p) @ vec(delta).
            h = pred.influence[q]
            dx = sum(h[0][i] * delta[i][0] for i in range(n))
            dy = sum(h[1][i] * delta[i][1] for i in range(n))
            self.assertAlmostEqual(pred.means[q][0] + dx, exact[0], places=9)
            self.assertAlmostEqual(pred.means[q][1] + dy, exact[1], places=9)

    def test_empirical_covariance_from_fixed_samples(self):
        # Deterministic pseudo-Gaussian cloud via Box-Muller on a fixed LCG;
        # checks H cov H^T against second moments of propagated samples.
        sources, targets = grid_problem()
        result = fit(sources, targets)
        n = result.merged_count
        sigma2 = 0.02
        rho = 0.6  # correlation between the two coordinates of each target
        cov = zero_cov(2 * n)
        for i in range(n):
            cov[2 * i][2 * i] = sigma2
            cov[2 * i + 1][2 * i + 1] = sigma2
            cov[2 * i][2 * i + 1] = cov[2 * i + 1][2 * i] = rho * sigma2
        query = [3.7, 5.1]
        pred = result.predict_with_covariance([query], cov)
        expected = pred.covariances[0]

        state = 12345

        def uniform():
            nonlocal state
            state = (1103515245 * state + 12345) % (2 ** 31)
            return state / 2 ** 31

        def gauss2():
            u1 = max(uniform(), 1e-12)
            u2 = uniform()
            r = math.sqrt(-2.0 * math.log(u1))
            return r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)

        nsamples = 4000
        sxx = sxy = syy = 0.0
        h = pred.influence[0]
        root = math.sqrt(sigma2)
        for _ in range(nsamples):
            dx = dy = 0.0
            for i in range(n):
                z1, z2 = gauss2()
                # Correlate the pair: ex = s z1, ey = s (rho z1 + sqrt(1-rho^2) z2)
                ex = root * z1
                ey = root * (rho * z1 + math.sqrt(1.0 - rho * rho) * z2)
                dx += h[0][i] * ex
                dy += h[1][i] * ey
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
        # ~1.2% relative standard error at 4000 samples; assert at 8%.
        self.assertAlmostEqual(sxx / nsamples, expected[0][0], delta=0.08 * expected[0][0])
        self.assertAlmostEqual(syy / nsamples, expected[1][1], delta=0.08 * expected[1][1])
        self.assertAlmostEqual(sxy / nsamples, expected[0][1], delta=0.08 * max(expected[0][0], 1e-12))


class TestCovarianceValidation(unittest.TestCase):
    def setUp(self):
        sources, targets = grid_problem()
        self.result = fit(sources, targets)
        self.n = self.result.merged_count
        self.dim = 2 * self.n

    def assert_rejected(self, cov):
        with self.assertRaises(TPSInputError):
            self.result.predict_with_covariance([[1.0, 1.0]], cov)

    def test_zero_matrix_allowed(self):
        pred = self.result.predict_with_covariance(
            [[1.0, 1.0]], zero_cov(self.dim)
        )
        self.assertEqual(pred.covariances[0], [[0.0, 0.0], [0.0, 0.0]])

    def test_zero_eigenvalue_direction_allowed(self):
        # Only target 0.x varies; every other direction has zero variance.
        cov = zero_cov(self.dim)
        cov[0][0] = 2.0
        pred = self.result.predict_with_covariance([[1.0, 1.0]], cov)
        h = pred.influence[0][0]
        self.assertAlmostEqual(pred.covariances[0][0][0], 2.0 * h[0] * h[0], places=12)
        self.assertAlmostEqual(pred.covariances[0][1][1], 0.0, places=15)

    def test_wrong_size_rejected(self):
        self.assert_rejected(zero_cov(self.dim - 1))
        self.assert_rejected(zero_cov(self.dim + 2))
        cov = zero_cov(self.dim)
        cov[3] = cov[3][:-1]
        self.assert_rejected(cov)

    def test_asymmetric_rejected(self):
        cov = zero_cov(self.dim)
        cov[0][2] = 0.1
        self.assert_rejected(cov)

    def test_indefinite_rejected(self):
        cov = independent_cov(self.dim, 1.0)
        cov[0][0] = -0.5
        self.assert_rejected(cov)
        cov = independent_cov(self.dim, 1.0)
        cov[0][2] = cov[2][0] = 5.0  # correlation > 1
        self.assert_rejected(cov)

    def test_non_finite_and_bool_rejected(self):
        cov = zero_cov(self.dim)
        cov[1][1] = float("nan")
        self.assert_rejected(cov)
        cov = zero_cov(self.dim)
        cov[1][1] = float("inf")
        self.assert_rejected(cov)
        cov = zero_cov(self.dim)
        cov[0][0] = True
        self.assert_rejected(cov)

    def test_bad_query_point_rejected_before_computation(self):
        with self.assertRaises(TPSInputError):
            self.result.predict_with_covariance(
                [[1.0, float("nan")]], independent_cov(self.dim, 1.0)
            )
        with self.assertRaises(TPSInputError):
            self.result.predict_with_covariance(
                [[1.0]], independent_cov(self.dim, 1.0)
            )

    def test_baseline_fit_unchanged(self):
        # The covariance interface must not disturb fit/predict outputs.
        sources, targets = grid_problem()
        r1 = fit(sources, targets)
        before = r1.warp.evaluate([2.5, 2.5])
        r1.predict_with_covariance([[0.1, 0.2]], independent_cov(2 * r1.merged_count, 1.0))
        after = r1.warp.evaluate([2.5, 2.5])
        self.assertEqual(before, after)
        r2 = fit(sources, targets)
        self.assertEqual(r1.weights, r2.weights)
        self.assertEqual(r1.affine, r2.affine)
        self.assertIsInstance(
            r1.predict_with_covariance([[0.0, 0.0]], zero_cov(2 * r1.merged_count)),
            TPSPrediction,
        )


if __name__ == "__main__":
    unittest.main()
