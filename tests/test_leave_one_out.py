"""Necessary tests for the leave-one-out diagnostic.

Expected values come from closed-form construction (known affine maps) or
from ``_reference`` (an independent from-scratch implementation that
rebuilds each reduced small system group by group).  The package under
test is never used to manufacture its own expectations.

Error accounting (same as the base suite): package-vs-independent-reference
tolerance 1e-9 on both output coordinates for O(1) data.
"""

import math
import unittest

from tps2d import fit, leave_one_out, TPSInputError, DegeneratePointsError

from _reference import reference_fit, reference_warp


def merged_pairs(src, tgt):
    """Merge same-source/same-target duplicates the way the contract does."""
    seen = {}
    order = []
    for s, t in zip(src, tgt):
        key = (float(s[0]), float(s[1]))
        val = (float(t[0]), float(t[1]))
        if key in seen:
            assert seen[key] == val
            continue
        seen[key] = val
        order.append(key)
    return order, [seen[k] for k in order]


class AffineLeaveOneOutTests(unittest.TestCase):
    """Pure affine data: every leave-one-out prediction is exact."""

    def affine_map(self, p):
        x, y = p
        return [2.0 + 3.0 * x - 1.0 * y, -1.0 + 0.5 * x + 2.0 * y]

    def setUp(self):
        self.src = [[0, 0], [1, 0], [0, 1], [2, 3], [-2, 1.5], [3.5, -2]]
        self.tgt = [self.affine_map(p) for p in self.src]

    def check_exact(self, lam):
        loo = leave_one_out(self.src, self.tgt, lam=lam)
        self.assertEqual(len(loo.entries), len(self.src))
        for e in loo.entries:
            self.assertTrue(e.computable, e.reason)
            self.assertIsNotNone(e.prediction)
            msg = "lam=%s residual at %s: %s" % (lam, e.point, e.residual)
            self.assertLess(e.residual, 1e-9, msg)
            expected = self.affine_map(e.point)
            self.assertAlmostEqual(e.prediction[0], expected[0], delta=1e-9)
            self.assertAlmostEqual(e.prediction[1], expected[1], delta=1e-9)

    def test_zero_regularisation(self):
        self.check_exact(0.0)

    def test_positive_regularisation(self):
        self.check_exact(0.5)


class SingleOutlierTests(unittest.TestCase):
    """One deviating point must stand out in leave-one-out residuals."""

    def setUp(self):
        self.src = [[3.0 * i, 3.0 * j] for j in range(4) for i in range(4)]
        # Smooth affine background, single point pushed away.
        self.tgt = [[x + 0.5, y - 0.25] for x, y in self.src]
        self.tgt[7] = [self.src[7][0] + 4.0, self.src[7][1] - 3.0]

    def test_outlier_has_largest_residual(self):
        for lam in (0.0, 1.0):
            loo = leave_one_out(self.src, self.tgt, lam=lam)
            top = loo.max_residual_entry
            self.assertIsNotNone(top)
            self.assertEqual(top.index, 7, "lam=%s" % lam)
            self.assertGreater(top.residual, 1.0, "lam=%s" % lam)
            others = [e for e in loo.computable_entries if e.index != 7]
            self.assertLess(max(e.residual for e in others), top.residual)

    def test_outlier_prediction_follows_background(self):
        # With the outlier removed, the fit is essentially the background
        # affine map, so the prediction at the outlier lands near it.
        loo = leave_one_out(self.src, self.tgt, lam=0.0)
        entry = loo.entries[7]
        self.assertAlmostEqual(entry.prediction[0],
                               self.src[7][0] + 0.5, delta=1e-6)
        self.assertAlmostEqual(entry.prediction[1],
                               self.src[7][1] - 0.25, delta=1e-6)


class DuplicateGroupTests(unittest.TestCase):
    """Merged duplicates are removed as one group: no target leakage."""

    def setUp(self):
        # One deviating point, duplicated with the SAME target (merged).
        self.src = [[0, 0], [4, 0], [0, 4], [4, 4], [2, 1], [2, 1], [1, 3]]
        self.tgt = [[0.5, -0.25], [4.5, -0.25], [0.5, 3.75], [4.5, 3.75],
                    [8.0, -7.0], [8.0, -7.0], [1.5, 2.75]]

    def test_group_removed_not_leaked(self):
        # lam=0 interpolates the merged outlier exactly in the full fit,
        # but its leave-one-out prediction must NOT see that target.
        loo = leave_one_out(self.src, self.tgt, lam=0.0)
        self.assertEqual(len(loo.entries), 6)  # duplicates merged
        outlier = loo.entries[4]
        self.assertEqual(outlier.point, [2.0, 1.0])
        self.assertTrue(outlier.computable, outlier.reason)
        # Prediction follows the affine background, far from the outlier
        # target [8, -7]; a leaked duplicate would give residual ~0.
        self.assertGreater(outlier.residual, 5.0)
        self.assertAlmostEqual(outlier.prediction[0], 2.5, delta=1e-6)
        self.assertAlmostEqual(outlier.prediction[1], 0.75, delta=1e-6)

    def test_matches_reference_group_rebuild(self):
        msrc, mtgt = merged_pairs(self.src, self.tgt)
        for lam in (0.0, 0.25):
            loo = leave_one_out(self.src, self.tgt, lam=lam)
            for e in loo.computable_entries:
                red_src = [list(p) for k, p in enumerate(msrc) if k != e.index]
                red_tgt = [list(t) for k, t in enumerate(mtgt) if k != e.index]
                params = reference_fit(red_src, red_tgt, lam)
                px, py = reference_warp(params, e.point[0], e.point[1])
                self.assertAlmostEqual(e.prediction[0], px, delta=1e-9,
                                       msg="lam=%s group %s x" % (lam, e.index))
                self.assertAlmostEqual(e.prediction[1], py, delta=1e-9,
                                       msg="lam=%s group %s y" % (lam, e.index))


class DegenerateReducedTests(unittest.TestCase):
    """Removing one of three points leaves 2 < 3: flagged, not zero."""

    def test_three_points_remove_one(self):
        src = [[0, 0], [1, 0], [0, 1]]
        tgt = [[1, 2], [2, 2], [1, 3]]
        for lam in (0.0, 1.0):
            loo = leave_one_out(src, tgt, lam=lam)
            self.assertEqual(len(loo.entries), 3)
            for e in loo.entries:
                self.assertFalse(e.computable)
                self.assertIsNone(e.prediction)
                self.assertIsNone(e.residual)
                self.assertIsNotNone(e.reason)
            self.assertIsNone(loo.max_residual_entry)

    def test_collinear_after_removal(self):
        # Four points, three of them collinear: removing the off-line
        # point leaves a collinear (degenerate) configuration.
        src = [[0, 0], [1, 0], [2, 0], [0.5, 2.0]]
        tgt = [[0, 1], [1, 1], [2, 1], [0.5, 3.0]]
        loo = leave_one_out(src, tgt, lam=0.0)
        self.assertFalse(loo.entries[3].computable)
        self.assertIsNone(loo.entries[3].residual)
        for e in loo.entries[:3]:
            self.assertTrue(e.computable, e.reason)


class ReferenceEquivalenceTests(unittest.TestCase):
    """Every computable entry equals an independent group-by-group rebuild."""

    def test_grid_both_lambdas(self):
        src = [[3.0 * i, 3.0 * j] for j in range(4) for i in range(4)]
        tgt = [[x + 1.2 * math.sin(0.35 * y), y + 1.2 * math.cos(0.30 * x)]
               for x, y in src]
        msrc, mtgt = merged_pairs(src, tgt)
        for lam in (0.0, 2.0):
            loo = leave_one_out(src, tgt, lam=lam)
            self.assertEqual(len(loo.entries), len(msrc))
            for e in loo.entries:
                self.assertTrue(e.computable, e.reason)
                red_src = [list(p) for k, p in enumerate(msrc) if k != e.index]
                red_tgt = [list(t) for k, t in enumerate(mtgt) if k != e.index]
                params = reference_fit(red_src, red_tgt, lam)
                px, py = reference_warp(params, e.point[0], e.point[1])
                self.assertAlmostEqual(e.prediction[0], px, delta=1e-9,
                                       msg="lam=%s group %s x" % (lam, e.index))
                self.assertAlmostEqual(e.prediction[1], py, delta=1e-9,
                                       msg="lam=%s group %s y" % (lam, e.index))
                expected_res = math.hypot(px - e.target[0], py - e.target[1])
                self.assertAlmostEqual(e.residual, expected_res, delta=1e-9)


class BaseFitUntouchedTests(unittest.TestCase):
    """The diagnostic must not mutate the base fit or later warp behaviour."""

    def test_warp_unchanged(self):
        src = [[3.0 * i, 3.0 * j] for j in range(3) for i in range(3)]
        tgt = [[x + math.sin(y), y + math.cos(x)] for x, y in src]
        base = fit(src, tgt, lam=0.1)
        probe = [1.7, 4.2]
        before = base.warp.evaluate(probe)
        jac_before = base.warp.jacobian(probe)
        loo = leave_one_out(src, tgt, lam=0.1)
        # The diagnostic's own base fit agrees with the independent one.
        self.assertEqual(before, loo.fit.warp.evaluate(probe))
        after = base.warp.evaluate(probe)
        jac_after = base.warp.jacobian(probe)
        self.assertEqual(before, after)
        self.assertEqual(jac_before, jac_after)
        # Base in-sample residuals are unchanged by the diagnostic.
        base2 = fit(src, tgt, lam=0.1)
        self.assertEqual(base.residuals, base2.residuals)

    def test_input_lists_not_mutated(self):
        src = [[0, 0], [1, 0], [0, 1], [2, 3]]
        tgt = [[1, 1], [2, 1], [1, 2], [3, 4]]
        src_copy = [list(p) for p in src]
        tgt_copy = [list(p) for p in tgt]
        leave_one_out(src, tgt, lam=0.0)
        self.assertEqual(src, src_copy)
        self.assertEqual(tgt, tgt_copy)


class LeaveOneOutInputContractTests(unittest.TestCase):
    """leave_one_out shares the fit input contract."""

    def test_conflicting_duplicates_rejected(self):
        src = [[0, 0], [1, 0], [0, 1], [0, 0]]
        tgt = [[0, 0], [1, 0], [0, 1], [5, 5]]
        with self.assertRaises(TPSInputError):
            leave_one_out(src, tgt)

    def test_degenerate_full_set_rejected(self):
        src = [[0, 0], [1, 0], [2, 0]]
        tgt = [[0, 0], [1, 0], [2, 0]]
        with self.assertRaises(DegeneratePointsError):
            leave_one_out(src, tgt)

    def test_nan_rejected(self):
        src = [[0, 0], [1, 0], [0, 1], [float("nan"), 2]]
        tgt = [[0, 0], [1, 0], [0, 1], [1, 2]]
        with self.assertRaises(TPSInputError):
            leave_one_out(src, tgt)


if __name__ == "__main__":
    unittest.main()
