"""Two-dimensional thin-plate spline warps (standard library only).

Implements the classical 2-D thin-plate spline

    f(p) = a0 + a1*x + a2*y + sum_i w_i * U(||p - c_i||),
    U(r) = r**2 * log(r),  U(0) = 0,

fitted independently for the two output coordinates by solving one block
system that determines the affine and bending coefficients jointly.

Interpolating case (``lam = 0``), P^T w = 0:

    [ K   P ] [w]   [v]
    [ P^T 0 ] [a] = [0],   P_i = [1, x_i, y_i].

Smoothing case (``lam > 0``) with per-point confidence weights ``sigma_i``
minimises  sum_i sigma_i (f(c_i) - v_i)^2 + lam * w^T K w, subject to
P^T w = 0.  Writing r = v - K w - P a and w = W r / lam gives the residual
form (never needed numerically, since it divides by lam):

    [ I + K W / lam   P ] [r]   [v]
    [ P^T W           0 ] [a] = [0],   W = diag(sigma_i).

The implementation instead solves the equivalent, uniformly well-scaled
KKT system for (w, a, mu) jointly (affine and bending parts in one solve):

    [ lam K + K W K   K W P   P ] [w ]   [K W v  ]
    [ P^T W K         P^T W P  0 ] [a ] = [P^T W v]
    [ P^T             0         0 ] [mu]   [0      ].

Points with sigma_i = 0 have w_i = 0 exactly and are dropped from that
system; for sigma_i == 1 and lam > 0 the result equals the classical
``K + lam I`` smoothing spline, and for lam = 0 the classical interpolation
system [K P; P^T 0] is solved directly.

Inputs are validated strictly: NaN/Infinity are rejected, booleans are not
accepted as numbers, duplicate source points with conflicting targets raise,
zero-confidence points at ``lam = 0`` raise (the interpolant through an
untrusted point is undefined), and configurations whose *trusted* points do
not affinely span the plane raise `DegeneratePointsError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._linalg import solve_linear, affine_rank, SingularMatrixError

__all__ = [
    "MAX_CONTROL_POINTS",
    "TPSInputError",
    "DegeneratePointsError",
    "TPSFitResult",
    "TPSWarp",
    "fit",
]

MAX_CONTROL_POINTS = 40


class TPSInputError(ValueError):
    """Invalid input to TPS fitting or evaluation (shape, NaN, duplicates...)."""


class DegeneratePointsError(TPSInputError):
    """Affine design matrix rank < 3 (collinear / insufficient points)."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_real_number(value):
    # bool is a subclass of int; the contract rejects booleans explicitly.
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _finite_float(value, what):
    if not _is_real_number(value):
        raise TPSInputError(f"{what} must be a real number, got {type(value).__name__}")
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        raise TPSInputError(f"{what} must be finite, got {fv}")
    return fv


def _nonneg_float(value, what):
    fv = _finite_float(value, what)
    if fv < 0.0:
        raise TPSInputError(f"{what} must be >= 0, got {fv}")
    return fv


def _check_points(points, name):
    if not isinstance(points, (list, tuple)):
        raise TPSInputError(f"{name} must be a list/tuple of 2-D points")
    if len(points) == 0:
        raise TPSInputError(f"{name} must contain at least one point")
    cleaned = []
    for k, pt in enumerate(points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise TPSInputError(
                f"{name}[{k}] must be a 2-element [x, y] pair, got {pt!r}"
            )
        x = _finite_float(pt[0], f"{name}[{k}].x")
        y = _finite_float(pt[1], f"{name}[{k}].y")
        cleaned.append((x, y))
    return cleaned


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _U(dx, dy):
    """Thin-plate kernel U(r) = r^2 log(r) with U(0) = 0 (2-D)."""
    r2 = dx * dx + dy * dy
    if r2 == 0.0:
        return 0.0
    # r^2 * (0.5 * ln(r^2)); avoids the extra sqrt and is exact algebraically.
    return 0.5 * r2 * math.log(r2)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TPSFitResult:
    """Outcome of `fit` and the fitted warp.

    Attributes
    ----------
    warp:
        A `TPSWarp` usable for point evaluation and Jacobians.
    weights:
        n x 2 non-affine coefficients (one row per merged control point),
        columns correspond to output x and y.
    affine:
        3 x 2 affine coefficients ``[[a0_x, a0_y], [a1_x, a1_y],
        [a2_x, a2_y]]`` with warp output f = a0 + a1*x + a2*y + sum w_i U_i.
    residuals:
        Per-control-point Euclidean residual ``||f(c_i) - target_i||``
        (length n, against the merged, deduplicated point set).
    rms_residual:
        RMS of the two-coordinate fitting residual over all points.
    bending_energy:
        Thin-plate bending energy w^T K w summed over the two output
        coordinates; zero for a pure affine transform.
    rank:
        Numerical rank of the affine design matrix P built from the points
        that actually carry information (positive-confidence points when
        smoothing, all merged points when interpolating); 3 for a valid fit.
    merged_count:
        Number of distinct source points after same-target duplicates were
        merged (may be < the number of input pairs).
    confidence:
        Per-merged-point non-negative confidence sigma_i actually used
        (duplicates were summed).  ``None`` when the caller did not pass
        explicit confidence weights (all treated as 1).
    effective_count:
        Number of merged points with strictly positive confidence (i.e. that
        influence the fit).  Equals ``merged_count`` for interpolating fits.
    """

    warp: "TPSWarp"
    weights: list
    affine: list
    residuals: list
    rms_residual: float
    bending_energy: float
    rank: int
    merged_count: int
    confidence: "list | None" = None
    effective_count: int = 0

    @property
    def coefficients(self):
        """Tuple ``(weights, affine)`` exactly as stored."""
        return self.weights, self.affine


@dataclass
class TPSWarp:
    """Fitted 2-D thin-plate spline: evaluate points or query Jacobians."""

    control_points: list
    weights: list          # n x 2
    affine: list           # 3 x 2

    # -- core evaluation ----------------------------------------------------

    def _basis_at(self, x, y):
        """Return (kernel values u[0..n-1], kernel grad pairs du[0..n-1])."""
        pts = self.control_points
        u = [0.0] * len(pts)
        du = [(0.0, 0.0)] * len(pts)
        for i, (cx, cy) in enumerate(pts):
            dx = x - cx
            dy = y - cy
            r2 = dx * dx + dy * dy
            if r2 == 0.0:
                # U = 0; directional derivative of r^2 log r is 0 at r = 0
                # (the scalar field U has a stationary point at the origin).
                continue
            log_r2 = math.log(r2)
            u[i] = 0.5 * r2 * log_r2
            # d/dx U = dx * (log(r^2) + 1)
            one_plus_log = log_r2 + 1.0
            du[i] = (dx * one_plus_log, dy * one_plus_log)
        return u, du

    def evaluate(self, point):
        """Warp a single ``[x, y]`` point -> ``[x', y']``."""
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise TPSInputError("point must be a 2-element [x, y] pair")
        x = _finite_float(point[0], "point.x")
        y = _finite_float(point[1], "point.y")
        u, _ = self._basis_at(x, y)
        w, a = self.weights, self.affine
        out_x = a[0][0] + a[1][0] * x + a[2][0] * y
        out_y = a[0][1] + a[1][1] * x + a[2][1] * y
        for i, ui in enumerate(u):
            out_x += w[i][0] * ui
            out_y += w[i][1] * ui
        return [out_x, out_y]

    def warp_points(self, points):
        """Warp a list/tuple of 2-D points; all-or-none validation."""
        if not isinstance(points, (list, tuple)):
            raise TPSInputError("points must be a list/tuple of 2-D points")
        # Validate everything first so a bad element never leaves a partially
        # consumed/processed batch (the operation is side-effect free, but
        # error location is reported before any computation).
        clean = _check_points(points, "points")
        return [self.evaluate(p) for p in clean]

    def __call__(self, point):
        return self.evaluate(point)

    def jacobian(self, point):
        """Return the 2 x 2 Jacobian ``[[dfx/dx, dfx/dy], [dfy/dx, dfy/dy]]``.

        Finite for every point, including the control points themselves.
        """
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise TPSInputError("point must be a 2-element [x, y] pair")
        x = _finite_float(point[0], "point.x")
        y = _finite_float(point[1], "point.y")
        _, du = self._basis_at(x, y)
        w, a = self.weights, self.affine
        j00 = a[1][0]
        j01 = a[2][0]
        j10 = a[1][1]
        j11 = a[2][1]
        for i, (gx, gy) in enumerate(du):
            j00 += w[i][0] * gx
            j01 += w[i][0] * gy
            j10 += w[i][1] * gx
            j11 += w[i][1] * gy
        return [[j00, j01], [j10, j11]]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _merge_duplicates(sources, targets, confidence):
    """Collapse repeated source points.

    Same source + same target -> one control point; independent confidence
    values add (precision of independent observations of the same point).
    Same source + different targets -> TPSInputError (no unique interpolant).
    Returns ``(points, targets, confidence)`` in first-occurrence order.
    Coordinate equality is exact (already-converted floats); callers needing
    tolerance-based merging must pre-snap their own data.
    """
    seen = {}
    order = []
    for s, t, sigma in zip(sources, targets, confidence):
        if s in seen:
            if seen[s][0] != t:
                raise TPSInputError(
                    f"duplicate source point {s} maps to conflicting targets "
                    f"{seen[s][0]} and {t}"
                )
            seen[s][1] += sigma  # combine independent evidence
            continue
        seen[s] = [t, sigma]
        order.append(s)
    return (
        order,
        [seen[s][0] for s in order],
        [seen[s][1] for s in order],
    )


def fit(sources, targets, lam=0.0, *, confidence=None):
    """Fit a 2-D thin-plate spline.

    Parameters
    ----------
    sources:
        1..40 control points ``[[x, y], ...]`` in the source/domain frame.
    targets:
        Corresponding points ``[[x', y'], ...]`` in the target frame.
    lam:
        Non-negative regularisation parameter lambda >= 0.  ``lam = 0``
        interpolates every *trusted* control point exactly (up to
        floating-point round-off); larger values trade fit for smoothness,
        minimising ``sum sigma_i (f(c_i) - v_i)^2 + lam * w^T K w``.
    confidence:
        Optional per-point non-negative confidence weights sigma_i, one per
        input pair (same length as ``sources``).  Defaults to all ones.
        ``sigma_i = 0`` means the point is not trusted at all: it never
        constrains the fit (only possible with ``lam > 0``).  Duplicate
        source points with identical targets have their confidence summed.

    Returns
    -------
    TPSFitResult

    Raises
    ------
    TPSInputError
        Bad shapes/types, non-finite numbers, booleans-as-numbers,
        conflicting duplicates, negative/NaN confidence, invalid lambda, a
        zero-confidence point with ``lam = 0``, or no positively weighted
        points.
    DegeneratePointsError
        Fewer than 3 positively weighted distinct points, or those points
        affinely rank deficient (rank of their P < 3, i.e. collinear).
    """
    src = _check_points(sources, "sources")
    tgt = _check_points(targets, "targets")
    if len(src) != len(tgt):
        raise TPSInputError(
            f"sources/targets length mismatch: {len(src)} != {len(tgt)}"
        )
    if len(src) > MAX_CONTROL_POINTS:
        raise TPSInputError(
            f"at most {MAX_CONTROL_POINTS} control points allowed, got {len(src)}"
        )
    # bool is rejected inside _finite_float; also guard lam explicitly.
    lam_f = _finite_float(lam, "lam")
    if lam_f < 0.0:
        raise TPSInputError(f"lam must be >= 0, got {lam_f}")

    # Per-point confidence weights.  None means "unit weight everywhere", in
    # which case the classical system is recovered exactly.
    confidence_explicit = confidence is not None
    if confidence_explicit:
        if not isinstance(confidence, (list, tuple)):
            raise TPSInputError(
                "confidence must be a list/tuple of non-negative numbers"
            )
        if len(confidence) != len(src):
            raise TPSInputError(
                f"confidence length {len(confidence)} does not match number of "
                f"points {len(src)}"
            )
        sigmas = [
            _nonneg_float(confidence[k], f"confidence[{k}]")
            for k in range(len(confidence))
        ]
    else:
        sigmas = [1.0] * len(src)

    src, tgt, sigmas = _merge_duplicates(src, tgt, sigmas)
    n = len(src)
    if n < 3:
        raise DegeneratePointsError(
            f"need at least 3 distinct source points after merging, got {n}"
        )

    # Internal normalisation for scale robustness.  Medical pixel coordinates
    # are commonly O(1e3..1e6), where U ~ r^2 log r makes K dominate the block
    # matrix and a global pivot tolerance would falsely reject a small Schur
    # complement pivot.  We centre the sources and divide by their diameter;
    # targets are divided by the same factor.  Using U(s*r) = s^2 U(r) and
    # sum(w)=0, the fitted coefficients are converted back exactly:
    #     w = w~ / s,  a1 = a~1, a2 = a~2,  a0 = s*a~0 - a~1*cx - a~2*cy.
    # The data misfit picks up a factor s^2 while the bending energy w^T K w
    # is dimensionless under uniform scaling; dividing the whole objective by
    # s^2 leaves the confidence weights unchanged and maps the regulariser to
    #     lam~ = lam / s^2.
    cx = sum(p[0] for p in src) / n
    cy = sum(p[1] for p in src) / n
    diameter = 0.0
    for i in range(n):
        xi, yi = src[i]
        for j in range(i + 1, n):
            d = math.hypot(xi - src[j][0], yi - src[j][1])
            if d > diameter:
                diameter = d
    if diameter == 0.0:
        raise DegeneratePointsError("all source points coincide")

    s = diameter
    s2 = s * s
    nsrc = [[(x - cx) / s, (y - cy) / s] for (x, y) in src]
    ntgt = [[vx / s, vy / s] for (vx, vy) in tgt]
    nsig = list(sigmas)
    nlam = lam_f / s2

    # Only positively weighted points constrain a smoothing fit; with lam = 0
    # every merged point must be interpolated, so zero confidence there is a
    # contradiction rather than a silently ignored point.
    if lam_f == 0.0:
        zero_idx = [i for i, sigma in enumerate(sigmas) if sigma == 0.0]
        if zero_idx:
            raise TPSInputError(
                f"confidence[{zero_idx[0]}] = 0 with lam = 0: an "
                f"untrusted point cannot be interpolated; use lam > 0 to fit "
                f"with confidence weights"
            )
        active = list(range(n))
    else:
        active = [i for i, sigma in enumerate(nsig) if sigma > 0.0]
        if len(active) < 3:
            raise DegeneratePointsError(
                f"need at least 3 distinct source points with positive "
                f"confidence to determine affine and bending terms, got "
                f"{len(active)}"
            )

    # Rank of the affine design P = [1, x, y] over the constraining points.
    Pn = [[1.0, nsrc[i][0], nsrc[i][1]] for i in active]
    rank = affine_rank(Pn)
    if rank < 3:
        raise DegeneratePointsError(
            f"affine design matrix rank is {rank} < 3 over the "
            f"{len(active)} positively weighted control points: they are "
            f"collinear or fewer than 3 distinct points"
        )

    # Assemble the (n+3) x (n+3) system in residual form.  For lam > 0:
    #   r_i + sum_j K_ij sigma_j r_j / lam + P_i a = v_i
    #   sum_i sigma_i P_i r_i = 0,   w_i = sigma_i r_i / lam.
    # Zero-confidence rows/columns decouple (they contain no K W products and
    # their bottom-row entries are zero), so no division by sigma occurs.
    size = n + 3
    A = [[0.0] * size for _ in range(size)]
    if lam_f == 0.0:
        # Classical interpolation: K w + P a = v, P^T w = 0.
        for i in range(n):
            xi, yi = nsrc[i]
            for j in range(i + 1, n):
                kij = _U(xi - nsrc[j][0], yi - nsrc[j][1])
                A[i][j] = kij
                A[j][i] = kij
            A[i][n] = 1.0
            A[i][n + 1] = xi
            A[i][n + 2] = yi
            A[n][i] = 1.0
            A[n + 1][i] = xi
            A[n + 2][i] = yi
        rhs_x = [t[0] for t in ntgt] + [0.0, 0.0, 0.0]
        rhs_y = [t[1] for t in ntgt] + [0.0, 0.0, 0.0]

        try:
            sol_x, sol_y = solve_linear(A, [rhs_x, rhs_y])
        except SingularMatrixError as exc:
            # rank(P) >= 3 guarantees the block system is invertible; reaching
            # here means the configuration is extremely close to degenerate.
            raise DegeneratePointsError(
                f"TPS system numerically singular despite rank(P)={rank}: {exc}"
            ) from exc

        wx_n, wy_n = sol_x[:n], sol_y[:n]
        ax_n, ay_n = sol_x[n:], sol_y[n:]
    else:
        # Weighted smoothing as a joint KKT system in (w, a, mu), obtained
        # from the Lagrangian  r^T W r + lam w^T K w + 2 mu^T P^T w with
        # r = v - K w - P a:
        #   (lam K + K W K) w + K W P a + P mu = K W v
        #   P^T W K w + P^T W P a             = P^T W v
        #   P^T w                              = 0
        # Solved jointly for bending and affine parts.  Every block is O(1)
        # in normalised units for all lam >= 0 (the lam K term simply fades
        # as lam -> 0, where the system continuously becomes the interpolant
        # through the positively weighted points), so it stays well pivoted.
        # A zero-confidence anchor has w_i = sigma_i r_i / lam = 0 exactly:
        # it contributes no kernel anywhere and no data row, so it is dropped
        # from the solved system (rather than weakly identified in it) and
        # re-inserted as a zero-coefficient row afterwards.
        m = len(active)
        Kmat = [[0.0] * m for _ in range(m)]
        for ii in range(m):
            xi, yi = nsrc[active[ii]]
            for jj in range(ii + 1, m):
                aj = active[jj]
                kij = _U(xi - nsrc[aj][0], yi - nsrc[aj][1])
                Kmat[ii][jj] = kij
                Kmat[jj][ii] = kij
        Pmat = [[1.0, nsrc[ai][0], nsrc[ai][1]] for ai in active]
        wsig = [nsig[ai] for ai in active]
        wtgt = [ntgt[ai] for ai in active]
        # K W (diagonal W, all weights strictly positive).
        KW = [
            [Kmat[ii][jj] * wsig[jj] for jj in range(m)]
            for ii in range(m)
        ]
        size = m + 6
        A = [[0.0] * size for _ in range(size)]
        for i in range(m):
            row = A[i]
            kwrow = KW[i]
            for j in range(m):
                acc = nlam * Kmat[i][j]
                for k in range(m):
                    kwk = kwrow[k]
                    if kwk != 0.0:
                        acc += kwk * Kmat[k][j]
                row[j] = acc
            for d in range(3):
                acc = 0.0
                for k in range(m):
                    kwk = kwrow[k]
                    if kwk != 0.0:
                        acc += kwk * Pmat[k][d]
                row[m + d] = acc
            row[m + 3] = 1.0
            row[m + 4] = Pmat[i][1]
            row[m + 5] = Pmat[i][2]
        for d in range(3):
            row = A[m + d]
            for j in range(m):
                acc = 0.0
                for i in range(m):
                    acc += wsig[i] * Pmat[i][d] * Kmat[i][j]
                row[j] = acc
            for e in range(3):
                acc = 0.0
                for i in range(m):
                    acc += wsig[i] * Pmat[i][d] * Pmat[i][e]
                row[m + e] = acc
        for d in range(3):
            for i in range(m):
                A[m + 3 + d][i] = Pmat[i][d]

        def _rhs(coord):
            b = [0.0] * size
            for i in range(m):
                acc = 0.0
                kwrow = KW[i]
                for k in range(m):
                    kwk = kwrow[k]
                    if kwk != 0.0:
                        acc += kwk * wtgt[k][coord]
                b[i] = acc
            for d in range(3):
                acc = 0.0
                for i in range(m):
                    acc += wsig[i] * Pmat[i][d] * wtgt[i][coord]
                b[m + d] = acc
            return b

        try:
            sol_x, sol_y = solve_linear(A, [_rhs(0), _rhs(1)])
        except SingularMatrixError as exc:
            raise DegeneratePointsError(
                f"weighted TPS system numerically singular despite "
                f"rank(P)={rank} over {m} weighted points: {exc}"
            ) from exc

        wx_active, wy_active = sol_x[:m], sol_y[:m]
        ax_n, ay_n = sol_x[m:m + 3], sol_y[m:m + 3]
        wx_n = [0.0] * n
        wy_n = [0.0] * n
        for ii, ai in enumerate(active):
            wx_n[ai] = wx_active[ii]
            wy_n[ai] = wy_active[ii]

    # Convert normalised coefficients back to the caller's units.
    # U(s*r) = s^2 U(r) + s^2 ln(s) ||r||^2; with P^T w = 0 the quadratic
    # extra term reduces, at every point, to the constant
    # s*ln(s) * sum(w~_i ||c~_i||^2), which is absorbed into a0.
    ln_s = math.log(s)

    def _back_to_user(w_n, a_n, cxu, cyu):
        const = ln_s * sum(
            w_n[i] * (nsrc[i][0] * nsrc[i][0] + nsrc[i][1] * nsrc[i][1])
            for i in range(n)
        )
        return [
            s * a_n[0] - a_n[1] * cxu - a_n[2] * cyu - s * const,
            a_n[1],
            a_n[2],
        ]

    wx = [wi / s for wi in wx_n]
    wy = [wi / s for wi in wy_n]
    ax = _back_to_user(wx_n, ax_n, cx, cy)
    ay = _back_to_user(wy_n, ay_n, cx, cy)
    weights = [[wx[i], wy[i]] for i in range(n)]
    affine = [[ax[0], ay[0]], [ax[1], ay[1]], [ax[2], ay[2]]]

    warp = TPSWarp(control_points=src, weights=weights, affine=affine)

    # Residuals at the (merged) control points, in original units.
    residuals = []
    sq_sum = 0.0
    for i, pt in enumerate(src):
        fx, fy = warp.evaluate(pt)
        ex = fx - tgt[i][0]
        ey = fy - tgt[i][1]
        residuals.append(math.hypot(ex, ey))
        sq_sum += ex * ex + ey * ey
    rms = math.sqrt(sq_sum / (2.0 * n))

    # Bending energy w~^T K~ w~ (identical to w^T K w in user units: the
    # s^-1 and s^2 scaling cancel), summed over the two output coordinates.
    energy = 0.0
    for w_coord in (wx_n, wy_n):
        acc = 0.0
        for i in range(n):
            xi, yi = nsrc[i]
            for j in range(i + 1, n):
                kij = _U(xi - nsrc[j][0], yi - nsrc[j][1])
                acc += 2.0 * w_coord[i] * w_coord[j] * kij
        energy += acc

    return TPSFitResult(
        warp=warp,
        weights=weights,
        affine=affine,
        residuals=residuals,
        rms_residual=rms,
        bending_energy=energy,
        rank=rank,
        merged_count=n,
        confidence=list(sigmas) if confidence_explicit else None,
        effective_count=len(active),
    )
