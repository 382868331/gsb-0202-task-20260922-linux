"""Two-dimensional thin-plate spline warps (standard library only).

Implements the classical 2-D thin-plate spline

    f(p) = a0 + a1*x + a2*y + sum_i w_i * U(||p - c_i||),
    U(r) = r**2 * log(r),  U(0) = 0,

fitted independently for the two output coordinates by solving

    [ K + lambda*I   P ] [w]   [v]
    [ P^T            0 ] [a] = [0],   P_i = [1, x_i, y_i],

which enforces P^T w = 0.  Inputs are validated strictly: NaN/Infinity are
rejected, booleans are not accepted as numbers, duplicate source points with
conflicting targets raise, and control-point configurations whose affine
part is rank deficient (all collinear, or fewer than 3 distinct points after
merging) raise `DegeneratePointsError`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._linalg import factorize, affine_rank, SingularMatrixError

__all__ = [
    "MAX_CONTROL_POINTS",
    "TPSInputError",
    "DegeneratePointsError",
    "TPSFitResult",
    "TPSPrediction",
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
# Covariance validation
# ---------------------------------------------------------------------------

def _check_covariance(cov, dim, name="target_cov"):
    """Validate a symmetric positive-semidefinite covariance matrix.

    Returns a cleaned list-of-lists of floats.  Zero eigenvalue directions
    are allowed (semidefinite); asymmetric or indefinite matrices, wrong
    shapes, non-finite entries and booleans are rejected *before* any
    computation with a locatable TPSInputError.
    """
    if not isinstance(cov, (list, tuple)) or len(cov) != dim:
        raise TPSInputError(
            f"{name} must be a {dim}x{dim} matrix, got "
            f"{len(cov) if isinstance(cov, (list, tuple)) else type(cov).__name__} rows"
        )
    mat = []
    for i, row in enumerate(cov):
        if not isinstance(row, (list, tuple)) or len(row) != dim:
            raise TPSInputError(
                f"{name}[{i}] must have {dim} columns, got "
                f"{len(row) if isinstance(row, (list, tuple)) else type(row).__name__}"
            )
        mat.append([_finite_float(v, f"{name}[{i}][{j}]") for j, v in enumerate(row)])

    scale = 0.0
    for row in mat:
        for value in row:
            if abs(value) > scale:
                scale = abs(value)
    sym_tol = 1e-9 * (scale if scale > 0.0 else 1.0)
    for i in range(dim):
        for j in range(i + 1, dim):
            if abs(mat[i][j] - mat[j][i]) > sym_tol:
                raise TPSInputError(
                    f"{name} must be symmetric: [{i}][{j}]={mat[i][j]} vs "
                    f"[{j}][{i}]={mat[j][i]}"
                )

    # Positive-semidefinite check via Cholesky with zero-pivot handling.
    # A (near-)zero pivot is only acceptable when the whole remaining
    # column of the Schur complement vanishes (zero eigenvalue direction).
    psd_tol = 1e-9 * (scale if scale > 0.0 else 1.0)
    lower = [[0.0] * dim for _ in range(dim)]
    for k in range(dim):
        d = mat[k][k]
        for j in range(k):
            d -= lower[k][j] * lower[k][j]
        if d < -psd_tol:
            raise TPSInputError(
                f"{name} is not positive semidefinite: negative pivot "
                f"{d:.6e} at row {k}"
            )
        if d <= psd_tol:
            # Zero eigenvalue direction: the residual column must vanish.
            for i in range(k + 1, dim):
                off = mat[i][k]
                for j in range(k):
                    off -= lower[i][j] * lower[k][j]
                if abs(off) > psd_tol:
                    raise TPSInputError(
                        f"{name} is not positive semidefinite: zero pivot at "
                        f"row {k} but residual column entry {off:.6e} at row {i}"
                    )
            continue
        lkk = math.sqrt(d)
        lower[k][k] = lkk
        for i in range(k + 1, dim):
            off = mat[i][k]
            for j in range(k):
                off -= lower[i][j] * lower[k][j]
            lower[i][k] = off / lkk
    return mat
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
        Numerical rank of the affine design matrix P (3 for a valid fit).
    merged_count:
        Number of distinct source points after same-target duplicates were
        merged (may be < the number of input pairs).
    """

    warp: "TPSWarp"
    weights: list
    affine: list
    residuals: list
    rms_residual: float
    bending_energy: float
    rank: int
    merged_count: int

    # Fixed-design system retained for predict_with_covariance().  These
    # are populated by fit() and are not part of the public contract.
    _factor: object = None
    _center: tuple = (0.0, 0.0)
    _scale: float = 1.0
    _nsrc: list = None

    @property
    def coefficients(self):
        """Tuple ``(weights, affine)`` exactly as stored."""
        return self.weights, self.affine

    # -- uncertainty propagation (fixed design, linear in the targets) -----

    def predict_with_covariance(self, points, target_cov):
        """Predict query points with covariance propagated from the targets.

        The control-point *positions* are fixed (the fitted system is the
        fixed design); only the target coordinates are uncertain, with
        covariance target_cov (a 2n x 2n symmetric positive-semidefinite
        matrix over the merged targets stacked as
        [x1, y1, x2, y2, ..., xn, yn]).  Correlated errors between
        control points are fully honoured through the off-diagonal blocks.

        Because the warp is linear in the targets, the prediction is exact
        linear propagation: f(p) = H(p) @ vec(targets) with a 2 x 2n
        influence matrix H, so Cov(f(p)) = H target_cov H^T.  The
        factorisation computed during fit() is reused; no refit happens.
        The query position itself is treated as exact.

        Returns a TPSPrediction with means, covariances (2x2 per query)
        and influence (per query [[hx_1..hx_n], [hy_1..hy_n]]).
        """
        clean = _check_points(points, 'points')
        n = self.merged_count
        cov = _check_covariance(target_cov, 2 * n)

        factor = self._factor
        if factor is None:  # defensive: result built without a system
            raise TPSInputError(
                "this TPSFitResult carries no factorisation; refit with fit()"
            )
        cx, cy = self._center
        s = self._scale
        size = n + 3

        # Influence columns M = A^-1 E (E selects the target rows), computed
        # once per call by re-solving against the stored factorisation.
        cols = [
            [1.0 if r == k else 0.0 for r in range(size)] for k in range(n)
        ]
        M = factor.solve(cols)

        means = []
        covariances = []
        influence = []
        for (x, y) in clean:
            nx = (x - cx) / s
            ny = (y - cy) / s
            basis = [
                _U(nx - px, ny - py) for (px, py) in self._nsrc
            ] + [1.0, nx, ny]
            h = [0.0] * n
            for k in range(n):
                col = M[k]
                acc = 0.0
                for r in range(size):
                    acc += basis[r] * col[r]
                h[k] = acc
            # f_user = s * f~ and v~ = v / s, so the same h applies to the
            # user-unit targets directly (the scale factors cancel).  The
            # two output coordinates share one system matrix, hence one h.
            influence.append([list(h), list(h)])

            # Mean from the fitted warp (identical to h . v up to round-off).
            means.append(self.warp.evaluate([x, y]))

            # Covariance: H is 2 x 2n with H[0][2k] = h[k] and
            # H[1][2k+1] = h[k]; C = H cov H^T (2x2).
            c00 = c01 = c11 = 0.0
            for i in range(n):
                hi = h[i]
                row_x = cov[2 * i]
                row_y = cov[2 * i + 1]
                for j in range(n):
                    hj = h[j]
                    c00 += hi * row_x[2 * j] * hj
                    c01 += hi * row_x[2 * j + 1] * hj
                    c11 += hi * row_y[2 * j + 1] * hj
            covariances.append([[c00, c01], [c01, c11]])

        return TPSPrediction(
            points=[[p[0], p[1]] for p in clean],
            means=means,
            covariances=covariances,
            influence=influence,
        )


@dataclass
class TPSPrediction:
    """Prediction with propagated target covariance at query points.

    Attributes
    ----------
    points:
        The validated query points (echoed back as floats).
    means:
        Predicted [x', y'] per query point; identical to warp.evaluate.
    covariances:
        2x2 covariance of the prediction per query point, propagated from
        the target covariance under the fixed-design linear model.
    influence:
        Per query point a 2 x n list of influence weights:
        influence[q][0][i] multiplies target x of control point i,
        influence[q][1][i] multiplies target y.  The query position
        itself carries no uncertainty.
    """

    points: list
    means: list
    covariances: list
    influence: list


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

def _merge_duplicates(sources, targets):
    """Collapse repeated source points.

    Same source + same target -> one control point.  Same source + different
    targets -> TPSInputError (no unique interpolant).  Returns merged lists.
    Coordinate equality is exact (already-converted floats); callers needing
    tolerance-based merging must pre-snap their own data.
    """
    seen = {}
    order = []
    for s, t in zip(sources, targets):
        if s in seen:
            if seen[s] != t:
                raise TPSInputError(
                    f"duplicate source point {s} maps to conflicting targets "
                    f"{seen[s]} and {t}"
                )
            # identical pair: silently merge
            continue
        seen[s] = t
        order.append(s)
    return order, [seen[s] for s in order]


def fit(sources, targets, lam=0.0):
    """Fit a 2-D thin-plate spline.

    Parameters
    ----------
    sources:
        1..40 control points ``[[x, y], ...]`` in the source/domain frame.
    targets:
        Corresponding points ``[[x', y'], ...]`` in the target frame.
    lam:
        Regularisation parameter lambda >= 0 added as ``lambda * I`` to K.
        ``lam = 0`` interpolates the (merged) control points exactly (up to
        floating-point round-off); larger values trade fit for smoothness.

    Returns
    -------
    TPSFitResult

    Raises
    ------
    TPSInputError
        Bad shapes/types, non-finite numbers, booleans-as-numbers,
        conflicting duplicates, or invalid lambda.
    DegeneratePointsError
        Fewer than 3 distinct source points or collinear configuration
        (rank of P < 3).
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

    src, tgt = _merge_duplicates(src, tgt)
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
    #     w = w~ / s,  a1 = a~1, a2 = a~2,  a0 = s*a~0 - a~1*cx - a~2*cy,
    # and the regulariser maps to  lam~ = lam / s^2.  Residuals and the
    # bending energy (w^T K w, scale invariant) are reported in user units.
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

    nsrc = [[(x - cx) / diameter, (y - cy) / diameter] for (x, y) in src]
    ntgt = [[vx / diameter, vy / diameter] for (vx, vy) in tgt]
    nlam = lam_f / (diameter * diameter)

    # Rank of the (normalised, scale-independent) affine design P = [1, x, y].
    Pn = [[1.0, sx, sy] for (sx, sy) in nsrc]
    rank = affine_rank(Pn)
    if rank < 3:
        raise DegeneratePointsError(
            f"affine design matrix rank is {rank} < 3: control points are "
            f"collinear or fewer than 3 distinct points after merging"
        )

    # Assemble the (n+3) x (n+3) TPS system  [K+lam~ I  P; P^T 0].
    size = n + 3
    A = [[0.0] * size for _ in range(size)]
    for i in range(n):
        xi, yi = nsrc[i]
        for j in range(i + 1, n):
            xj, yj = nsrc[j]
            kij = _U(xi - xj, yi - yj)
            A[i][j] = kij
            A[j][i] = kij
        A[i][i] = nlam
        # P block
        A[i][n] = 1.0
        A[i][n + 1] = xi
        A[i][n + 2] = yi
        # P^T block
        A[n][i] = 1.0
        A[n + 1][i] = xi
        A[n + 2][i] = yi

    rhs_x = [t[0] for t in ntgt] + [0.0, 0.0, 0.0]
    rhs_y = [t[1] for t in ntgt] + [0.0, 0.0, 0.0]

    try:
        factor = factorize(A)
        sol_x, sol_y = factor.solve([rhs_x, rhs_y])
    except SingularMatrixError as exc:
        # rank(P) >= 3 guarantees the block system is invertible for
        # lam~ >= 0 with the positive semidefinite TPS K; reaching here means
        # the configuration is so close to degenerate that pivoting failed.
        raise DegeneratePointsError(
            f"TPS system numerically singular despite rank(P)={rank}: {exc}"
        ) from exc

    # Convert normalised coefficients back to the caller's units.
    # U(s*r) = s^2 U(r) + s^2 ln(s) ||r||^2; with P^T w = 0 the quadratic
    # extra term reduces, at every point, to the constant
    # s*ln(s) * sum(w~_i ||c~_i||^2), which is absorbed into a0.
    s = diameter
    ln_s = math.log(s)
    wx_n, ax_n = sol_x[:n], sol_x[n:]
    wy_n, ay_n = sol_y[:n], sol_y[n:]

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
                xj, yj = nsrc[j]
                kij = _U(xi - xj, yi - yj)
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
        _factor=factor,
        _center=(cx, cy),
        _scale=s,
        _nsrc=nsrc,
    )
