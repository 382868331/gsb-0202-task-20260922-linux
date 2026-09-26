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
from dataclasses import dataclass, field

from ._linalg import (
    lu_factor,
    lu_solve,
    affine_rank,
    is_symmetric_psd,
    SingularMatrixError,
)

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


_NUMERIC_SYM_TOL = 100.0 * 2.220446049250313e-16


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


def _check_covariance_matrix(matrix, size, name):
    if not isinstance(matrix, (list, tuple)):
        raise TPSInputError(f"{name} must be a list/tuple of numeric rows")
    if len(matrix) != size:
        raise TPSInputError(
            f"{name} must be {size} x {size} for {size // 2} target points, "
            f"got {len(matrix)} rows"
        )

    cleaned = []
    scale = 0.0
    for i, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)) or len(row) != size:
            raise TPSInputError(
                f"{name}[{i}] must be a {size}-element numeric row, got {row!r}"
            )
        clean_row = []
        for j, value in enumerate(row):
            clean_row.append(_finite_float(value, f"{name}[{i}][{j}]"))
            scale = max(scale, abs(clean_row[-1]))
        cleaned.append(clean_row)

    sym_tol = max(_NUMERIC_SYM_TOL * scale, 100.0 * _NUMERIC_SYM_TOL)
    for i in range(size):
        for j in range(i + 1, size):
            if abs(cleaned[i][j] - cleaned[j][i]) > sym_tol:
                raise TPSInputError(
                    f"{name} must be symmetric: [{i}][{j}]={cleaned[i][j]} "
                    f"differs from [{j}][{i}]={cleaned[j][i]}"
                )
            avg = 0.5 * (cleaned[i][j] + cleaned[j][i])
            cleaned[i][j] = cleaned[j][i] = avg

    if not is_symmetric_psd(cleaned, tolerance=sym_tol):
        raise TPSInputError(f"{name} must be symmetric positive semidefinite")
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


def _snap_psd_2x2(mat):
    """Remove only round-off-scale negative eigenvalues from H C H^T."""
    a, b = mat[0]
    c, d = mat[1]
    b = c = 0.5 * (b + c)
    a += 0.0
    d += 0.0
    trace = a + d
    discriminant = (a - d) * (a - d) + 4.0 * b * b
    spread = math.sqrt(max(0.0, discriminant))
    lambda_min = 0.5 * (trace - spread)
    scale = max(1.0, abs(a), abs(d), abs(b))
    rounding_tol = 1000.0 * 2.220446049250313e-16 * scale

    if lambda_min >= -rounding_tol and lambda_min < 0.0:
        if spread <= rounding_tol:
            a = d = max(0.0, 0.5 * trace)
            b = c = 0.0
        else:
            correction = -lambda_min
            a += correction
            d += correction
    return [[max(0.0, a), b], [c, max(0.0, d)]]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TPSPrediction:
    """Mean prediction and target-error covariance at one query point."""

    mean: list
    covariance: list
    influence: list

    @property
    def influence_matrix(self):
        """2 x 2n influence matrix in point-major target order.

        Columns are ordered ``[x_0, y_0, x_1, y_1, ...]``.  Thus
        ``mean_error = influence_matrix @ target_error``.
        """
        row_x = []
        row_y = []
        for block in self.influence:
            row_x.extend((block[0][0], block[0][1]))
            row_y.extend((block[1][0], block[1][1]))
        return [row_x, row_y]


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
    _fixed_design: object = field(default=None, repr=False, compare=False)

    @property
    def coefficients(self):
        """Tuple ``(weights, affine)`` exactly as stored."""
        return self.weights, self.affine

    def influence_at(self, point):
        """Return per-control-point influence blocks for one query point.

        Each block is a 2 x 2 Jacobian of the predicted x/y with respect to
        that control point's target x/y.  Source positions are treated as
        fixed; only target-coordinate uncertainty is propagated.
        """
        if self._fixed_design is None:
            raise TPSInputError("this fitted result does not contain a fixed design")

        factor, nsrc, center, scale = self._fixed_design
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise TPSInputError("point must be a 2-element [x, y] pair")
        x = _finite_float(point[0], "point.x")
        y = _finite_float(point[1], "point.y")

        nx = (x - center[0]) / scale
        ny = (y - center[1]) / scale
        n = len(nsrc)
        query_basis = [_U(nx - sx, ny - sy) for sx, sy in nsrc]
        query_basis.extend((1.0, nx, ny))
        right_sides = []
        for j in range(n):
            rhs = [0.0] * (n + 3)
            rhs[j] = 1.0
            right_sides.append(rhs)
        responses = lu_solve(factor, right_sides)

        influences = []
        for response in responses:
            value = sum(
                query_basis[k] * response[k] for k in range(n + 3)
            )
            influences.append([[value, 0.0], [0.0, value]])
        return influences

    def predict(self, point, target_covariance):
        """Predict a point and propagate fixed-design target covariance.

        ``target_covariance`` is a ``2n x 2n`` finite, real, symmetric positive
        semidefinite matrix using point-major order
        ``[x_0, y_0, x_1, y_1, ...]`` for the ``merged_count`` target points.
        Correlations between different control points and between x/y are
        allowed.  The query point itself is deterministic and is not included
        in the returned covariance.
        """
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise TPSInputError("point must be a 2-element [x, y] pair")
        x = _finite_float(point[0], "point.x")
        y = _finite_float(point[1], "point.y")

        n = self.merged_count
        size = 2 * n
        covariance = _check_covariance_matrix(
            target_covariance, size, "target_covariance"
        )

        # Validate/query with the fixed design before performing propagation.
        influence = self.influence_at([x, y])
        hx = [block[0][0] for block in influence]
        hy = [block[1][1] for block in influence]

        cxx = 0.0
        cxy = 0.0
        cyy = 0.0
        for i in range(n):
            for j in range(n):
                cxx += hx[i] * covariance[2 * i][2 * j] * hx[j]
                cxy += hx[i] * covariance[2 * i][2 * j + 1] * hy[j]
                cyy += hy[i] * covariance[2 * i + 1][2 * j + 1] * hy[j]

        mean = self.warp.evaluate([x, y])
        out_cov = _snap_psd_2x2([[cxx, cxy], [cxy, cyy]])
        return TPSPrediction(
            mean=mean,
            covariance=out_cov,
            influence=influence,
        )


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
        factor = lu_factor(A)
        sol_x, sol_y = lu_solve(factor, [rhs_x, rhs_y])
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
        _fixed_design=(factor, nsrc, (cx, cy), diameter),
    )
