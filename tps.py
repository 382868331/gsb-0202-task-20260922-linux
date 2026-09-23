"""Thin-plate spline (TPS) deformation for 2D surface measurements.

Pure-Python (standard library only) implementation of the classical
Bookstein thin-plate spline for point-based surface registration, e.g.
offline medical image/surface alignment.

Model
-----
Given ``n`` control points ``c_i = (x_i, y_i)`` and target positions
``t_i``, the warp for each output coordinate independently is

    f(p) = a0 + a1*x + a2*y + sum_i w_i * U(|p - c_i|)

with radial kernel ``U(r) = r^2 * log(r)`` and ``U(0) = 0``.  The
coefficients ``[w, a]`` solve the block system

    [ K + lambda*I   P ] [ w ]   [ t ]
    [     P^T        0 ] [ a ] = [ 0 ]

where ``K_ij = U(|c_i - c_j|)`` and ``P = [1, x, y]`` (n x 3).  The
bottom block enforces the side condition ``P^T w = 0``.  The two output
coordinates share the same system matrix and are solved independently.

Error semantics
---------------
* ``TPSInputError``     -- invalid user input (type, finiteness, shape,
  limits, conflicting duplicates, negative lambda).  Raised before any
  computation; nothing is silently corrected.
* ``TPSDegenerateError`` -- the distinct control points do not span the
  affine part (fewer than 3 points, or all collinear), so no unique TPS
  solution exists.
* ``TPSSingularError``  -- the configuration passed the degeneracy check
  but the (n+3) x (n+3) system was numerically singular to solve
  (severely ill-conditioned input).

All numeric inputs must be real numbers (``int``/``float``; ``bool`` is
rejected) and finite (NaN and Infinity are rejected).
"""

from __future__ import annotations

import math

__all__ = [
    "TPSError",
    "TPSInputError",
    "TPSDegenerateError",
    "TPSSingularError",
    "TPS",
    "fit_tps",
    "kernel",
    "MAX_CONTROL_POINTS",
    "MIN_CONTROL_POINTS",
]

MAX_CONTROL_POINTS = 40
MIN_CONTROL_POINTS = 3

# Relative thresholds.  Matrix entries are O(1)..O(L^2 |log L|) for
# coordinate extent L; these tolerances sit far above double-precision
# round-off for n <= 40 and far below pivots of well-posed problems.
_RANK_RTOL = 1e-12
_PIVOT_RTOL = 1e-13


class TPSError(Exception):
    """Base class for all errors raised by this module."""


class TPSInputError(TPSError, ValueError):
    """Invalid user input: type, finiteness, shape, limits, duplicates."""


class TPSDegenerateError(TPSError):
    """Control points do not span the affine part (rank of P < 3)."""


class TPSSingularError(TPSError, ArithmeticError):
    """TPS block system numerically singular despite non-degenerate input."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TPSInputError(
            f"{name}: expected a real number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise TPSInputError(f"{name}: value must be finite, got {value!r}")
    return v


def _parse_point(p, name):
    try:
        x, y = p
    except (TypeError, ValueError):
        raise TPSInputError(
            f"{name}: expected an (x, y) pair, got {p!r}") from None
    return (_finite_float(x, f"{name}[0]"), _finite_float(y, f"{name}[1]"))


def _parse_points(points, name):
    if isinstance(points, (str, bytes)):
        raise TPSInputError(f"{name}: expected an iterable of (x, y) pairs")
    try:
        items = list(points)
    except TypeError:
        raise TPSInputError(
            f"{name}: expected an iterable of (x, y) pairs") from None
    return [_parse_point(p, f"{name}[{i}]") for i, p in enumerate(items)]


def _parse_lambda(lam):
    v = _finite_float(lam, "lambda_")
    if v < 0.0:
        raise TPSInputError(f"lambda_: must be >= 0, got {v!r}")
    return v


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def kernel(r):
    """TPS radial kernel U(r) = r^2 * log(r), with U(0) = 0.

    ``r`` must be a finite real number >= 0.
    """
    r = _finite_float(r, "r")
    if r < 0.0:
        raise TPSInputError(f"r: radius must be >= 0, got {r!r}")
    if r == 0.0:
        return 0.0
    return r * r * math.log(r)


# ---------------------------------------------------------------------------
# Linear algebra (dense, pure Python; n + 3 <= 43 unknowns)
# ---------------------------------------------------------------------------

def _solve(a, b):
    """Solve a x = b by Gaussian elimination with partial pivoting."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    scale = max((abs(v) for row in a for v in row), default=0.0)
    tol = _PIVOT_RTOL * (scale if scale > 0.0 else 1.0)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) <= tol:
            raise TPSSingularError(
                f"TPS system numerically singular at column {col}")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] * inv
            if f == 0.0:
                continue
            m[r][col] = 0.0
            for c in range(col + 1, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n]
        for c in range(r + 1, n):
            s -= m[r][c] * x[c]
        x[r] = s / m[r][r]
    return x


def _check_affine_rank(points):
    """Raise TPSDegenerateError unless points span the 2D affine part."""
    x0, y0 = points[0]
    vecs = [(x - x0, y - y0) for x, y in points[1:]]
    scale = 0.0
    for vx, vy in vecs:
        scale = max(scale, abs(vx), abs(vy))
    if scale == 0.0:
        raise TPSDegenerateError(
            "all control points coincide; affine rank < 3")
    best = 0.0
    for i in range(len(vecs)):
        vix, viy = vecs[i]
        for j in range(i + 1, len(vecs)):
            cross = abs(vix * vecs[j][1] - viy * vecs[j][0])
            if cross > best:
                best = cross
    if best <= _RANK_RTOL * scale * scale:
        raise TPSDegenerateError(
            "control points are collinear; affine rank < 3, "
            "TPS solution is not unique")


def _merge_duplicates(src, tgt):
    """Merge exact duplicate source points.

    Identical (source, target) pairs collapse to one control point;
    identical sources with different targets are rejected.
    """
    seen = {}
    for i, (s, t) in enumerate(zip(src, tgt)):
        if s in seen:
            j, t0 = seen[s]
            if t != t0:
                raise TPSInputError(
                    f"duplicate source point {s!r} at indices {j} and {i} "
                    f"with conflicting targets {t0!r} and {t!r}")
        else:
            seen[s] = (i, t)
    points = list(seen.keys())
    targets = [seen[s][1] for s in points]
    return points, targets


def _bending_energy(kmat, w):
    n = len(w)
    total = 0.0
    for i in range(n):
        wi = w[i]
        if wi == 0.0:
            continue
        row = kmat[i]
        acc = 0.0
        for j in range(n):
            acc += row[j] * w[j]
        total += wi * acc
    return total


# ---------------------------------------------------------------------------
# Public fit entry point
# ---------------------------------------------------------------------------

def fit_tps(source, target, lambda_=0.0):
    """Fit a 2D thin-plate spline from control points to target points.

    Parameters
    ----------
    source, target : iterable of (x, y) pairs
        Corresponding control/target positions.  Between
        ``MIN_CONTROL_POINTS`` and ``MAX_CONTROL_POINTS`` pairs; the
        limit applies to the pairs as supplied (before duplicate
        merging).  Exact duplicate sources with identical targets are
        merged; with different targets they raise ``TPSInputError``.
    lambda_ : real number >= 0
        Tikhonov regularization weight (diagonal of the K block).
        ``lambda_ = 0`` interpolates exactly.

    Returns
    -------
    TPS
        Fitted model with warp/jacobian methods and fit diagnostics.

    Raises
    ------
    TPSInputError, TPSDegenerateError, TPSSingularError
    """
    src = _parse_points(source, "source")
    tgt = _parse_points(target, "target")
    lam = _parse_lambda(lambda_)
    if len(src) != len(tgt):
        raise TPSInputError(
            f"source/target length mismatch: {len(src)} vs {len(tgt)}")
    if len(src) > MAX_CONTROL_POINTS:
        raise TPSInputError(
            f"at most {MAX_CONTROL_POINTS} control points allowed, "
            f"got {len(src)}")
    points, targets = _merge_duplicates(src, tgt)
    n = len(points)
    if n < MIN_CONTROL_POINTS:
        raise TPSInputError(
            f"need at least {MIN_CONTROL_POINTS} distinct control points, "
            f"got {n}")
    _check_affine_rank(points)

    # K block (without lambda) is kept for the bending energy.
    kmat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            dx = xi - points[j][0]
            dy = yi - points[j][1]
            r = math.hypot(dx, dy)
            u = r * r * math.log(r) if r > 0.0 else 0.0
            kmat[i][j] = u
            kmat[j][i] = u

    size = n + 3
    amat = [[0.0] * size for _ in range(size)]
    for i in range(n):
        row = amat[i]
        for j in range(n):
            row[j] = kmat[i][j]
        row[i] += lam
        row[n] = 1.0
        row[n + 1] = points[i][0]
        row[n + 2] = points[i][1]
        amat[n][i] = 1.0
        amat[n + 1][i] = points[i][0]
        amat[n + 2][i] = points[i][1]

    rhs_x = [t[0] for t in targets] + [0.0, 0.0, 0.0]
    rhs_y = [t[1] for t in targets] + [0.0, 0.0, 0.0]
    sol_x = _solve(amat, rhs_x)
    sol_y = _solve(amat, rhs_y)

    weights = (tuple(sol_x[:n]), tuple(sol_y[:n]))
    affine = (tuple(sol_x[n:]), tuple(sol_y[n:]))

    tps = TPS(
        control_points=tuple(points),
        targets=tuple(targets),
        lambda_=lam,
        weights=weights,
        affine=affine,
    )

    residuals = []
    for p, t in zip(points, targets):
        fx, fy = tps._warp_one(p)
        residuals.append((fx - t[0], fy - t[1]))
    tps.residuals = tuple(residuals)
    sq = [dx * dx + dy * dy for dx, dy in residuals]
    tps.rms_error = math.sqrt(sum(sq) / n)
    tps.max_error = math.sqrt(max(sq))

    be_x = _bending_energy(kmat, weights[0])
    be_y = _bending_energy(kmat, weights[1])
    tps.bending_energy_per_axis = (be_x, be_y)
    tps.bending_energy = be_x + be_y
    return tps


# ---------------------------------------------------------------------------
# Fitted model
# ---------------------------------------------------------------------------

class TPS:
    """A fitted thin-plate spline.  Construct via :func:`fit_tps`.

    Attributes
    ----------
    control_points : tuple of (x, y)
        Distinct source control points after duplicate merging.
    targets : tuple of (x, y)
        Target positions aligned with ``control_points``.
    lambda_ : float
        Regularization weight used in the fit.
    weights : (tuple, tuple)
        Kernel weights ``(w_x, w_y)`` for the two output coordinates.
    affine : (tuple, tuple)
        Affine coefficients ``((a0, a1, a2), (b0, b1, b2))``; the warp is
        ``a0 + a1*x + a2*y`` plus the kernel sum, per coordinate.
    residuals : tuple of (dx, dy)
        ``warped control point - target`` at every control point.
    rms_error, max_error : float
        RMS / max Euclidean residual norm over the control points.
    bending_energy, bending_energy_per_axis : float / (float, float)
        ``w^T K w`` (unregularized K), total and per output coordinate.
    """

    __slots__ = (
        "control_points", "targets", "lambda_", "weights", "affine",
        "residuals", "rms_error", "max_error",
        "bending_energy", "bending_energy_per_axis",
    )

    def __init__(self, control_points, targets, lambda_, weights, affine):
        self.control_points = control_points
        self.targets = targets
        self.lambda_ = lambda_
        self.weights = weights
        self.affine = affine
        self.residuals = ()
        self.rms_error = 0.0
        self.max_error = 0.0
        self.bending_energy = 0.0
        self.bending_energy_per_axis = (0.0, 0.0)

    def __repr__(self):
        return (f"TPS(n={len(self.control_points)}, "
                f"lambda_={self.lambda_!r}, "
                f"rms_error={self.rms_error:.3e}, "
                f"bending_energy={self.bending_energy:.6e})")

    def _warp_one(self, p):
        x, y = p
        (a0, a1, a2), (b0, b1, b2) = self.affine
        fx = a0 + a1 * x + a2 * y
        fy = b0 + b1 * x + b2 * y
        for (cx, cy), wx, wy in zip(
                self.control_points, self.weights[0], self.weights[1]):
            dx = x - cx
            dy = y - cy
            r2 = dx * dx + dy * dy
            if r2 > 0.0:
                u = 0.5 * r2 * math.log(r2)
                fx += wx * u
                fy += wy * u
        return (fx, fy)

    def warp_point(self, point):
        """Warp a single ``(x, y)`` point; returns an ``(x, y)`` tuple."""
        return self._warp_one(_parse_point(point, "point"))

    def warp(self, points):
        """Warp an iterable of ``(x, y)`` points; returns a list of tuples."""
        pts = _parse_points(points, "points")
        return [self._warp_one(p) for p in pts]

    def jacobian(self, point):
        """Analytic Jacobian of the warp at ``point``.

        Returns ``((dfx/dx, dfx/dy), (dfy/dx, dfy/dy))``.  For
        ``U(r) = r^2 log r`` the derivative term per control point is
        ``(log(r^2) + 1) * (p - c)``, which vanishes at ``r = 0``.
        """
        x, y = _parse_point(point, "point")
        (a0, a1, a2), (b0, b1, b2) = self.affine
        jxx, jxy = a1, a2
        jyx, jyy = b1, b2
        for (cx, cy), wx, wy in zip(
                self.control_points, self.weights[0], self.weights[1]):
            dx = x - cx
            dy = y - cy
            r2 = dx * dx + dy * dy
            if r2 > 0.0:
                c = math.log(r2) + 1.0
                jxx += wx * c * dx
                jxy += wx * c * dy
                jyx += wy * c * dx
                jyy += wy * c * dy
        return ((jxx, jxy), (jyx, jyy))
