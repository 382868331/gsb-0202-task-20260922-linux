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

from ._linalg import (
    lu_factor,
    lu_solve,
    affine_rank,
    SingularMatrixError,
)

__all__ = [
    "MAX_CONTROL_POINTS",
    "TPSInputError",
    "DegeneratePointsError",
    "TPSFitResult",
    "TPSWarp",
    "TPSFitter",
    "fit",
]

MAX_CONTROL_POINTS = 40


class TPSInputError(ValueError):
    """Invalid input to TPS fitting or evaluation (shape, NaN, duplicates...)."""


class DegeneratePointsError(TPSInputError):
    """Affine design matrix rank < 3 (collinear / insufficient points)."""


# Computations that depend only on the source control points and lambda.  When
# targets are updated without touching the sources or the regularisation, all
# of these are reused and their names are reported in TPSFitResult.reused.
CACHED_COMPUTATIONS = (
    "source_validation",
    "source_dedup",
    "normalization",
    "affine_rank",
    "system_matrix_assembly",
    "lu_factorization",
    "energy_kernel_table",
)


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

    reused: tuple = ()

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

def _merge_source_only(sources):
    """Collapse repeated source points without checking targets.

    Returns the distinct sources in first-seen order plus, for every
    distinct source, the list of input indices mapping to it.  The merge
    rule is exactly `_merge_duplicates`: identical (x, y) floats collapse to
    one control point; conflicting targets are checked separately against
    each distinct target group.
    """
    distinct = []
    index_of = {}
    groups = []
    for k, s in enumerate(sources):
        if s in index_of:
            groups[index_of[s]].append(k)
        else:
            index_of[s] = len(distinct)
            distinct.append(s)
            groups.append([k])
    return distinct, groups


def _prepare_source_model(sources, lam_f):
    """Build every quantity that depends only on sources and lambda.

    This is the cacheable half of a fit.  It performs source validation,
    duplicate merging, normalisation, the affine-rank check, assembly of the
    ``(n+3) x (n+3)`` TPS system matrix, its LU factorisation, and the kernel
    table reused by the bending-energy quadratic form.  A raised error leaves
    no partial cache behind (the caller simply discards the result).
    """
    src, groups = _merge_source_only(sources)
    n = len(src)
    if n < 3:
        raise DegeneratePointsError(
            f"need at least 3 distinct source points after merging, got {n}"
        )

    # Internal normalisation (see the module docstring of fit) for numerical
    # robustness with pixel-scale coordinates.
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
    nlam = lam_f / (diameter * diameter)

    # Rank of the (normalised, scale-independent) affine design P = [1,x,y].
    Pn = [[1.0, sx, sy] for (sx, sy) in nsrc]
    rank = affine_rank(Pn)
    if rank < 3:
        raise DegeneratePointsError(
            f"affine design matrix rank is {rank} < 3: control points are "
            f"collinear or fewer than 3 distinct points after merging"
        )

    # Assemble the (n+3) x (n+3) TPS system  [K+lam~ I  P; P^T 0].  The
    # off-diagonal kernel entries are stored once (in A); the diagonal is
    # nlam and the bending-energy sum skips the U(0)=0 diagonal.
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
        A[i][n] = 1.0
        A[i][n + 1] = xi
        A[i][n + 2] = yi
        A[n][i] = 1.0
        A[n + 1][i] = xi
        A[n + 2][i] = yi

    try:
        factor = lu_factor(A)
    except SingularMatrixError as exc:
        raise DegeneratePointsError(
            f"TPS system numerically singular despite rank(P)={rank}: {exc}"
        ) from exc

    # Symmetric kernel table for the bending energy (diagonal is U(0)=0).
    kernel_table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = nsrc[i]
        for j in range(i + 1, n):
            kij = A[i][j]
            kernel_table[i][j] = kij
            kernel_table[j][i] = kij

    return {
        "sources": src,
        "groups": groups,
        "n": n,
        "cx": cx,
        "cy": cy,
        "diameter": diameter,
        "nsrc": nsrc,
        "rank": rank,
        "factor": factor,
        "kernel_table": kernel_table,
    }


def _check_targets_against_groups(targets, groups, distinct_sources, raw_count):
    """Validate targets and the unchanged duplicate-merge conflict rule."""
    tgt = _check_points(targets, "targets")
    if len(tgt) != raw_count:
        raise TPSInputError(
            f"sources/targets length mismatch: {raw_count} != {len(tgt)}"
        )
    merged_targets = []
    for gi, group in enumerate(groups):
        first = tgt[group[0]]
        for idx in group[1:]:
            if tgt[idx] != first:
                raise TPSInputError(
                    f"duplicate source point {distinct_sources[gi]} maps to "
                    f"conflicting targets "
                    f"{first} and {tgt[idx]}"
                )
        merged_targets.append(first)
    return tgt, merged_targets


def _solve_for_targets(model, merged_targets, reused):
    """Solve the two coordinate systems for one set of (merged) targets.

    Everything in ``model`` (normalisation, assembled matrix, LU factor,
    kernel table) is reused; only the right-hand sides, back-conversion,
    residuals and bending energy are (re)computed.
    """
    n = model["n"]
    diameter = model["diameter"]
    nsrc = model["nsrc"]
    ntgt = [[vx / diameter, vy / diameter] for (vx, vy) in merged_targets]

    rhs_x = [t[0] for t in ntgt] + [0.0, 0.0, 0.0]
    rhs_y = [t[1] for t in ntgt] + [0.0, 0.0, 0.0]

    # The cached LU factor is reused directly; it cannot become singular
    # between factorisations, so the only failure mode here is bad inputs,
    # already rejected above.
    sol_x, sol_y = lu_solve(model["factor"], [rhs_x, rhs_y])

    # Convert normalised coefficients back to the caller's units.
    # U(s*r) = s^2 U(r) + s^2 ln(s) ||r||^2; with P^T w = 0 the quadratic
    # extra term reduces, at every point, to the constant
    # s*ln(s) * sum(w~_i ||c~_i||^2), absorbed into a0.
    s = diameter
    cx, cy = model["cx"], model["cy"]
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

    src = model["sources"]
    warp = TPSWarp(control_points=src, weights=weights, affine=affine)

    residuals = []
    sq_sum = 0.0
    for i, pt in enumerate(src):
        fx, fy = warp.evaluate(pt)
        ex = fx - merged_targets[i][0]
        ey = fy - merged_targets[i][1]
        residuals.append(math.hypot(ex, ey))
        sq_sum += ex * ex + ey * ey
    rms = math.sqrt(sq_sum / (2.0 * n))

    # Bending energy via the cached normalised kernel table (identical to
    # w^T K w in user units), summed over the two output coordinates.
    table = model["kernel_table"]
    energy = 0.0
    for w_coord in (wx_n, wy_n):
        acc = 0.0
        for i in range(n):
            wi = w_coord[i]
            row = table[i]
            for j in range(i + 1, n):
                acc += 2.0 * wi * w_coord[j] * row[j]
        energy += acc

    return TPSFitResult(
        warp=warp,
        weights=weights,
        affine=affine,
        residuals=residuals,
        rms_residual=rms,
        bending_energy=energy,
        rank=model["rank"],
        merged_count=n,
        reused=tuple(reused),
    )


def fit(sources, targets, lam=0.0):
    """Fit a 2-D thin-plate spline (stateless one-shot entry point).

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
    if len(src) > MAX_CONTROL_POINTS:
        raise TPSInputError(
            f"at most {MAX_CONTROL_POINTS} control points allowed, got {len(src)}"
        )
    # bool is rejected inside _finite_float; also guard lam explicitly.
    lam_f = _finite_float(lam, "lam")
    if lam_f < 0.0:
        raise TPSInputError(f"lam must be >= 0, got {lam_f}")

    # Basic target validation (shape, finiteness, length) before any source
    # geometry work, so shape errors are reported exactly as before.
    raw_targets = _check_points(targets, "targets")
    if len(src) != len(raw_targets):
        raise TPSInputError(
            f"sources/targets length mismatch: {len(src)} != {len(raw_targets)}"
        )

    model = _prepare_source_model(src, lam_f)
    _, merged_targets = _check_targets_against_groups(
        targets, model["groups"], model["sources"], len(src)
    )
    # One-shot fits perform every computation once; nothing is reused.
    return _solve_for_targets(model, merged_targets, ())


class TPSFitter:
    """Reusable TPS solver for a fixed source configuration.

    Source control points and the regularisation parameter normally stay
    fixed while a user drags target points interactively.  Constructing the
    fitter performs, exactly once, every source/lambda-dependent computation
    (validation, duplicate merging, normalisation, affine-rank check, system
    matrix assembly and LU factorisation, and the bending-energy kernel
    table).  Each `fit` with new targets only substitutes new right-hand
    sides and back-converts the coefficients.

    Cache invalidation
    ------------------
    The cache is tied to the ``(sources, lam)`` given at construction.  A
    change to source positions or to the regularisation requires a new
    fitter (there is no in-place mutation API), so a stale factorisation can
    never be used.  A failed target update raises and leaves the previous
    valid `result`/`warp` untouched.
    """

    def __init__(self, sources, lam=0.0, targets=None):
        src = _check_points(sources, "sources")
        if len(src) > MAX_CONTROL_POINTS:
            raise TPSInputError(
                f"at most {MAX_CONTROL_POINTS} control points allowed, got "
                f"{len(src)}"
            )
        lam_f = _finite_float(lam, "lam")
        if lam_f < 0.0:
            raise TPSInputError(f"lam must be >= 0, got {lam_f}")

        # Build the cache before validating any targets: source/model errors
        # must surface even if no targets were supplied.
        model = _prepare_source_model(src, lam_f)

        self._sources = src
        self._lam = lam_f
        self._model = model
        self._result = None
        self._warp = None

        if targets is not None:
            self.fit(targets)

    @property
    def sources(self):
        """Validated source points as accepted at construction (raw input)."""
        return [list(p) for p in self._sources]

    @property
    def control_points(self):
        """Distinct (merged) source points the spline is anchored at."""
        return [list(p) for p in self._model["sources"]]

    @property
    def lam(self):
        """Regularisation parameter the cached factorisation was built with."""
        return self._lam

    @property
    def rank(self):
        """Numerical affine-rank recorded by the cached model."""
        return self._model["rank"]

    @property
    def merged_count(self):
        """Number of distinct source control points after merging."""
        return self._model["n"]

    @property
    def result(self):
        """Last valid `TPSFitResult`, or ``None`` before the first fit."""
        return self._result

    @property
    def warp(self):
        """Warp of the last successful fit, or ``None`` before the first."""
        return self._warp

    def cached_computations(self):
        """Names of computations reused by target-only refits."""
        return list(CACHED_COMPUTATIONS)

    def fit(self, targets):
        """Refit for new target positions, reusing the cached source model.

        Targets go through the same validation and the same conflicting-
        duplicate rule as `tps2d.fit`.  On failure the call raises before any
        cached state is replaced, so the previous result/warp remain usable.
        """
        _, merged_targets = _check_targets_against_groups(
            targets, self._model["groups"], self._model["sources"],
            len(self._sources),
        )
        result = _solve_for_targets(
            self._model, merged_targets, CACHED_COMPUTATIONS
        )
        self._result = result
        self._warp = result.warp
        return result

    def update_targets(self, targets):
        """Alias for `fit` (update target positions in place on the fitter)."""
        return self.fit(targets)
