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

from ._linalg import lu_factor, lu_solve, affine_rank, SingularMatrixError

__all__ = [
    "MAX_CONTROL_POINTS",
    "TPSInputError",
    "DegeneratePointsError",
    "TPSFitResult",
    "TPSWarp",
    "TPSFitter",
    "TPSFitCache",
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
    reused_components: tuple = ()
    cache_hit: bool = False

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

@dataclass
class _PreparedProblem:
    raw_sources: list
    source_points: list
    source_indices: list
    normalized_sources: list
    center: tuple
    diameter: float
    rank: int
    factorization: tuple
    kernel_entries: list


def _prepare_sources(sources, lam=0.0):
    """Build and factorise every part that depends only on sources and lam."""
    src = _check_points(sources, "sources")
    if len(src) > MAX_CONTROL_POINTS:
        raise TPSInputError(
            f"at most {MAX_CONTROL_POINTS} control points allowed, got {len(src)}"
        )
    lam_f = _finite_float(lam, "lam")
    if lam_f < 0.0:
        raise TPSInputError(f"lam must be >= 0, got {lam_f}")

    unique = []
    unique_index = {}
    source_indices = []
    for point in src:
        if point not in unique_index:
            unique_index[point] = len(unique)
            unique.append(point)
        source_indices.append(unique_index[point])

    n = len(unique)
    if n < 3:
        raise DegeneratePointsError(
            f"need at least 3 distinct source points after merging, got {n}"
        )

    cx = sum(p[0] for p in unique) / n
    cy = sum(p[1] for p in unique) / n
    diameter = 0.0
    for i in range(n):
        xi, yi = unique[i]
        for j in range(i + 1, n):
            d = math.hypot(xi - unique[j][0], yi - unique[j][1])
            if d > diameter:
                diameter = d
    if diameter == 0.0:
        raise DegeneratePointsError("all source points coincide")

    normalized = [[(x - cx) / diameter, (y - cy) / diameter]
                  for (x, y) in unique]
    rank = affine_rank([[1.0, x, y] for x, y in normalized])
    if rank < 3:
        raise DegeneratePointsError(
            f"affine design matrix rank is {rank} < 3: control points are "
            f"collinear or fewer than 3 distinct points after merging"
        )

    size = n + 3
    matrix = [[0.0] * size for _ in range(size)]
    kernel_entries = []
    nlam = lam_f / (diameter * diameter)
    for i in range(n):
        xi, yi = normalized[i]
        for j in range(i + 1, n):
            xj, yj = normalized[j]
            kij = _U(xi - xj, yi - yj)
            matrix[i][j] = kij
            matrix[j][i] = kij
            kernel_entries.append((i, j, kij))
        matrix[i][i] = nlam
        matrix[i][n] = 1.0
        matrix[i][n + 1] = xi
        matrix[i][n + 2] = yi
        matrix[n][i] = 1.0
        matrix[n + 1][i] = xi
        matrix[n + 2][i] = yi

    try:
        factorization = lu_factor(matrix)
    except SingularMatrixError as exc:
        raise DegeneratePointsError(
            f"TPS system numerically singular despite rank(P)={rank}: {exc}"
        ) from exc

    return _PreparedProblem(
        raw_sources=src,
        source_points=unique,
        source_indices=source_indices,
        normalized_sources=normalized,
        center=(cx, cy),
        diameter=diameter,
        rank=rank,
        factorization=factorization,
        kernel_entries=kernel_entries,
    )


REUSED_COMPONENTS = (
    "duplicate_source_grouping",
    "source_normalization",
    "affine_rank",
    "kernel_matrix_assembly",
    "regularized_system_assembly",
    "lu_factorization",
    "pivot_permutation",
)


def _solve_for_targets(prepared, targets, reused_components=()):
    """Perform only target-dependent RHS assembly, substitution and metrics."""
    checked_targets = _check_points(targets, "targets")
    if len(checked_targets) != len(prepared.raw_sources):
        raise TPSInputError(
            f"sources/targets length mismatch: {len(prepared.raw_sources)} "
            f"!= {len(checked_targets)}"
        )

    merged_targets = [None] * len(prepared.source_points)
    for source_index, target in zip(prepared.source_indices, checked_targets):
        existing = merged_targets[source_index]
        if existing is None:
            merged_targets[source_index] = target
        elif existing != target:
            raise TPSInputError(
                f"duplicate source point {prepared.source_points[source_index]} "
                f"maps to conflicting targets {existing} and {target}"
            )

    scale = prepared.diameter
    normalized_targets = [[x / scale, y / scale] for x, y in merged_targets]
    rhs_x = [t[0] for t in normalized_targets] + [0.0, 0.0, 0.0]
    rhs_y = [t[1] for t in normalized_targets] + [0.0, 0.0, 0.0]
    sol_x, sol_y = lu_solve(prepared.factorization, [rhs_x, rhs_y])

    n = len(prepared.source_points)
    nsrc = prepared.normalized_sources
    cx, cy = prepared.center
    ln_s = math.log(scale)
    wx_n, ax_n = sol_x[:n], sol_x[n:]
    wy_n, ay_n = sol_y[:n], sol_y[n:]

    def affine_back(w_n, affine_n):
        constant = ln_s * sum(
            w_n[i] * (nsrc[i][0] * nsrc[i][0] + nsrc[i][1] * nsrc[i][1])
            for i in range(n)
        )
        return [
            scale * affine_n[0]
            - affine_n[1] * cx
            - affine_n[2] * cy
            - scale * constant,
            affine_n[1],
            affine_n[2],
        ]

    weights_x = [value / scale for value in wx_n]
    weights_y = [value / scale for value in wy_n]
    affine_x = affine_back(wx_n, ax_n)
    affine_y = affine_back(wy_n, ay_n)
    weights = [[weights_x[i], weights_y[i]] for i in range(n)]
    affine = [
        [affine_x[0], affine_y[0]],
        [affine_x[1], affine_y[1]],
        [affine_x[2], affine_y[2]],
    ]

    warp = TPSWarp(prepared.source_points, weights, affine)
    residuals = []
    squared_sum = 0.0
    for i, point in enumerate(prepared.source_points):
        fx, fy = warp.evaluate(point)
        ex = fx - merged_targets[i][0]
        ey = fy - merged_targets[i][1]
        residuals.append(math.hypot(ex, ey))
        squared_sum += ex * ex + ey * ey
    rms = math.sqrt(squared_sum / (2.0 * n))

    energy = 0.0
    for w_coord in (wx_n, wy_n):
        for i, j, kij in prepared.kernel_entries:
            energy += 2.0 * w_coord[i] * w_coord[j] * kij

    return TPSFitResult(
        warp=warp,
        weights=weights,
        affine=affine,
        residuals=residuals,
        rms_residual=rms,
        bending_energy=energy,
        rank=prepared.rank,
        merged_count=n,
        reused_components=tuple(reused_components),
        cache_hit=bool(reused_components),
    )


class TPSFitCache:
    """Reusable factorisation keyed by source points and regularisation."""

    def __init__(self):
        self._prepared = None
        self._sources = None
        self._lam = None
        self.last_reused_components = ()

    def get(self, sources, lam):
        candidate_sources = _check_points(sources, "sources")
        candidate_lam = _finite_float(lam, "lam")
        if (
            self._prepared is not None
            and self._sources == candidate_sources
            and self._lam == candidate_lam
        ):
            self.last_reused_components = REUSED_COMPONENTS
            return self._prepared, REUSED_COMPONENTS

        prepared = _prepare_sources(candidate_sources, candidate_lam)
        self._prepared = prepared
        self._sources = candidate_sources
        self._lam = candidate_lam
        self.last_reused_components = ()
        return prepared, ()

    def clear(self):
        self._prepared = None
        self._sources = None
        self._lam = None
        self.last_reused_components = ()


class TPSFitter:
    """Stateful fitter for repeated target drags with fixed source points."""

    def __init__(self, sources=None, targets=None, lam=0.0):
        self._cache = TPSFitCache()
        self._sources = None
        self._lam = None
        self.last_result = None
        if sources is not None:
            self.set_problem(sources, targets=targets, lam=lam)

    def set_sources(self, sources, lam=None):
        checked = _check_points(sources, "sources")
        lam_value = self._lam if lam is None else _finite_float(lam, "lam")
        prepared = _prepare_sources(checked, lam_value)
        self._cache._prepared = prepared
        self._cache._sources = checked
        self._cache._lam = lam_value
        self._cache.last_reused_components = ()
        self._sources = checked
        self._lam = lam_value
        self.last_result = None
        return self

    def set_regularization(self, lam):
        if self._sources is None:
            raise TPSInputError("cannot set lam before sources are initialised")
        return self.set_sources(self._sources, lam=lam)

    set_regularisation = set_regularization

    def set_problem(self, sources, targets, lam=0.0):
        self.set_sources(sources, lam=lam)
        if targets is None:
            raise TPSInputError("targets is required")
        self.last_result = _solve_for_targets(self._cache._prepared, targets, ())
        return self.last_result

    def update_targets(self, targets):
        if self._cache._prepared is None:
            raise TPSInputError("fitter has no source points")
        result = _solve_for_targets(
            self._cache._prepared, targets, REUSED_COMPONENTS
        )
        self.last_result = result
        return result

    def fit(self, sources=None, targets=None, lam=None):
        if sources is not None or lam is not None:
            if targets is None:
                raise TPSInputError("targets is required when sources or lam changes")
            if sources is not None:
                self.set_sources(sources, lam=self._lam if lam is None else lam)
            else:
                self.set_regularization(lam)
            self.last_result = _solve_for_targets(
                self._cache._prepared, targets, ()
            )
            return self.last_result
        if targets is None:
            raise TPSInputError("targets is required")
        return self.update_targets(targets)


def fit(sources, targets, lam=0.0, cache=None):
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
    cache:
        Optional `TPSFitCache`. When source points and lam are unchanged, the
        cached normalised K/P assembly and LU factorisation are reused; only
        RHS assembly, substitution and result metrics are recomputed.

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
    if cache is None:
        prepared = _prepare_sources(sources, lam)
        return _solve_for_targets(prepared, targets)
    prepared, reused = cache.get(sources, lam)
    return _solve_for_targets(prepared, targets, reused)
