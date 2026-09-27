"""Small dense linear algebra helpers for tps2d (standard library only).

Direct Gaussian elimination with partial pivoting.  The TPS system has at
most ``n + 3 <= 43`` unknowns, so an O(n**3) elimination in pure Python is
plenty fast and keeps the project free of third-party dependencies.
"""

from __future__ import annotations

_EPS = 2.220446049250313e-16  # IEEE-754 binary64 machine epsilon


class SingularMatrixError(ValueError):
    """Raised when elimination encounters a numerically zero pivot."""


def lu_factor(a):
    """Factorise ``a`` once into an LU decomposition with partial pivoting.

    The returned object can be reused for any number of right-hand sides via
    `lu_solve`; this is the cacheable part of a TPS refit when the source
    control points and regularisation (hence the system matrix) are fixed and
    only the targets change.  The factorisation stores the elimination
    multipliers below the diagonal, the upper factor on/above it, and the
    pivot row swaps applied during elimination.
    """
    matrix = [list(row) for row in a]
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("lu_factor expects a square matrix")

    scale = 0.0
    for row in matrix:
        for value in row:
            if abs(value) > scale:
                scale = abs(value)
    pivot_tol = n * _EPS * (scale if scale > 0.0 else 1.0)

    swaps = []
    for k in range(n):
        pivot_row = k
        pivot_abs = abs(matrix[k][k])
        for i in range(k + 1, n):
            cand = abs(matrix[i][k])
            if cand > pivot_abs:
                pivot_abs = cand
                pivot_row = i
        if pivot_abs <= pivot_tol:
            raise SingularMatrixError(
                f"linear system is (numerically) singular: pivot {pivot_abs:.3e} "
                f"<= tolerance {pivot_tol:.3e}"
            )
        if pivot_row != k:
            matrix[k], matrix[pivot_row] = matrix[pivot_row], matrix[k]
            swaps.append((k, pivot_row))

        pivot = matrix[k][k]
        for i in range(k + 1, n):
            factor = matrix[i][k] / pivot
            if factor == 0.0:
                continue
            row_i = matrix[i]
            row_k = matrix[k]
            for j in range(k + 1, n):
                row_i[j] -= factor * row_k[j]
            row_i[k] = factor  # store the L multiplier below the diagonal

    return {"lu": matrix, "swaps": swaps, "n": n, "pivot_tol": pivot_tol}


def lu_solve(factor, rhs):
    """Solve ``a @ x = rhs`` using a factorisation from `lu_factor`.

    ``rhs`` is a flat sequence (returns a flat list) or a sequence of equal
    length columns (returns a list of flat solution lists).  The factorisation
    is never modified, so one cached factor can serve repeated target updates.
    """
    matrix = factor["lu"]
    n = factor["n"]

    if rhs and isinstance(rhs[0], (list, tuple)):
        columns = [list(col) for col in rhs]
    else:
        columns = [list(rhs)]
    for col in columns:
        if len(col) != n:
            raise ValueError("rhs length must match matrix size")

    # Apply the recorded pivot row swaps to every rhs column.
    for r0, r1 in factor["swaps"]:
        for col in columns:
            col[r0], col[r1] = col[r1], col[r0]

    # Forward substitution: L y = P b, with unit diagonal in L.
    for k in range(n):
        row_k = matrix[k]
        for col in columns:
            for i in range(k + 1, n):
                multiplier = matrix[i][k]
                if multiplier != 0.0:
                    col[i] -= multiplier * col[k]

    # Back substitution: U x = y.
    solutions = []
    for col in columns:
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            total = col[i]
            row = matrix[i]
            for j in range(i + 1, n):
                total -= row[j] * x[j]
            x[i] = total / row[i]
        solutions.append(x)

    return solutions[0] if len(solutions) == 1 else solutions


def solve_linear(a, rhs):
    """Solve ``a @ x = rhs`` for one or several right-hand sides.

    ``a`` is an n x n matrix (list of lists), ``rhs`` is either a flat
    sequence (returns a flat list) or a sequence of equal-length columns
    (returns a list of flat solution lists).  Inputs are copied; the caller's
    matrices are never modified.

    A pivot no larger than ``n * eps * max|a_ij|`` is treated as singular.
    """
    return lu_solve(lu_factor(a), rhs)


def affine_rank(mat, tol=1e-12):
    """Numerical rank of an ``m x 3`` design matrix ``P = [1, x, y]``.

    Gaussian elimination with row pivoting.  A pivot is declared zero when it
    is no larger than ``tol`` times the largest absolute entry in the whole
    matrix (the configuration's coordinate diameter, including the constant
    column), so the criterion is scale independent and treats points whose
    off-line deviation is below ~1e-12 diameters as collinear.  ``tol`` is
    intentionally tight (about 4.5e3 machine epsilons).
    """
    m = len(mat)
    if m == 0:
        return 0
    work = [list(row) for row in mat]
    ncols = min(3, len(work[0]))
    global_scale = 0.0
    for row in work:
        for value in row:
            if abs(value) > global_scale:
                global_scale = abs(value)
    if global_scale == 0.0:
        return 0
    pivot_tol = tol * global_scale
    rank = 0
    for col in range(ncols):
        pivot_row = None
        pivot_abs = pivot_tol
        for i in range(rank, m):
            s = abs(work[i][col])
            if s > pivot_abs:
                pivot_abs = s
                pivot_row = i
        if pivot_row is None:
            continue
        work[rank], work[pivot_row] = work[pivot_row], work[rank]
        pivot = work[rank][col]
        for i in range(rank + 1, m):
            factor = work[i][col] / pivot
            if factor == 0.0:
                continue
            for j in range(col, ncols):
                work[i][j] -= factor * work[rank][j]
        rank += 1
    return rank
