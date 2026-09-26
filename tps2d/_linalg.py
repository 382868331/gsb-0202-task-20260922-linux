"""Small dense linear algebra helpers for tps2d (standard library only).

Direct Gaussian elimination with partial pivoting.  The TPS system has at
most ``n + 3 <= 43`` unknowns, so an O(n**3) elimination in pure Python is
plenty fast and keeps the project free of third-party dependencies.
"""

from __future__ import annotations

_EPS = 2.220446049250313e-16  # IEEE-754 binary64 machine epsilon


class SingularMatrixError(ValueError):
    """Raised when elimination encounters a numerically zero pivot."""


class LUFactorization:
    """Reusable LU factorisation of a square matrix (PA = LU).

    Created by factorize().  solve() accepts one or several right-hand
    sides, so additional solves (e.g. influence-coefficient columns) never
    repeat the O(n**3) elimination.
    """

    __slots__ = ("_lu", "_perm", "_n")

    def __init__(self, lu, perm):
        self._lu = lu
        self._perm = perm
        self._n = len(lu)

    @property
    def size(self):
        return self._n

    def solve(self, rhs):
        """Solve the factorised system for one or several rhs columns.

        Same calling convention as solve_linear(): a flat sequence returns
        a flat list, a sequence of columns returns a list of flat lists.
        """
        n = self._n
        lu = self._lu
        if rhs and isinstance(rhs[0], (list, tuple)):
            columns = [list(col) for col in rhs]
        else:
            columns = [list(rhs)]
        for col in columns:
            if len(col) != n:
                raise ValueError("rhs length must match matrix size")

        solutions = []
        for col in columns:
            # Apply the row permutation recorded during elimination.
            b = [col[self._perm[i]] for i in range(n)]
            # Forward substitution with the unit lower triangle.
            for i in range(n):
                row = lu[i]
                total = b[i]
                for j in range(i):
                    total -= row[j] * b[j]
                b[i] = total
            # Back substitution with the upper triangle.
            for i in range(n - 1, -1, -1):
                row = lu[i]
                total = b[i]
                for j in range(i + 1, n):
                    total -= row[j] * b[j]
                b[i] = total / row[i]
            solutions.append(b)

        return solutions[0] if len(solutions) == 1 else solutions


def factorize(a):
    """Factorise a square matrix for repeated solves.

    LU decomposition with partial pivoting; a pivot no larger than
    ``n * eps * max|a_ij|`` is treated as singular.  The input is copied
    and never modified.
    """
    matrix = [list(row) for row in a]
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("factorize expects a square matrix")

    scale = 0.0
    for row in matrix:
        for value in row:
            if abs(value) > scale:
                scale = abs(value)
    pivot_tol = n * _EPS * (scale if scale > 0.0 else 1.0)

    perm = list(range(n))
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
            perm[k], perm[pivot_row] = perm[pivot_row], perm[k]

        pivot = matrix[k][k]
        for i in range(k + 1, n):
            factor = matrix[i][k] / pivot
            if factor == 0.0:
                continue
            row_i = matrix[i]
            row_k = matrix[k]
            for j in range(k + 1, n):
                row_i[j] -= factor * row_k[j]
            row_i[k] = factor  # unit-diagonal L stored below the diagonal

    return LUFactorization(matrix, perm)


def solve_linear(a, rhs):
    """Solve ``a @ x = rhs`` for one or several right-hand sides.

    ``a`` is an n x n matrix (list of lists), ``rhs`` is either a flat
    sequence (returns a flat list) or a sequence of equal-length columns
    (returns a list of flat solution lists).  Inputs are copied; the caller's
    matrices are never modified.

    A pivot no larger than ``n * eps * max|a_ij|`` is treated as singular.
    """
    return factorize(a).solve(rhs)


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
