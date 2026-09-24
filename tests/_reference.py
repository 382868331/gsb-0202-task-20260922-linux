"""Independent reference implementation used ONLY by the test suite.

Everything here is written from scratch for the tests and does not import
``tps2d`` at all: its own kernel, its own Gaussian elimination and its own
TPS assembly.  Expected values in the tests come from this module or from
closed-form construction; the package under test is never used to produce
its own expectations.
"""

import math


def kernel(dx, dy):
    r2 = dx * dx + dy * dy
    if r2 == 0.0:
        return 0.0
    return 0.5 * r2 * math.log(r2)


def gauss_solve(a, b):
    """Plain Gauss-Jordan solve with partial pivoting (independent code)."""
    n = len(a)
    m = [list(a[i]) + [b[i]] for i in range(n)]
    for k in range(n):
        piv = k
        for i in range(k + 1, n):
            if abs(m[i][k]) > abs(m[piv][k]):
                piv = i
        m[k], m[piv] = m[piv], m[k]
        pivot = m[k][k]
        if abs(pivot) < 1e-14:
            raise ZeroDivisionError("reference solver hit a singular matrix")
        for j in range(k, n + 1):
            m[k][j] /= pivot
        for i in range(n):
            if i == k:
                continue
            factor = m[i][k]
            if factor == 0.0:
                continue
            for j in range(k, n + 1):
                m[i][j] -= factor * m[k][j]
    return [m[i][n] for i in range(n)]


def reference_fit(src, tgt, lam=0.0):
    """Return (wx, wy, ax, ay, merged_src) using an independent assembly."""
    seen = {}
    for s, t in zip(src, tgt):
        key = (float(s[0]), float(s[1]))
        tv = (float(t[0]), float(t[1]))
        if key in seen:
            assert seen[key] == tv
            continue
        seen[key] = tv
    pts = list(seen.keys())
    n = len(pts)
    size = n + 3
    a = [[0.0] * size for _ in range(size)]
    for i in range(n):
        for j in range(i + 1, n):
            kij = kernel(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            a[i][j] = a[j][i] = kij
        a[i][i] = float(lam)
        a[i][n] = a[n][i] = 1.0
        a[i][n + 1] = a[n + 1][i] = pts[i][0]
        a[i][n + 2] = a[n + 2][i] = pts[i][1]
    bx = [seen[p][0] for p in pts] + [0.0, 0.0, 0.0]
    by = [seen[p][1] for p in pts] + [0.0, 0.0, 0.0]
    sx = gauss_solve(a, bx)
    sy = gauss_solve(a, by)
    return sx[:n], sy[:n], sx[n:], sy[n:], pts


def reference_fit_weighted(src, tgt, lam, sigmas):
    """Weighted smoothing spline via an independent from-scratch assembly.

    Solves the joint stationarity system
    ``[K + diag(lam/sigma_i)  P; P^T 0] [w; a] = [v; 0]`` directly (no
    normalisation), so it shares no code path with the package.  Every
    confidence weight must be strictly positive; the sigma = 0 limit is
    checked separately by deleting the point.  Returns the same tuple
    layout as `reference_fit`.
    """
    seen = {}
    sw = {}
    for s, t, sg in zip(src, tgt, sigmas):
        key = (float(s[0]), float(s[1]))
        tv = (float(t[0]), float(t[1]))
        if key in seen:
            assert seen[key] == tv
            sw[key] += float(sg)
            continue
        seen[key] = tv
        sw[key] = float(sg)
    pts = list(seen.keys())
    n = len(pts)
    size = n + 3
    a = [[0.0] * size for _ in range(size)]
    for i in range(n):
        for j in range(i + 1, n):
            kij = kernel(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            a[i][j] = a[j][i] = kij
        a[i][i] = float(lam) / sw[pts[i]]
        a[i][n] = a[n][i] = 1.0
        a[i][n + 1] = a[n + 1][i] = pts[i][0]
        a[i][n + 2] = a[n + 2][i] = pts[i][1]
    bx = [seen[p][0] for p in pts] + [0.0, 0.0, 0.0]
    by = [seen[p][1] for p in pts] + [0.0, 0.0, 0.0]
    sx = gauss_solve(a, bx)
    sy = gauss_solve(a, by)
    return sx[:n], sy[:n], sx[n:], sy[n:], pts


def reference_warp(params, x, y):
    wx, wy, ax, ay, pts = params
    ox = ax[0] + ax[1] * x + ax[2] * y
    oy = ay[0] + ay[1] * x + ay[2] * y
    for i, (cx, cy) in enumerate(pts):
        u = kernel(x - cx, y - cy)
        ox += wx[i] * u
        oy += wy[i] * u
    return ox, oy


def finite_diff_jacobian(eval2d, x, y, h=1e-6):
    """Central-difference Jacobian of any callable (x, y) -> (ox, oy)."""
    xp = eval2d(x + h, y)
    xm = eval2d(x - h, y)
    yp = eval2d(x, y + h)
    ym = eval2d(x, y - h)
    return [
        [(xp[0] - xm[0]) / (2 * h), (yp[0] - ym[0]) / (2 * h)],
        [(xp[1] - xm[1]) / (2 * h), (yp[1] - ym[1]) / (2 * h)],
    ]
