"""Offline demo of target-covariance propagation in tps2d.

Run with:  python demo_iteration.py

Fits a deterministic TPS warp, propagates a correlated target covariance to
query points (means, 2x2 covariances, influence weights), cross-checks the
linear propagation against an actual refit on a fixed perturbation, and
triggers a genuine rejection of an invalid (indefinite) covariance.
Everything is computed live: no network, no sleep, no pre-recorded output.
"""

import math
import time

from tps2d import fit, TPSInputError


def build_problem():
    sources, targets = [], []
    for j in range(4):
        for i in range(4):
            x, y = 3.0 * i, 3.0 * j
            sources.append([x, y])
            targets.append([
                x + 1.2 * math.sin(0.35 * y),
                y + 1.2 * math.cos(0.30 * x),
            ])
    return sources, targets


def main():
    t0 = time.perf_counter()
    sources, targets = build_problem()
    result = fit(sources, targets)
    n = result.merged_count
    print(f"fitted {n} control points, rms residual {result.rms_residual:.3e}")

    # Correlated target covariance: independent per-point noise (sigma^2)
    # plus a shared, fully correlated survey-level translation (tau^2).
    sigma2, tau2 = 0.04, 0.10
    dim = 2 * n
    cov = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        for j in range(dim):
            if i % 2 == j % 2:
                cov[i][j] = tau2
    for i in range(dim):
        cov[i][i] += sigma2

    queries = [[1.5, 1.5], [4.5, 4.5], [7.5, 7.5]]
    pred = result.predict_with_covariance(queries, cov)
    print("\ncorrelated target covariance propagated to query points:")
    for q, pt in enumerate(pred.points):
        c = pred.covariances[q]
        m = pred.means[q]
        print(
            f"  p={pt}  mean=({m[0]: .6f}, {m[1]: .6f})  "
            f"std=({math.sqrt(c[0][0]):.4f}, {math.sqrt(c[1][1]):.4f})  "
            f"corr={c[0][1] / math.sqrt(c[0][0] * c[1][1]): .3f}"
        )
    h = pred.influence[0][0]
    print(f"  influence weights at {pred.points[0]} sum to {sum(h):.12f} "
          f"(partition of unity), min={min(h): .4f}, max={max(h): .4f}")

    # Cross-check: refit with a fixed deterministic perturbation and compare
    # against the linear propagation f + H @ delta (must match exactly).
    delta = [
        [0.02 * math.sin(1.7 * k + 0.3), 0.02 * math.cos(1.1 * k - 0.8)]
        for k in range(n)
    ]
    refit = fit(sources, [
        [targets[k][0] + delta[k][0], targets[k][1] + delta[k][1]]
        for k in range(n)
    ])
    worst = 0.0
    for q, pt in enumerate(queries):
        exact = refit.warp.evaluate(pt)
        hx, hy = pred.influence[q]
        lin = [
            pred.means[q][0] + sum(hx[i] * delta[i][0] for i in range(n)),
            pred.means[q][1] + sum(hy[i] * delta[i][1] for i in range(n)),
        ]
        worst = max(worst, abs(lin[0] - exact[0]), abs(lin[1] - exact[1]))
    print(f"\nlinear propagation vs refit on fixed perturbation: "
          f"max abs deviation {worst:.3e}")

    # Genuine rejection: an indefinite covariance is refused before compute.
    bad = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        bad[i][i] = 1.0
    bad[0][2] = bad[2][0] = 5.0  # correlation > 1 -> not semidefinite
    try:
        result.predict_with_covariance([[0.0, 0.0]], bad)
        print("ERROR: indefinite covariance was accepted")
    except TPSInputError as exc:
        print(f"\nindefinite covariance rejected as required:\n  {exc}")

    print(f"\ntotal elapsed {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
