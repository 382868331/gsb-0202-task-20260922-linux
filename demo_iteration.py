"""Offline iteration demo for fixed-design TPS covariance propagation.

Run from the repository root:

    python demo_iteration.py

The script constructs targets, fits once, reuses the cached factorization,
and validates the linear covariance propagation by applying deterministic
perturbations and fitting the perturbed targets.  It has no network, sleep,
credentials, or pre-recorded results.
"""

import math
import time

from tps2d import fit


def build_problem():
    sources = []
    targets = []
    for j in range(4):
        for i in range(4):
            x = 3.0 * i
            y = 3.0 * j
            sources.append([x, y])
            targets.append([
                x + 0.7 * math.sin(0.31 * x + 0.17 * y),
                y + 0.6 * math.cos(0.23 * y - 0.11 * x),
            ])
    return sources, targets


def covariance(rows):
    return [list(map(float, row)) for row in rows]


def independent_covariance(n, variance_x, variance_y):
    mat = [[0.0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        mat[2 * i][2 * i] = variance_x
        mat[2 * i + 1][2 * i + 1] = variance_y
    return mat


def common_translation_covariance(n, vxx, vxy, vyy):
    mat = [[0.0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            mat[2 * i][2 * j] = vxx
            mat[2 * i][2 * j + 1] = vxy
            mat[2 * i + 1][2 * j] = vxy
            mat[2 * i + 1][2 * j + 1] = vyy
    return mat


def rank_one_covariance(vector):
    return [
        [vector[i] * vector[j] for j in range(len(vector))]
        for i in range(len(vector))
    ]


def main():
    started = time.perf_counter()
    sources, targets = build_problem()
    result = fit(sources, targets, lam=0.0)
    query = [4.2, 5.7]
    n = len(targets)

    print("=== tps2d fixed-design covariance propagation ===")
    print(f"control points: {n}; query: {query}; network/sleep: none")
    print(f"mean prediction: {[round(v, 9) for v in result.warp.evaluate(query)]}")

    independent = independent_covariance(n, 0.04, 0.09)
    pred_independent = result.predict(query, independent)
    print("\nindependent target noise (variances 0.04, 0.09):")
    print(f"  covariance: {pred_independent.covariance}")

    common = common_translation_covariance(n, 0.04, 0.01, 0.09)
    pred_common = result.predict(query, common)
    print("\nfully correlated common target translation:")
    print(f"  covariance: {pred_common.covariance}")

    perturbations = [
        [
            [0.11 * math.sin(0.7 * i + 0.2), 0.09 * math.cos(0.5 * i)]
            for i in range(n)
        ],
        [
            [0.07 * math.cos(0.4 * i + 1.1), 0.12 * math.sin(0.9 * i + 0.3)]
            for i in range(n)
        ],
    ]

    print("\ndeterministic perturbation checks (fit once per check for verification):")
    max_error = 0.0
    for sample_id, delta in enumerate(perturbations, start=1):
        flat_delta = [value for pair in delta for value in pair]
        prediction = result.predict(query, rank_one_covariance(flat_delta))
        perturbed_targets = [
            [target[0] + shift[0], target[1] + shift[1]]
            for target, shift in zip(targets, delta)
        ]
        direct = fit(sources, perturbed_targets, lam=0.0).warp.evaluate(query)
        influence_matrix = prediction.influence_matrix
        linear_shift = [
            sum(influence_matrix[a][k] * flat_delta[k] for k in range(2 * n))
            for a in range(2)
        ]
        outer_shift = [
            [linear_shift[0] ** 2, linear_shift[0] * linear_shift[1]],
            [linear_shift[0] * linear_shift[1], linear_shift[1] ** 2],
        ]
        sample_error = max(
            abs(direct[a] - prediction.mean[a] - linear_shift[a])
            for a in range(2)
        )
        cov_error = max(
            abs(prediction.covariance[a][b] - outer_shift[a][b])
            for a in range(2)
            for b in range(2)
        )
        max_error = max(max_error, sample_error, cov_error)
        print(f"  sample {sample_id}: direct-linear error = {sample_error:.3e}, "
              f"covariance error = {cov_error:.3e}")

    if max_error > 1e-9:
        raise SystemExit(f"propagation check failed: max error {max_error}")
    print(f"\nall fixed-design linear propagation checks passed")
    print(f"elapsed: {time.perf_counter() - started:.3f} s")


if __name__ == "__main__":
    main()
