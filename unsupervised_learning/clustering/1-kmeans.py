#!/usr/bin/env python3
import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    C = initialize(X, k)

    for _ in range(iterations):

        distances = np.linalg.norm(
            X[:, np.newaxis] - C,
            axis=2
        )

        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)

        for j in range(k):
            points = X[clss == j]

            if len(points) == 0:
                new_C[j] = np.random.uniform(
                    np.min(X, axis=0),
                    np.max(X, axis=0)
                )
            else:
                new_C[j] = np.mean(points, axis=0)

        if np.allclose(C, new_C):
            return new_C, clss

        C = new_C

    return C, clss
