#!/usr/bin/env python3
"""K-means clustering"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    return np.random.uniform(mins, maxs, (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """performs K-means clustering"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    if k > n:
        return None, None

    C = initialize(X, k)

    if C is None:
        return None, None

    for i in range(iterations):

        distances = np.linalg.norm(
            X[:, np.newaxis] - C,
            axis=2
        )

        clss = np.argmin(distances, axis=1)

        new_C = np.copy(C)

        for j in range(k):

            points = X[clss == j]

            if points.shape[0] == 0:

                new_C[j] = np.random.uniform(
                    np.min(X, axis=0),
                    np.max(X, axis=0),
                    (1, d)
                )

            else:

                new_C[j] = np.mean(points, axis=0)

        if np.allclose(C, new_C):

            return C, clss

        C = np.copy(new_C)

    distances = np.linalg.norm(
        X[:, np.newaxis] - C,
        axis=2
    )

    clss = np.argmin(distances, axis=1)

    return C, clss
