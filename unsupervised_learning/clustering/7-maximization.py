#!/usr/bin/env python3
import numpy as np


def maximization(X, g):

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    weights = np.sum(g, axis=1)

    pi = weights / n

    m = (g @ X) / weights[:, np.newaxis]

    S = np.zeros((k, d, d))

    for i in range(k):
        diff = X - m[i]

        weighted = g[i][:, np.newaxis] * diff

        S[i] = (weighted.T @ diff) / weights[i]

    return pi, m, S
