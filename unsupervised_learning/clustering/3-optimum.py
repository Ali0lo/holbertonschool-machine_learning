#!/usr/bin/env python3
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmax, int) or kmax <= kmin:
        return None, None

    results = []
    d_vars = []

    base_var = None

    for k in range(kmin, kmax + 1):

        C, clss = kmeans(X, k, iterations)

        var = variance(X, C)

        if base_var is None:
            base_var = var

        results.append((C, clss))
        d_vars.append(base_var - var)

    return results, d_vars
