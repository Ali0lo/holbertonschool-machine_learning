#!/usr/bin/env python3
import numpy as np

expectation_maximization = \
    __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None,
        iterations=1000,
        tol=1e-5,
        verbose=False):

    n, d = X.shape

    if kmax is None:
        kmax = n

    l_vals = []
    b_vals = []
    results = []

    for k in range(kmin, kmax + 1):

        pi, m, S, g, l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        p = (k * d) + \
            (k * d * (d + 1) / 2) + \
            (k - 1)

        bic = p * np.log(n) - 2 * l

        l_vals.append(l)
        b_vals.append(bic)
        results.append((pi, m, S))

    b_vals = np.array(b_vals)

    idx = np.argmin(b_vals)

    best_k = kmin + idx

    return best_k, results[idx], \
        np.array(l_vals), b_vals
