#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using the Bayesian Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC.

    Args:
        X:          numpy.ndarray of shape (n, d) - the dataset
        kmin:       minimum number of clusters to check (inclusive)
        kmax:       maximum number of clusters to check (inclusive)
        iterations: max iterations for the EM algorithm
        tol:        tolerance for the EM algorithm
        verbose:    whether EM should print info

    Returns:
        best_k, best_result, l, b  or  None, None, None, None on failure
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2
            or not isinstance(kmin, int) or kmin < 1
            or (kmax is not None and (not isinstance(kmax, int) or kmax < 1))
            or not isinstance(iterations, int) or iterations < 1
            or not isinstance(tol, float) or tol < 0
            or not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmin >= kmax:
        return None, None, None, None

    num_k = kmax - kmin + 1
    l = np.zeros(num_k)
    b = np.zeros(num_k)
    results = []

    for i, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_like = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose
        )
        if pi is None:
            return None, None, None, None

        # Number of free parameters:
        #   pi:  k - 1  (priors sum to 1)
        #   m:   k * d  (means)
        #   S:   k * d*(d+1)/2  (symmetric covariance matrices)
        p = (k - 1) + k * d + k * d * (d + 1) // 2

        l[i] = log_like
        b[i] = p * np.log(n) - 2 * log_like
        results.append((pi, m, S))

    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
