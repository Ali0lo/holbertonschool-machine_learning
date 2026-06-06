#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using the Bayesian Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using the Bayesian Information Criterion.

    Args:
        X (numpy.ndarray): shape (n, d) dataset.
        kmin (int): minimum number of clusters to check (inclusive).
        kmax (int or None): maximum number of clusters to check (inclusive).
            If None, set to the maximum number of clusters possible.
        iterations (int): maximum number of iterations for the EM algorithm.
        tol (float): tolerance for the EM algorithm.
        verbose (bool): if True, EM prints info to standard output.

    Returns:
        tuple: (best_k, best_result, l, b) or (None, None, None, None) on failure.
            best_k (int): best value for k based on BIC.
            best_result (tuple): (pi, m, S) for the best k.
            l (numpy.ndarray): shape (kmax - kmin + 1,) log likelihoods.
            b (numpy.ndarray): shape (kmax - kmin + 1,) BIC values.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < 1):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    if kmin >= kmax:
        return None, None, None, None

    l_vals = []
    b_vals = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None:
            return None, None, None, None
        p = (k * d) + (k * d * (d + 1) // 2) + (k - 1)
        bic = p * np.log(n) - 2 * log_l
        l_vals.append(log_l)
        b_vals.append(bic)
        results.append((pi, m, S))

    l = np.array(l_vals)
    b = np.array(b_vals)
    best_idx = np.argmin(b)
    best_k = kmin + best_idx

    return best_k, results[best_idx], l, b
