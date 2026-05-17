#!/usr/bin/env python3
"""Initialize variables for P affinities."""
import numpy as np


def P_init(X, perplexity):
    """
    Initialize variables needed to compute P affinities.

    Args:
        X (np.ndarray): Shape (n, d).
        perplexity (float): Target perplexity.

    Returns:
        tuple: (D, P, betas, H)
    """
    n = X.shape[0]
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
