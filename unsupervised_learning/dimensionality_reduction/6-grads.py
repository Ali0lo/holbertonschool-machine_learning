#!/usr/bin/env python3
"""Compute gradients for t-SNE."""
import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Compute gradients of Y.

    Args:
        Y (np.ndarray): Shape (n, ndim).
        P (np.ndarray): Shape (n, n).

    Returns:
        tuple: (dY, Q)
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros_like(Y)

    for i in range(n):
        diff = Y[i] - Y
        weights = (P[i] - Q[i]) * num[i]
        dY[i] = np.sum(weights[:, None] * diff, axis=0)

    return dY, Q
