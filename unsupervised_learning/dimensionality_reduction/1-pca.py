#!/usr/bin/env python3
"""PCA with fixed output dimensionality."""
import numpy as np


def pca(X, ndim):
    """
    Perform PCA and return transformed data.

    Args:
        X (np.ndarray): Shape (n, d), mean-centered.
        ndim (int): Output dimensionality.

    Returns:
        np.ndarray: Transformed data of shape (n, ndim).
    """
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt.T[:, :ndim]
    T = np.matmul(X, W)
    return T
