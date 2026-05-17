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
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    W = eigvecs[:, :ndim]
    T = np.matmul(X, W)
    return T
