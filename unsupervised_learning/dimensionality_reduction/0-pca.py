#!/usr/bin/env python3
"""PCA that retains a fraction of variance."""
import numpy as np


def pca(X, var=0.95):
    """
    Perform PCA on a dataset and retain a fraction of variance.

    Args:
        X (np.ndarray): Shape (n, d), mean-centered.
        var (float): Fraction of variance to retain.

    Returns:
        np.ndarray: Projection matrix W of shape (d, nd).
    """
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    total = np.sum(eigvals)
    cumvar = np.cumsum(eigvals) / total
    nd = np.searchsorted(cumvar, var) + 1

    W = eigvecs[:, :nd]
    return W
