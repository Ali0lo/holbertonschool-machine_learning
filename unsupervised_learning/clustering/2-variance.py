#!/usr/bin/env python3
"""Calculates intra-cluster variance"""
import numpy as np


def variance(X, C):
    """calculates total intra-cluster variance"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(
        X[:, np.newaxis] - C,
        axis=2
    )

    min_dist = np.min(distances, axis=1)

    return np.sum(min_dist ** 2)
