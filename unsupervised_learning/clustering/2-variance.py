#!/usr/bin/env python3
import numpy as np


def variance(X, C):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    distances = np.linalg.norm(
        X[:, np.newaxis] - C,
        axis=2
    )

    min_dist = np.min(distances, axis=1)

    return np.sum(min_dist ** 2)
