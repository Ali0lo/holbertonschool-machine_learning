#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    return np.random.uniform(mins, maxs, (k, d))
