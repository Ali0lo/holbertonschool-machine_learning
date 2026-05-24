#!/usr/bin/env python3
import numpy as np


def pdf(X, m, S):

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None

    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    det = np.linalg.det(S)

    inv = np.linalg.inv(S)

    norm = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    diff = X - m

    expo = -0.5 * np.sum(
        (diff @ inv) * diff,
        axis=1
    )

    P = norm * np.exp(expo)

    return np.maximum(P, 1e-300)
