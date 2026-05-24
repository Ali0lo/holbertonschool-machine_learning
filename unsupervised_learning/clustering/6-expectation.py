#!/usr/bin/env python3
"""Expectation step for GMM"""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates expectation step"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d):
        return None, None

    if S.shape != (k, d, d):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    probs = np.zeros((k, n))

    for i in range(k):

        P = pdf(X, m[i], S[i])

        if P is None:
            return None, None

        probs[i] = pi[i] * P

    total = np.sum(probs, axis=0)

    g = probs / total

    l = np.sum(np.log(total))

    return g, l
