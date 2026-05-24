#!/usr/bin/env python3
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    k = pi.shape[0]

    probs = np.array([
        pi[i] * pdf(X, m[i], S[i])
        for i in range(k)
    ])

    total = np.sum(probs, axis=0)

    g = probs / total

    l = np.sum(np.log(total))

    return g, l
