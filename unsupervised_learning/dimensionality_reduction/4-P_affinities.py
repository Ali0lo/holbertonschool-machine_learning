#!/usr/bin/env python3
"""Compute symmetric P affinities."""
import numpy as np

P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Compute symmetric P affinities.

    Args:
        X (np.ndarray): Shape (n, d).
        tol (float): Tolerance for entropy matching.
        perplexity (float): Target perplexity.

    Returns:
        np.ndarray: Symmetric P affinities of shape (n, n).
    """
    D, P, betas, H = P_init(X, perplexity)
    n = X.shape[0]

    for i in range(n):
        beta = betas[i]
        Di = np.delete(D[i], i)

        Hi, Pi = HP(Di, beta)
        Hdiff = Hi - H
        low = None
        high = None

        while abs(Hdiff) > tol:
            if Hdiff > 0:
                low = beta
                if high is None:
                    beta = beta * 2
                else:
                    beta = (beta + high) / 2
            else:
                high = beta
                if low is None:
                    beta = beta / 2
                else:
                    beta = (beta + low) / 2

            Hi, Pi = HP(Di, beta)
            Hdiff = Hi - H

        betas[i] = beta
        P[i, np.arange(n) != i] = Pi

    P = (P + P.T) / (2 * n)
    return P
