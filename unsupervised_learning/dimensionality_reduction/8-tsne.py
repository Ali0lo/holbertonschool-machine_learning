#!/usr/bin/env python3
"""t-SNE implementation."""
import numpy as np

pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Perform t-SNE transformation.

    Args:
        X (np.ndarray): Shape (n, d).
        ndims (int): Output dimensionality.
        idims (int): PCA dimensionality.
        perplexity (float): Target perplexity.
        iterations (int): Number of iterations.
        lr (float): Learning rate.

    Returns:
        np.ndarray: Low-dimensional embedding of shape (n, ndims).
    """
    X = pca(X, idims)
    P = P_affinities(X, perplexity=perplexity)
    P = P * 4

    n = X.shape[0]
    Y = np.random.randn(n, ndims)
    iY = np.zeros_like(Y)

    for it in range(iterations):
        dY, Q = grads(Y, P)

        momentum = 0.5 if it < 20 else 0.8
        iY = momentum * iY - lr * dY
        Y = Y + iY

        Y = Y - np.mean(Y, axis=0)

        if it == 100:
            P = P / 4

        if (it + 1) % 100 == 0:
            C = cost(P, Q)
            print(f"Cost at iteration {it + 1}: {C}")

    return Y
