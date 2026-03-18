#!/usr/bin/env python3
"""Gradient descent with dropout regularization."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Update weights and biases using dropout gradient descent."""
    m = Y.shape[1]
    dz = cache["A{}".format(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A{}".format(i - 1)]
        W_key = "W{}".format(i)
        b_key = "b{}".format(i)

        W = weights[W_key].copy()
        dW = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights[W_key] = weights[W_key] - alpha * dW
        weights[b_key] = weights[b_key] - alpha * db

        if i > 1:
            da = np.matmul(W.T, dz)
            da = (da * cache["D{}".format(i - 1)]) / keep_prob
            A = cache["A{}".format(i - 1)]
            dz = da * (1 - np.power(A, 2))
