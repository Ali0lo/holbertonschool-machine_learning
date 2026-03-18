#!/usr/bin/env python3
"""L2 regularization gradient descent module."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Update the weights and biases using gradient descent with L2."""
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]

        dW = (np.matmul(dz, A_prev.T) / m) + ((lambtha / m) * W)
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)

        weights["W" + str(i)] = W - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db
