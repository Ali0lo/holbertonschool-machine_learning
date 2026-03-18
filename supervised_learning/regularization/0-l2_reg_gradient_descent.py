#!/usr/bin/env python3
"""Gradient descent with L2 regularization for deep neural networks."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Update weights and biases using gradient descent with L2 regularization.

    Y is a one-hot numpy.ndarray of shape (classes, m) with correct labels.
    weights is a dictionary containing W1, b1, ..., WL, bL.
    cache is a dictionary containing A0, A1, ..., AL.
    alpha is the learning rate.
    lambtha is the L2 regularization parameter.
    L is the number of layers.
    """
    m = Y.shape[1]
    dz = cache["A{}".format(L)] - Y

    for i in range(L, 0, -1):
        a_prev = cache["A{}".format(i - 1)]
        w_key = "W{}".format(i)
        b_key = "b{}".format(i)

        W = weights[w_key].copy()
        dW = (np.matmul(dz, a_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            da = np.matmul(W.T, dz)
            a_curr = cache["A{}".format(i - 1)]
            dz = da * (1 - np.power(a_curr, 2))

        weights[w_key] = weights[w_key] - alpha * dW
        weights[b_key] = weights[b_key] - alpha * db
