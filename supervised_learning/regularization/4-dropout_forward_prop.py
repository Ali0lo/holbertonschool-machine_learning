#!/usr/bin/env python3
"""Forward propagation with inverted dropout."""
import numpy as np


def softmax(z):
    """Softmax activation."""
    t = np.exp(z - np.max(z, axis=0, keepdims=True))
    return t / np.sum(t, axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conduct forward propagation using dropout."""
    cache = {}
    cache["A0"] = X

    for i in range(1, L + 1):
        W = weights["W{}".format(i)]
        b = weights["b{}".format(i)]
        A_prev = cache["A{}".format(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            A = softmax(Z)
            cache["A{}".format(i)] = A
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A = (A * D) / keep_prob
            cache["D{}".format(i)] = D
            cache["A{}".format(i)] = A

    return cache
