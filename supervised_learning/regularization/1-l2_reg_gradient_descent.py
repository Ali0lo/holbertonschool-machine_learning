#!/usr/bin/env python3
"""Module for gradient descent with L2 regularization."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Update the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y: One-hot numpy.ndarray of shape (classes, m) containing correct
           labels for the data
        weights: Dictionary of the weights and biases of the neural network
        cache: Dictionary of the outputs of each layer of the neural network
        alpha: The learning rate
        lambtha: The L2 regularization parameter
        L: The number of layers of the network

    The neural network uses tanh activations on each layer except the last,
    which uses a softmax activation. The weights and biases are updated
    in place.
    """
    m = Y.shape[1]

    # Make a copy of weights to use original values during backprop
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        # Get activation from previous layer
        A_prev = cache['A' + str(layer - 1)]

        # Calculate gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Add L2 regularization term to weight gradient
        dW += (lambtha / m) * weights['W' + str(layer)]

        # Calculate dZ for previous layer before updating weights
        if layer > 1:
            # Use current (not yet updated) weights
            W_current = weights['W' + str(layer)]
            # Derivative of tanh: 1 - tanh^2(z) = 1 - A^2
            dZ = np.matmul(
                W_current.T, dZ) * (1 - np.square(cache['A' + str(layer - 1)])
                )

        # Update weights and biases in place
        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
