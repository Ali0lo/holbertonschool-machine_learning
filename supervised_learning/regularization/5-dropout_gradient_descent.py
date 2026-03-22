#!/usr/bin/env python3
"""Module for gradient descent with dropout."""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Update the weights of a neural network with Dropout regularization using
    gradient descent.

    Args:
        Y: One-hot numpy.ndarray of shape (classes, m) containing correct
           labels for the data
        weights: Dictionary of the weights and biases of the neural network
        cache: Dictionary of the outputs and dropout masks of each layer
        alpha: The learning rate
        keep_prob: The probability that a node will be kept
        L: The number of layers of the network

    The weights of the network are updated in place.
    """
    m = Y.shape[1]

    # Backward propagation
    # Start with output layer (softmax)
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):
        # Get activation from previous layer
        A_prev = cache['A' + str(layer - 1)]

        # Calculate gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Calculate dZ for previous layer (if not at input layer)
        if layer > 1:
            # Backpropagate through weights
            dA = np.matmul(weights['W' + str(layer)].T, dZ)
            # Apply dropout mask to gradient
            dA = dA * cache['D' + str(layer - 1)]
            # Scale by keep_prob (inverted dropout)
            dA = dA / keep_prob

            # Derivative of tanh: 1 - tanh^2(z) = 1 - A^2
            dZ = dA * (1 - np.square(cache['A' + str(layer - 1)]))

        # Update weights and biases in place
        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
