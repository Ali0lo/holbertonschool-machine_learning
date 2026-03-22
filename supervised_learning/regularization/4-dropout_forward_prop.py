#!/usr/bin/env python3
"""Module for forward propagation with dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conduct forward propagation using Dropout.

    Args:
        X: numpy.ndarray of shape (nx, m) containing the input data
        weights: Dictionary of the weights and biases of the neural network
        L: The number of layers in the network
        keep_prob: The probability that a node will be kept

    Returns:
        A dictionary containing the outputs of each layer and the dropout
        mask used on each layer
    """
    cache = {}
    cache['A0'] = X
    
    for layer in range(1, L + 1):
        # Get previous activation
        A_prev = cache['A' + str(layer - 1)]
        
        # Compute Z = W * A_prev + b
        Z = np.matmul(weights['W' + str(layer)], A_prev) + \
            weights['b' + str(layer)]
        
        if layer == L:
            # Last layer: softmax activation
            exp_Z = np.exp(Z)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache['A' + str(layer)] = A
        else:
            # Hidden layers: tanh activation
            A = np.tanh(Z)
            
            # Apply dropout
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            D = D.astype(int)
            
            # Apply mask and scale (inverted dropout)
            A = A * D
            A = A / keep_prob
            
            cache['A' + str(layer)] = A
            cache['D' + str(layer)] = D
    
    return cache