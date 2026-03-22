#!/usr/bin/env python3
"""Module for calculating L2 regularization cost."""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculate the cost of a neural network with L2 regularization.

    Args:
        cost: The cost of the network without L2 regularization
        lambtha: The regularization parameter
        weights: A dictionary of the weights and biases (numpy.ndarrays)
                 of the neural network
        L: The number of layers in the neural network
        m: The number of data points used

    Returns:
        The cost of the network accounting for L2 regularization
    """
    l2_sum = 0
    
    # Sum the Frobenius norm (squared L2 norm) of all weight matrices
    for layer in range(1, L + 1):
        weight_key = 'W' + str(layer)
        # Calculate the Frobenius norm squared (sum of all squared elements)
        l2_sum += np.sum(np.square(weights[weight_key]))
    
    # Add L2 regularization term to the original cost
    l2_cost = cost + (lambtha / (2 * m)) * l2_sum
    
    return l2_cost
