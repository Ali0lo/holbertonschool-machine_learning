#!/usr/bin/env python3
"""L2 regularization cost for a neural network."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the total cost of a neural network with L2 regularization.

    Parameters:
        cost (float): Cost without L2 regularization
        lambtha (float): Regularization parameter
        weights (dict): Dictionary of weights/biases ("W1", "b1", ..., "WL", "bL")
        L (int): Number of layers
        m (int): Number of data points

    Returns:
        float: Cost including L2 regularization
    """
    l2_sum = 0.0

    # Sum squared values of weight matrices only (exclude biases)
    for i in range(1, L + 1):
        W = weights.get(f"W{i}")
        if W is not None:
            l2_sum += np.sum(np.square(W))

    return cost + (lambtha / (2 * m)) * l2_sum
