#!/usr/bin/env python3
"""L2 regularization cost for a neural network."""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculate neural network cost with L2 regularization.

    Parameters:
        cost (float): Cost without L2 regularization.
        lambtha (float): Regularization parameter.
        weights (dict): Dictionary with network parameters
            (e.g., "W1", "b1", ..., "WL", "bL").
        L (int): Number of layers.
        m (int): Number of data points.

    Returns:
        float: Cost including L2 regularization.
    """
    l2_sum = 0.0

    for i in range(1, L + 1):
        w_key = "W{}".format(i)
        W = weights.get(w_key)
        if W is not None:
            l2_sum += np.sum(np.square(W))

    return cost + (lambtha / (2 * m)) * l2_sum
