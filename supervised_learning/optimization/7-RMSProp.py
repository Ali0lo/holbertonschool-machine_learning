#!/usr/bin/env python3
"""Module to update variables using RMSProp optimization."""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
    - alpha: learning rate
    - beta2: RMSProp decay rate (weight)
    - epsilon: small number to avoid division by zero
    - var: numpy.ndarray containing the variable to be updated
    - grad: numpy.ndarray containing the gradient of var
    - s: numpy.ndarray containing the previous second moment of var

    Returns:
    - var: updated variable
    - s: updated second moment
    """
    # Update the moving average of squared gradients
    s = beta2 * s + (1 - beta2) * np.square(grad)

    # Update the variable
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
