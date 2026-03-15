#!/usr/bin/env python3
"""Module to update variables using the Adam optimization algorithm."""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Parameters:
    - alpha: learning rate
    - beta1: weight for the first moment (momentum)
    - beta2: weight for the second moment (RMSProp)
    - epsilon: small number to avoid division by zero
    - var: numpy.ndarray containing the variable to update
    - grad: numpy.ndarray containing the gradient of var
    - v: numpy.ndarray containing the previous first moment
    - s: numpy.ndarray containing the previous second moment
    - t: time step (iteration) used for bias correction

    Returns:
    - var: updated variable
    - v: updated first moment
    - s: updated second moment
    """
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # Update biased second moment estimate
    s = beta2 * s + (1 - beta2) * np.square(grad)

    # Correct bias for first and second moments
    v_corrected = v / (1 - beta1**t)
    s_corrected = s / (1 - beta2**t)

    # Update variable
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
