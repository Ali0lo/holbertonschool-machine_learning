#!/usr/bin/env python3
"""Module to perform batch normalization on neural network outputs."""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network using batch normalization.

    Parameters:
    - Z: numpy.ndarray of shape (m, n) to normalize
         m is the number of data points
         n is the number of features
    - gamma: numpy.ndarray of shape (1, n) containing the scales
    - beta: numpy.ndarray of shape (1, n) containing the offsets
    - epsilon: small number to avoid division by zero

    Returns:
    - Z_norm: numpy.ndarray of shape (m, n) containing the normalized outputs
    """
    # Compute mean and variance across the batch
    mu = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)

    # Normalize
    Z_hat = (Z - mu) / np.sqrt(var + epsilon)

    # Scale and shift
    Z_norm = gamma * Z_hat + beta

    return Z_norm
