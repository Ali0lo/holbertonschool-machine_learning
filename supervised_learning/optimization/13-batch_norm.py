#!/usr/bin/env python3
"""Applies batch normalization to an unactivated layer output."""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Returns the batch-normalized matrix."""
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    return gamma * Z_norm + beta
