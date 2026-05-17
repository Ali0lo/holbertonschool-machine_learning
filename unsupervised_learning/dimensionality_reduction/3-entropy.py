#!/usr/bin/env python3
"""Compute Shannon entropy and P affinities."""
import numpy as np


def HP(Di, beta):
    """
    Compute Shannon entropy and P affinities for one point.

    Args:
        Di (np.ndarray): Shape (n - 1,), distances to other points.
        beta (np.ndarray): Shape (1,), beta value.

    Returns:
        tuple: (Hi, Pi)
    """
    Pi = np.exp(-Di * beta)
    sumP = np.sum(Pi)

    if sumP == 0:
        Pi = np.zeros_like(Di)
        Hi = 0.0
    else:
        Pi = Pi / sumP
        Hi = -np.sum(Pi * np.log2(Pi + 1e-12))

    return Hi, Pi
