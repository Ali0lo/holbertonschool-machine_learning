#!/usr/bin/env python3
"""Compute t-SNE cost."""
import numpy as np


def cost(P, Q):
    """
    Compute KL divergence cost.

    Args:
        P (np.ndarray): Shape (n, n).
        Q (np.ndarray): Shape (n, n).

    Returns:
        float: Cost value.
    """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    return np.sum(P * np.log(P / Q))
