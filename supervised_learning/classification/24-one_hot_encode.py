#!/usr/bin/env python3
"""Module that defines a one-hot encoding function."""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix."""
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes < 2:
        return None
    if classes < np.max(Y) + 1:
        return None
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1.0
        return one_hot
    except Exception:
        return None
