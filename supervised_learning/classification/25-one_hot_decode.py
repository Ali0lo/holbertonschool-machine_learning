#!/usr/bin/env python3
"""Module that defines a one-hot decoding function."""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of numeric class labels.

    Args:
        one_hot: a numpy.ndarray with shape (classes, m) that is a one-hot
                 encoding of the numeric labels.

    Returns:
        A numpy.ndarray with shape (m,) containing the numeric labels,
        or None on failure.
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    if one_hot.shape[0] == 0 or one_hot.shape[1] == 0:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
