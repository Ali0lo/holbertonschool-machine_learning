#!/usr/bin/env python3
""" This module will define a function for one hot encoding """
import numpy as np


def one_hot_encode(Y, classes):
    """Converts numeric labels to one-hot encoding (shape: classes x m)."""
    
    # check type first
    if not isinstance(Y, np.ndarray):
        return None  # return None immediately if Y is not a numpy.ndarray

    try:
        Y = Y.astype(int)
        m = Y.shape[0]
        if m == 0 or classes <= 0:
            return None

        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot

    except Exception:
        return None
