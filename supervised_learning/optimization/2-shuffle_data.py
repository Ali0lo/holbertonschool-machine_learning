#!/usr/bin/env python3
"""Shuffles two data matrices in the same way."""

import numpy as np


def shuffle_data(X, Y):
    """Returns shuffled X and Y."""
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]
