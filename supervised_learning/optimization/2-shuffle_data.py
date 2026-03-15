#!/usr/bin/env python3
"""Module to shuffle two datasets in unison."""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles two matrices the same way along first axis.

    Parameters:
    - X: numpy.ndarray of shape (m, nx) to shuffle
         m is the number of data points
         nx is the number of features in X
    - Y: numpy.ndarray of shape (m, ny) to shuffle
         m is the number of data points
         ny is the number of features in Y

    Returns:
    - X_shuffled: shuffled X matrix
    - Y_shuffled: shuffled Y matrix
    """
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled