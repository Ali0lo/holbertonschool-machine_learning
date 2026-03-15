#!/usr/bin/env python3
"""Module for normalizing a dataset using given mean and standard deviation."""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a dataset.

    Parameters:
    - X: numpy.ndarray of shape (d, nx) containing the data to normalize
         d is the number of data points
         nx is the number of features
    - m: numpy.ndarray of shape (nx,) containing the mean of each feature
    - s: numpy.ndarray of shape (nx,) containing the standard deviation of each feature

    Returns:
    - X_norm: numpy.ndarray of shape (d, nx) containing the normalized data
    """
    X_norm = (X - m) / s
    return X_norm
