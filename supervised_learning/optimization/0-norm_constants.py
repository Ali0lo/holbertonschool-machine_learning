#!/usr/bin/env python3
"""Module to calculate normalization constants (mean and std) for a dataset."""
import numpy as np


def normalization_constants(X):
    """
    Calculates the mean and standard deviation of each feature in a dataset.

    Parameters:
    - X: numpy.ndarray of shape (m, nx) containing the data to normalize
         m is the number of data points
         nx is the number of features

    Returns:
    - mean: numpy.ndarray of shape (nx,) containing the mean of each feature
    - std: numpy.ndarray of shape (nx,) containing the standard deviation
           of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
