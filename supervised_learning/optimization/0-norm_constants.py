#!/usr/bin/env python3
"""Calculates normalization constants for a dataset."""

import numpy as np


def normalization_constants(X):
    """Returns feature-wise mean and standard deviation."""
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)
    return mean, stddev
