#!/usr/bin/env python3
"""Convert labels to one-hot matrix"""
import numpy as np


def one_hot(labels, classes=None):
    """Converts a label vector to a one-hot matrix"""

    labels = np.array(labels)

    if classes is None:
        classes = np.max(labels) + 1

    one_hot_matrix = np.zeros((labels.shape[0], classes))
    one_hot_matrix[np.arange(labels.shape[0]), labels] = 1

    return one_hot_matrix
