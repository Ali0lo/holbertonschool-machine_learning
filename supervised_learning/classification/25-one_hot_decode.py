#!/usr/bin/env python3
""" one hot decoding """
import numpy as np


def one_hot_decode(one_hot):
    """ Converts a one-hot encoded matrix into a vector of numeric labels. """
    if not isinstance(one_hot, np.ndarray):
        return None

    if one_hot.ndim != 2:
        return None

    try:
        # The index of the max in each column is the class label
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
