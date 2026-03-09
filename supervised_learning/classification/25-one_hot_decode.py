#!/usr/bin/env python3
""" one hot decoding """
import numpy as np


def one_hot_decode(one_hot):
    """ Converts a one-hot encoded matrix into a vector of numeric labels. """

    # check if input is a numpy ndarray
    if not isinstance(one_hot, np.ndarray):
        return None

    try:
        # argmax along axis 0 gives the index of the 1 in each column
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
