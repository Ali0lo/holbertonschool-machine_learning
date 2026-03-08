#!/usr/bin/env python3
"""One-hot decode"""
import numpy as np


def one_hot_decode(one_hot):
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
