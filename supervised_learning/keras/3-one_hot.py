#!/usr/bin/env python3
"""Converts labels to one-hot matrix."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: Vector of labels.
        classes: Number of classes.

    Returns:
        The one-hot matrix.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
