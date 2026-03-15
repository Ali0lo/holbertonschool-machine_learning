#!/usr/bin/env python3
"""Calculates precision for each class in a confusion matrix."""

import numpy as np


def precision(confusion):
    """Returns the precision of each class."""
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    return true_positives / (true_positives + false_positives)
