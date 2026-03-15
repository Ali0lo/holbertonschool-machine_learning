#!/usr/bin/env python3
"""Calculates sensitivity for each class in a confusion matrix."""

import numpy as np


def sensitivity(confusion):
    """Returns the sensitivity (recall) of each class."""
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    return true_positives / (true_positives + false_negatives)
