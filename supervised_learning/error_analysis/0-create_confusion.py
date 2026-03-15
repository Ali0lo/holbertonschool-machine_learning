#!/usr/bin/env python3
"""Creates a confusion matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Returns a confusion matrix from one-hot labels and predictions."""
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)
    classes = labels.shape[1]

    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (true_classes, pred_classes), 1)

    return confusion
