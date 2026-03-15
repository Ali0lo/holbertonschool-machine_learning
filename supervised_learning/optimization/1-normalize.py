#!/usr/bin/env python3
"""Normalizes a matrix using provided mean and standard deviation."""

import numpy as np


def normalize(X, m, s):
    """Returns the normalized matrix."""
    return (X - m) / s
