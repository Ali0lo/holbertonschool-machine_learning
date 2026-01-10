#!/usr/bin/env python3
"""0-from_numpy

This module provides a helper function to create a pandas DataFrame from a
NumPy ndarray with columns named in capital alphabetical order.
"""

import numpy as np
import pandas as pd


def from_numpy(array):
    """Create a pandas DataFrame from a NumPy ndarray.

    Columns are labeled in capital alphabetical order starting from 'A'.

    Args:
        array (np.ndarray): The NumPy ndarray to convert.

    Returns:
        pd.DataFrame: A DataFrame with labeled columns.

    Raises:
        ValueError: If array has more than 26 columns.
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if array.ndim == 1:
        array = array.reshape(-1, 1)

    n_cols = array.shape[1] if array.ndim > 1 else 1
    if n_cols > 26:
        raise ValueError("array must have no more than 26 columns")

    cols = [chr(ord('A') + i) for i in range(n_cols)]
    return pd.DataFrame(array, columns=cols)
