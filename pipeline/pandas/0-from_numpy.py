#!/usr/bin/env python3
"""Creates a pandas.DataFrame from a numpy.ndarray."""

import pandas as pd


def from_numpy(array):
    """Creates a pandas.DataFrame from a numpy.ndarray.

    Args:
        array (numpy.ndarray): numpy array from which to create the DataFrame.

    Raises:
        ValueError: If the number of columns in array is greater than 26.

    Returns:
        pandas.DataFrame: DataFrame created from array with columns labeled
            'A', 'B', ...
    """
    cols = array.shape[1]
    if cols > 26:
        raise ValueError("array has too many columns")

    columns = [chr(ord('A') + i) for i in range(cols)]
    return pd.DataFrame(array, columns=columns)
