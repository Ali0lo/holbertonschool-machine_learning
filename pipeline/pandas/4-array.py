#!/usr/bin/env python3
"""Select values from a DataFrame and return them as a NumPy array."""


def array(df):
    """Select the last 10 rows of the High and Close columns as a NumPy array.

    Args:
        df: pandas DataFrame containing 'High' and 'Close' columns.

    Returns:
        numpy.ndarray: The selected values as a NumPy array.
    """
    return df[["High", "Close"]].tail(10).to_numpy()
