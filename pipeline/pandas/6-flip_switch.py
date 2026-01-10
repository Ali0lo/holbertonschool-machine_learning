#!/usr/bin/env python3
"""Sort a DataFrame in reverse chronological order and transpose it."""


def flip_switch(df):
    """Sort the data in reverse chronological order, then transpose.

    Args:
        df: pandas DataFrame.

    Returns:
        pandas.DataFrame: The transformed DataFrame.
    """
    return df.sort_index(ascending=False).T
