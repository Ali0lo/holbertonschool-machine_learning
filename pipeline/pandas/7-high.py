#!/usr/bin/env python3
"""Sort a DataFrame by the High price in descending order."""


def high(df):
    """Sort a DataFrame by the High column in descending order.

    Args:
        df: pandas DataFrame containing a 'High' column.

    Returns:
        pandas.DataFrame: The sorted DataFrame.
    """
    return df.sort_values(by="High", ascending=False)
