#!/usr/bin/env python3
"""Remove rows where Close is NaN."""


def prune(df):
    """Remove any entries where the Close column has NaN values.

    Args:
        df: pandas DataFrame containing a 'Close' column.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    return df.dropna(subset=["Close"])
