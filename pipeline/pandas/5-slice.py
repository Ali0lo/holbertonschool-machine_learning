#!/usr/bin/env python3
"""Slice specific columns and select every 60th row."""


def slice(df):
    """Extract selected columns and return every 60th row.

    Extracts the columns High, Low, Close, and Volume_(BTC), then selects
    every 60th row from these columns.

    Args:
        df: pandas DataFrame.

    Returns:
        pandas.DataFrame: The sliced DataFrame.
    """
    cols = ["High", "Low", "Close", "Volume_(BTC)"]
    return df[cols].iloc[::60]
