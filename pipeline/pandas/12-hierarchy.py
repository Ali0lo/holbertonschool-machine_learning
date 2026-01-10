#!/usr/bin/env python3
"""Concatenate two DataFrames with a MultiIndex ordered by Timestamp."""

import pandas as pd

index = __import__('10-index').index


def hierarchy(df1, df2):
    """Rearrange MultiIndex to have Timestamp first and concat a time window.

    - Indexes both dataframes on their Timestamp columns.
    - Selects rows from 1417411980 to 1417417980 (inclusive) from both.
    - Concatenates with keys 'bitstamp' (df2) and 'coinbase' (df1).
    - Swaps MultiIndex levels so Timestamp is first.
    - Sorts to ensure chronological order.

    Args:
        df1: coinbase pandas DataFrame containing a 'Timestamp' column.
        df2: bitstamp pandas DataFrame containing a 'Timestamp' column.

    Returns:
        pandas.DataFrame: Concatenated DataFrame with MultiIndex (Timestamp,
            source) sorted chronologically.
    """
    df1 = index(df1).loc[1417411980:1417417980]
    df2 = index(df2).loc[1417411980:1417417980]

    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    df = df.swaplevel(0, 1)
    df = df.sort_index(level=0)

    return df
