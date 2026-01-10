#!/usr/bin/env python3
"""Concatenate two DataFrames with keys after indexing on Timestamp."""

import pandas as pd

index = __import__('10-index').index


def concat(df1, df2):
    """Index on Timestamp, slice df2, then concat with keys.

    Indexes both dataframes on their Timestamp columns.
    Keeps all timestamps from df2 up to and including 1417411920.
    Concatenates the selected df2 rows above df1.
    Adds keys: 'bitstamp' for df2 and 'coinbase' for df1.

    Args:
        df1: coinbase pandas DataFrame containing a 'Timestamp' column.
        df2: bitstamp pandas DataFrame containing a 'Timestamp' column.

    Returns:
        pandas.DataFrame: Concatenated DataFrame with keys.
    """
    df1 = index(df1)
    df2 = index(df2)

    df2 = df2.loc[:1417411920]

    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
