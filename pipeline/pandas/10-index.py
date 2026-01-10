#!/usr/bin/env python3
"""Set the Timestamp column as the DataFrame index."""


def index(df):
    """Set the Timestamp column as the index of the DataFrame.

    Args:
        df: pandas DataFrame containing a 'Timestamp' column.

    Returns:
        pandas.DataFrame: The modified DataFrame indexed by 'Timestamp'.
    """
    return df.set_index("Timestamp")
