#!/usr/bin/env python3
"""Rename Timestamp to Datetime and format columns."""

import pandas as pd


def rename(df):
    """Renames Timestamp column to Datetime, converts to datetime, and filters.

    Args:
        df (pd.DataFrame): DataFrame containing 'Timestamp' and 'Close' columns.

    Returns:
        pd.DataFrame: Modified DataFrame with only 'Datetime' and 'Close'.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    return df[["Datetime", "Close"]]
