#!/usr/bin/env python3
"""Compute descriptive statistics for all columns except Timestamp."""

import pandas as pd


def analyze(df):
    """Compute descriptive statistics for all columns except Timestamp.

    Args:
        df: pandas DataFrame that includes a 'Timestamp' column.

    Returns:
        pandas.DataFrame: Descriptive statistics for non-Timestamp columns.
    """
    df = df.drop(columns=["Timestamp"])
    return df.describe()
