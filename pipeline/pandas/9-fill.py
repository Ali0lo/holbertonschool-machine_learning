#!/usr/bin/env python3
"""Fill missing values in price/volume columns and drop Weighted_Price."""


def fill(df):
    """Modify the DataFrame by dropping and filling specific columns.

    Removes the Weighted_Price column.

    Fills missing values:
    - Close: with previous row's value.
    - High, Low, Open: with the corresponding Close in the same row.
    - Volume_(BTC), Volume_(Currency): with 0.

    Args:
        df: pandas DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].fillna(method="ffill")

    for col in ["Open", "High", "Low"]:
        df[col] = df[col].fillna(df["Close"])

    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)

    return df
