#!/usr/bin/env python3
"""Clean and plot BTC data as daily values from 2017+."""


def visualize(df):
    """Transform the dataframe, plot it, and return it."""
    # Drop Weighted_Price
    df = df.drop(columns=["Weighted_Price"])

    # Timestamp -> Date, convert, set index
    df = df.rename(columns={"Timestamp": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], unit="s")
    df = df.set_index("Date")

    # Fill missing values
    df["Close"] = df["Close"].ffill()
    df[["High", "Low", "Open"]] = df[["High", "Low", "Open"]].fillna(df["Close"])
    df[["Volume_(BTC)", "Volume_(Currency)"]] = df[
        ["Volume_(BTC)", "Volume_(Currency)"]
    ].fillna(0)

    # Keep 2017+
    df = df.loc["2017-01-01":]

    # Daily groups
    df = df.resample("D").agg(
        {
            "High": "max",
            "Low": "min",
            "Open": "mean",
            "Close": "mean",
            "Volume_(BTC)": "sum",
            "Volume_(Currency)": "sum",
        }
    )

    # Plot
    df.plot(subplots=True, figsize=(12, 8))

    return df
