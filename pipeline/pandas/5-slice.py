#!/usr/bin/env python3
"""
Module for slicing a DataFrame: select specific columns and every 60th row.
"""


def slice(df):
    """
    Extracts columns 'High', 'Low', 'Close', and 'Volume_(BTC)',
    and selects every 60th row from these columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Sliced DataFrame with columns and every 60th row.
    """
    df_subset = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    return df_subset.iloc[::60]
