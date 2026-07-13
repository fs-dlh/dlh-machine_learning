#!/usr/bin/env python3
"""
Module for sorting a DataFrame by the 'High' column in descending order.
"""


def high(df):
    """
    Sorts the DataFrame by the 'High' price in descending order.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'High' column.

    Returns:
        df  The sorted DataFrame (highest High first).
    """
    return df.sort_values(by='High', ascending=False)
