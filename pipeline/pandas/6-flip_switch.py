#!/usr/bin/env python3
"""
Module for sorting a DataFrame in reverse chronological order and transposing.
"""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order by the 'Timestamp'
    column and then transposes the sorted DataFrame.

    Args:
        df : Input DataFrame with a 'Timestamp' column.

    Returns:
        df_transposed: The transformed DataFrame.
    """

    df_sorted = df.sort_values(by='Timestamp', ascending=False)
    df_transposed = df_sorted.T
    return df_transposed
