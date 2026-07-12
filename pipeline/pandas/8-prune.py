#!/usr/bin/env python3
""" Module for removing rows with NaN values in the 'Close' column. """


def prune(df):
    """
    Removes any entries where the 'Close' column has NaN values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        df: Modified DataFrame with rows containing NaN in 'Close' removed.
    """

    return df.dropna(subset=['Close'])
