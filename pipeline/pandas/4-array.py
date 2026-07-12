#!/usr/bin/env python3
""" Module to extract last 10 rows of High and Close columns as numpy array.
"""


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns and converts
    them to a numpy.ndarray.

    Args:
        df (pd.DataFrame): Input DataFrame contains 'High' and 'Close' columns.

    Returns:
        np.ndarray: The selected values as a numpy array.
    """

    return df[['High', 'Close']].tail(10).to_numpy()
