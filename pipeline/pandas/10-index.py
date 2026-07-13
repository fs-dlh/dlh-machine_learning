#!/usr/bin/env python3
""" Module for setting the Timestamp column as the DataFrame index. """


def index(df):
    """ Sets the 'Timestamp' column as the index of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'Timestamp' column.

    Returns:
        df: New DataFrame with 'Timestamp' as the index.
    """
    return df.set_index('Timestamp')
