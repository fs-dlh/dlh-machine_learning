#!/usr/bin/env python3
""" Module for computing descriptive statistics of a DataFrame,
excluding the Timestamp column. """


def analyze(df):
    """ Computes descriptive statistics for all columns except the Timestamp.

    Args:
        df (pd.DataFrame): Input DataFrame with a Timestamp column.

    Returns:
        df: A DataFrame containing statistics
            (count, mean, std, min, 25%, 50%, 75%, max)
            for all numeric columns other than Timestamp.
    """

    df_without_timestamp = df.drop(columns=['Timestamp'])
    stats = df_without_timestamp.describe()

    return stats
