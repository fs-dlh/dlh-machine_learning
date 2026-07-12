#!/usr/bin/env python3
""" Module for concatenating two DataFrames with keys. """

import pandas as pd

# Import the index function from task 10
index = __import__('10-index').index


def concat(df1, df2):
    """ Concatenates two DataFrames indexed by Timestamp.

    - df1 (coinbase) is kept as is.
    - df2 (bitstamp) is filtered to include only rows up to and including
      timestamp 1417411920.
    - Both DataFrames are indexed by their Timestamp column.
    - The selected rows from df2 are placed at the top, followed by df1.
    - Adds a top-level key: 'bitstamp' for df2 rows, 'coinbase' for df1 rows.

    Args:
        df1 (pd.DataFrame): Coinbase data.
        df2 (pd.DataFrame): Bitstamp data.

    Returns:
        pd.DataFrame: Concatenated DataFrame with a MultiIndex.
    """
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    df2_filtered = df2_indexed.loc[df2_indexed.index <= 1417411920]

    result = pd.concat(
        [df2_filtered, df1_indexed],
        keys=['bitstamp', 'coinbase']
    )

    return result
