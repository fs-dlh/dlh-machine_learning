#!/usr/bin/env python3
""" Module for creating a hierarchical DataFrame from two exchange DataFrames,
with Timestamp as the first level and exchange key as second level. """

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """ Takes two DataFrames, sets their Timestamp as index,
    filters rows between timestamps 1417411980 and 1417417980 inclusive,
    concatenates them with keys 'bitstamp' and 'coinbase', and rearranges the
    MultiIndex so that Timestamp is the first level,
    and ensures chronological order.

    Args:
        df1 (pd.DataFrame): Coinbase data.
        df2 (pd.DataFrame): Bitstamp data.

    Returns:
        pd.DataFrame: Concatenated DataFrame with MultiIndex.  """

    df1_idx = index(df1)
    df2_idx = index(df2)

    start_ts = 1417411980
    end_ts = 1417417980

    mask1 = (df1_idx.index >= start_ts) & (df1_idx.index <= end_ts)
    mask2 = (df2_idx.index >= start_ts) & (df2_idx.index <= end_ts)
    df1_filtered = df1_idx.loc[mask1]
    df2_filtered = df2_idx.loc[mask2]

    concat_df = pd.concat(
        [df2_filtered, df1_filtered],
        keys=['bitstamp', 'coinbase']
    )

    concat_df = concat_df.swaplevel(0, 1)
    concat_df = concat_df.sort_index(level=0)

    return concat_df
