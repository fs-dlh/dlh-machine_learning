#!/usr/bin/env python3
""" Module for filling missing values in a DataFrame. """


def fill(df):
    """ Performs data cleaning on a DataFrame:
        - Removes the 'Weighted_Price' column.
        - Forward-fills missing values in the 'Close' column.
        - Fills missing values in 'High', 'Low', and 'Open' with the
          corresponding 'Close' value from the same row.
        - Fills missing values in 'Volume_(BTC)' and 'Volume_(Currency)'
          with 0.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        df: Modified DataFrame.
    """

    df = df.drop(columns=['Weighted_Price'])
    df['Close'] = df['Close'].fillna(method='ffill')
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
