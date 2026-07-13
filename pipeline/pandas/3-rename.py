#!/usr/bin/env python3
""" Module to rename Timestamp column to Datetime and convert the lues. """

import pandas as pd


def rename(df):
    """ Renames the 'Timestamp' column to 'Datetime', converts to datetime,
    and returns only the 'Datetime' and 'Close' columns.

    Args: df (pd.DataFrame): Input DataFrame with a 'Timestamp' column.

    Returns: pd.DataFrame: Modified df with 'Datetime' and 'Close' columns."""

    df = df.rename(columns={'Timestamp': 'Datetime'})

    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    return df[['Datetime', 'Close']]
