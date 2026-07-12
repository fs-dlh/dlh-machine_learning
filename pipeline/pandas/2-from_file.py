#!/usr/bin/env python3
""" Module for loading a pandas DataFrame from a file. """

import pandas as pd


def from_file(filename, delimiter):
    """     Loads data from a file as a pd.DataFrame.

    Args:
        filename (str): Path to the file to load.
        delimiter (str): Column separator.

    Returns:
        pd.DataFrame: Loaded DataFrame.     """

    return pd.read_csv(filename, delimiter=delimiter)
