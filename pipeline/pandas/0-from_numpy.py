#!/usr/bin/env python3

""" Module for creating a pandas DataFrame from a numpy array. """

import pandas as pd


def from_numpy(array):
    """ Creates a pd.DataFrame from a np.ndarray.

    Args:
        array (np.ndarray): The numpy array to convert.

    Returns:
        pd.DataFrame: The newly created DataFrame with columns labeled
    """
    num_cols = array.shape[1]
    cols = list(chr(65 + i) for i in range(num_cols))
    return pd.DataFrame(array, columns=cols)
