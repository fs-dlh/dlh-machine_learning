#!/usr/bin/env python3
""" 3-mini_batch module """
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """ Creates mini-batches from shuffled data for mini-batch gd.

    Args:
        X : shape (m, nx) containing input data.
        Y : shape (m, ny) containing labels.
        batch_size : number of data points per batch.

    Returns:
        list: of tuples (X_batch, Y_batch) where each batch is a slice of the
              shuffled data. The last batch may be smaller than batch_size.
    """

    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    batches = []

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        batches.append((X_batch, Y_batch))

    return batches
