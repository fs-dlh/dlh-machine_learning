#!/usr/bin/env python3
""" 3-one_hot module """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Converts a label vector into a one-hot matrix.

    Args:
        labels : integer labels to convert.
        classes : number of classes. If None, inferred from
                                 the maximum label value + 1.

    Returns:
        numpy.ndarray: one-hot matrix of shape (len(labels), classes) with
                       dtype float32.
    """
    return K.utils.to_categorical(labels, num_classes=classes, dtype='float32')
