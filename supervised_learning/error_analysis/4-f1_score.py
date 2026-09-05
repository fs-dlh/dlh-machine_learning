#!/usr/bin/env python3
""" Module for calculating F1 score from a confusion matrix. """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ Calculate the F1 score for each class from a confusion matrix.

    F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    Args:
        confusion : Confusion matrix of shape (classes, classes),
                                   where rows are true labels and columns are
                                   predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the F1 score
                       for each class.
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)
    numerator = 2 * prec * sens
    denominator = prec + sens

    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(denominator != 0, numerator / denominator, 0.0)

    return f1
