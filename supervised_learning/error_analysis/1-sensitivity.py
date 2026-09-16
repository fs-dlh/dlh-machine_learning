#!/usr/bin/env python3
""" Module for calculating sensitivity from a confusion matrix. """
import numpy as np


def sensitivity(confusion):
    """ Calculate the sensitivity (true positive rate) for each class.

    Sensitivity for class i = TP_i / (TP_i + FN_i)
    where TP_i = confusion[i][i] and FN_i = sum(row i) - TP_i.

    Args:
        confusion : Confusion matrix of shape (classes, classes),
                                   where rows are true labels and columns are
                                   predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the sensitivity
                       for each class.
    """

    tp = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = np.where(actual_positives != 0, tp / actual_positives, 0.0)
    return sens
