#!/usr/bin/env python3
""" Module for calculating precision from a confusion matrix. """
import numpy as np


def precision(confusion):
    """ Calculate the precision (positive predictive value) for each class.

    Precision for class i = TP_i / (TP_i + FP_i)
    where TP_i = confusion[i][i] and FP_i = sum(column i) - TP_i.

    Args:
        confusion : Confusion matrix of shape (classes, classes),
                                   where rows are true labels and columns are
                                   predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the precision
                       for each class.
    """

    tp = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(
            predicted_positives != 0, tp / predicted_positives, 0.0)
    return prec
