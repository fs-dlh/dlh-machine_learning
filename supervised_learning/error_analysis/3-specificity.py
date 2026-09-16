#!/usr/bin/env python3
""" Module for calculating specificity from a confusion matrix. """
import numpy as np


def specificity(confusion):
    """ Calculate the specificity (true negative rate) for each class
        in a multiclass confusion matrix.

    For each class i, specificity is defined as:
    TN_i / (TN_i + FP_i), where:
      - TP_i = confusion[i, i]
      - FP_i = sum(column i) - TP_i
      - FN_i = sum(row i) - TP_i
      - TN_i = total_samples - (TP_i + FN_i + FP_i)

    Args:
        confusion : Confusion matrix of shape (classes, classes),
                                   with rows = true labels,
                                   columns = predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing the specificity
                       for each class.
    """
    total = np.sum(confusion)
    row_sums = np.sum(confusion, axis=1)
    col_sums = np.sum(confusion, axis=0)
    tp = np.diag(confusion)

    tn = total - row_sums - col_sums + tp

    denominator = total - row_sums

    with np.errstate(divide='ignore', invalid='ignore'):
        spec = np.where(denominator != 0, tn / denominator, 0.0)

    return spec
