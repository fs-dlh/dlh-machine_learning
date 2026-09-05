#!/usr/bin/env python3
""" Module to create a confusion matrix. """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ Create a confusion matrix from one-hot encoded labels and predictions.

    Args:
        labels : one-hot array of shape (m, classes)
                                containing the true labels.
        logits : one-hot array of shape (m, classes)
                                containing the predicted labels.

    Returns:
        numpy.ndarray: confusion matrix of shape (classes, classes)
                       with rows representing true labels and columns
                       representing predicted labels.
    """
    m, classes = labels.shape
    true = np.argmax(labels, axis=1)
    pred = np.argmax(logits, axis=1)
    confusion = np.zeros((classes, classes), dtype=int)
    np.add.at(confusion, (true, pred), 1)
    return confusion
