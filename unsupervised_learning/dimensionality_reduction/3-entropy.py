#!/usr/bin/env python3
""" 3-entropy module """
import numpy as np


def HP(Di, beta):
    """ Computes the Shannon entropy and P affinities relative to a data point.

    Args:
        Di : squared distances from the point to all other points.
        beta : beta value for the Gaussian distribution.

    Returns:
        Hi : Shannon entropy of the points.
        Pi : containing the P affinities of the points.
    """

    P = np.exp(-Di * beta)
    Pi = P / np.sum(P)
    Hi = -np.sum(Pi * np.log2(Pi + 1e-12))

    return Hi, Pi
