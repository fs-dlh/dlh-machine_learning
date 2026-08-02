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
        Pi : shape (n - 1,) containing the P affinities of the points.
    """

    P = np.exp(-Di * beta)
    sum_P = np.sum(P)
    Pi = P / sum_P
    Hi = np.log(sum_P) + beta * np.sum(Di * P) / sum_P
    Hi /= np.log(2)
 
    return np.float32(Hi), np.float32(Pi)
