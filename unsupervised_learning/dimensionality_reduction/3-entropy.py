#!/usr/bin/env python3
""" 3-entropy module """
import numpy as np


def HP(Di, beta):
    """ Computes the Shannon entropy and P affinities relative to a data point.

    Args:
        Di : squared distances from the point to all other points.
        beta : beta value for the Gaussian distribution.

    Returns:
        Hi : Shannon entropy of the conditional distribution (base 2).
        Pi : shape (n - 1,), conditional probabilities p_{j|i} for all j ≠ i.
    """

    if isinstance(beta, np.ndarray):
        beta = beta.item() if beta.size == 1 else beta

    P = np.exp(-Di * beta)

    sum_P = np.sum(P)

    Pi = P / sum_P

    mask = Pi > 0
    if np.any(mask):
        Hi = np.log2(sum_P) + beta * np.sum(Di * Pi) * np.log2(np.e)
    else:
        Hi = 0.0

    return Hi, Pi
