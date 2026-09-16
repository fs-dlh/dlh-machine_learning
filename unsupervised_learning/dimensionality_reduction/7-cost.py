#!/usr/bin/env python3
""" that calculates the cost of the t-SNE transformation """
import numpy as np


def cost(P, Q):
    """ Compute the cost of the t-SNE transformation:

    Parameters:
    P : containing the P affinities.
    Q : containing the Q affinities

    Returns:
    C : the cost of the transformation,
        C = sum_{i≠j} p_ij * log(p_ij / q_ij).
    """
    eps = 1e-12

    P_safe = np.maximum(P, eps)
    Q_safe = np.maximum(Q, eps)

    C = np.sum(P_safe * np.log(P_safe / Q_safe))

    return C
