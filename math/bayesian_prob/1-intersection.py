#!/usr/bin/env python3
"""Module to calculate the intersection.

x is the number of patients that develop severe side effects
n is the total number of patients observed
P is an array of probabilities of developing severe side effects
Pr is a 1D numpy.ndarray containing the prior beliefs of P
"""

import numpy as np


def likelihood(x, n, P):
    """ Calculate the likelihood """

    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")

    if len(P.shape) != 1:
        for i in range(len(P.shape)):
            raise TypeError("P must be a 1D numpy.ndarray")
            return None

    if not (isinstance(n, int) and (n > 0)):
        raise ValueError("n must be a positive integer")

    message = "x must be an integer that is greater than or equal to 0"

    if not (isinstance(x, int) and (x >= 0)):
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    for i in range(len(P)):
        if not (0 <= P[i] <= 1):
            raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the binomial coefficient
    coeff = 1.0
    for i in range(1, x + 1):
        coeff *= (n - x + i) / i

    # Calculate the likelihood
    likelihood = coeff * (P ** x) * ((1 - P) ** (n - x))
    return likelihood


def intersection(x, n, P, Pr):
    """ Calculate the intersection """

    if not (isinstance(n, int) and (n > 0)):
        raise ValueError("n must be a positive integer")

    message = "x must be an integer that is greater than or equal to 0"
    if not (isinstance(x, int) and (x >= 0)):
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")

    if len(P.shape) != 1:
        for i in range(len(P.shape)):
            raise TypeError("P must be a 1D numpy.ndarray")
            return None

    message = "Pr must be a numpy.ndarray with the same shape as P"
    if not isinstance(Pr, np.ndarray):
        raise TypeError(message)
    if Pr.shape != P.shape:
        raise TypeError(message)

    for i in range(len(P)):
        if not (0 <= P[i] <= 1):
            raise ValueError("All values in P must be in the range [0, 1]")

    for i in range(len(Pr)):
        if not (0 <= Pr[i] <= 1):
            raise ValueError("All values in Pr must be in the range [0, 1]")

    # Calculate the intersection
    intersection = likelihood(x, n, P) * Pr
    return intersection
