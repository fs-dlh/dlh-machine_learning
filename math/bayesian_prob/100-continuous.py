#!/usr/bin/env python3
"""Module to calculate the Continuous Posterior probability.
"""

from scipy import special as spc


def posterior(x, n, p1, p2):
    """ Calculate the intersection """

    if not (isinstance(n, int) and (n > 0)):
        raise ValueError("n must be a positive integer")

    message = "x must be an integer that is greater than or equal to 0"
    if not (isinstance(x, int) and (x >= 0)):
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float) or p1 <= 0 or p1 >= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float) or p2 <= 0 or p2 >= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Posterior is Beta(x+1, n-x+1)
    alpha = x + 1
    beta = n - x + 1

    # CDF of Beta distribution at p2 and p1 using regularized incomplete beta
    cdf_p2 = spc.betainc(alpha, beta, p2)
    cdf_p1 = spc.betainc(alpha, beta, p1)

    return cdf_p2 - cdf_p1
