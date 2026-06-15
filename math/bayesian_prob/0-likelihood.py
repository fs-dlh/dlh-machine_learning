#!/usr/bin/env python3
"""Module to calculate the likelihood of a binomial distribution.

x is the number of patients that develop severe side effects
n is the total number of patients observed
P is an array of probabilities of developing severe side effects

"""

import numpy as np


def factorial(n):
    """
    Calculate the factorial of a non-negative integer n.

    n! = n * (n - 1) * (n - 2) * ... * 1

    """
    if not isinstance(n, int):
        raise TypeError("n must be an integer")

    if n < 0:
        raise ValueError("n must be a non-negative integer")

    if n == 0 or n == 1:
        return 1

    result = 1
    for i in range(2, n + 1):
        result *= i

    return result   


def coefficient(n, x):
    """
    Calculate the binomial coefficient "n choose x".

    C(n,k) = n! / (k! * (n - k)!)

    """
    if not (isinstance(n, int) and isinstance(x, int)):
        raise TypeError("n and x must be integers")

    if not (0 <= x <= n):
        raise ValueError("x must be between 0 and n")

    coefficient = factorial(n) / (factorial(x) * factorial(n - x))

    return coefficient  




def likelihood(x, n, P):
    """
    Calculate the likelihood of observing x successes in n trials
    with a probability of success P.
    """
    if not (isinstance(x, int) and isinstance(n, int)):
        raise TypeError("x and n must be integers")

    if not (0 <= x <= n):
        raise ValueError("x must be between 0 and n")

    for i in P:
        if not (0 <= i <= 1):
            raise ValueError("P must be between 0 and 1")

    # Calculate the binomial coefficient
    binom_coeff = coefficient(n, x)

    # Calculate the likelihood
    likelihood = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihood


