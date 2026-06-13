#!/usr/bin/env python3
"""Module for Poisson distribution."""


class Poisson:
    """Class that represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """ Initialize the Poisson distribution.

        Args:
            data (list, optional): List of data points.
            lambtha (float, optional): Expected number of occurrences.

        Raises:
            TypeError: If data is provided but is not a list.
            ValueError: If lambtha <= 0 or data has fewer than 2 points.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """      Calculate the value of the Probability Mass Function.

        Args:    k (int or float): Number of "successes".

        Returns: float: PMF value for k.
        """
        if not isinstance(k, int):
            k = int(k)
        if k <= 0 :
            return 0.0
        e = 2.7182818285
        result = 1.0
        for i in range(1, k + 1):
            result *= self.lambtha / i
        result *= e ** (-self.lambtha)
        return result
    
    def cdf(self, k):
        """       Calculate the value of the Cumulative Distribution Function.

        Args:     k (int or float): Number of "successes".

        Returns:  float: CDF value for k.
        """
        k = int(k)
        if k < 0:
            return 0.0
        cumulative = 0.0
        for i in range(k + 1):
            cumulative += self.pmf(i)
        return cumulative
