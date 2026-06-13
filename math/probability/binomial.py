#!/usr/bin/env python3
"""Module for Binomial distribution."""


class Binomial:
    """Class that represents a Binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution.

        Args:
            data (list, optional): List of data points.
            n (int, optional): Number of Bernoulli trials. Defaults to 1.
            p (float, optional): Probability of success. Defaults to 0.5.

        Raises:
            TypeError: If data is provided but is not a list.
            ValueError: If n <= 0, or p <= 0 or p >= 1,
                        or data has fewer than 2 points.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            n_data = len(data)
            mean = sum(data) / n_data
            variance = sum((x - mean) ** 2 for x in data) / n_data

            p_est = 1.0 - (variance / mean)
            self.p = float(p_est)

            n_est = mean / self.p
            self.n = int(round(n_est))

            self.p = float(mean / self.n)

    def pmf(self, k):
        """      Calculate the value of the Probability Mass Function.

        Args:    k (int or float): Number of successes.

        Returns: float: PMF value for k.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        comb = 1.0
        for i in range(1, k + 1):
            comb *= (self.n - k + i) / i

        pmf_value = comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf_value
