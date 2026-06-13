#!/usr/bin/env python3
"""Module for Normal distribution."""


class Normal:
    """Class that represents a Normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the Normal distribution.

        Args:
            data (list, optional): List of data points.
            mean (float, optional): Mean of the distribution. Defaults to 0.
            stddev (float, optional): Standard deviation of the distribution.

        Raises:
            TypeError: If data is provided but is not a list.
            ValueError: If stddev <= 0 or data has fewer than 2 points.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            self.mean = float(sum(data) / n)
            variance = sum((x - self.mean) ** 2 for x in data) / n
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """      Calculate the z-score of a given x-value.

        Args:    x (float): The x-value.

        Returns: float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """      Calculate the x-value of a given z-score.

        Args:    z (float): The z-score.

        Returns: float: The x-value corresponding to the z-score.
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """      Calculate the value of the Probability Density Function.

        Args:    x (float): The x-value.

        Returns: float: PDF value for x.
        """
        pi = 3.1415926536
        e = 2.7182818285
        coefficient = 1.0 / (self.stddev * (2.0 * pi) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2.0 * self.stddev ** 2)
        return coefficient * (e ** exponent)

    def cdf(self, x):
        """      Calculate the value of the Cumulative Distribution Function.

        Args:    x (float): The x-value.

        Returns: float: CDF value for x.
        """
        pi = 3.1415926536
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        erf = (2 / pi ** 0.5) * (z - (z ** 3) / 3 + (z ** 5) / 10 - (z ** 7)
                                 / 42 + (z ** 9) / 216)

        return 0.5 * (1 + erf)
