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
