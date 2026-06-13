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
