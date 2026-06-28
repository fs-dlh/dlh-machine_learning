#!/usr/bin/env python3
"""Module for defining the MultiNormal class for multivariate  distribution."""

import numpy as np


class MultiNormal:
    """ Class for multivariate normal distribution. """

    def __init__(self, data):
        """ Initialize """

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        center = data - self.mean
        self.cov = (center @ center.T) / (n - 1)
        self.data = data
        