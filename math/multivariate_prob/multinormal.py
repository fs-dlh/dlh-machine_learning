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
        Xcen = data - self.mean
        self.cov = (Xcen @ Xcen.T) / (n - 1)
        self.data = data

    def pdf(self, x):
        """ Calculate the probability density function. """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        
        diff = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        exponent = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
        coefficient = 1 / np.sqrt((2 * np.pi) ** len(x) * det_cov)
        return float(coefficient * np.exp(exponent))
