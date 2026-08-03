#!/usr/bin/env python3
""" Module calculates the expectation step in the EM algorithm for a GMM. """
import numpy as np


def expectation(X, pi, m, S):
    """ Performs the expectation step for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data points.
        pi: numpy.ndarray of shape (k,) containing the prior probabilities.
        m: numpy.ndarray of shape (k, d) containing the centroid means.
        S: numpy.ndarray of shape (k, d, d) containing the covariance matrices.

    Returns:
        g: numpy.ndarray of shape (k, n) with posterior probabilities.
        lh: total log likelihood of the data.
        On failure, returns (None, None).
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None
    if not np.isclose(np.sum(pi), 1.0):
        return None, None

    pdf = __import__('5-pdf').pdf
    weighted = np.zeros((k, n))

    for j in range(k):
        P = pdf(X, m[j], S[j])
        if P is None:
            return None, None
        weighted[j] = pi[j] * P

    sum_weighted = np.sum(weighted, axis=0)
    if np.any(sum_weighted == 0):
        return None, None
    lh = np.sum(np.log(sum_weighted))
    g = weighted / sum_weighted

    return g, lh
