#!/usr/bin/env python3
""" Module calculates the maximization step in the EM algorithm for a GMM. """
import numpy as np


def maximization(X, g):
    """ Performs the maximization step of the EM algorithm for a GMM.

    Parameters:
    X : numpy.ndarray of shape (n, d) The dataset.
    g : numpy.ndarray of shape (k, n) Posterior probabilities.

    Returns:
    pi : numpy.ndarray of shape (k,) Updated priors for each cluster.
    m : numpy.ndarray of shape (k, d) Updated centroid means for each cluster.
    S : numpy.ndarray of shape (k, d, d) Updated covariance matrices.
    Returns (None, None, None) on failure.
    """

    if not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray):
        return None, None, None
    if X.ndim != 2 or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape
    if n != n_g:
        return None, None, None

    N_k = np.sum(g, axis=1)
    if np.any(N_k == 0):
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    pi = N_k / n

    m = (g @ X) / N_k[:, None]

    S = np.zeros((k, d, d))
    for i in range(k):
        centered = X - m[i]
        weighted_sum = (centered.T * g[i, :]) @ centered   # shape (d, d)
        S[i] = weighted_sum / N_k[i]

    return pi, m, S
