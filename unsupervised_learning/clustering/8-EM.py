#!/usr/bin/env python3
""" Module calculates the expectation maximization for a GMM. """
import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ Performs the expectation maximization for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer number of clusters.
        iterations: positive integer maximum number of iterations.
        tol: non-negative float tolerance for log likelihood difference.
        verbose: boolean; if True print log likelihood every 10 iterations
                 and after the last iteration.

    Returns:
        pi: numpy.ndarray of shape (k,) priors for each cluster.
        m: numpy.ndarray of shape (k, d) centroid means.
        S: numpy.ndarray of shape (k, d, d) covariance matrices.
        g: numpy.ndarray of shape (k, n) posterior probabilities.
        lh: float log likelihood of the model.
        Or None, None, None, None, None on failure.
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    try:
        # Import required functions
        initialize = __import__('4-initialize').initialize
        expectation = __import__('6-expectation').expectation
        maximization = __import__('7-maximization').maximization
    except Exception:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    prev_l = None

    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print("Log Likelihood after {} iterations: {:.5f}"
                  .format(i, l))

        if prev_l is not None and abs(l - prev_l) <= tol:
            return pi, m, S, g, l

        if i == iterations:
            break

        prev_l = l

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

    g, l = expectation(X, pi, m, S)
    if pi is None or m is None or S is None or g is None or lh is None:
        return None, None, None, None, None

    return pi, m, S, g, l
