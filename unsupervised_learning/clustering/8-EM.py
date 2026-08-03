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
        initialize = __import__('4-initialize').initialize
        expectation = __import__('6-expectation').expectation
        maximization = __import__('7-maximization').maximization
    except Exception:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, l_prev = expectation(X, pi, m, S)
    if g is None or l_prev is None:
        return None, None, None, None, None

    if verbose:
        print(f"Log Likelihood after 0 iterations: {round(l_prev, 5)}")

    i = 0
    while i < iterations:
        i += 1

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, lh = expectation(X, pi, m, S)
        if g is None or lh is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations):
            print(f"Log Likelihood after {i} iterations: {round(lh, 5)}")

        if abs(lh - l_prev) <= tol:
            if verbose and i % 10 != 0 and i != iterations:
                print(f"Log Likelihood after {i} iterations: {round(lh, 5)}")
            return pi, m, S, g, lh

        l_prev = lh

    return pi, m, S, g, l_prev
