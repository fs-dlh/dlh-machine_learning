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
        l: float log likelihood of the model.
        Or None, None, None, None, None on failure.
    """
    try:
        # Import required functions
        initialize = __import__('4-initialize').initialize
        expectation = __import__('6-expectation').expectation
        maximization = __import__('7-maximization').maximization
    except Exception:
        return None, None, None, None, None

    try:
        pi, m, S = initialize(X, k)
    except Exception:
        return None, None, None, None, None

    n, d = X.shape
    prev_l = None
    converged = False

    for i in range(iterations):
        # Expectation step
        try:
            g, lh = expectation(X, pi, m, S)
        except Exception:
            return None, None, None, None, None

        converged = False
        if prev_l is not None:
            # Using absolute difference is a best practice here
            if abs(lh - prev_l) <= tol:
                converged = True

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {lh:.5f}")

        if converged:
            return pi, m, S, g, lh

        try:
            pi, m, S = maximization(X, g)
        except Exception:
            return None, None, None, None, None

        prev_l = lh

    try:
        g, lh = expectation(X, pi, m, S)
    except Exception:
        return None, None, None, None, None

    return pi, m, S, g, lh
