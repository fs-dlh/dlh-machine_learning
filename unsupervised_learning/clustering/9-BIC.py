#!/usr/bin/env python3
""" Module calculates the expectation maximization for a GMM. """
import numpy as np


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a Gaussian Mixture Model using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        kmin: positive int, minimum number of clusters to check (inclusive).
        kmax: positive int or None, maximum number of clusters to check.
              If None, set to the maximum possible (number of data points).
        iterations: positive int, maximum number of EM iterations.
        tol: non-negative float, tolerance for the EM algorithm.
        verbose: bool, whether EM should print information.

    Returns:
        best_k: int, the best number of clusters based on BIC.
        best_result: tuple (pi, m, S) for the best model.
        l: numpy.ndarray of shape (kmax-kmin+1) containing log‑likelihoods.
        b: numpy.ndarray of shape (kmax-kmin+1) containing BIC values.
        Or (None, None, None, None) on failure.
    """

    expectation_maximization = __import__('8-EM').expectation_maximization
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape
    if n == 0 or d == 0:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if kmax is None:
        kmax = n
    else:
        if not isinstance(kmax, int) or kmax < 1:
            return None, None, None, None

    if kmin > kmax:
        return None, None, None, None
    l_list = []
    b_list = []
    best_k = None
    best_result = None
    best_bic = float('inf')
    for k in range(kmin, kmax + 1):
        try:
            result = expectation_maximization(X, k, iterations, tol, verbose)
            if result is None:
                raise ValueError("EM returned None")

            if len(result) == 4:
                pi, m, S, logl = result
            elif len(result) == 5:
                pi, m, S, g, logl = result
            else:
                raise ValueError("Unexpected return length from EM")

        except Exception:
            return None, None, None, None
        p = (k - 1) + k * d + k * d * (d + 1) // 2

        bic = p * np.log(n) - 2.0 * logl

        l_list.append(logl)
        b_list.append(bic)
        if bic < best_bic or (bic == best_bic and best_k is None):
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    if best_k is None:
        return None, None, None, None

    return best_k, best_result, np.array(l_list), np.array(b_list)
