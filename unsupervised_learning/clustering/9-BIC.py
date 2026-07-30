#!/usr/bin/env python3
""" Module calculates the expectation maximization for a GMM. """
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ Finds the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray (n, d) containing the dataset.
        kmin: minimum number of clusters to check (inclusive).
        kmax: maximum number of clusters to check.
        iterations: maximum number of EM iterations.
        tol: tolerance for the EM algorithm.
        verbose: whether EM should print information.

    Returns:
        best_k: int, the best number of clusters based on BIC.
        best_result: tuple (pi, m, S) for the best model.
        l_list: numpy.ndarray (kmax-kmin+1) containing log‑likelihoods.
        b_list: numpy.ndarray (kmax-kmin+1) containing BIC values.
        Or (None, None, None, None) on failure.
    """

    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            type(kmin) is not int or kmin <= 0 or
            (kmax is not None and (type(kmax) is not int or kmax <= 0)) or
            type(iterations) is not int or iterations <= 0 or
            type(tol) is not float or tol < 0 or
            type(verbose) is not bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    if kmin >= kmax:
        return None, None, None, None

    l_list = []
    b_list = []
    best_results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None or ll is None:
            return None, None, None, None

        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)

        bic = (p * np.log(n)) - (2 * ll)

        b_list.append(bic)
        l_list.append(ll)

        best_results.append((pi, m, S))

    b_list = np.array(b_list)
    l_list = np.array(l_list)

    best_idx = np.argmin(b_list)
    best_k = kmin + best_idx
    best_result = best_results[best_idx]

    return best_k, best_result, l_list, b_list
