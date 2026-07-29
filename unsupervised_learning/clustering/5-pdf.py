#!/usr/bin/env python3
""" Module calculates the PDF of a Gaussian distribution. """
import numpy as np


def pdf(X, m, S):
    """ Calculates the PDF of a multivariate Gaussian distribution.

    Parameters:
    X : numpy.ndarray of shape (n, d)
    m : numpy.ndarray of shape (d,) Mean of the distribution.
    S : numpy.ndarray of shape (d, d) Covariance matrix.

    Returns:
    P : numpy.ndarray of shape (n,) PDF values for each data point
    None on failure.
    """
    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray):
        return None
    if not isinstance(S, np.ndarray):
        return None
    if X.ndim != 2 or m.ndim != 1 or S.ndim != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None

    try:
        detS = np.linalg.det(S)
        if detS <= 0:  # Covariance must be positive definite
            return None
        invS = np.linalg.inv(S)
        centered = X - m   # shape (n, d)
        quad = np.sum((centered @ invS) * centered, axis=1)   # shape (n,)
        norm = 1.0 / np.sqrt((2 * np.pi) ** d * detS)
        P = norm * np.exp(-0.5 * quad)
        P = np.maximum(P, 1e-300)
        return P

    except np.linalg.LinAlgError:
        return None
    except Exception:
        return None
