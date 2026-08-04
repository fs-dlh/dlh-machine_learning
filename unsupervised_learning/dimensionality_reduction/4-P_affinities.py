#!/usr/bin/env python3
""" that calculates the symmetric P affinities of a data set  """
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """ Calculates the symmetric P affinities of a data set for t-SNE.

    Args:
        X: numpy.ndarray containing the dataset
        tol: maximum tolerance for the difference in Shannon entropy from
         perplexity for all Gaussian distributions
        perplexity: perplexity that all Gaussian distributions should have

    Returns:
        P: numpy.ndarray containing the symmetric P affinities
    """
    n = X.shape[0]

    D, _, _, _ = P_init(X, perplexity)

    target_H = np.log2(perplexity)

    P_cond = np.zeros((n, n))

    for i in range(n):
        beta = 1.0
        low = None
        high = None

        Di = np.delete(D[i, :], i)
        H, p_i = HP(Di, beta)

        p = np.insert(p_i, i, 0.0)

        if np.abs(H - target_H) <= tol:
            P_cond[i, :] = p
            continue

        if H > target_H:
            low = beta
            while True:
                beta *= 2
                H, _ = HP(Di, beta)
                if H <= target_H:
                    high = beta
                    break
                low = beta
        else:
            high = beta
            while True:
                beta /= 2
                H, _ = HP(Di, beta)
                if H >= target_H:
                    low = beta
                    break
                high = beta

        while True:
            beta = (low + high) / 2
            H, p_i = HP(Di, beta)
            p = np.insert(p_i, i, 0.0)
            if np.abs(H - target_H) <= tol:
                P_cond[i, :] = p
                break
            if H > target_H:
                low = beta
            else:
                high = beta

    # p_ij = (p_{j|i} + p_{i|j}) / (2 * n)
    P = (P_cond + P_cond.T) / (2 * n)

    return P
