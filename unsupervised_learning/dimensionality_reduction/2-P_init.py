#!/usr/bin/env python3
""" 2-P_init module """
import numpy as np


def P_init(X, perplexity):
    """ Initializes variables for computing the P affinities in t-SNE.

    Args:
        X : shape (n, d), the dataset to be transformed.
        perplexity : target perplexity for the Gaussian distributions.

    Returns:
        D : squared pairwise distances (diagonal = 0).
        P : initialized to zeros.
        betas : all ones (beta = 1/(2*sigma^2)).
        H : Shannon entropy for the given perplexity (log2(perplexity)).
    """
    n, d = X.shape

    sum_X = np.sum(X ** 2, axis=1, keepdims=True)

    D = sum_X + sum_X.T - 2 * np.dot(X, X.T)

    np.fill_diagonal(D, 0.0)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return D, P, betas, H
