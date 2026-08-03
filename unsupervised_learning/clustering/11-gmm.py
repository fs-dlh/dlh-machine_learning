#!/usr/bin/env python3
""" Module for K-means clustering on the dataset X. """
import sklearn.mixture as skm


def gmm(X, k):
    """
    Fit a Gaussian Mixture Model with k components and return the parameters,
    cluster assignments, and BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: int, number of clusters.

    Returns:
        pi: numpy.ndarray (k,) containing the cluster priors (weights).
        m:  numpy.ndarray (k, d) containing the centroid means.
        S:  numpy.ndarray (k, d, d) containing the covariance matrices.
        clss: numpy.ndarray (n,) containing the cluster index for each point.
        bic: float, the Bayesian Information Criterion for the fitted model.
    """

    model = skm.GaussianMixture(n_components=k, covariance_type='full')
    model.fit(X)

    # Extract parameters
    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
