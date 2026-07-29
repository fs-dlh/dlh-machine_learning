#!/usr/bin/env python3
""" Module for K-means clustering on the dataset X. """
import sklearn.cluster as skc


def kmeans(X, k):
    """ Performs K-means clustering on the dataset X.

    Args:
        X : shape (n, d) containing the dataset.
        k : number of clusters.

    Returns:
        C : shape (k, d) containing centroid means.
        clss : shape (n,) containing cluster indices for each point.
    """
    model = skc.KMeans(n_clusters=k).fit(X)
    return model.cluster_centers_, model.labels_
