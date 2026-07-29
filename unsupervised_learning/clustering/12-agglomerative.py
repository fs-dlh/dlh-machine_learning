#!/usr/bin/env python3
""" Module for K-means clustering on the dataset X. """
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage and cuts the dendrogram
    at a given cophenetic distance threshold.

    Parameters:
        X : (n, d) containing the dataset.
        dist : maximum cophenetic distance for all clusters.

    Returns:
        numpy.ndarray: containing cluster indices for each data point.
    """

    Z = sch.linkage(X, method='ward')

    clss = sch.fcluster(Z, t=dist, criterion='distance')

    plt.figure()
    sch.dendrogram(Z, color_threshold=dist)
    plt.show()

    return clss
