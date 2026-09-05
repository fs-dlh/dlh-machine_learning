#!/usr/bin/env python3
""" L2 regularization cost module for neural networks. """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculate the cost of a neural network with L2 regularization.

    The L2 regularization penalty is added to the provided cost:
    L2_penalty = (lambtha / (2 * m)) * sum_{l=1}^{L} ||W_l||_F^2

    Args:
        cost : cost of the network without L2 regularization.
        lambtha : regularization parameter.
        weights : dictionary containing the weights and biases of the network.
                    Only weights (keys 'W1', 'W2', ..., 'WL') are used.
        L : number of layers in the neural network.
        m : number of data points used.

    Returns:
        float or numpy.ndarray: the cost of the network accounting for
                                L2 regularization, same type as `cost`.
    """

    l2_penalty = 0.0
    for i in range(1, L + 1):
        w = weights['W' + str(i)]
        l2_penalty += np.sum(w ** 2)

    total_cost = cost + (lambtha / (2 * m)) * l2_penalty
    return total_cost
