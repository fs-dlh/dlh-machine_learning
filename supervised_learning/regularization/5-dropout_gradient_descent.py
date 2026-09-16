#!/usr/bin/env python3
""" Gradient descent with Dropout regularization for a neural network. """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Update weights using gradient descent with Dropout.

    Args:
        Y : one‑hot labels of shape (classes, m).
        weights : contains 'W1','b1',...,'WL','bL'.
        cache : from dropout_forward_prop, contains 'A0'..'AL'
                      and 'D1'..'D{L-1}'.
        alpha : learning rate.
        keep_prob : probability of keeping a node.
        L : number of layers.
    """

    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA_prev = np.matmul(W.T, dZ) * cache['D' + str(i - 1)] / keep_prob
            dZ = dA_prev * (1 - np.power(A_prev, 2))

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
