#!/usr/bin/env python3
""" Forward propagation with Dropout for a neural network. """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Perform forward propagation with Dropout.

    Args:
        X : input data of shape (nx, m).
        weights : weights and biases, keys 'W1','b1',...,'WL','bL'.
        L  number of layers.
        keep_prob : probability of keeping a node.

    Returns:
        dict: cache containing:
              - 'A0' to 'AL' (activations, /w dropout applied on hidden layers)
              - 'D1' to 'D{L-1}' (dropout masks for hidden layers)
    """
    cache = {'A0': X}

    for i in range(1, L):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(W, A_prev) + b
        A = np.tanh(Z)

        D = (np.random.rand(*A.shape) < keep_prob).astype(int)
        A = A * D / keep_prob

        cache['A' + str(i)] = A
        cache['D' + str(i)] = D

    W = weights['W' + str(L)]
    b = weights['b' + str(L)]
    A_prev = cache['A' + str(L - 1)]
    Z = np.matmul(W, A_prev) + b

    exp_Z = np.exp(Z)
    A_L = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    cache['A' + str(L)] = A_L

    return cache
