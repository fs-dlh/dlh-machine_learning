#!/usr/bin/env python3
""" Gradient descent with L2 regularization for a neural network. """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Update weights and biases using gradient descent with L2 regularization

    The network has L layers, with tanh activation on hidden layers and softmax
    on the output layer. The weights are regularized with L2 penalty.

    Args:
        Y : one-hot labels of shape (classes, m).
        weights : contains 'W1', 'b1', ..., 'WL', 'bL' (numpy.ndarrays).
        cache : contains 'A0', 'A1', ..., 'AL' (numpy.ndarrays),
                      where A0 is the input and AL is the softmax output.
        alpha : learning rate.
        lambtha : L2 regularization parameter.
        L : number of layers.
    """

    m = Y.shape[1]

    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - np.power(A_prev, 2))

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
