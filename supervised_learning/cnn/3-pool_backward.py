#!/usr/bin/env python3
""" Pooling backward propagation module. """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Performs BP over a pooling layer of a neural network.

    Args:
        dA: numpy.ndarray of shape (m, h_new, w_new, c) containing the
            partial derivatives with respect to the output of the pooling layer
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
            output of the previous layer.
        kernel_shape: tuple (kh, kw) containing the size of the pooling kernel.
        stride: tuple (sh, sw) containing the strides for the pooling.
        mode: string, either 'max' or 'avg', indicating pooling type.

    Returns:
        dA_prev: partial derivatives with respect to the previous layer.
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            h_start = h * sh
            h_end = h_start + kh
            for w in range(w_new):
                w_start = w * sw
                w_end = w_start + kw
                for ch in range(c):
                    if mode == 'max':
                        window = A_prev[i, h_start:h_end, w_start:w_end, ch]
                        max_val = np.max(window)
                        mask = (window == max_val)
                        dA_prev[i, h_start:h_end, w_start:w_end, ch] += \
                            dA[i, h, w, ch] * mask
                    elif mode == 'avg':
                        dA_prev[i, h_start:h_end, w_start:w_end, ch] += \
                            dA[i, h, w, ch] / (kh * kw)
                    else:
                        raise ValueError("mode must be 'max' or 'avg'")

    return dA_prev
